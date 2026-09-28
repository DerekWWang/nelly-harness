#![cfg(unix)]

use nelly_harness::layers::{JobStatus, MemorableRequest, MemoryConfig, MemoryLayers, MemoryReply};
use nelly_harness::memory::{MemorableCli, MemoryStore};
use std::fs;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Sandbox(PathBuf);

impl Sandbox {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "nelly-layers-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn store(&self) -> MemoryStore {
        MemoryStore::open(self.0.join("episodes")).unwrap()
    }

    fn cli(&self) -> MemorableCli {
        let path = self.0.join("fake-cli");
        fs::write(
            &path,
            r##"#!/bin/sh
base=$(dirname "$0")
if [ "$1" = status ]; then
  if [ -f "$base/consent" ]; then
    cat "$base/consent"
  else
    printf 'write consent: read-write'
  fi
  exit 0
fi
printf '%s:%s\n' "$1" "$2" >> "$base/calls"
case "$2" in
  blocked*)
    touch "$base/started"
    while [ ! -f "$base/release" ]; do sleep 0.01; done
    ;;
  fail)
    printf 'fake failure: externally supplied content' >&2
    exit 7
    ;;
  oversized)
    printf '%0600d' 0
    exit 0
    ;;
esac
case "$1" in
  ingest) printf 'CLI acknowledged; no workflow extracted' ;;
  chain)
    case "$2" in
      malformed) printf 'not JSON' ;;
      wrong-shape) printf '{}' ;;
      *) printf '{"items":[]}' ;;
    esac
    ;;
  list)
    if [ -f "$base/bad-list" ]; then printf '{}'; else printf '[]'; fi
    ;;
  *) printf '{"operation":"%s","query":"%s"}\n' "$1" "$2" ;;
esac
"##,
        )
        .unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        MemorableCli {
            program: path.into_os_string(),
            prefix_args: vec![],
            timeout: Duration::from_secs(3),
            max_output_bytes: 1024,
        }
    }

    fn layers(&self, config: MemoryConfig) -> MemoryLayers {
        MemoryLayers::new(self.store(), self.cli(), config).unwrap()
    }

    fn release(&self) {
        fs::write(self.0.join("release"), b"ready").unwrap();
    }

    fn wait_started(&self) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !self.0.join("started").exists() {
            assert!(Instant::now() < deadline, "fake CLI did not start");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn calls(&self) -> usize {
        fs::read_to_string(self.0.join("calls"))
            .unwrap_or_default()
            .lines()
            .count()
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn recall(query: &str) -> MemorableRequest {
    MemorableRequest::Recall {
        query: query.into(),
    }
}

fn wait(layers: &mut MemoryLayers, id: u64) -> MemoryReply {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let reply = layers.poll(id).unwrap();
        if reply.status != JobStatus::Pending {
            return reply;
        }
        assert!(Instant::now() < deadline, "memory job did not finish");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn pending_reads_are_nonblocking_single_flight_and_cache_shares_the_result() {
    let sandbox = Sandbox::new();
    let mut layers = sandbox.layers(MemoryConfig::default());
    let first = layers.submit(recall("blocked")).unwrap();
    assert_eq!(first.status, JobStatus::Pending);
    sandbox.wait_started();
    let start = Instant::now();
    let duplicate = layers.submit(recall("blocked")).unwrap();
    assert_eq!(duplicate.job_id, first.job_id);
    assert_eq!(
        layers.poll(first.job_id).unwrap().status,
        JobStatus::Pending
    );
    assert!(start.elapsed() < Duration::from_millis(500));
    sandbox.release();
    let finished = wait(&mut layers, first.job_id);
    assert_eq!(finished.status, JobStatus::Ready);
    let data = finished.data.unwrap();
    assert_eq!(data["trusted"], false);
    assert_eq!(data["source"], "memorable");
    assert_eq!(data["json"]["query"], "blocked");
    let hit = layers.submit(recall("blocked")).unwrap();
    assert!(hit.cached);
    assert_eq!(hit.job_id, first.job_id);
    assert!(std::sync::Arc::ptr_eq(&data, &hit.data.unwrap()));
    assert_eq!(sandbox.calls(), 1);
    layers.shutdown().unwrap();
}

#[test]
fn queue_rejects_overflow_and_shutdown_drains_full_completion_channel() {
    let sandbox = Sandbox::new();
    let mut layers = sandbox.layers(MemoryConfig {
        queue_capacity: 1,
        ..MemoryConfig::default()
    });
    layers.submit(recall("blocked-first")).unwrap();
    sandbox.wait_started();
    layers.submit(recall("second")).unwrap();
    assert_eq!(
        layers.submit(recall("third")).unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
    assert_eq!(layers.stats()["pending"], 2);
    sandbox.release();
    // No polling: the shutdown barrier itself must empty the completion queue.
    layers.shutdown().unwrap();
    assert_eq!(sandbox.calls(), 2);
}

#[test]
fn pending_jobs_are_never_evicted_to_make_room() {
    let sandbox = Sandbox::new();
    let mut layers = sandbox.layers(MemoryConfig {
        max_jobs: 1,
        max_cache_entries: 1,
        ..MemoryConfig::default()
    });
    let accepted = layers.submit(recall("blocked")).unwrap();
    assert_eq!(
        layers.submit(recall("new")).unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
    assert_eq!(
        layers.poll(accepted.job_id).unwrap().status,
        JobStatus::Pending
    );
    sandbox.release();
    wait(&mut layers, accepted.job_id);
    let next = layers.submit(recall("new")).unwrap();
    assert_eq!(
        layers.poll(accepted.job_id).unwrap_err().kind(),
        io::ErrorKind::NotFound
    );
    wait(&mut layers, next.job_id);
    layers.shutdown().unwrap();
}

#[test]
fn ttl_and_fifo_limits_prevent_unbounded_cache_and_retained_jobs() {
    let sandbox = Sandbox::new();
    let mut layers = sandbox.layers(MemoryConfig {
        ttl: Duration::from_millis(40),
        max_jobs: 2,
        max_cache_entries: 1,
        ..MemoryConfig::default()
    });
    let one = layers.submit(recall("one")).unwrap();
    wait(&mut layers, one.job_id);
    assert!(layers.submit(recall("one")).unwrap().cached);
    std::thread::sleep(Duration::from_millis(65));
    let two = layers.submit(recall("one")).unwrap();
    assert!(!two.cached);
    assert_ne!(two.job_id, one.job_id);
    wait(&mut layers, two.job_id);
    let three = layers.submit(recall("three")).unwrap();
    wait(&mut layers, three.job_id);
    assert_eq!(layers.stats()["jobs"], 2);
    assert_eq!(layers.stats()["cache_entries"], 1);
    assert_eq!(
        layers.poll(one.job_id).unwrap_err().kind(),
        io::ErrorKind::NotFound
    );
    let four = layers.submit(recall("one")).unwrap();
    assert!(!four.cached);
    wait(&mut layers, four.job_id);
    layers.shutdown().unwrap();
}

#[test]
fn result_byte_pressure_evicts_completed_data_and_oversized_results_fail() {
    let sandbox = Sandbox::new();
    let mut layers = sandbox.layers(MemoryConfig {
        max_result_bytes: 256,
        ..MemoryConfig::default()
    });
    let first = layers.submit(recall(&"a".repeat(90))).unwrap();
    assert_eq!(wait(&mut layers, first.job_id).status, JobStatus::Ready);
    let second = layers.submit(recall(&"b".repeat(90))).unwrap();
    assert_eq!(wait(&mut layers, second.job_id).status, JobStatus::Ready);
    assert_eq!(
        layers.poll(first.job_id).unwrap_err().kind(),
        io::ErrorKind::NotFound
    );
    let stats = layers.stats();
    assert!(stats["result_bytes"].as_u64().unwrap() <= 256);
    let oversized = layers.submit(recall("oversized")).unwrap();
    assert_eq!(
        wait(&mut layers, oversized.job_id).status,
        JobStatus::Failed
    );
    assert!(layers.stats()["result_bytes"].as_u64().unwrap() <= 256);
    assert!(layers.shutdown().is_err());
}

#[test]
fn failures_are_pollable_but_never_become_read_hits() {
    let sandbox = Sandbox::new();
    let mut layers = sandbox.layers(MemoryConfig::default());
    let failed = layers.submit(recall("fail")).unwrap();
    let reply = wait(&mut layers, failed.job_id);
    assert_eq!(reply.status, JobStatus::Failed);
    assert!(reply.error.unwrap().contains("fake failure"));
    assert!(reply.data.is_none());
    let retry = layers.submit(recall("fail")).unwrap();
    assert!(!retry.cached);
    assert_ne!(failed.job_id, retry.job_id);
    assert_eq!(wait(&mut layers, retry.job_id).status, JobStatus::Failed);
    assert_eq!(sandbox.calls(), 2);
    assert_eq!(layers.stats()["cache_entries"], 0);
    assert!(layers.shutdown().is_err());
}

#[test]
fn invalidation_during_pending_read_prevents_stale_single_flight_and_cache_fill() {
    let sandbox = Sandbox::new();
    let mut layers = sandbox.layers(MemoryConfig::default());
    let old = layers.submit(recall("blocked")).unwrap();
    sandbox.wait_started();
    let generation = layers.invalidate();
    let new = layers.submit(recall("blocked")).unwrap();
    assert_ne!(new.job_id, old.job_id);
    assert_eq!(new.generation, generation);
    sandbox.release();
    let old_reply = wait(&mut layers, old.job_id);
    assert_eq!(old_reply.generation, 0);
    let new_reply = wait(&mut layers, new.job_id);
    assert_eq!(new_reply.generation, generation);
    let hit = layers.submit(recall("blocked")).unwrap();
    assert!(hit.cached);
    assert_eq!(hit.job_id, new.job_id);
    assert_eq!(sandbox.calls(), 2);
    layers.shutdown().unwrap();
}

#[test]
fn finished_ingest_acknowledges_invalidates_and_remains_idempotent() {
    let sandbox = Sandbox::new();
    let store = sandbox.store();
    store.begin("episode-1", "Capture a task").unwrap();
    store
        .finish("episode-1", "Long-form episode information")
        .unwrap();
    let mut layers = MemoryLayers::new(store, sandbox.cli(), MemoryConfig::default()).unwrap();
    let read = layers.submit(recall("prior")).unwrap();
    wait(&mut layers, read.job_id);
    let request = MemorableRequest::Ingest {
        episode_id: "episode-1".into(),
    };
    let ingest = layers.submit(request.clone()).unwrap();
    let reply = wait(&mut layers, ingest.job_id);
    assert_eq!(reply.status, JobStatus::Ready);
    assert_eq!(reply.generation, 1);
    assert_eq!(layers.stats()["cache_entries"], 0);
    let data = reply.data.unwrap();
    assert_eq!(data["acknowledged"], true);
    assert_eq!(data["already_synced"], false);
    assert!(data["workflow_stored"].is_null());
    let again = layers.submit(request).unwrap();
    assert!(!again.cached);
    assert_eq!(
        wait(&mut layers, again.job_id).data.unwrap()["already_synced"],
        true
    );
    assert_eq!(sandbox.calls(), 2); // One recall and exactly one CLI ingest.
    let fresh = layers.submit(recall("prior")).unwrap();
    assert!(!fresh.cached);
    wait(&mut layers, fresh.job_id);
    layers.shutdown().unwrap();
}

#[test]
fn status_is_never_cached_and_other_reads_are_wrapped_as_untrusted_data() {
    let sandbox = Sandbox::new();
    let mut layers = sandbox.layers(MemoryConfig::default());
    for request in [
        MemorableRequest::Chain {
            query: "a chain".into(),
        },
        MemorableRequest::Show {
            slug: "procedures/a-note".into(),
        },
        MemorableRequest::List { all: true },
        MemorableRequest::Status,
    ] {
        let pending = layers.submit(request.clone()).unwrap();
        let reply = wait(&mut layers, pending.job_id);
        assert_eq!(reply.status, JobStatus::Ready);
        assert_eq!(reply.data.unwrap()["trusted"], false);
        let next = layers.submit(request.clone()).unwrap();
        if request == MemorableRequest::Status {
            assert!(!next.cached);
            assert_ne!(next.job_id, pending.job_id);
            assert_eq!(
                wait(&mut layers, next.job_id).data.unwrap()["text"],
                "write consent: read-write"
            );
        } else {
            assert!(next.cached);
        }
    }
    assert_eq!(sandbox.calls(), 3);
    layers.shutdown().unwrap();
}

#[test]
fn invalid_requests_are_rejected_before_any_cli_is_enqueued() {
    let sandbox = Sandbox::new();
    let mut layers = sandbox.layers(MemoryConfig::default());
    for request in [
        recall(" "),
        recall("-x"),
        recall("bad\0query"),
        recall(&"x".repeat(16385)),
        MemorableRequest::Chain { query: "".into() },
        MemorableRequest::Show {
            slug: "procedures/../other".into(),
        },
        MemorableRequest::Show {
            slug: "procedures/".into(),
        },
        MemorableRequest::Ingest {
            episode_id: "../episode".into(),
        },
    ] {
        assert_eq!(
            layers.submit(request).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }
    assert_eq!(layers.stats()["jobs"], 0);
    layers.shutdown().unwrap();
    assert_eq!(sandbox.calls(), 0);
}

#[test]
fn disabling_removes_retained_results_and_rejects_pending_and_future_access() {
    let sandbox = Sandbox::new();
    let mut layers = sandbox.layers(MemoryConfig::default());
    let cached = layers.submit(recall("cached")).unwrap();
    wait(&mut layers, cached.job_id);
    let running = layers.submit(recall("blocked")).unwrap();
    sandbox.wait_started();
    let queued = layers.submit(recall("must-not-run")).unwrap();
    layers.disable();
    assert!(!layers.is_enabled());
    assert_eq!(layers.stats()["cache_entries"], 0);
    assert_eq!(
        layers.poll(cached.job_id).unwrap_err().kind(),
        io::ErrorKind::NotFound
    );
    for id in [running.job_id, queued.job_id] {
        let reply = layers.poll(id).unwrap();
        assert_eq!(reply.status, JobStatus::Failed);
        assert!(reply.data.is_none());
        assert!(reply.error.unwrap().contains("disabled"));
    }
    assert_eq!(
        layers.submit(recall("future")).unwrap_err().kind(),
        io::ErrorKind::PermissionDenied
    );
    sandbox.release();
    assert!(layers.shutdown().is_err());
    assert_eq!(sandbox.calls(), 2); // Cached read plus the already-running read.
}

#[test]
fn backend_read_denial_revokes_cached_and_retained_results() {
    let sandbox = Sandbox::new();
    let mut layers = sandbox.layers(MemoryConfig::default());
    let old = layers.submit(recall("previously-allowed")).unwrap();
    wait(&mut layers, old.job_id);
    fs::write(sandbox.0.join("consent"), "write consent: deny").unwrap();
    let denied = layers.submit(recall("fresh-read")).unwrap();
    assert_eq!(wait(&mut layers, denied.job_id).status, JobStatus::Failed);
    assert!(!layers.is_enabled());
    assert_eq!(layers.stats()["cache_entries"], 0);
    assert_eq!(
        layers.poll(old.job_id).unwrap_err().kind(),
        io::ErrorKind::NotFound
    );
    assert_eq!(sandbox.calls(), 1); // Consent failure happened before retrieval.
    assert!(layers.shutdown().is_err());
}

#[test]
fn fresh_status_revokes_prior_cache_when_provider_consent_changes() {
    let sandbox = Sandbox::new();
    let mut layers = sandbox.layers(MemoryConfig::default());
    let old = layers.submit(recall("cached-before-status")).unwrap();
    wait(&mut layers, old.job_id);
    fs::write(sandbox.0.join("consent"), "  write consent  unset\n").unwrap();
    let status = layers.submit(MemorableRequest::Status).unwrap();
    assert_eq!(wait(&mut layers, status.job_id).status, JobStatus::Failed);
    assert!(!layers.is_enabled());
    assert_eq!(
        layers.poll(old.job_id).unwrap_err().kind(),
        io::ErrorKind::NotFound
    );
    assert_eq!(layers.stats()["cache_entries"], 0);
    assert!(layers.shutdown().is_err());
}

#[test]
fn read_only_consent_rejects_ingest_without_revoking_reads() {
    let sandbox = Sandbox::new();
    let store = sandbox.store();
    store.begin("read-only", "A task").unwrap();
    store.finish("read-only", "An episode").unwrap();
    fs::write(sandbox.0.join("consent"), "write consent: read-only").unwrap();
    let mut layers = MemoryLayers::new(store, sandbox.cli(), MemoryConfig::default()).unwrap();
    let write = layers
        .submit(MemorableRequest::Ingest {
            episode_id: "read-only".into(),
        })
        .unwrap();
    assert_eq!(wait(&mut layers, write.job_id).status, JobStatus::Failed);
    assert!(layers.is_enabled());
    let read = layers.submit(recall("allowed")).unwrap();
    assert_eq!(wait(&mut layers, read.job_id).status, JobStatus::Ready);
    assert_eq!(sandbox.calls(), 1);
    assert!(layers.shutdown().is_err());
}

#[test]
fn declared_json_operations_reject_malformed_and_wrong_shape_results() {
    let sandbox = Sandbox::new();
    let mut layers = sandbox.layers(MemoryConfig::default());
    for query in ["malformed", "wrong-shape"] {
        let request = MemorableRequest::Chain {
            query: query.into(),
        };
        let job = layers.submit(request.clone()).unwrap();
        assert_eq!(wait(&mut layers, job.job_id).status, JobStatus::Failed);
        let again = layers.submit(request).unwrap();
        assert!(!again.cached);
        assert_eq!(wait(&mut layers, again.job_id).status, JobStatus::Failed);
    }
    fs::write(sandbox.0.join("bad-list"), "bad").unwrap();
    let list = layers
        .submit(MemorableRequest::List { all: false })
        .unwrap();
    assert_eq!(wait(&mut layers, list.job_id).status, JobStatus::Failed);
    assert_eq!(layers.stats()["cache_entries"], 0);
    assert!(layers.shutdown().is_err());
}
