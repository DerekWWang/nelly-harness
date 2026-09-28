#![cfg(unix)]

use nelly_harness::{
    layered::LayeredExecutor,
    layers::{MemoryConfig, MemoryLayers},
    memory::{MemorableCli, MemoryStore},
    model::{AudioChunk, ModelEvent, VoiceModel},
    session::{ToolExecutor, VoiceSession},
    Harness, ToolCall,
};
use serde_json::{json, Value};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Sandbox(PathBuf);

impl Sandbox {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "nelly-layered-executor-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn store(&self) -> MemoryStore {
        MemoryStore::open(self.0.join("episodes")).unwrap()
    }

    fn executor(&self, script: &str) -> LayeredExecutor<Harness> {
        let path = self.0.join("memorable");
        fs::write(
            &path,
            format!(
                "#!/bin/sh\nset -eu\nif test \"$1\" = status; then printf '%s' 'write consent: read-write'; exit 0; fi\n{script}\n"
            ),
        )
        .unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        let cli = MemorableCli {
            program: path.into_os_string(),
            prefix_args: vec![],
            timeout: Duration::from_secs(3),
            max_output_bytes: 4096,
        };
        let memory = MemoryLayers::new(self.store(), cli, MemoryConfig::default()).unwrap();
        LayeredExecutor::new(Harness::default(), Some(memory))
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn await_ready(executor: &mut LayeredExecutor<Harness>, mut reply: Value) -> Value {
    let deadline = Instant::now() + Duration::from_secs(3);
    while reply["status"] == "pending" {
        assert!(Instant::now() < deadline, "memory job did not finish");
        std::thread::sleep(Duration::from_millis(5));
        let job_id = reply["job_id"].as_u64().unwrap();
        reply = executor
            .execute(&ToolCall::MemorablePoll { job_id }, false)
            .unwrap()
            .value
            .as_ref()
            .clone();
    }
    assert_eq!(reply["status"], "ready", "{reply}");
    reply
}

#[test]
fn disabled_memory_preserves_app_tools_and_rejects_all_speculative_writes() {
    let mut executor = LayeredExecutor::new(Harness::default(), None);
    let put = ToolCall::NotesPut {
        topic: "work".into(),
        name: "agenda".into(),
        note: "Design review".into(),
    };
    assert_eq!(
        executor.execute(&put, true).unwrap_err(),
        "speculative writes are forbidden"
    );
    executor.execute(&put, false).unwrap();
    assert_eq!(executor.revision(), 1);
    assert_eq!(executor.inner().revision(), 1);
    for call in [
        ToolCall::MemorableRecall {
            query: "work".into(),
        },
        ToolCall::MemorableShow {
            slug: "procedures/work".into(),
        },
        ToolCall::MemorableChain {
            query: "work".into(),
        },
        ToolCall::MemorableList { all: false },
        ToolCall::MemorableStatus,
        ToolCall::MemorablePoll { job_id: 1 },
        ToolCall::MemorableIngest {
            episode_id: "work".into(),
        },
        ToolCall::MemorableInvalidate,
        ToolCall::MemorableDisable,
    ] {
        assert!(executor
            .execute(&call, false)
            .unwrap_err()
            .contains("--memorable"));
        if !call.is_read() {
            assert_eq!(
                executor.execute(&call, true).unwrap_err(),
                "speculative writes are forbidden"
            );
        }
    }
    let result = executor
        .execute(
            &ToolCall::NotesGet {
                topic: "work".into(),
                name: "agenda".into(),
            },
            true,
        )
        .unwrap();
    assert_eq!(result.value["note"], "Design review");
    assert_eq!(result.revision, 1);
    assert!(executor.memory_mut().is_none());
    executor.shutdown().unwrap();
}

#[test]
fn notes_and_calendar_remain_available_while_memorable_is_blocked() {
    let sandbox = Sandbox::new();
    let mut executor = sandbox.executor(
        r#"touch "${0}.started"
while test ! -e "${0}.release"; do sleep 0.01; done
printf '%s' 'procedures/work'"#,
    );
    let pending = executor
        .execute(
            &ToolCall::MemorableRecall {
                query: "work".into(),
            },
            true,
        )
        .unwrap();
    assert_eq!(pending.value["status"], "pending");
    assert!(executor
        .execute(&ToolCall::MemorablePoll { job_id: u64::MAX }, false)
        .unwrap_err()
        .contains("unknown"));
    let deadline = Instant::now() + Duration::from_secs(2);
    while !sandbox.0.join("memorable.started").exists() {
        assert!(Instant::now() < deadline, "fake memory CLI never started");
        std::thread::sleep(Duration::from_millis(5));
    }

    let started = Instant::now();
    executor
        .execute(
            &ToolCall::NotesPut {
                topic: "work".into(),
                name: "agenda".into(),
                note: "Local result".into(),
            },
            false,
        )
        .unwrap();
    for _ in 0..256 {
        let result = executor
            .execute(
                &ToolCall::NotesGet {
                    topic: "work".into(),
                    name: "agenda".into(),
                },
                true,
            )
            .unwrap();
        assert_eq!(result.value["note"], "Local result");
        assert_eq!(result.revision, 1);
        assert!(executor
            .execute(
                &ToolCall::ScheduleIsFree {
                    start_minute: 0,
                    end_minute: 60,
                },
                true,
            )
            .unwrap()
            .value["free"]
            .as_bool()
            .unwrap());
    }
    // The worker is still gated, so this is a deadlock/regression guard, not a
    // microbenchmark. Application work must complete well before its deadline.
    assert!(started.elapsed() < Duration::from_secs(1));
    let invalidated = executor
        .execute(&ToolCall::MemorableInvalidate, false)
        .unwrap();
    fs::write(sandbox.0.join("memorable.release"), b"").unwrap();
    let completed = await_ready(&mut executor, pending.value.as_ref().clone());
    assert_eq!(completed["status"], "ready");
    assert_eq!(completed["generation"], pending.value["generation"]);
    assert_ne!(completed["generation"], invalidated.value["generation"]);
    let fresh = executor
        .execute(
            &ToolCall::MemorableRecall {
                query: "work".into(),
            },
            false,
        )
        .unwrap();
    assert!(!fresh.cached);
    assert_eq!(fresh.value["generation"], invalidated.value["generation"]);
    assert_eq!(executor.revision(), 1);
    executor.shutdown().unwrap();
}

#[derive(Default)]
struct MemoryModel {
    phase: u8,
    job_id: u64,
    ready: bool,
    observations: Vec<(Value, Value)>,
}

impl VoiceModel for MemoryModel {
    fn sample_rate_hz(&self) -> u32 {
        16_000
    }

    fn chunk_samples(&self) -> usize {
        1
    }

    fn infer(&mut self, _audio: AudioChunk<'_>) -> Result<Vec<ModelEvent>, String> {
        let event = match self.phase {
            0 => {
                self.phase = 1;
                ModelEvent::Prefetch {
                    call: json!({"tool": "memorable_recall", "query": "plan a review"}),
                }
            }
            1 if self.ready => {
                self.phase = 2;
                self.ready = false;
                ModelEvent::ToolCall {
                    call: json!({"tool": "memorable_show", "slug": "procedures/review"}),
                }
            }
            2 if self.ready => {
                self.phase = 3;
                return Ok(vec![
                    ModelEvent::Prefetch {
                        call: json!({"tool": "memorable_ingest", "episode_id": "session"}),
                    },
                    ModelEvent::Prefetch {
                        call: json!({"tool": "memorable_invalidate"}),
                    },
                    ModelEvent::Prefetch {
                        call: json!({"tool": "memorable_disable"}),
                    },
                ]);
            }
            1 | 2 => ModelEvent::ToolCall {
                call: json!({"tool": "memorable_poll", "job_id": self.job_id}),
            },
            _ => ModelEvent::Reply {
                text: "Memory loaded".into(),
            },
        };
        Ok(vec![event])
    }

    fn observe_tool_result(&mut self, call: &Value, result: &Value) -> Result<bool, String> {
        if result["ok"] == true {
            self.job_id = result["result"]["job_id"].as_u64().unwrap();
            self.ready = result["result"]["status"] == "ready";
        }
        self.observations.push((call.clone(), result.clone()));
        Ok(true)
    }

    fn reset(&mut self) -> Result<(), String> {
        *self = Self::default();
        Ok(())
    }
}

#[test]
fn generated_memory_calls_feed_the_model_and_speculative_ingest_never_runs() {
    let sandbox = Sandbox::new();
    let executor = sandbox.executor(
        r#"printf '%s\n' "$1" >> "${0}.calls"
sleep 0.02
case "$1" in
recall) printf '%s' 'procedures/review' ;;
show) printf '%s' 'Read the agenda before the design review.' ;;
*) exit 99 ;;
esac"#,
    );
    let mut session = VoiceSession::new(MemoryModel::default(), executor);
    let deadline = Instant::now() + Duration::from_secs(3);
    while session.model().phase != 3 {
        assert!(
            Instant::now() < deadline,
            "voice model never received memory"
        );
        session
            .process(AudioChunk {
                sample_rate_hz: 16_000,
                pcm: &[0.0],
            })
            .unwrap();
        std::thread::sleep(Duration::from_millis(5));
    }
    let observations = &session.model().observations;
    assert!(observations.iter().any(|(call, result)| {
        call["tool"] == "memorable_recall" && result["result"]["status"] == "pending"
    }));
    assert!(observations.iter().any(|(call, result)| {
        call["tool"] == "memorable_show" && result["result"]["status"] == "pending"
    }));
    assert!(observations.iter().any(|(call, result)| {
        call["tool"] == "memorable_poll" && result["result"]["status"] == "ready"
    }));
    let (_, retrieved) = observations
        .iter()
        .find(|(_, result)| {
            result["result"]["data"]["text"] == "Read the agenda before the design review."
        })
        .unwrap();
    assert_eq!(retrieved["result"]["data"]["source"], "memorable");
    assert_eq!(retrieved["result"]["data"]["trusted"], false);
    for tool in [
        "memorable_ingest",
        "memorable_invalidate",
        "memorable_disable",
    ] {
        let (_, result) = observations
            .iter()
            .find(|(call, _)| call["tool"] == tool)
            .unwrap();
        assert_eq!(result["ok"], false);
        assert_eq!(result["error"], "speculative writes are forbidden");
    }
    assert_eq!(session.engine().revision(), 0);
    let (_, executor) = session.into_parts();
    executor.shutdown().unwrap();
    assert_eq!(
        fs::read_to_string(sandbox.0.join("memorable.calls")).unwrap(),
        "recall\nshow\n"
    );
}

#[test]
fn completed_ingest_invalidates_prefetched_recall_without_mutating_core_revision() {
    let sandbox = Sandbox::new();
    let store = sandbox.store();
    store.begin("finished", "Remember a design review").unwrap();
    store.finish("finished", "Review completed").unwrap();
    let mut executor = sandbox.executor(
        r#"case "$1" in
recall) printf '%s' 'procedures/review' ;;
ingest) test -f "$2"; printf '%s' 'stored' ;;
*) exit 99 ;;
esac"#,
    );
    let recall = ToolCall::MemorableRecall {
        query: "review".into(),
    };
    let initial = executor.execute(&recall, true).unwrap();
    let ready = await_ready(&mut executor, initial.value.as_ref().clone());
    let generation = ready["generation"].as_u64().unwrap();
    let cached = executor.execute(&recall, false).unwrap();
    assert!(cached.cached);
    assert_eq!(cached.value["status"], "ready");

    let ingest = executor
        .execute(
            &ToolCall::MemorableIngest {
                episode_id: "finished".into(),
            },
            false,
        )
        .unwrap();
    await_ready(&mut executor, ingest.value.as_ref().clone());
    let fresh = executor.execute(&recall, false).unwrap();
    assert!(!fresh.cached);
    assert!(fresh.value["generation"].as_u64().unwrap() > generation);
    assert_eq!(fresh.revision, 0);
    let ready = await_ready(&mut executor, fresh.value.as_ref().clone());
    let invalidated = executor
        .execute(&ToolCall::MemorableInvalidate, false)
        .unwrap();
    assert_eq!(invalidated.value["invalidated"], true);
    assert_eq!(
        invalidated.value["generation"].as_u64().unwrap(),
        ready["generation"].as_u64().unwrap() + 1
    );
    assert!(!executor.execute(&recall, true).unwrap().cached);
    assert_eq!(executor.revision(), 0);
    executor.shutdown().unwrap();
}

#[test]
fn disabling_memory_discards_cached_content_and_keeps_app_tools_available() {
    let sandbox = Sandbox::new();
    let mut executor = sandbox.executor("printf '%s' 'procedures/review'");
    let recall = ToolCall::MemorableRecall {
        query: "review".into(),
    };
    let pending = executor.execute(&recall, true).unwrap();
    let ready = await_ready(&mut executor, pending.value.as_ref().clone());
    assert!(executor.execute(&recall, false).unwrap().cached);
    assert_eq!(
        executor
            .execute(&ToolCall::MemorableDisable, true)
            .unwrap_err(),
        "speculative writes are forbidden"
    );
    assert!(executor.memory_mut().unwrap().is_enabled());
    let disabled = executor
        .execute(&ToolCall::MemorableDisable, false)
        .unwrap();
    assert_eq!(disabled.value["enabled"], false);
    assert_eq!(disabled.revision, 0);
    assert!(!executor.memory_mut().unwrap().is_enabled());
    assert!(executor.execute(&recall, false).is_err());
    assert!(executor
        .execute(
            &ToolCall::MemorablePoll {
                job_id: ready["job_id"].as_u64().unwrap(),
            },
            false,
        )
        .is_err());
    executor
        .execute(
            &ToolCall::NotesPut {
                topic: "local".into(),
                name: "still available".into(),
                note: "yes".into(),
            },
            false,
        )
        .unwrap();
    assert_eq!(executor.revision(), 1);
    executor.shutdown().unwrap();
}
