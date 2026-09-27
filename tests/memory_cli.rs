use nelly_harness::memory::{
    Action, AudioSource, BackgroundRecorder, MemorableCli, MemoryStore, StepInput,
};
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Sandbox(PathBuf);
impl Sandbox {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "nelly-memory-{}-{}-{}",
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
    #[cfg(unix)]
    fn cli(&self, script: &str) -> MemorableCli {
        use std::os::unix::fs::PermissionsExt;
        let path = self
            .0
            .join(format!("fake-cli-{}", NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::write(&path, format!("#!/bin/sh\n{script}\n")).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        MemorableCli {
            program: path.into_os_string(),
            prefix_args: vec![],
            timeout: Duration::from_secs(3),
            max_output_bytes: 1024,
        }
    }
}
impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn step(started_ms: u64) -> StepInput {
    StepInput {
        pre_state: json!({"topic":"garden","note":null}),
        action: Action {
            name: "notes.create".into(),
            input: json!({"topic":"garden","body":"water basil"}),
        },
        result: Some(json!({"created":true})),
        post_state: json!({"note":"water basil"}),
        started_ms,
        ended_ms: started_ms + 10,
        audio: vec![],
    }
}

#[test]
fn durable_audio_pair_export_and_official_trace_schema() {
    let sandbox = Sandbox::new();
    let store = sandbox.store();
    store.begin("session-1", "Remember to water basil").unwrap();
    let source = sandbox.0.join("speech.wav");
    fs::write(&source, b"fixture PCM bytes").unwrap();
    let mut input = step(0);
    input.audio.push(AudioSource {
        path: source.clone(),
        role: "observation".into(),
        media_type: "audio/wav".into(),
        sample_rate_hz: Some(24_000),
        channels: Some(1),
    });
    let saved = store.append("session-1", input).unwrap();
    fs::remove_file(source).unwrap();
    assert_eq!(
        fs::read(store.root().join(&saved.audio[0].path)).unwrap(),
        b"fixture PCM bytes"
    );
    let reopened = sandbox.store();
    assert_eq!(reopened.append("session-1", step(10)).unwrap().sequence, 1);
    let metadata = reopened
        .finish(
            "session-1",
            "The user planted basil and wants a regular reminder.",
        )
        .unwrap();
    assert_eq!(metadata.step_count, 2);
    assert!(reopened.append("session-1", step(20)).is_err());
    let mut exported = Vec::new();
    assert_eq!(
        reopened.export_jsonl("session-1", &mut exported).unwrap(),
        2
    );
    let rows: Vec<Value> = std::str::from_utf8(&exported)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows[0]["step"]["pre_state"]["topic"], "garden");
    assert_eq!(rows[0]["step"]["audio"][0]["sample_rate_hz"], 24_000);
    assert_eq!(rows[1]["step"]["sequence"], 1);
    assert_eq!(rows[0]["episode"]["summary"], metadata.summary.unwrap());
    let trace: Value = serde_json::from_slice(
        &fs::read(store.root().join("session-1/memorable-trace.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(trace["session_id"], "session-1");
    assert_eq!(trace["harness"], "nelly-rust");
    assert_eq!(
        trace["tool_calls"][0],
        json!({"name":"notes.create", "input":{"topic":"garden","body":"water basil"}, "result":{"created":true}})
    );
    assert!(trace.get("audio").is_none());
}

#[test]
fn failed_audio_copy_rolls_back_and_retry_keeps_sequence() {
    let sandbox = Sandbox::new();
    let store = sandbox.store();
    store.begin("retry", "A task").unwrap();
    let source = sandbox.0.join("audio.pcm");
    fs::write(&source, [1, 2, 3]).unwrap();
    let audio = AudioSource {
        path: source,
        role: "observation".into(),
        media_type: "audio/pcm".into(),
        sample_rate_hz: None,
        channels: None,
    };
    let mut input = step(0);
    input.audio.push(audio.clone());
    input.audio.push(AudioSource {
        path: sandbox.0.join("missing"),
        ..audio
    });
    assert!(store.append("retry", input).is_err());
    assert_eq!(
        fs::read_dir(store.root().join("retry/audio"))
            .unwrap()
            .count(),
        0
    );
    assert_eq!(store.append("retry", step(0)).unwrap().sequence, 0);
}

#[test]
fn crash_tail_is_recovered_and_unknown_results_stay_unknown() {
    let sandbox = Sandbox::new();
    let store = sandbox.store();
    store.begin("crashed", "A task").unwrap();
    store.append("crashed", step(0)).unwrap();
    OpenOptions::new()
        .append(true)
        .open(store.root().join("crashed/steps.jsonl"))
        .unwrap()
        .write_all(b"{\"partial\":")
        .unwrap();
    let mut unknown = step(10);
    unknown.result = None;
    assert_eq!(
        sandbox.store().append("crashed", unknown).unwrap().sequence,
        1
    );
    store.finish("crashed", "No guessed outcome").unwrap();
    let trace: Value = serde_json::from_slice(
        &fs::read(store.root().join("crashed/memorable-trace.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(trace["tool_calls"].as_array().unwrap().len(), 2);
    assert!(trace["tool_calls"][1].get("result").is_none());
}

#[test]
fn rejects_path_traversal_oversized_steps_and_reversed_timing() {
    let sandbox = Sandbox::new();
    let store = sandbox.store();
    for id in ["../escape", "/tmp/escape", "x/y", ".", ""] {
        assert!(store.begin(id, "task").is_err());
    }
    store.begin("valid", "task").unwrap();
    let mut huge = step(0);
    huge.pre_state = json!("x".repeat(1_048_576));
    assert!(store.append("valid", huge).is_err());
    store.append("valid", step(5)).unwrap();
    // Multiple actions can share an observation interval.
    store.append("valid", step(5)).unwrap();
    let mut reversed = step(20);
    reversed.ended_ms = 19;
    assert!(store.append("valid", reversed).is_err());
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&sandbox.0, store.root().join("link")).unwrap();
        assert!(store.metadata("link").is_err());
    }
}

#[test]
#[cfg(unix)]
fn sync_failure_retains_episode_and_success_is_idempotent() {
    let sandbox = Sandbox::new();
    let store = sandbox.store();
    store.begin("sync", "task").unwrap();
    store.append("sync", step(0)).unwrap();
    store.finish("sync", "finished").unwrap();
    let failed = sandbox.cli("printf 'consent is disabled' >&2; exit 9");
    assert!(store
        .sync("sync", &failed)
        .unwrap_err()
        .to_string()
        .contains("consent is disabled"));
    assert!(!store.root().join("sync/memorable-synced.json").exists());
    assert!(store.root().join("sync/steps.jsonl").exists());
    let successful = sandbox.cli(
        "test \"$1\" = ingest || exit 2\ntest -f \"$2\" || exit 3\nprintf 'stored procedure'\n",
    );
    assert!(!store.sync("sync", &successful).unwrap().already_synced);
    // If the marker works, the failing CLI is never invoked again.
    assert!(store.sync("sync", &failed).unwrap().already_synced);
}

#[test]
#[cfg(unix)]
fn cli_has_bounded_output_deadline_and_literal_arguments() {
    let sandbox = Sandbox::new();
    let cli = sandbox.cli("printf '%s' \"$2\"");
    assert_eq!(
        cli.recall("basil; $(exit 77)").unwrap().stdout,
        "basil; $(exit 77)"
    );
    assert!(cli.recall("--help").is_err());
    let noisy = sandbox.cli("while :; do printf 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'; done");
    assert!(noisy
        .recall("task")
        .unwrap_err()
        .to_string()
        .contains("output limit"));
    let mut slow = sandbox.cli("sleep 10");
    slow.timeout = Duration::from_millis(75);
    let start = Instant::now();
    assert_eq!(
        slow.recall("task").unwrap_err().kind(),
        std::io::ErrorKind::TimedOut
    );
    assert!(start.elapsed() < Duration::from_secs(2));
}

#[test]
fn background_worker_flushes_and_preserves_storage_errors() {
    let sandbox = Sandbox::new();
    let store = sandbox.store();
    store.begin("background", "task").unwrap();
    let recorder = BackgroundRecorder::start(store.clone(), "background", 2).unwrap();
    recorder.try_append(step(0)).unwrap();
    recorder.try_append(step(10)).unwrap();
    recorder.flush().unwrap();
    assert_eq!(
        fs::read_to_string(store.root().join("background/steps.jsonl"))
            .unwrap()
            .lines()
            .count(),
        2
    );
    assert_eq!(
        recorder.finish("persisted on worker").unwrap().step_count,
        2
    );
    assert!(BackgroundRecorder::start(store.clone(), "background", 2).is_err());
    store.begin("worker-failure", "task").unwrap();
    let recorder = BackgroundRecorder::start(store.clone(), "worker-failure", 2).unwrap();
    let mut invalid = step(0);
    invalid.audio.push(AudioSource {
        path: sandbox.0.join("not-a-file"),
        role: "observation".into(),
        media_type: "audio/wav".into(),
        sample_rate_hz: None,
        channels: None,
    });
    recorder.try_append(invalid).unwrap();
    assert!(recorder.flush().is_err());
    assert!(recorder.check_error().is_err());
    assert!(recorder.try_append(step(10)).is_err());
    assert!(recorder.finish("must fail").is_err());
    assert!(store
        .metadata("worker-failure")
        .unwrap()
        .finished_at_ms
        .is_none());
}

#[test]
#[cfg(unix)]
fn sync_rejects_symlinked_controls_and_copy_rejects_fifo() {
    let sandbox = Sandbox::new();
    let store = sandbox.store();
    store.begin("links", "task").unwrap();
    store.finish("links", "finished").unwrap();
    let outside = sandbox.0.join("outside.json");
    fs::write(&outside, b"{}").unwrap();
    for name in ["memorable-trace.json", "memorable-synced.json", ".lock"] {
        let path = store.root().join("links").join(name);
        if path.exists() {
            fs::remove_file(&path).unwrap();
        }
        std::os::unix::fs::symlink(&outside, &path).unwrap();
        let cli = sandbox.cli("exit 88");
        assert_eq!(
            store.sync("links", &cli).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        fs::remove_file(path).unwrap();
    }
    store.begin("fifo", "task").unwrap();
    let fifo = sandbox.0.join("audio.fifo");
    assert!(std::process::Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .unwrap()
        .success());
    let mut input = step(0);
    input.audio.push(AudioSource {
        path: fifo,
        role: "observation".into(),
        media_type: "audio/pcm".into(),
        sample_rate_hz: None,
        channels: None,
    });
    assert_eq!(
        store.append("fifo", input).unwrap_err().kind(),
        std::io::ErrorKind::InvalidInput
    );
}

#[test]
#[cfg(unix)]
fn timeout_terminates_grandchildren_before_they_can_write() {
    let sandbox = Sandbox::new();
    let mut cli = sandbox.cli("(sleep 0.4; printf escaped > \"${0}.survived\") &\nwait");
    cli.timeout = Duration::from_millis(50);
    let mut marker = cli.program.clone();
    marker.push(".survived");
    assert_eq!(
        cli.recall("task").unwrap_err().kind(),
        std::io::ErrorKind::TimedOut
    );
    std::thread::sleep(Duration::from_millis(450));
    assert!(!PathBuf::from(marker).exists());
}
