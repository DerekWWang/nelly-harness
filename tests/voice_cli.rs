#![cfg(feature = "onnx")]

//! End-to-end tests that load actual ONNX weights in the Rust CLI. Run with:
//! ORT_DYLIB_PATH=/absolute/path/to/runtime cargo test --features onnx --test voice_cli -- --ignored

use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

struct TempDirectory(PathBuf);

impl TempDirectory {
    fn new() -> Self {
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "nelly-voice-cli-test-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/model-fixture")
}

fn json_lines(bytes: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(bytes)
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn run_voice(actions: Option<Value>) -> (Vec<Value>, Vec<Value>) {
    assert!(
        std::env::var_os("ORT_DYLIB_PATH").is_some(),
        "set ORT_DYLIB_PATH to ONNX Runtime 1.24; see MODEL.md"
    );
    let directory = TempDirectory::new();
    let source = directory.0.join("source.wav");
    let manifest_path = directory.0.join("model.json");
    let data = directory.0.join("data");
    let original_audio = fs::read(fixture().join("audio.wav")).unwrap();
    fs::write(&source, &original_audio).unwrap();
    let mut manifest: Value =
        serde_json::from_slice(&fs::read(fixture().join("model.json")).unwrap()).unwrap();
    manifest["model_path"] = json!(fixture().join("weights.onnx"));
    if let Some(actions) = actions {
        manifest["actions"] = actions;
    }
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_nelly-harness"))
        .arg("voice")
        .arg(&manifest_path)
        .arg(&source)
        .arg("--data")
        .arg(&data)
        .args([
            "--episode",
            "test-session",
            "--task",
            "Local ONNX integration",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "voice CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Training export must depend on the episode-owned audio, not the original
    // source path or a symlink that stops working when that source disappears.
    fs::remove_file(&source).unwrap();
    let export = Command::new(env!("CARGO_BIN_EXE_nelly-harness"))
        .args(["memory", "export", "test-session", "--data"])
        .arg(&data)
        .output()
        .unwrap();
    assert!(
        export.status.success(),
        "memory export failed: {}",
        String::from_utf8_lossy(&export.stderr)
    );
    let rows = json_lines(&export.stdout);
    assert!(!rows.is_empty());
    assert!(rows[0]["episode"]["finished_at_ms"].is_u64());
    assert_eq!(rows[0]["step"]["action"]["name"], "audio_observation");
    let attachment = &rows[0]["step"]["audio"][0];
    let stored_path = data
        .join("episodes")
        .join(attachment["path"].as_str().unwrap());
    assert_eq!(fs::read(&stored_path).unwrap(), original_audio);
    assert!(!fs::symlink_metadata(&stored_path)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(attachment["role"], "observation");
    for row in &rows[1..] {
        assert_eq!(row["step"]["pre_state"]["audio"][0], *attachment);
        assert_eq!(row["step"]["post_state"]["audio"][0], *attachment);
    }
    (json_lines(&output.stdout), rows)
}

#[test]
#[ignore = "requires ONNX Runtime 1.24 via ORT_DYLIB_PATH"]
fn native_weights_prefetch_and_export_owned_audio_with_state_action_times() {
    let (events, rows) = run_voice(None);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["event"], "tool_result");
    assert_eq!(events[0]["call"]["tool"], "notes_get");
    assert_eq!(events[0]["speculative"], true);
    assert_eq!(events[0]["result"]["ok"], true);
    assert_eq!(events[0]["observation_consumed"], false);
    assert_eq!(events[0]["revision_before"], 0);
    assert_eq!(events[0]["revision_after"], 0);
    assert_eq!(events[0]["at_ms"], 40);
    assert_eq!(rows.len(), 2);
    let step = &rows[1]["step"];
    assert_eq!(step["action"]["name"], "notes_get");
    assert_eq!(
        step["action"]["input"],
        json!({"topic": "demo", "name": "welcome"})
    );
    assert_eq!(step["started_ms"], 20);
    assert_eq!(step["ended_ms"], 40);
    assert_eq!(step["pre_state"]["revision"], 0);
    assert_eq!(step["post_state"]["revision"], 0);
    assert_eq!(step["result"]["ok"], true);
}

#[test]
#[ignore = "requires ONNX Runtime 1.24 via ORT_DYLIB_PATH"]
fn reply_only_model_still_records_audio_and_reply_action() {
    let (events, rows) = run_voice(Some(json!([null, {"event": "reply", "text": "Ready"}])));
    assert_eq!(
        events,
        vec![json!({"event": "reply", "text": "Ready", "at_ms": 40})]
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(
        rows[1]["step"]["action"],
        json!({"name": "reply", "input": {"text": "Ready"}})
    );
    assert_eq!(rows[1]["step"]["result"]["generated"], true);
    assert_eq!(rows[1]["step"]["started_ms"], 20);
    assert_eq!(rows[1]["step"]["ended_ms"], 40);
}

#[test]
#[ignore = "requires ONNX Runtime 1.24 via ORT_DYLIB_PATH"]
fn silent_model_still_records_an_owned_audio_episode() {
    let (events, rows) = run_voice(Some(json!([null, null])));
    assert!(events.is_empty());
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["episode"]["step_count"], 1);
}
