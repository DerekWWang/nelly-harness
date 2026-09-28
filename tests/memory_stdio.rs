#![cfg(unix)]

use nelly_harness::{memory::MemoryStore, persistence::MAX_REQUEST_BYTES};
use serde_json::{json, Value};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, ExitStatus, Stdio},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError},
    },
    thread::JoinHandle,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Sandbox(PathBuf);

impl Sandbox {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "nelly-memory-stdio-{}-{}-{}",
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

    fn cli(&self, script: &str) -> PathBuf {
        let path = self.0.join("memorable");
        fs::write(
            &path,
            format!(
                "#!/bin/sh\nset -eu\nif test \"$1\" = status; then printf '%s' 'write consent: read-write'; exit 0; fi\n{script}\n"
            ),
        )
        .unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    fn data(&self) -> PathBuf {
        self.0.join("data")
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Server {
    child: Option<Child>,
    input: Option<ChildStdin>,
    output: Receiver<String>,
    reader: Option<JoinHandle<()>>,
    next_id: u64,
}

impl Server {
    fn start(sandbox: &Sandbox, cli: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_nelly-harness"))
            .arg("serve")
            .arg("--data")
            .arg(sandbox.data())
            .arg("--memorable")
            .arg("--memorable-bin")
            .arg(cli)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let input = child.stdin.take();
        let stdout = child.stdout.take().unwrap();
        let (sender, output) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        Self {
            child: Some(child),
            input,
            output,
            reader: Some(reader),
            next_id: 1,
        }
    }

    fn request(&mut self, call: Value, speculative: bool) -> Value {
        let request_id = self.next_id;
        self.next_id += 1;
        let request = json!({"request_id": request_id, "call": call, "speculative": speculative});
        writeln!(self.input.as_mut().unwrap(), "{request}").unwrap();
        let line = self
            .output
            .recv_timeout(Duration::from_secs(2))
            .expect("serve did not answer while its memory worker was pending");
        let response: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(response["request_id"], request_id);
        response
    }

    fn finish(mut self) -> (ExitStatus, String) {
        drop(self.input.take());
        let deadline = Instant::now() + Duration::from_secs(5);
        let status = loop {
            if let Some(status) = self.child.as_mut().unwrap().try_wait().unwrap() {
                break status;
            }
            assert!(Instant::now() < deadline, "serve did not drain and exit");
            std::thread::sleep(Duration::from_millis(5));
        };
        let mut stderr = String::new();
        self.child
            .as_mut()
            .unwrap()
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut stderr)
            .unwrap();
        self.reader.take().unwrap().join().unwrap();
        self.child.take();
        (status, stderr)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        drop(self.input.take());
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn await_file(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while !path.exists() {
        assert!(
            Instant::now() < deadline,
            "missing marker {}",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

// The gate times out independently of the harness, including if a test fails
// and drops the parent process before its Memorable subprocess finishes.
const GATE: &str = r#"
touch "${0}.started"
attempt=0
while test ! -e "${0}.release"; do
    attempt=$((attempt + 1))
    test "$attempt" -lt 300 || exit 98
    sleep 0.01
done
"#;

#[test]
fn streaming_client_polls_prefetch_then_reuses_cache_and_disables_memory() {
    let sandbox = Sandbox::new();
    let cli = sandbox.cli(&format!(
        "test \"$1\" = recall\n{GATE}\nprintf '%s' 'procedures/review'\ntouch \"${{0}}.completed\""
    ));
    let mut server = Server::start(&sandbox, &cli);
    let recall = json!({"tool": "memorable_recall", "query": "review"});
    let pending = server.request(recall.clone(), true);
    assert_eq!(pending["ok"], true);
    assert_eq!(pending["result"]["status"], "pending");
    let job_id = pending["result"]["job_id"].as_u64().unwrap();
    await_file(&sandbox.0.join("memorable.started"));

    let write = server.request(
        json!({"tool": "notes_put", "topic": "work", "name": "agenda", "note": "Local note"}),
        false,
    );
    assert_eq!(write["ok"], true);
    assert_eq!(write["revision"], 1);
    let read = json!({"tool": "notes_get", "topic": "work", "name": "agenda"});
    assert_eq!(
        server.request(read.clone(), true)["result"]["note"],
        "Local note"
    );
    let poll = json!({"tool": "memorable_poll", "job_id": job_id});
    assert_eq!(
        server.request(poll.clone(), false)["result"]["status"],
        "pending"
    );

    fs::write(sandbox.0.join("memorable.release"), b"").unwrap();
    await_file(&sandbox.0.join("memorable.completed"));
    assert!(matches!(
        server.output.recv_timeout(Duration::from_millis(50)),
        Err(RecvTimeoutError::Timeout)
    ));
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let result = server.request(poll.clone(), false);
        if result["result"]["status"] == "ready" {
            assert_eq!(result["result"]["data"]["text"], "procedures/review");
            assert_eq!(result["result"]["data"]["trusted"], false);
            assert_eq!(result["revision"], 1);
            break;
        }
        assert!(Instant::now() < deadline);
        assert_eq!(result["result"]["status"], "pending");
        std::thread::sleep(Duration::from_millis(5));
    }
    let cached = server.request(recall.clone(), false);
    assert_eq!(cached["cached"], true);
    assert_eq!(cached["result"]["status"], "ready");
    assert_eq!(cached["result"]["job_id"], job_id);
    let disabled = server.request(json!({"tool": "memorable_disable"}), false);
    assert_eq!(disabled["result"]["enabled"], false);
    let denied = server.request(recall, false);
    assert_eq!(denied["ok"], false);
    assert!(denied["error"].as_str().unwrap().contains("disabled"));
    assert_eq!(server.request(read, false)["result"]["note"], "Local note");
    let (status, stderr) = server.finish();
    assert!(status.success(), "{stderr}");
}

#[test]
fn oversized_input_drains_accepted_ingest_before_reporting_protocol_failure() {
    let sandbox = Sandbox::new();
    let store = MemoryStore::open(sandbox.data().join("episodes")).unwrap();
    store.begin("finished", "A completed local task").unwrap();
    store.finish("finished", "Task completed").unwrap();
    let cli = sandbox.cli(&format!(
        "test \"$1\" = ingest\ntest -f \"$2\"\n{GATE}\nprintf '%s' 'acknowledged'\ntouch \"${{0}}.ingested\""
    ));
    let mut server = Server::start(&sandbox, &cli);
    let pending = server.request(
        json!({"tool": "memorable_ingest", "episode_id": "finished"}),
        false,
    );
    assert_eq!(pending["result"]["status"], "pending");
    await_file(&sandbox.0.join("memorable.started"));
    let mut oversized = vec![b'x'; MAX_REQUEST_BYTES + 1];
    oversized.push(b'\n');
    server
        .input
        .as_mut()
        .unwrap()
        .write_all(&oversized)
        .unwrap();
    drop(server.input.take());
    std::thread::sleep(Duration::from_millis(50));
    assert!(server.child.as_mut().unwrap().try_wait().unwrap().is_none());
    fs::write(sandbox.0.join("memorable.release"), b"").unwrap();
    let (status, stderr) = server.finish();
    assert!(!status.success());
    assert!(stderr.contains("line exceeds byte limit"), "{stderr}");
    assert!(sandbox.0.join("memorable.ingested").exists());
    let marker: Value = serde_json::from_slice(
        &fs::read(store.root().join("finished/memorable-synced.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(marker["stdout"], "acknowledged");
    assert_eq!(
        store.metadata("finished").unwrap().summary.as_deref(),
        Some("Task completed")
    );
}

#[test]
fn memory_command_routes_chain_list_status_and_layer_description() {
    let sandbox = Sandbox::new();
    let cli = sandbox.cli(
        r#"case "$1" in
chain)
    test "$2" = 'review notes'
    test "$3" = --json
    printf '%s' '{"segments":["review notes"],"items":[],"coverage":0,"score":0}'
    ;;
list)
    test "$2" = --json
    test "$3" = --all
    printf '%s' '[]'
    ;;
*) exit 99 ;;
esac"#,
    );
    for (args, expected) in [
        (
            vec!["chain", "review notes"],
            json!({"segments":["review notes"],"items":[],"coverage":0,"score":0}),
        ),
        (vec!["list", "--all"], json!([])),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_nelly-harness"))
            .arg("memory")
            .args(args)
            .arg("--memorable-bin")
            .arg(&cli)
            .current_dir(&sandbox.0)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&output.stdout).unwrap(),
            expected
        );
    }
    let status = Command::new(env!("CARGO_BIN_EXE_nelly-harness"))
        .args(["memory", "status", "--memorable-bin"])
        .arg(&cli)
        .current_dir(&sandbox.0)
        .output()
        .unwrap();
    assert!(status.status.success());
    assert_eq!(status.stdout, b"write consent: read-write");
    let layers = Command::new(env!("CARGO_BIN_EXE_nelly-harness"))
        .args(["memory", "layers"])
        .current_dir(&sandbox.0)
        .output()
        .unwrap();
    assert!(layers.status.success());
    let layers: Value = serde_json::from_slice(&layers.stdout).unwrap();
    assert_eq!(layers["layers"].as_array().unwrap().len(), 4);
    assert_eq!(layers["provider_checked"], false);
    assert!(!sandbox.0.join("data").exists());
}
