use serde_json::{json, Value};
use std::{
    io::Write,
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
fn run(args: &[&str], requests: &[Value]) -> Vec<Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_nelly-harness"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    for request in requests {
        writeln!(input, "{request}").unwrap();
    }
    drop(input);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}
fn request(id: u64, speculative: bool, call: Value) -> Value {
    json!({"request_id":id,"speculative":speculative,"call":call})
}

#[test]
fn stdio_prefetch_crud_and_restart() {
    let directory = std::env::temp_dir().join(format!(
        "nelly-stdio-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let path = directory.to_str().unwrap();
    let read = json!({"tool":"notes_get","topic":"people","name":"Ada"});
    let put = json!({"tool":"notes_put","topic":"people","name":"Ada","note":"tea"});
    let rows = run(
        &["serve", "--data", path],
        &[
            request(0, true, read.clone()),
            request(1, false, read.clone()),
            request(2, true, put.clone()),
            request(3, false, put),
            request(4, false, read.clone()),
            request(
                5,
                false,
                json!({"tool":"schedule_create","title":"call","start_minute":100,"end_minute":130}),
            ),
            request(
                6,
                true,
                json!({"tool":"schedule_is_free","start_minute":120,"end_minute":150}),
            ),
        ],
    );
    assert_eq!(rows[0]["result"]["note"], Value::Null);
    assert_eq!(rows[1]["cached"], true);
    assert_eq!(rows[2]["ok"], false);
    assert_eq!(rows[3]["revision"], 1);
    assert_eq!(rows[4]["result"]["note"], "tea");
    assert_eq!(rows[4]["cached"], false);
    assert_eq!(rows[6]["result"]["free"], false);
    let id = rows[5]["result"]["id"].as_u64().unwrap();
    let rows = run(
        &["serve", "--data", path],
        &[
            request(7, false, read),
            request(8, false, json!({"tool":"schedule_delete","id":id})),
            request(
                9,
                false,
                json!({"tool":"schedule_is_free","start_minute":120,"end_minute":150}),
            ),
            request(
                10,
                false,
                json!({"tool":"notes_delete","topic":"people","name":"Ada"}),
            ),
        ],
    );
    assert_eq!(rows[0]["request_id"], 7);
    assert_eq!(rows[0]["result"]["note"], "tea");
    assert_eq!(rows[0]["revision"], 2);
    assert_eq!(rows[2]["result"]["free"], true);
    assert_eq!(rows[3]["revision"], 4);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn unknown_tools_and_fields_return_errors_without_stopping_stream() {
    let rows = run(
        &["serve", "--ephemeral"],
        &[
            request(
                1,
                false,
                json!({"tool":"shell","command":"touch unexpected"}),
            ),
            json!({"call":{"tool":"stats"},"extra":1}),
            request(3, false, json!({"tool":"stats"})),
        ],
    );
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["ok"], false);
    assert_eq!(rows[1]["ok"], false);
    assert_eq!(rows[2]["ok"], true);
    assert_eq!(rows[2]["revision"], 0);
}

#[test]
fn all_function_schemas_include_read_classification() {
    let rows = run(&["tools"], &[]);
    let tools = rows[0].as_array().unwrap();
    assert_eq!(tools.len(), 21);
    for tool in tools {
        assert!(tool["read_only"].is_boolean());
        assert!(tool["function"]["name"].is_string());
        assert_eq!(
            tool["function"]["parameters"]["additionalProperties"],
            false
        );
    }
}
