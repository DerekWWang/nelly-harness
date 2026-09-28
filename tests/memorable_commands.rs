#![cfg(unix)]

use nelly_harness::memory::MemorableCli;
use serde_json::{json, Value};
use std::fs;
use std::io::ErrorKind;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Sandbox(PathBuf);

impl Sandbox {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "nelly-memorable-commands-{}-{}-{}",
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

    fn cli(&self, script: &str) -> MemorableCli {
        let path = self.0.join("memorable");
        fs::write(&path, format!("#!/bin/sh\nset -eu\n{script}\n")).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        MemorableCli {
            program: path.into_os_string(),
            prefix_args: vec![],
            timeout: Duration::from_secs(2),
            max_output_bytes: 4096,
        }
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn chain_uses_structured_output_and_keeps_query_literal() {
    let sandbox = Sandbox::new();
    let cli = sandbox.cli(
        r#"if test "$1" = status; then printf 'write consent: read-write\n'; exit 0; fi
test "$#" = 3
test "$1" = chain
test "$2" = 'create a note; $(exit 99) and then book a meeting'
test "$3" = --json
printf '%s' '{"segments":["create a note","book a meeting"],"items":[{"kind":"gap","segment":"book a meeting"}],"coverage":0.5,"score":1.2}'"#,
    );
    let output = cli
        .chain("create a note; $(exit 99) and then book a meeting")
        .unwrap();
    let plan: Value = serde_json::from_str(&output.stdout).unwrap();
    assert_eq!(plan["coverage"], 0.5);
    assert_eq!(plan["items"][0]["kind"], "gap");
}

#[test]
fn list_and_status_follow_pinned_cli_output_contract() {
    let sandbox = Sandbox::new();
    let cli = sandbox.cli(
        r#"case "$1" in
list)
    test "$2" = --json
    if test "$#" = 3; then test "$3" = --all; else test "$#" = 2; fi
    printf '%s' '[{"intent":"note","preferred":"procedures/note","revisions":[{"slug":"procedures/note","revision":1,"title":"Note","verified":true,"stale":null,"recalled":2,"ok":1,"fail":0}]}]'
    ;;
status)
    test "$#" = 1
    printf '%s' 'write consent  read-only'
    ;;
*) exit 90 ;;
esac"#,
    );
    for all in [false, true] {
        let output = cli.list(all).unwrap();
        let list: Value = serde_json::from_str(&output.stdout).unwrap();
        assert_eq!(list[0]["preferred"], json!("procedures/note"));
    }
    assert_eq!(cli.status().unwrap().stdout, "write consent  read-only");
}

#[test]
fn chain_rejects_empty_oversized_and_option_like_queries_before_spawning() {
    let cli = MemorableCli {
        program: "/nonexistent/never-execute".into(),
        prefix_args: vec![],
        ..MemorableCli::default()
    };
    for query in ["", "--render", "-query", &"x".repeat(16 * 1024 + 1)] {
        assert_eq!(
            cli.chain(query).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
}

#[test]
fn new_commands_preserve_common_process_bounds_and_failures() {
    let sandbox = Sandbox::new();
    let mut cli = sandbox.cli("printf 'denied by backend' >&2\nexit 3");
    for result in [cli.chain("task"), cli.list(false), cli.status()] {
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("denied by backend"));
    }
    cli.timeout = Duration::ZERO;
    assert_eq!(cli.status().unwrap_err().kind(), ErrorKind::InvalidInput);
    cli.timeout = Duration::from_secs(2);
    cli.max_output_bytes = 0;
    assert_eq!(cli.list(true).unwrap_err().kind(), ErrorKind::InvalidInput);
}

#[test]
fn retrieval_fails_closed_when_provider_consent_is_missing_or_denied() {
    for mode in ["deny", "unset", "changed-format"] {
        let sandbox = Sandbox::new();
        let cli = sandbox.cli(&format!(
            "test \"$1\" = status\nprintf 'write consent: {mode}\\n'"
        ));
        for result in [
            cli.recall("task"),
            cli.chain("task"),
            cli.show("procedures/note"),
            cli.list(false),
        ] {
            assert_eq!(result.unwrap_err().kind(), ErrorKind::PermissionDenied);
        }
    }
}
