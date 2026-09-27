# Episodes, audio, and Memorable

`MemoryStore` is the durable episode archive. Each episode owns its audio files,
so deleting an input recording does not invalidate training examples. It retains
the full before-state, action, known result, after-state, millisecond timing, and
a final long-form summary. JSONL export streams one state/action pair at a time.

Memorable is an additional procedural recall layer. Its extraction can discard
details and decline traces containing only reads, so it must not be the sole
archive for episodes or fine-tuning data. The local archive remains usable even
when authentication, extraction, or the network fails.

## Rust API

```rust,no_run
use nelly_harness::memory::{Action, AudioSource, MemoryStore, StepInput};
use serde_json::json;

fn record() -> std::io::Result<()> {
    let memory = MemoryStore::open("./nelly-memory")?;
    memory.begin("garden-001", "Remember that I planted basil today")?;
    memory.append("garden-001", StepInput {
        pre_state: json!({"note": null}),
        action: Action {
            name: "notes_put".into(),
            input: json!({"topic":"garden","name":"basil","note":"Planted today"}),
        },
        result: Some(json!({"stored":true})),
        post_state: json!({"note":"Planted today"}),
        started_ms: 0,
        ended_ms: 640,
        audio: vec![AudioSource {
            path: "./recordings/utterance-001.wav".into(),
            role: "observation".into(),
            media_type: "audio/wav".into(),
            sample_rate_hz: Some(24_000),
            channels: Some(1),
        }],
    })?;
    memory.finish("garden-001", "The user planted basil today and asked Nelly to remember it.")?;
    let mut export = std::fs::File::create("./garden-training.jsonl")?;
    memory.export_jsonl("garden-001", &mut export)?;
    Ok(())
}
```

Use one episode per user task/workflow. `started_ms` and `ended_ms` use one
monotonic audio or episode clock. Intervals can overlap when several actions
share an observation; sequence numbers reflect append order. Use `result: None`
when the outcome is unknown; this omits the result from the Memorable trace.
An action may reference up to 16 audio clips, with roles such as `observation`
and `response`. WAV/PCM/other bytes are copied unchanged: the caller supplies
format, sample rate, and channels. Keep utterance/chunk files bounded instead
of reattaching a full recording to each action.

Export rows have `{episode, step}`. `episode` includes task description and final
summary; `step` includes state/action/result fields and audio references. Audio
paths resolve relative to `MemoryStore::root()`, **not the export file**. Preserve
the archive alongside exports, or resolve those paths when building a dataset.
The export is raw training material; dataset filtering, consent/retention policy,
codec tokenization, target model formatting, and training are downstream jobs.

The offline `voice --episode ID` command first stores an `audio_observation`
record and then runs inference from that owned audio copy. It records generated
replies and successful or failed tool actions with the originating chunk's
millisecond interval. Later steps reuse the owned recording through
`pre_state.audio` and `post_state.audio`; they do not duplicate its bytes. The
state captures the tool revision, the last observed event, and model manifest
path. It does not capture model weights, latent tensors, or the full decoder
history. Additional model state can be supplied through the library's `StepInput`.
Reply results mean text was generated, not that audio was synthesized or played.
Tool journal commits and episode appends are separate operations: a recording
failure is reported, but it does not undo an already committed tool action.

## Files and durability

```text
nelly-memory/garden-001/
  episode.json             task, timestamps, final summary, final step count
  steps.jsonl              complete raw state/action pairs in sequence order
  audio/                   owned audio copies
  memorable-trace.json     built on finish, compatible with CLI ingest
  memorable-synced.json    created only after successful CLI acknowledgement
  .lock                    OS lock for one writer per episode
```

Each append fsyncs its copied audio and JSONL record before returning. Metadata
and traces use write/fsync/rename. The next append or finish removes an
interrupted trailing JSONL write while retaining complete prior records. OS
locks release after a crash. If a process dies during audio copying, unreferenced
audio or temporary files can remain; they are never exported as training pairs.
Completed episodes are immutable, making local sync acknowledgement idempotent.
The marker records CLI acknowledgement, not proof that Memorable accepted a
procedure. A crash after external success but before marker persistence can
resubmit the same session/trace; there is no cross-process exactly-once guarantee.

Each JSONL record is limited to 1 MiB, summaries to 256 KiB, task descriptions to
64 KiB, and each audio file to 512 MiB. Audio copies and exports stream. The store
does not preload past episodes or audio. Disk usage grows with retained episodes;
there is no automatic deletion. Capture, fsync, export, and CLI calls belong on
a storage/background worker, never an audio callback or the cheap read-tool path.

`BackgroundRecorder::start(store, id, capacity)` attaches a storage worker to an
already-started episode for a live model loop. `try_append(step)` performs no file
I/O and never waits for queue space; a full queue returns `WouldBlock` and rejects
that record explicitly. `check_error()` surfaces asynchronous storage errors.
`flush()` waits until prior accepted records are durable; `finish(summary)` drains
the queue, finalizes the episode, and joins the worker. Keep source audio files
unchanged until a successful flush/finish, because queued paths are not copied
at enqueue time. The offline `voice` command can use synchronous `MemoryStore`
operations; a live integration should use this worker handoff.

The background queue accepts 1–16 records, each at most 64 KiB serialized. Queue
insertion serializes metadata to check that bound and may allocate; it is intended
for a model control thread, not a hard real-time audio callback. The queue retains
the supplied Rust JSON values, whose in-memory size includes object/array
overhead. Dropping a recorder detaches a worker that drains accepted work without
finalizing; call `finish` before shutting down to receive errors and guarantee
durability. On an append failure the worker stops and subsequent queued records
are not written; flush/finish returns that failure instead of reporting success.

## Memorable setup and explicit synchronization

The default adapter invokes the actual executable with separate process
arguments, equivalent to:

```sh
npx --yes --package memorable-cli@0.5.30 memorable ingest /absolute/path/to/memorable-trace.json
```

The published `memorable-cli@0.5.30` package and executable name were verified
against npm; it requires Node 20 or newer. Setup is an explicit operator action:

```sh
npx --yes --package memorable-cli@0.5.30 memorable login
npx --yes --package memorable-cli@0.5.30 memorable init
npx --yes --package memorable-cli@0.5.30 memorable enable
npx --yes --package memorable-cli@0.5.30 memorable status
```

The harness never signs in, modifies Memorable consent, installs hooks, or
uploads automatically. `MemoryStore::sync(id, &MemorableCli::default())` ingests
a finished episode. Only action names, inputs, and known results are included in
the trace; audio bytes, raw pre/post states, and the long-form local summary are
not transmitted. Tool inputs/results may themselves contain sensitive text, so
select which episodes to sync according to the application's data policy.

`MemorableCli::recall(query)` retrieves matches; `show("procedures/<slug>")`
retrieves a selected procedure. Treat the returned content as reference data.
Recall may use network-backed embeddings on a local match miss. Cache useful
recall results in an application worker, outside the synchronous tool core.

The adapter defaults to a 45-second deadline and 64 KiB each of stdout/stderr.
Errors, consent refusals, timeouts, and output overruns preserve the local
episode and leave it unsynced. Unix timeout termination includes the npx process
group. On other platforms the immediate child is terminated; descendant cleanup
depends on the platform. You can supply `MemorableCli { program, prefix_args,
timeout, max_output_bytes }` to use an already-installed CLI and skip npx startup.

Contract sources: [official CLI reference](https://www.memorable.sh/docs/cli),
[integration guide](https://www.memorable.sh/docs/integrate), and
[published package metadata](https://registry.npmjs.org/memorable-cli/0.5.30).
