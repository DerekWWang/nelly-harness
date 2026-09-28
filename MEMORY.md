# Episodes, audio, and Memorable

`MemoryStore` is the durable episode archive. Each episode owns its audio files,
so deleting an input recording does not invalidate training examples. It retains
the full before-state, action, known result, after-state, millisecond timing, and
a final long-form summary. JSONL export streams one state/action pair at a time.

Memorable is an additional procedural recall layer. Its extraction can discard
details and decline traces containing only reads, so it must not be the sole
archive for episodes or fine-tuning data. The local archive remains usable even
when authentication, extraction, or the network fails.

## Four connected layers

The harness connects [Memorable's four layers](https://www.memorable.sh/) through
an optional background worker. Workflow synthesis and graph composition run in
the configured Memorable provider; raw audio episodes remain local.

| Layer | Nelly implementation | Entry points |
| --- | --- | --- |
| 1. Traces | Durable audio, state/action/result pairs, lossless JSONL export | `MemoryStore`, `BackgroundRecorder`, `memory begin/append/finish/export` |
| 2. Workflow synthesis | Project completed episodes into a provider trace and ingest it; retain failures for retry | `memory sync`, `memorable_ingest` |
| 3. Graph assembly | Ask the provider to compose stored workflows, with dependencies, coverage, and gaps | `memory chain`, `memorable_chain` |
| 4. Retrieval | Background recall/show/list, deduplicated pending work, bounded cache | `memorable_recall`, `memorable_show`, `memorable_list`, `memorable_poll` |

Inspect the wiring without accessing the provider:

```sh
cargo run -- memory layers
```

Enable it for the live tool loop:

```sh
cargo run --release -- serve --data ./data --memorable
# Or use an already-installed CLI:
cargo run --release -- serve --data ./data --memorable --memorable-bin /path/to/memorable
# The same two flags work with the voice command.
```

No worker is created without `--memorable`, and a worker launches no provider
process until a memory request arrives. All subprocess calls, provider status
checks, disk reads, and ingestion happen on that worker. Notes and scheduling
delegate directly to their existing core. A cold query returns a job immediately:

```json
{"request_id":1,"speculative":true,"call":{"tool":"memorable_recall","query":"prepare my weekly review"}}
{"request_id":2,"call":{"tool":"notes_get","topic":"work","name":"agenda"}}
{"request_id":3,"call":{"tool":"memorable_poll","job_id":1}}
```

The memory result has `job_id`, `status` (`pending`, `ready`, or `failed`),
`generation`, `data`, and `error`. Reissue the original read or poll its job when
the model is ready for context; pending responses never block. Duplicate pending
requests share one job. Completed reads return from cache until TTL/eviction or
invalidation. Failed jobs are observable and a repeated request can retry them.
The outer tool `ok` means the request was handled; the job's `status` reports
whether provider work succeeded. `data` marks provider content with
`source: "memorable"` and `trusted: false`. The model receives recalled procedures
and graph plans as reference data and commits application actions through the
normal tool route.

| Tool | Arguments | Behavior |
| --- | --- | --- |
| `memorable_recall` | `query` | Cached automatic single/chained recall |
| `memorable_show` | `slug` such as `procedures/abc-review` | Cached rendered procedure |
| `memorable_chain` | `query` | Cached JSON plan from graph composition |
| `memorable_list` | Optional `all` | Cached JSON workflow/revision inventory |
| `memorable_status` | None | Fresh background provider/consent status |
| `memorable_poll` | `job_id` | Nonblocking job result |
| `memorable_ingest` | `episode_id` | Committed-only synthesis request for a finished local episode |
| `memorable_invalidate` | None | Committed-only retrieval cache invalidation |
| `memorable_disable` | None | Committed-only local revocation and retained-result clearing |

`memorable_ingest` and cache controls cannot run speculatively. An acknowledged
ingest advances the memory generation and invalidates retrieval cache entries,
including when the acknowledgement payload exceeds the result budget. It does
not advance the notes/calendar revision. Pending reads submitted before an
invalidation keep their original generation and cannot populate the new cache;
compare generations before using retained results.

Default limits are four queued requests, 32 retained jobs, 32 cached reads,
256 KiB of retained serialized result/error payload, and a five-minute TTL.
`MemoryConfig` configures these bounds. Query text is limited to 16 KiB. Queues,
keys, JSON tree overhead, one queued completion, the active worker's buffers,
and caller-retained `Arc` replies are additional bounded/consumer-owned memory.
Full queues return an explicit error before accepting work. Completed job
results can be evicted; polling an evicted job returns an error. Use the Rust
`MemoryLayers` API to share cached `Arc` results; JSONL transport necessarily
serializes the returned context.

The synchronous convenience commands are also available:

```sh
cargo run -- memory chain 'prepare my weekly review'
cargo run -- memory list --all
cargo run -- memory status
```

The `serve` stream stays request/response: keep stdin open and poll to receive
completed content. EOF drains accepted work but does not emit unsolicited job
results. `voice` likewise drains after replay. Operational errors still drain
accepted memory work before exit. A failed accepted job makes shutdown report
failure, even if a caller already read its failed result. In library code, call
`LayeredExecutor::shutdown` or `MemoryLayers::shutdown` outside the inference
loop; it may wait for the bounded provider calls.

For embedding, wrap any existing executor with
`LayeredExecutor::new(engine, Some(MemoryLayers::new(store, cli, config)?))` and
pass it to `VoiceSession`. Model-emitted memory calls and polling then follow the
same tool-result feedback path as application tools.

## Provider contract and access

The pinned CLI checks are supplemented with the public `status` command before
cold reads or writes. Retrieval requires `read-only` or `read-write`; ingestion
requires `read-write`. Denied, unset, or unrecognized status fails closed. These
checks happen on the worker; cached hits use their last verified access state
until expiry or revocation. A worker-observed read denial clears local access and
cached data. Explicit `memorable_disable` immediately rejects new and queued
requests, discards in-flight retrieval results, and clears retained replies.
An already-running ingest can still finish, and previously returned caller-owned
copies cannot be recalled. There is no model-callable enable operation.

When an operator changes provider consent outside Nelly, disable or restart the
live layer immediately; cached entries cannot discover external changes without
a fresh provider call. `memorable_disable` affects this Nelly process, while the
provider's own consent and deletion controls manage persisted workflows.

The provider filters custom tool arguments. The derived trace adds a supported,
identifier-only `description` (topic/name/event IDs and times), retains actual
tool names and known results, and excludes audio-capture, reply, and Memorable
control events. Original raw episodes remain intact. Note bodies and transcripts
are not copied into metadata to defeat provider filtering. Unknown outcomes stay
unknown. Unprocessed older episodes are reprojected during sync.

**Provider limitation:** the pinned generic-harness extractor may classify
Nelly-specific actions as `other` and decline workflow synthesis. Successful CLI
acknowledgement does not prove a workflow was stored; inspect `memory list` or
`memory show` after ingestion. No fabricated shell execution or renamed provider
harness is used to force acceptance. Graph plans require stored workflows in the
provider and use its project/artifact dependency rules. Details and source links
are in [the pinned contract notes](docs/MEMORABLE_CONTRACT.md).

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

The harness never signs in, modifies provider consent, or installs hooks.
`MemoryStore::sync(id, &MemorableCli::default())`, `memory sync`, or a committed
`memorable_ingest` request ingests a finished episode. The derived trace contains
tool names, inputs with metadata descriptions, and known results; audio bytes,
raw pre/post states, and the long-form local summary are not transmitted. Tool
inputs/results may themselves contain sensitive text, so select which episodes
to sync according to the application's data policy.

`MemorableCli::recall(query)` retrieves matches; `show("procedures/<slug>")`
retrieves a selected procedure. Treat the returned content as reference data.
Recall may use network-backed embeddings on a local match miss. `MemoryLayers`
handles the background execution and bounded cache outside the synchronous core.

The adapter defaults to a 45-second deadline per subprocess and 64 KiB each of
stdout/stderr. A cold guarded operation can run one status process plus one
operation process, each with that deadline.
Errors, consent refusals, timeouts, and output overruns preserve the local
episode and leave it unsynced. Unix timeout termination includes the npx process
group. On other platforms the immediate child is terminated; descendant cleanup
depends on the platform. You can supply `MemorableCli { program, prefix_args,
timeout, max_output_bytes }` to use an already-installed CLI and skip npx startup.

Contract sources: [official CLI reference](https://www.memorable.sh/docs/cli),
[integration guide](https://www.memorable.sh/docs/integrate), and
[published package metadata](https://registry.npmjs.org/memorable-cli/0.5.30).
