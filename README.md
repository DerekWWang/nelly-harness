# Nelly harness

A small Rust tool runtime for a local voice agent. Embed `Harness` directly for
the cheapest calls, or run the JSONL executable. Rust 1.88+ is required.

- Exact topic/name note CRUD, with borrowed, allocation-free reads.
- Calendar CRUD, overlap queries, availability, and earliest free interval.
- Sparse daily bitmaps: **184 bytes per occupied day**, one bit per minute.
- Speculative reads with a bounded cache. Writes invalidate cached results and
  cannot run speculatively.
- A durable mutation journal with one process owning each data directory.
- A local model interface, optional native ONNX inference, and PCM16 WAV replay.
- Owned episode audio, long-form summaries, state/action/result JSONL, training
  exports, and an explicit Memorable CLI adapter.

## Run

```sh
cd nelly-harness
cargo build --release
target/release/nelly-harness serve --ephemeral < examples/tools.jsonl
target/release/nelly-harness serve --data ./data
target/release/nelly-harness tools
```

`serve` reads one request per line and flushes one response per line. It defaults
to durable storage in `./data`; `--ephemeral` skips all disk work. Stdout contains
protocol results and stderr contains startup/protocol failures. Requests are
limited to 64 KiB. Unknown tools/fields return errors; oversized lines terminate
the stream without an unbounded allocation.

```json
{"request_id":1,"speculative":true,"call":{"tool":"notes_get","topic":"people","name":"Ada"}}
{"request_id":2,"call":{"tool":"notes_put","topic":"people","name":"Ada","note":"Prefers tea."}}
{"request_id":3,"call":{"tool":"notes_get","topic":"people","name":"Ada"}}
```

```json
{"ok":true,"request_id":3,"result":{"note":"Prefers tea."},"cached":false,"revision":1}
```

`request_id` is an optional opaque correlation value; it is **not** an idempotency
key. A client that loses a write response should reconcile state before retrying
`schedule_create`. `speculative` defaults to false. Feed a function-calling
model's name/arguments as `call: {"tool": name, ...arguments}`. The `tools`
command emits function schemas and a `read_only` classification.

| Tool | Arguments | Result |
| --- | --- | --- |
| `notes_put` | `topic`, `name`, `note` | Create or replace, `stored` |
| `notes_get` | `topic`, `name` | `note` string or null |
| `notes_delete` | `topic`, `name` | `deleted` boolean |
| `notes_list` | Optional `topic`, `offset`, `limit` | Sorted note rows |
| `schedule_create` | `title`, `start_minute`, `end_minute` | Event with stable ID |
| `schedule_update` | `id`, `title`, `start_minute`, `end_minute` | Replaced event |
| `schedule_get` | `id` | Event or null |
| `schedule_delete` | `id` | `deleted` boolean |
| `schedule_list` | `start_minute`, `end_minute`, optional `offset`, `limit` | Overlapping events |
| `schedule_is_free` | `start_minute`, `end_minute` | `free` boolean |
| `schedule_first_free` | `start_minute`, `end_minute`, `duration_minutes` | Earliest `start_minute` or null |
| `stats` | None | Revision, counts, cache and bitmap payload sizes |

Pages default to 32 rows, maximum 256. Keys use exact UTF-8 matching; casing and
normalization belong to the model adapter. Calendar intervals are half-open
`[start_minute, end_minute)` in **UTC Unix minutes**. Convert spoken/local dates
with the user's timezone before calling tools. Adjacent events do not overlap;
overlapping reservations are allowed. Deleting one overlapping event preserves
the other event's busy bits. Recurrence and external calendar sync are outside
this first version. Natural-language tool calls require a
normalization/translation adapter before they can invoke this typed contract.

## Embed and prefetch

```rust
use nelly_harness::{Harness, ToolCall};

let mut h = Harness::default();
let read = ToolCall::NotesGet { topic: "people".into(), name: "Ada".into() };
let prefetched = h.execute(&read, true)?;
let result = h.execute(&read, false)?; // same Arc<Value>, cache hit
assert!(result.cached);

// The smallest path bypasses JSON and result allocation altogether:
let note: Option<&str> = h.notes().get("people", "Ada");
let free = h.schedule().is_free(30_000_000, 30_000_030)?;
Ok::<(), String>(())
```

The core uses single-owner mutable state, without a thread pool, async runtime,
network server, database, or locks on in-memory reads. Prefetch as soon as the
voice model has enough intent; subsequent requests use the same key. The cache
holds up to 128 entries and 128 KiB of serialized key/value payload by default,
with FIFO eviction and no timer maintenance. `Harness::new` customizes limits.
Consumers may retain old `Arc` results: compare their revision with the current
revision before reusing them after a mutation. Retained results are memory owned
by the consumer and are outside the cache's budget.

Writes have different costs: durable mode appends and syncs a journal record
before acknowledgement. Startup replays the journal. A torn final line is
discarded, a complete corrupt record causes an error, and a failed write restores
the previous state or makes the instance reject further operations. File locking
prevents two processes from owning a journal. The journal grows with writes and
does not yet compact; startup work is proportional to mutation history.

## Local model and episodes

See [MODEL.md](MODEL.md) for native model loading, the runnable ONNX fixture, and
the `VoiceModel` interface. The ONNX dependency is feature-gated; the default
binary contains no inference runtime. A model session loads weights once and
feeds chunks and tool observations through a persistent adapter.

```sh
cargo build --release --features onnx
# Point ORT_DYLIB_PATH at an installed ONNX Runtime 1.24 shared library.
target/release/nelly-harness voice \
  examples/model-fixture/model.json examples/model-fixture/audio.wav \
  --data ./data --episode demo-001 --task 'Test voice prefetch'
target/release/nelly-harness memory export demo-001 --data ./data > training.jsonl
```

No trained Nelly checkpoint or architecture was supplied. The included ONNX
fixture is a hand-weighted waveform classifier that validates actual native
weight loading and inference. It does not understand speech or generate a
conversation. Connect the eventual decoder, codec, tool-result context, and
speech output through `VoiceModel`. WAV replay runs offline; the native interface
accepts incremental audio chunks from a microphone integration as well.

See [MEMORY.md](MEMORY.md) for manual recording and the Memorable adapter. Local
episodes are the lossless archive; Memorable extracts procedural memory from a
trace. Sync is explicit, uses a pinned CLI, retains data after failures, and
requires the user's configured Memorable account. No remote service is needed
for notes, scheduling, inference, or local episode export.

```sh
npx --yes --package memorable-cli@0.5.30 memorable login
npx --yes --package memorable-cli@0.5.30 memorable enable
target/release/nelly-harness memory sync demo-001 --data ./data
target/release/nelly-harness memory recall 'Test voice prefetch'
```

## Bounds and performance

Notes are limited to 4,096 entries, 256-byte key components, 16 KiB per value,
and 8 MiB total key/value payload. Long-form content belongs in episodes. The
schedule permits 4,096 events, 4,096 occupied days, 1,024-byte titles, and windows
up to 366 days. Empty calendars allocate no bitmap days. At the day cap, bitmap
payload is 736 KiB; map nodes, strings, model memory and allocator overhead are
additional. Queries allocate nothing for borrowed note lookup and bitmap
availability; sorted/paginated listings allocate references plus returned rows.

Initial release measurements on an Apple M4 Pro (2026-09-27):

| Operation | Mean time |
| --- | ---: |
| Borrowed note lookup, 1,000 notes | 21.5 ns |
| Bitmap availability, 30-minute interval | 3.9 ns |
| First free hour in a 30-day window | 7.9 ns |
| Cached typed tool dispatch | 19.9 ns |
| JSON parse + cached dispatch + result serialization | 268.8 ns |

These microbenchmarks exercise warm in-process operations and exclude transport,
disk sync, cold caches, inference, and Memorable. Only the explicitly labeled JSON
row includes parsing and serialization.
They are measurements of specific inputs, not end-to-end latency promises. The
default release binary was about 889 KiB, with 1.59 MiB maximum RSS reported
by macOS `time -l` for the eight-request ephemeral example. Binary/OS/runtime
versions and model weights change the footprint substantially. Reproduce:

```sh
cargo test --all-targets
cargo test --all-targets --features onnx
cargo clippy --all-targets --all-features -- -D warnings
cargo bench --bench hot_path
/usr/bin/time -l target/release/nelly-harness serve --ephemeral < examples/tools.jsonl
```

Tests include randomized calendar operations against a minute-level oracle,
overlap deletion, negative timestamps, bounds, speculative-write rejection,
cache invalidation, journal restart/torn-tail recovery, JSONL transport, owned
audio, streaming training export, and CLI failures/timeouts using a fake
Memorable executable. The actual ONNX inference test is explicitly enabled with
an installed native runtime as described in MODEL.md.
