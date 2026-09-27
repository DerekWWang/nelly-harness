# Local model integration

The Rust `VoiceModel` trait is the model boundary. Its optional `OnnxVoiceModel`
implementation loads ONNX weights once and runs inference in the same process;
there is no Python subprocess, HTTP server, or weight download on that path.
The default build excludes ONNX Runtime entirely.

**No trained Nelly checkpoint or model architecture was supplied.** The included
adapter supports a waveform-to-action classification head with a documented
export contract. It does not implement speech recognition, an autoregressive
conversation decoder, a speech codec, or speech synthesis. The tiny included
fixture has hand-set weights and synthetic audio, and tests the real weight
loading/inference/tool dispatch path; it does not understand speech.

## Run the native integration fixture

Build with Rust 1.88 or later:

```sh
cd nelly-harness
cargo build --release --features onnx
```

Provide an ONNX Runtime **1.24.x** shared library for your machine. Set
`ORT_DYLIB_PATH` to its absolute filename (on macOS, `libonnxruntime.1.24.3.dylib`;
on Linux, `libonnxruntime.so.1.24.3`; on Windows, `onnxruntime.dll`). Runtime
installation is explicit: Cargo neither bundles nor downloads it.

For a disposable installation using `uv`, run the following from the repository root.
Python is used here only to find a runtime library inside its package; the Rust
executable loads that library directly:

```sh
uv run --with onnxruntime==1.24.3 python -c 'import os,pathlib,subprocess,onnxruntime; p=pathlib.Path(onnxruntime.__file__).parent/"capi"; libs=sorted(p.glob("libonnxruntime*.dylib"))+sorted(p.glob("libonnxruntime.so*"))+sorted(p.glob("onnxruntime.dll")); assert libs,"runtime library not found"; subprocess.run(["target/release/nelly-harness","voice","examples/model-fixture/model.json","examples/model-fixture/audio.wav","--data","./data"],env=dict(os.environ,ORT_DYLIB_PATH=str(libs[0])),check=True)'
```

Or, after setting `ORT_DYLIB_PATH` yourself:

```sh
cargo run --release --features onnx -- voice \
  examples/model-fixture/model.json examples/model-fixture/audio.wav \
  --data ./data --episode native-model-demo --task 'Test local inference and prefetch'

cargo test --features onnx actual_onnx_weights_infer_and_suppress_duplicate_actions -- --ignored
cargo test --features onnx --test voice_cli -- --ignored
```

The fixture reads three 20 ms mono PCM frames. A negative frame maps to no-op;
a positive frame emits `notes_get` for topic `demo`, name `welcome`; a repeated
positive frame is suppressed. If the note does not exist, the tool returns its
normal missing-note result. That confirms dispatch without requiring initial
application data. The recording flags additionally preserve input audio and
model/tool state-action events for later export.

The CLI integration tests also exercise reply-only and silent models and verify
that exported episode audio remains available after deleting its original file.

## Export contract

The ONNX graph must have exactly one float32 input `[1, chunk_samples]` and a
float32 logits output `[1, number_of_actions]`. Dynamic dimensions are accepted
in graph metadata, but the actual runtime shapes must match the manifest. Audio
is mono float32 PCM normalized to `[-1, 1]` at the declared sample rate; resampling
and any model-specific preprocessing must happen before this adapter or inside
the graph. The CLI accepts mono PCM16 WAV.

The manifest is ordinary JSON, with paths relative to the manifest file:

```json
{
  "model_path": "nelly-action-head.onnx",
  "sample_rate_hz": 16000,
  "chunk_samples": 320,
  "input_name": "audio",
  "output_name": "logits",
  "intra_threads": 1,
  "min_confidence": 0.8,
  "emit_on_change": true,
  "actions": [
    null,
    {"event":"prefetch","call":{"tool":"notes_get","topic":"work","name":"agenda"}},
    {"event":"tool_call","call":{"tool":"notes_put","topic":"work","name":"status","note":"Ready"}},
    {"event":"reply","text":"Ready."}
  ]
}
```

Optional `runtime_library` selects a shared library relative to this manifest;
`ORT_DYLIB_PATH` takes precedence. ONNX Runtime is initialized once per process,
so multiple sessions in that process share the first loaded runtime library.

The adapter computes stable softmax confidence, selects at most one class per
chunk, and emits nothing below the threshold. A `null` action is required for
silence/no-op. With `emit_on_change: true`, repeated identical classes emit once
until another class or a below-threshold chunk intervenes. Calling `reset()`
clears this latch while retaining the loaded weights. For an application where
each frame is an independent decision, set `emit_on_change: false`.

`Prefetch` uses the same speculative-read enforcement as an external tool call.
It cannot write notes or modify the schedule. A `ToolCall` is a committed action
and is dispatched normally. The fixture only uses `Prefetch`.

These action arguments are fixed by the manifest. A production voice model
that generates arbitrary note text or topics needs its own decoder adapter.

## Connect the conversational model

`VoiceSession<M, E>` is the reusable live loop. It accepts any `VoiceModel` and
either `Harness` (in-memory tools), `DurableHarness` (journaled tools), or a custom
`ToolExecutor`. It is independent of WAV input: pass PCM from a microphone or
another local audio source into `process(AudioChunk)` and handle its
`SessionEvent` values. The model and tool engine remain loaded between chunks.

```rust,ignore
use nelly_harness::{Harness, model::AudioChunk, session::VoiceSession};

let mut session = VoiceSession::new(my_loaded_model, Harness::default());
// Repeat with each mono PCM chunk from the device's reusable audio buffer.
let events = session.process(AudioChunk {
    sample_rate_hz: 16_000,
    pcm: &audio_buffer,
})?;
// Render replies and record/inspect completed tool events here.
```

The session parses generated tool requests, enforces speculative read-only
execution, dispatches the tool, and returns success or error observations to the
model before processing the next event. Completed tool events include their
before/after revisions. If feedback consumption fails after a tool commits,
the returned event preserves the result and reports `observation_error` so a
caller can record the action and decide how to recover. `reset()` clears the
model's conversation while preserving the loaded weights and tool state.

Implement `VoiceModel` for the actual model architecture:

1. Load its weights, tokenizer, and codec once in the adapter's constructor.
2. Reuse codec buffers and decoder/KV state across `infer(AudioChunk)` calls.
3. Emit `ModelEvent::Prefetch` for tentative reads, `ToolCall` for committed
   requests, and `Reply` for decoded text. Parse generated tool JSON into a
   `serde_json::Value`; the harness validates the tool and arguments.
4. Implement `observe_tool_result` to put retrieved notes, schedule results,
   and errors back into model context. Return `true` only after consumption.
5. Clear conversational state in `reset()` while retaining weights.

The reference ONNX classifier is stateless and returns `false` from
`observe_tool_result`; it cannot use retrieved context in subsequent inference.
For a stateful conversational model, codec inputs, recurrent/cache tensors,
tool-result tokenization, and speech generation depend on its architecture and
are intentionally owned by that adapter.

The tool core stays independent of the model adapter, so native model integration
does not add IPC or JSON serialization to internal note/schedule access. The
adapter borrows the input PCM without a waveform copy. The default ONNX session
uses one intra-op thread, sequential graph execution, graph optimization, and
disabled thread spinning. Model weights, activations, runtime allocators, and
decoder cache will dominate memory once a real voice model is loaded; the small
tool-harness footprint does not imply a small model footprint.

To regenerate the checked-in fixture from the repository root:

```sh
uv run --with onnx==1.20.1 examples/model_fixture.py
```

Reference APIs: [ort Session](https://docs.rs/ort/2.0.0-rc.12/ort/session/struct.Session.html),
[ort TensorRef](https://docs.rs/ort/2.0.0-rc.12/ort/value/type.TensorRef.html),
[ort runtime initialization](https://docs.rs/ort/2.0.0-rc.12/ort/fn.init_from.html),
and [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
