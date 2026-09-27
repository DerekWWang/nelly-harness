//! Model boundary and an optional in-process ONNX waveform/action-head adapter.
//!
//! Model-specific codecs, tokenizers, and conversational state belong behind
//! `VoiceModel`; the tool engine does not depend on an inference runtime.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Borrowed mono, normalized f32 PCM. The caller owns the reusable audio buffer.
#[derive(Debug, Clone, Copy)]
pub struct AudioChunk<'a> {
    pub sample_rate_hz: u32,
    pub pcm: &'a [f32],
}

/// Only `ToolCall` may commit a mutation. `Prefetch` must go through the tool
/// engine's speculative-read path, including when emitted by a local model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelEvent {
    Prefetch { call: Value },
    ToolCall { call: Value },
    Reply { text: String },
}

/// In-process integration point for a real Nelly codec/decoder implementation.
/// Load weights once when constructing the implementation, then reuse it.
pub trait VoiceModel {
    fn sample_rate_hz(&self) -> u32;
    fn chunk_samples(&self) -> usize;
    fn infer(&mut self, audio: AudioChunk<'_>) -> Result<Vec<ModelEvent>, String>;

    /// Return `true` when the adapter actually consumed the observation. A
    /// stateless classifier returns `false`, so callers cannot mistake it for
    /// a conversational model that has received retrieved context.
    fn observe_tool_result(&mut self, call: &Value, result: &Value) -> Result<bool, String>;

    /// Start another conversation without reloading weights.
    fn reset(&mut self) -> Result<(), String>;
}

/// Export contract for the reference waveform action-head adapter. The graph
/// takes one float32 `[1, chunk_samples]` input and returns `[1, actions.len()]`
/// float32 logits. A null action is a no-op/silence class.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    pub model_path: std::path::PathBuf,
    #[serde(default)]
    pub runtime_library: Option<std::path::PathBuf>,
    pub sample_rate_hz: u32,
    pub chunk_samples: usize,
    #[serde(default = "default_input")]
    pub input_name: String,
    #[serde(default = "default_output")]
    pub output_name: String,
    #[serde(default = "default_threads")]
    pub intra_threads: usize,
    #[serde(default = "default_confidence")]
    pub min_confidence: f32,
    #[serde(default = "default_true")]
    pub emit_on_change: bool,
    pub actions: Vec<Option<ModelEvent>>,
}

fn default_input() -> String {
    "audio".into()
}
fn default_output() -> String {
    "logits".into()
}
fn default_threads() -> usize {
    1
}
fn default_confidence() -> f32 {
    0.8
}
fn default_true() -> bool {
    true
}

impl ModelManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.model_path.as_os_str().is_empty() {
            return Err("model_path must not be empty".into());
        }
        if !(8_000..=192_000).contains(&self.sample_rate_hz) {
            return Err("sample_rate_hz must be between 8000 and 192000".into());
        }
        if !(1..=1_048_576).contains(&self.chunk_samples) {
            return Err("chunk_samples must be between 1 and 1048576".into());
        }
        if !(1..=64).contains(&self.intra_threads) {
            return Err("intra_threads must be between 1 and 64".into());
        }
        if !self.min_confidence.is_finite() || !(0.0..=1.0).contains(&self.min_confidence) {
            return Err("min_confidence must be finite and between 0 and 1".into());
        }
        if self.input_name.is_empty() || self.output_name.is_empty() {
            return Err("tensor names must not be empty".into());
        }
        if !(2..=4096).contains(&self.actions.len()) {
            return Err("actions must contain between 2 and 4096 classes".into());
        }
        if !self.actions.iter().any(Option::is_none) {
            return Err("actions must contain a null no-op/silence class".into());
        }
        for action in self.actions.iter().flatten() {
            match action {
                ModelEvent::Prefetch { call } | ModelEvent::ToolCall { call } => {
                    if call.get("tool").and_then(Value::as_str).is_none() {
                        return Err(
                            "model tool actions must be objects containing a tool string".into(),
                        );
                    }
                }
                ModelEvent::Reply { text } if text.len() > 65_536 => {
                    return Err("model reply exceeds 65536 bytes".into());
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(any(feature = "onnx", test))]
fn validate_audio(
    audio: AudioChunk<'_>,
    sample_rate_hz: u32,
    samples: usize,
) -> Result<(), String> {
    if audio.sample_rate_hz != sample_rate_hz {
        return Err(format!(
            "expected {sample_rate_hz} Hz PCM; received {} Hz (resample before inference)",
            audio.sample_rate_hz
        ));
    }
    if audio.pcm.len() != samples {
        return Err(format!(
            "expected {samples} PCM samples; received {}",
            audio.pcm.len()
        ));
    }
    if audio.pcm.iter().any(|x| !x.is_finite() || x.abs() > 1.0) {
        return Err("audio samples must be finite normalized PCM in [-1, 1]".into());
    }
    Ok(())
}

#[cfg(any(feature = "onnx", test))]
fn choose_action(logits: &[f32], min_confidence: f32) -> Result<Option<usize>, String> {
    if logits.len() < 2 || logits.iter().any(|value| !value.is_finite()) {
        return Err("model returned invalid or non-finite logits".into());
    }
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    let max = f64::from(logits[best]);
    let sum: f64 = logits.iter().map(|&x| (f64::from(x) - max).exp()).sum();
    Ok((1.0 / sum >= f64::from(min_confidence)).then_some(best))
}

#[cfg(feature = "onnx")]
mod onnx {
    use super::*;
    use ort::{
        session::Session,
        value::{TensorElementType, TensorRef, ValueType},
    };
    use std::{
        fs::File,
        io::Read,
        path::{Path, PathBuf},
    };

    /// Optional native ONNX Runtime backend, dynamically linked at load time.
    /// This is a stateless action classifier, not a general speech/LLM decoder.
    pub struct OnnxVoiceModel {
        manifest: ModelManifest,
        session: Session,
        previous_class: Option<usize>,
    }

    impl OnnxVoiceModel {
        /// Resolve model/runtime paths relative to the manifest. An environment
        /// `ORT_DYLIB_PATH` overrides the manifest's runtime_library.
        pub fn load(manifest_path: impl AsRef<Path>) -> Result<Self, String> {
            let manifest_path = manifest_path.as_ref();
            let mut bytes = Vec::new();
            File::open(manifest_path)
                .map_err(|e| format!("open model manifest: {e}"))?
                .take(1_048_577)
                .read_to_end(&mut bytes)
                .map_err(|e| format!("read model manifest: {e}"))?;
            if bytes.len() > 1_048_576 {
                return Err("model manifest exceeds 1 MiB".into());
            }
            let manifest: ModelManifest =
                serde_json::from_slice(&bytes).map_err(|e| format!("parse model manifest: {e}"))?;
            manifest.validate()?;
            let directory = manifest_path.parent().unwrap_or_else(|| Path::new("."));
            let weights = directory.join(&manifest.model_path);
            if !weights.is_file() {
                return Err(format!("model weights do not exist: {}", weights.display()));
            }
            let runtime_library = std::env::var_os("ORT_DYLIB_PATH")
                .map(PathBuf::from)
                .or_else(|| {
                    manifest
                        .runtime_library
                        .as_ref()
                        .map(|path| directory.join(path))
                })
                .unwrap_or_else(|| PathBuf::from(default_runtime_library()));
            // Unlike implicit initialization, init_from reports missing or
            // incompatible runtime libraries as an ordinary error.
            ort::init_from(&runtime_library)
                .map_err(|e| {
                    format!("load ONNX Runtime (set ORT_DYLIB_PATH to the 1.24 library): {e}")
                })?
                .with_name("nelly")
                .commit();
            let session = Session::builder()
                .map_err(ort_error)?
                .with_intra_threads(manifest.intra_threads)
                .map_err(ort_error)?
                .with_parallel_execution(false)
                .map_err(ort_error)?
                .with_intra_op_spinning(false)
                .map_err(ort_error)?
                .with_inter_op_spinning(false)
                .map_err(ort_error)?
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
                .map_err(ort_error)?
                .commit_from_file(weights)
                .map_err(ort_error)?;
            if session.inputs().len() != 1 {
                return Err("ONNX action-head adapter requires exactly one audio input; implement VoiceModel for models with codec/cache/context inputs".into());
            }
            let input = &session.inputs()[0];
            if input.name() != manifest.input_name {
                return Err(format!(
                    "model input is {:?}, manifest expects {:?}",
                    input.name(),
                    manifest.input_name
                ));
            }
            validate_shape(input.dtype(), &[1, manifest.chunk_samples as i64], "input")?;
            let output = session
                .outputs()
                .iter()
                .find(|output| output.name() == manifest.output_name)
                .ok_or_else(|| format!("model has no output named {:?}", manifest.output_name))?;
            validate_shape(
                output.dtype(),
                &[1, manifest.actions.len() as i64],
                "logits",
            )?;
            Ok(Self {
                manifest,
                session,
                previous_class: None,
            })
        }

        pub fn manifest(&self) -> &ModelManifest {
            &self.manifest
        }
    }

    fn default_runtime_library() -> &'static str {
        if cfg!(target_os = "windows") {
            "onnxruntime.dll"
        } else if cfg!(target_os = "macos") {
            "libonnxruntime.dylib"
        } else {
            "libonnxruntime.so"
        }
    }

    fn ort_error<T>(error: ort::Error<T>) -> String {
        format!("ONNX inference runtime: {error}")
    }

    fn validate_shape(dtype: &ValueType, expected: &[i64], label: &str) -> Result<(), String> {
        match dtype {
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                shape,
                ..
            } if shape.len() == expected.len()
                && shape
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| *actual == -1 || actual == expected) =>
            {
                Ok(())
            }
            _ => Err(format!(
                "{label} must be float32 with shape {expected:?}; got {dtype:?}"
            )),
        }
    }

    impl VoiceModel for OnnxVoiceModel {
        fn sample_rate_hz(&self) -> u32 {
            self.manifest.sample_rate_hz
        }
        fn chunk_samples(&self) -> usize {
            self.manifest.chunk_samples
        }

        fn infer(&mut self, audio: AudioChunk<'_>) -> Result<Vec<ModelEvent>, String> {
            validate_audio(audio, self.sample_rate_hz(), self.chunk_samples())?;
            // TensorRef borrows the caller's PCM without copying the waveform.
            let input =
                TensorRef::from_array_view(([1, audio.pcm.len()], audio.pcm)).map_err(ort_error)?;
            let outputs = self
                .session
                .run(ort::inputs![self.manifest.input_name.as_str() => input])
                .map_err(ort_error)?;
            let (shape, logits) = outputs[self.manifest.output_name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(ort_error)?;
            if shape.as_ref() != [1, self.manifest.actions.len() as i64] {
                return Err(format!("model logits shape changed at runtime: {shape:?}"));
            }
            let selected = choose_action(logits, self.manifest.min_confidence)?;
            let duplicate = self.manifest.emit_on_change && selected == self.previous_class;
            self.previous_class = selected;
            if duplicate {
                return Ok(Vec::new());
            }
            Ok(selected
                .and_then(|index| self.manifest.actions[index].clone())
                .into_iter()
                .collect())
        }

        fn observe_tool_result(&mut self, _call: &Value, _result: &Value) -> Result<bool, String> {
            Ok(false)
        }

        fn reset(&mut self) -> Result<(), String> {
            self.previous_class = None;
            Ok(())
        }
    }
}

#[cfg(feature = "onnx")]
pub use onnx::OnnxVoiceModel;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_softmax_rejects_invalid_and_uncertain_logits() {
        assert_eq!(choose_action(&[10_000.0, 10_020.0], 0.9).unwrap(), Some(1));
        assert_eq!(choose_action(&[0.0, 0.0], 0.8).unwrap(), None);
        assert!(choose_action(&[f32::NAN, 0.0], 0.8).is_err());
        assert!(choose_action(&[f32::INFINITY, 0.0], 0.8).is_err());
    }

    #[test]
    fn malformed_or_unbounded_audio_is_rejected_before_runtime() {
        assert!(validate_audio(
            AudioChunk {
                sample_rate_hz: 8_000,
                pcm: &[0.0]
            },
            16_000,
            1
        )
        .is_err());
        assert!(validate_audio(
            AudioChunk {
                sample_rate_hz: 16_000,
                pcm: &[0.0]
            },
            16_000,
            2
        )
        .is_err());
        assert!(validate_audio(
            AudioChunk {
                sample_rate_hz: 16_000,
                pcm: &[f32::NAN]
            },
            16_000,
            1
        )
        .is_err());
        assert!(validate_audio(
            AudioChunk {
                sample_rate_hz: 16_000,
                pcm: &[1.1]
            },
            16_000,
            1
        )
        .is_err());
    }

    #[test]
    fn manifest_requires_a_noop_and_valid_tool_payloads() {
        let mut manifest: ModelManifest = serde_json::from_value(serde_json::json!({
            "model_path": "weights.onnx", "sample_rate_hz": 16000, "chunk_samples": 320,
            "actions": [null, {"event": "prefetch", "call": {"tool": "notes_get", "topic": "work", "name": "todo"}}]
        })).unwrap();
        manifest.validate().unwrap();
        manifest.actions[1] = Some(ModelEvent::ToolCall {
            call: serde_json::json!({"invalid": true}),
        });
        assert!(manifest.validate().is_err());
        manifest.actions = vec![
            Some(ModelEvent::Reply {
                text: "hello".into()
            });
            2
        ];
        assert!(manifest.validate().is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    #[ignore = "requires ONNX Runtime 1.24 via ORT_DYLIB_PATH; see MODEL.md"]
    fn actual_onnx_weights_infer_and_suppress_duplicate_actions() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples/model-fixture/model.json");
        let mut model = OnnxVoiceModel::load(manifest).unwrap();
        let positive = vec![0.75_f32; model.chunk_samples()];
        let negative = vec![-0.75_f32; model.chunk_samples()];
        let sample_rate_hz = model.sample_rate_hz();
        let first = model
            .infer(AudioChunk {
                sample_rate_hz,
                pcm: &positive,
            })
            .unwrap();
        assert_eq!(
            first,
            vec![ModelEvent::Prefetch {
                call: serde_json::json!({"tool": "notes_get", "topic": "demo", "name": "welcome"})
            }]
        );
        assert!(model
            .infer(AudioChunk {
                sample_rate_hz,
                pcm: &positive
            })
            .unwrap()
            .is_empty());
        assert!(model
            .infer(AudioChunk {
                sample_rate_hz,
                pcm: &negative
            })
            .unwrap()
            .is_empty());
        assert_eq!(
            model
                .infer(AudioChunk {
                    sample_rate_hz,
                    pcm: &positive
                })
                .unwrap(),
            first
        );
        assert!(!model
            .observe_tool_result(&serde_json::json!({}), &serde_json::json!({}))
            .unwrap());
        model.reset().unwrap();
        assert_eq!(
            model
                .infer(AudioChunk {
                    sample_rate_hz,
                    pcm: &positive
                })
                .unwrap(),
            first
        );
    }
}
