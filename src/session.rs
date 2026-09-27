//! Reusable model/tool feedback loop, independent of WAV files and any runtime.

use crate::{
    model::{AudioChunk, ModelEvent, VoiceModel},
    persistence::DurableHarness,
    Harness, ToolCall, ToolResult,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Limit one inference batch so a faulty model cannot commit unbounded actions.
pub const MAX_EVENTS_PER_CHUNK: usize = 256;

pub trait ToolExecutor {
    fn execute(&mut self, call: &ToolCall, speculative: bool) -> Result<ToolResult, String>;
    fn revision(&self) -> u64;
}

impl ToolExecutor for Harness {
    fn execute(&mut self, call: &ToolCall, speculative: bool) -> Result<ToolResult, String> {
        Harness::execute(self, call, speculative)
    }
    fn revision(&self) -> u64 {
        Harness::revision(self)
    }
}

impl ToolExecutor for DurableHarness {
    fn execute(&mut self, call: &ToolCall, speculative: bool) -> Result<ToolResult, String> {
        DurableHarness::execute(self, call, speculative)
    }
    fn revision(&self) -> u64 {
        self.core().revision()
    }
}

impl<T: ToolExecutor + ?Sized> ToolExecutor for &mut T {
    fn execute(&mut self, call: &ToolCall, speculative: bool) -> Result<ToolResult, String> {
        (**self).execute(call, speculative)
    }
    fn revision(&self) -> u64 {
        (**self).revision()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum SessionEvent {
    Reply {
        text: String,
    },
    ToolResult {
        call: Value,
        speculative: bool,
        result: Value,
        revision_before: u64,
        revision_after: u64,
        observation_consumed: bool,
        /// Feedback may fail after a tool commits. Preserve the completed tool
        /// event so the caller can record it and decide how to recover.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        observation_error: Option<String>,
    },
}

/// Hold a loaded model and tool engine for a live conversation. Feed borrowed
/// audio chunks from a microphone, network stream, or file using the same API.
/// This type performs no file I/O or episode recording itself.
pub struct VoiceSession<M, E> {
    model: M,
    engine: E,
}

impl<M: VoiceModel, E: ToolExecutor> VoiceSession<M, E> {
    pub fn new(model: M, engine: E) -> Self {
        Self { model, engine }
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn engine(&self) -> &E {
        &self.engine
    }

    pub fn into_parts(self) -> (M, E) {
        (self.model, self.engine)
    }

    /// Reset conversational state while keeping loaded weights and app state.
    pub fn reset(&mut self) -> Result<(), String> {
        self.model.reset()
    }

    pub fn process(&mut self, audio: AudioChunk<'_>) -> Result<Vec<SessionEvent>, String> {
        if audio.sample_rate_hz != self.model.sample_rate_hz() {
            return Err("audio sample rate differs from the model".into());
        }
        if audio.pcm.len() != self.model.chunk_samples() {
            return Err("audio chunk length differs from the model".into());
        }
        if audio
            .pcm
            .iter()
            .any(|sample| !sample.is_finite() || sample.abs() > 1.0)
        {
            return Err("audio samples must be finite normalized PCM in [-1, 1]".into());
        }
        let generated = self.model.infer(audio)?;
        if generated.len() > MAX_EVENTS_PER_CHUNK {
            return Err(format!(
                "model emitted more than {MAX_EVENTS_PER_CHUNK} events in one audio chunk"
            ));
        }
        let mut events = Vec::with_capacity(generated.len());
        for event in generated {
            let (call, speculative) = match event {
                ModelEvent::Reply { text } => {
                    events.push(SessionEvent::Reply { text });
                    continue;
                }
                ModelEvent::Prefetch { call } => (call, true),
                ModelEvent::ToolCall { call } => (call, false),
            };
            let revision_before = self.engine.revision();
            let result = match serde_json::from_value::<ToolCall>(call.clone()) {
                Err(error) => {
                    json!({"ok": false, "error": format!("invalid model tool call: {error}")})
                }
                // Enforce this at the session boundary even with a custom
                // ToolExecutor implementation that omits the core's check.
                Ok(tool) if speculative && !tool.is_read() => {
                    json!({"ok": false, "error": "speculative writes are forbidden"})
                }
                Ok(tool) => match self.engine.execute(&tool, speculative) {
                    Ok(result) => {
                        json!({"ok": true, "result": result.value, "revision": result.revision, "cached": result.cached})
                    }
                    Err(error) => json!({"ok": false, "error": error}),
                },
            };
            let revision_after = self.engine.revision();
            let (observation_consumed, observation_error) =
                match self.model.observe_tool_result(&call, &result) {
                    Ok(consumed) => (consumed, None),
                    Err(error) => (false, Some(error)),
                };
            events.push(SessionEvent::ToolResult {
                call,
                speculative,
                result,
                revision_before,
                revision_after,
                observation_consumed,
                observation_error,
            });
        }
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct TestModel {
        frames: VecDeque<Vec<ModelEvent>>,
        observations: Vec<(Value, Value)>,
        fail_observation: bool,
    }

    impl VoiceModel for TestModel {
        fn sample_rate_hz(&self) -> u32 {
            16_000
        }
        fn chunk_samples(&self) -> usize {
            1
        }
        fn infer(&mut self, _audio: AudioChunk<'_>) -> Result<Vec<ModelEvent>, String> {
            Ok(self.frames.pop_front().unwrap_or_default())
        }
        fn observe_tool_result(&mut self, call: &Value, result: &Value) -> Result<bool, String> {
            if self.fail_observation {
                return Err("decoder feedback failed".into());
            }
            self.observations.push((call.clone(), result.clone()));
            Ok(true)
        }
        fn reset(&mut self) -> Result<(), String> {
            self.observations.clear();
            Ok(())
        }
    }

    fn chunk() -> AudioChunk<'static> {
        AudioChunk {
            sample_rate_hz: 16_000,
            pcm: &[0.0],
        }
    }

    fn model(frames: Vec<Vec<ModelEvent>>) -> TestModel {
        TestModel {
            frames: frames.into(),
            observations: Vec::new(),
            fail_observation: false,
        }
    }

    #[test]
    fn live_session_consumes_prefetch_errors_and_updated_context() {
        let get = json!({"tool": "notes_get", "topic": "work", "name": "agenda"});
        let put = json!({"tool": "notes_put", "topic": "work", "name": "agenda", "note": "Review at noon"});
        let voice = model(vec![
            vec![ModelEvent::Prefetch { call: get.clone() }],
            vec![
                ModelEvent::Prefetch { call: put.clone() },
                ModelEvent::ToolCall { call: put },
            ],
            vec![
                ModelEvent::ToolCall { call: get.clone() },
                ModelEvent::Reply {
                    text: "Ready".into(),
                },
            ],
            vec![ModelEvent::Prefetch { call: get }],
        ]);
        let mut session = VoiceSession::new(voice, Harness::default());
        session.process(chunk()).unwrap();
        assert_eq!(
            session.model().observations[0].1["result"]["note"],
            Value::Null
        );
        let writes = session.process(chunk()).unwrap();
        assert_eq!(writes.len(), 2);
        assert_eq!(session.engine().revision(), 1);
        assert_eq!(session.model().observations[1].1["ok"], false);
        assert_eq!(session.model().observations[2].1["ok"], true);
        let events = session.process(chunk()).unwrap();
        assert_eq!(
            events.last(),
            Some(&SessionEvent::Reply {
                text: "Ready".into()
            })
        );
        assert_eq!(
            session.model().observations[3].1["result"]["note"],
            "Review at noon"
        );
        assert_eq!(session.model().observations[3].1["cached"], false);
        session.process(chunk()).unwrap();
        assert_eq!(session.model().observations[4].1["cached"], true);
        session.reset().unwrap();
        assert!(session.model().observations.is_empty());
        assert_eq!(session.engine().revision(), 1);
    }

    #[test]
    fn malformed_generated_json_is_observed_as_a_tool_error() {
        let voice = model(vec![vec![ModelEvent::ToolCall {
            call: json!({"tool": "unknown_tool"}),
        }]]);
        let mut session = VoiceSession::new(voice, Harness::default());
        session.process(chunk()).unwrap();
        assert_eq!(session.model().observations.len(), 1);
        assert_eq!(session.model().observations[0].1["ok"], false);
        assert_eq!(session.engine().revision(), 0);
    }

    #[test]
    fn observation_failure_preserves_the_committed_action_event() {
        let mut voice = model(vec![vec![ModelEvent::ToolCall {
            call: json!({"tool": "notes_put", "topic": "x", "name": "y", "note": "z"}),
        }]]);
        voice.fail_observation = true;
        let mut session = VoiceSession::new(voice, Harness::default());
        let events = session.process(chunk()).unwrap();
        assert!(
            matches!(&events[0], SessionEvent::ToolResult { result, observation_consumed: false, observation_error: Some(error), .. } if result["ok"] == true && error == "decoder feedback failed")
        );
        assert_eq!(session.engine().revision(), 1);
    }

    #[test]
    fn invalid_audio_and_excessive_event_batches_do_not_execute_tools() {
        let voice = model(vec![vec![
            ModelEvent::ToolCall {
                call: json!({"tool": "notes_put", "topic": "x", "name": "y", "note": "z"})
            };
            MAX_EVENTS_PER_CHUNK + 1
        ]]);
        let mut session = VoiceSession::new(voice, Harness::default());
        assert!(session
            .process(AudioChunk {
                sample_rate_hz: 8000,
                pcm: &[0.0]
            })
            .is_err());
        assert!(session.process(chunk()).is_err());
        assert_eq!(session.engine().revision(), 0);
    }
}
