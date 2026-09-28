//! Vendor trace projection. Raw episode rows are never rewritten or reduced.
use crate::memory::EpisodeStep;
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Serialize)]
pub(crate) struct TraceCall {
    pub name: String,
    pub input: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
}

pub(crate) fn project(step: &EpisodeStep) -> Option<TraceCall> {
    if matches!(step.action.name.as_str(), "audio_observation" | "reply")
        || step.action.name.starts_with("memorable_")
    {
        return None;
    }
    let mut input = step.action.input.clone();
    // The provider drops custom argument keys. Preserve identifiers in its
    // supported description field without moving note bodies/transcripts into
    // a field intended for metadata. Never invent a shell command or tool kind.
    let mut description = step.action.name.clone();
    for key in [
        "topic",
        "name",
        "id",
        "title",
        "start_minute",
        "end_minute",
        "duration_minutes",
    ] {
        if let Some(value) = step.action.input.get(key) {
            if value.is_string() || value.is_number() {
                let rendered = value.to_string();
                if description.len() + key.len() + rendered.len() + 2 <= 2048 {
                    description.push(' ');
                    description.push_str(key);
                    description.push('=');
                    description.push_str(&rendered);
                }
            }
        }
    }
    if let Some(object) = input.as_object_mut() {
        // An explicitly provided description may include user content. Keep it
        // in the local episode; the derived description contains metadata only.
        object.insert("description".into(), json!(description));
    } else {
        input = json!({"description":description});
    }
    // Forward only observed outcomes; an absent `ok` remains unknown. In
    // particular, successful capture/enqueue is not successful tool execution.
    Some(TraceCall {
        name: step.action.name.clone(),
        input,
        result: step.result.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Action;

    fn step(name: &str, input: Value, result: Option<Value>) -> EpisodeStep {
        EpisodeStep {
            schema_version: 1,
            episode_id: "test".into(),
            sequence: 0,
            pre_state: Value::Null,
            action: Action {
                name: name.into(),
                input,
            },
            result,
            post_state: Value::Null,
            started_ms: 0,
            ended_ms: 1,
            audio: vec![],
        }
    }

    #[test]
    fn description_preserves_identifiers_without_smuggling_note_content() {
        let original = step(
            "notes_put",
            json!({"topic":"people","name":"Ada","note":"private long-form note","description":"raw transcript"}),
            Some(json!({"ok":true,"result":{"stored":true}})),
        );
        let trace = project(&original).unwrap();
        assert_eq!(trace.name, "notes_put");
        assert_eq!(
            trace.input["description"],
            "notes_put topic=\"people\" name=\"Ada\""
        );
        assert_eq!(trace.result.unwrap()["ok"], true);
        assert_eq!(original.action.input["description"], "raw transcript");
        assert!(trace.input.get("command").is_none());
    }

    #[test]
    fn failed_unknown_and_non_tool_events_stay_honest() {
        assert_eq!(
            project(&step("notes_put", json!({}), Some(json!({"ok":false}))))
                .unwrap()
                .result
                .unwrap()["ok"],
            false
        );
        assert!(project(&step("notes_put", json!({}), None))
            .unwrap()
            .result
            .is_none());
        for name in [
            "reply",
            "audio_observation",
            "memorable_recall",
            "memorable_ingest",
        ] {
            assert!(project(&step(name, json!({}), Some(json!({"ok":true})))).is_none());
        }
    }
}
