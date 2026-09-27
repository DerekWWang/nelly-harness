//! Exact UTF-8 topic/name lookup. Borrowed reads do not allocate.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const MAX_COMPONENT_BYTES: usize = 256;
pub const MAX_NOTE_BYTES: usize = 16 * 1024;
pub const MAX_NOTES: usize = 4096;
pub const MAX_TOTAL_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Note {
    pub topic: String,
    pub name: String,
    pub note: String,
}

#[derive(Default)]
pub struct Notes {
    topics: HashMap<String, HashMap<String, String>>,
    len: usize,
    bytes: usize,
}

impl Notes {
    pub fn get(&self, topic: &str, name: &str) -> Option<&str> {
        self.topics.get(topic)?.get(name).map(String::as_str)
    }

    pub fn put(&mut self, topic: String, name: String, note: String) -> Result<(), String> {
        for (label, value) in [("topic", &topic), ("name", &name)] {
            if value.trim().is_empty() || value.len() > MAX_COMPONENT_BYTES {
                return Err(format!(
                    "{label} must contain 1..={MAX_COMPONENT_BYTES} UTF-8 bytes"
                ));
            }
        }
        if note.len() > MAX_NOTE_BYTES {
            return Err(format!(
                "note exceeds {MAX_NOTE_BYTES} bytes; use episodic memory for long-form content"
            ));
        }
        let old = self.get(&topic, &name);
        let new_entry = old.is_none();
        if new_entry && self.len >= MAX_NOTES {
            return Err("note count limit reached".into());
        }
        let new_bytes = self.bytes - old.map_or(0, str::len)
            + note.len()
            + if new_entry {
                topic.len() + name.len()
            } else {
                0
            };
        if new_bytes > MAX_TOTAL_BYTES {
            return Err("note byte limit reached".into());
        }
        self.topics.entry(topic).or_default().insert(name, note);
        self.bytes = new_bytes;
        self.len += usize::from(new_entry);
        Ok(())
    }

    pub fn delete(&mut self, topic: &str, name: &str) -> bool {
        let Some(notes) = self.topics.get_mut(topic) else {
            return false;
        };
        let Some(note) = notes.remove(name) else {
            return false;
        };
        self.len -= 1;
        self.bytes -= topic.len() + name.len() + note.len();
        if notes.is_empty() {
            self.topics.remove(topic);
        }
        true
    }

    /// Deterministic pagination; only the selected page is cloned.
    pub fn list(&self, topic: Option<&str>, offset: usize, limit: usize) -> Vec<Note> {
        let mut rows: Vec<(&String, &String, &String)> = Vec::new();
        if let Some(topic) = topic {
            if let Some((stored_topic, notes)) = self.topics.get_key_value(topic) {
                rows.extend(notes.iter().map(|(name, note)| (stored_topic, name, note)));
            }
        } else {
            for (topic, notes) in &self.topics {
                rows.extend(notes.iter().map(|(name, note)| (topic, name, note)));
            }
        }
        rows.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        rows.into_iter()
            .skip(offset)
            .take(limit)
            .map(|(topic, name, note)| Note {
                topic: topic.clone(),
                name: name.clone(),
                note: note.clone(),
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn payload_bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crud_exact_keys_and_accounting() {
        let mut n = Notes::default();
        n.put("a/b".into(), "c".into(), "one".into()).unwrap();
        n.put("a".into(), "b/c".into(), "two".into()).unwrap();
        n.put("a/b".into(), "c".into(), "updated".into()).unwrap();
        assert_eq!(n.get("a/b", "c"), Some("updated"));
        assert_eq!(n.get("a", "b/c"), Some("two"));
        assert_eq!(n.len(), 2);
        assert_eq!(n.list(None, 1, 1)[0].note, "updated");
        assert!(n.delete("a/b", "c"));
        assert!(!n.delete("a/b", "c"));
        assert!(n.delete("a", "b/c"));
        assert_eq!(n.payload_bytes(), 0);
        assert!(n.is_empty());
    }
    #[test]
    fn rejected_updates_preserve_value() {
        let mut n = Notes::default();
        n.put("x".into(), "y".into(), "old".into()).unwrap();
        assert!(n
            .put("x".into(), "y".into(), "z".repeat(MAX_NOTE_BYTES + 1))
            .is_err());
        assert_eq!(n.get("x", "y"), Some("old"));
        assert!(n.put(" ".into(), "y".into(), "v".into()).is_err());
    }
}
