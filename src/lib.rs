//! Nelly's synchronous, single-owner tool core. Embed this directly for the lowest
//! overhead; JSONL transport, journaling, model inference and memory are adapters.
pub mod audio;
pub mod memory;
pub mod model;
pub mod notes;
pub mod persistence;
pub mod schedule;
pub mod session;

use notes::Notes;
use schedule::Schedule;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

const MAX_PAGE: usize = 256;
const DEFAULT_PAGE: usize = 32;
fn page_size() -> usize {
    DEFAULT_PAGE
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
#[serde(tag = "tool", rename_all = "snake_case", deny_unknown_fields)]
pub enum ToolCall {
    NotesGet {
        topic: String,
        name: String,
    },
    NotesList {
        #[serde(default)]
        topic: Option<String>,
        #[serde(default)]
        offset: usize,
        #[serde(default = "page_size")]
        limit: usize,
    },
    NotesPut {
        topic: String,
        name: String,
        note: String,
    },
    NotesDelete {
        topic: String,
        name: String,
    },
    ScheduleGet {
        id: u64,
    },
    ScheduleList {
        start_minute: i64,
        end_minute: i64,
        #[serde(default)]
        offset: usize,
        #[serde(default = "page_size")]
        limit: usize,
    },
    ScheduleIsFree {
        start_minute: i64,
        end_minute: i64,
    },
    ScheduleFirstFree {
        start_minute: i64,
        end_minute: i64,
        duration_minutes: u32,
    },
    ScheduleCreate {
        title: String,
        start_minute: i64,
        end_minute: i64,
    },
    ScheduleUpdate {
        id: u64,
        title: String,
        start_minute: i64,
        end_minute: i64,
    },
    ScheduleDelete {
        id: u64,
    },
    Stats,
}

impl ToolCall {
    pub fn is_read(&self) -> bool {
        !matches!(
            self,
            Self::NotesPut { .. }
                | Self::NotesDelete { .. }
                | Self::ScheduleCreate { .. }
                | Self::ScheduleUpdate { .. }
                | Self::ScheduleDelete { .. }
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::NotesGet { .. } => "notes_get",
            Self::NotesList { .. } => "notes_list",
            Self::NotesPut { .. } => "notes_put",
            Self::NotesDelete { .. } => "notes_delete",
            Self::ScheduleGet { .. } => "schedule_get",
            Self::ScheduleList { .. } => "schedule_list",
            Self::ScheduleIsFree { .. } => "schedule_is_free",
            Self::ScheduleFirstFree { .. } => "schedule_first_free",
            Self::ScheduleCreate { .. } => "schedule_create",
            Self::ScheduleUpdate { .. } => "schedule_update",
            Self::ScheduleDelete { .. } => "schedule_delete",
            Self::Stats => "stats",
        }
    }

    fn validate_read(&self) -> Result<(), String> {
        match self {
            Self::NotesGet { topic, name } | Self::NotesDelete { topic, name } => {
                if topic.len() > notes::MAX_COMPONENT_BYTES
                    || name.len() > notes::MAX_COMPONENT_BYTES
                {
                    return Err("note key too long".into());
                }
            }
            Self::NotesList { topic, limit, .. } => {
                if topic
                    .as_ref()
                    .is_some_and(|v| v.len() > notes::MAX_COMPONENT_BYTES)
                {
                    return Err("topic too long".into());
                }
                validate_limit(*limit)?;
            }
            Self::ScheduleList { limit, .. } => validate_limit(*limit)?,
            _ => {}
        }
        Ok(())
    }
}

fn validate_limit(limit: usize) -> Result<(), String> {
    if limit == 0 || limit > MAX_PAGE {
        Err(format!("limit must be 1..={MAX_PAGE}"))
    } else {
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct ToolResult {
    pub value: Arc<Value>,
    pub cached: bool,
    pub revision: u64,
}

struct CacheEntry {
    value: Arc<Value>,
    bytes: usize,
}

/// FIFO eviction avoids a linked-list allocation/touch on every hit. Limits
/// cover serialized keys+values; HashMap/Arc bookkeeping adds modest overhead.
struct ReadCache {
    entries: HashMap<ToolCall, CacheEntry>,
    order: VecDeque<ToolCall>,
    bytes: usize,
    max_entries: usize,
    max_bytes: usize,
}

impl ReadCache {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            max_entries,
            max_bytes,
        }
    }
    fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.bytes = 0;
    }
    fn insert(&mut self, call: &ToolCall, value: Arc<Value>) {
        if self.max_entries == 0 || self.max_bytes == 0 {
            return;
        }
        let bytes = serde_json::to_vec(call)
            .map_or(self.max_bytes, |s| s.len())
            .saturating_mul(2)
            .saturating_add(serde_json::to_vec(value.as_ref()).map_or(self.max_bytes, |s| s.len()));
        if bytes > self.max_bytes {
            return;
        }
        while self.entries.len() >= self.max_entries || self.bytes + bytes > self.max_bytes {
            let Some(old) = self.order.pop_front() else {
                break;
            };
            if let Some(entry) = self.entries.remove(&old) {
                self.bytes -= entry.bytes;
            }
        }
        self.bytes += bytes;
        self.order.push_back(call.clone());
        self.entries
            .insert(call.clone(), CacheEntry { value, bytes });
    }
}

pub struct Harness {
    notes: Notes,
    schedule: Schedule,
    cache: ReadCache,
    revision: u64,
}

impl Default for Harness {
    fn default() -> Self {
        Self::new(128, 128 * 1024)
    }
}

impl Harness {
    pub fn new(cache_entries: usize, cache_bytes: usize) -> Self {
        Self {
            notes: Notes::default(),
            schedule: Schedule::default(),
            cache: ReadCache::new(cache_entries, cache_bytes),
            revision: 0,
        }
    }
    pub fn notes(&self) -> &Notes {
        &self.notes
    }
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Prefetch calls use the same cache key as later committed reads. A write
    /// cannot be executed speculatively, even if emitted by a model in error.
    pub fn execute(&mut self, call: &ToolCall, speculative: bool) -> Result<ToolResult, String> {
        if speculative && !call.is_read() {
            return Err("speculative writes are forbidden".into());
        }
        call.validate_read()?;
        if call.is_read() && !matches!(call, ToolCall::Stats) {
            if let Some(entry) = self.cache.entries.get(call) {
                return Ok(ToolResult {
                    value: Arc::clone(&entry.value),
                    cached: true,
                    revision: self.revision,
                });
            }
        }
        if !call.is_read() && self.revision == u64::MAX {
            return Err("revision exhausted".into());
        }
        let value = match call {
            ToolCall::NotesGet { topic, name } => json!({"note":self.notes.get(topic,name)}),
            ToolCall::NotesList {
                topic,
                offset,
                limit,
            } => json!(self.notes.list(topic.as_deref(), *offset, *limit)),
            ToolCall::NotesPut { topic, name, note } => {
                self.notes.put(topic.clone(), name.clone(), note.clone())?;
                json!({"stored":true})
            }
            ToolCall::NotesDelete { topic, name } => {
                json!({"deleted":self.notes.delete(topic,name)})
            }
            ToolCall::ScheduleGet { id } => json!(self.schedule.get(*id)),
            ToolCall::ScheduleList {
                start_minute,
                end_minute,
                offset,
                limit,
            } => {
                json!(self
                    .schedule
                    .events_page(*start_minute, *end_minute, *offset, *limit)?)
            }
            ToolCall::ScheduleIsFree {
                start_minute,
                end_minute,
            } => json!({"free":self.schedule.is_free(*start_minute,*end_minute)?}),
            ToolCall::ScheduleFirstFree {
                start_minute,
                end_minute,
                duration_minutes,
            } => {
                json!({"start_minute":self.schedule.first_free(*start_minute,*end_minute,*duration_minutes)?})
            }
            ToolCall::ScheduleCreate {
                title,
                start_minute,
                end_minute,
            } => json!(self
                .schedule
                .create(title.clone(), *start_minute, *end_minute)?),
            ToolCall::ScheduleUpdate {
                id,
                title,
                start_minute,
                end_minute,
            } => json!(self
                .schedule
                .update(*id, title.clone(), *start_minute, *end_minute)?),
            ToolCall::ScheduleDelete { id } => {
                json!({"deleted":self.schedule.delete(*id).is_some()})
            }
            ToolCall::Stats => {
                json!({"revision":self.revision,"notes":self.notes.len(),"note_payload_bytes":self.notes.payload_bytes(),
                "events":self.schedule.len(),"bitmap_bytes":self.schedule.bitmap_bytes(),
                "cached_reads":self.cache.entries.len(),"cache_payload_bytes":self.cache.bytes})
            }
        };
        let value = Arc::new(value);
        if call.is_read() {
            if !matches!(call, ToolCall::Stats) {
                self.cache.insert(call, Arc::clone(&value));
            }
        } else {
            self.revision += 1;
            self.cache.clear();
        }
        Ok(ToolResult {
            value,
            cached: false,
            revision: self.revision,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefetch_is_reused_but_never_survives_a_write() {
        let mut h = Harness::default();
        let read = ToolCall::NotesGet {
            topic: "people".into(),
            name: "Ada".into(),
        };
        let a = h.execute(&read, true).unwrap();
        let b = h.execute(&read, false).unwrap();
        assert!(b.cached);
        assert!(Arc::ptr_eq(&a.value, &b.value));
        let write = ToolCall::NotesPut {
            topic: "people".into(),
            name: "Ada".into(),
            note: "likes tea".into(),
        };
        assert!(h.execute(&write, true).is_err());
        assert_eq!(h.revision(), 0);
        h.execute(&write, false).unwrap();
        let c = h.execute(&read, false).unwrap();
        assert!(!c.cached);
        assert_eq!(c.value["note"], "likes tea");
        assert_eq!(c.revision, 1);
        // Old shared results can be held by consumers, with their original revision.
        assert_eq!(a.value["note"], Value::Null);
    }
    #[test]
    fn cache_is_bounded_and_fifo() {
        let mut h = Harness::new(2, 1024);
        let read = |name: &str| ToolCall::NotesGet {
            topic: "x".into(),
            name: name.into(),
        };
        for n in ["a", "b", "c"] {
            h.execute(&read(n), true).unwrap();
        }
        assert_eq!(h.cache.entries.len(), 2);
        assert!(!h.execute(&read("a"), false).unwrap().cached);
        assert!(h.cache.bytes <= 1024);
        let mut h = Harness::new(2, 1);
        assert!(!h.execute(&read("a"), true).unwrap().cached);
        assert!(!h.execute(&read("a"), false).unwrap().cached);
    }
    #[test]
    fn malformed_calls_are_rejected() {
        assert!(serde_json::from_value::<ToolCall>(
            json!({"tool":"notes_get","topic":"a","name":"b","extra":1})
        )
        .is_err());
        assert!(Harness::default()
            .execute(
                &ToolCall::NotesList {
                    topic: None,
                    offset: 0,
                    limit: usize::MAX
                },
                false
            )
            .is_err());
    }
}
