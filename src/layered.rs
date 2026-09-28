//! Compose the synchronous application core with optional background memory.
//!
//! Notes and scheduling calls delegate directly to the inner executor. Only
//! Memorable calls touch the worker or its cache, so a cold recall never blocks
//! the application tools or the voice model's inference thread.

use crate::{
    layers::{MemorableRequest, MemoryLayers, MemoryReply},
    session::ToolExecutor,
    ToolCall, ToolResult,
};
use serde_json::json;
use std::{io, sync::Arc};

pub struct LayeredExecutor<E> {
    engine: E,
    memory: Option<MemoryLayers>,
}

impl<E: ToolExecutor> LayeredExecutor<E> {
    pub fn new(engine: E, memory: Option<MemoryLayers>) -> Self {
        Self { engine, memory }
    }

    pub fn inner(&self) -> &E {
        &self.engine
    }

    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    pub fn memory_mut(&mut self) -> Option<&mut MemoryLayers> {
        self.memory.as_mut()
    }

    /// Drain accepted Memorable work outside the real-time inference loop.
    pub fn shutdown(self) -> io::Result<()> {
        match self.memory {
            Some(memory) => memory.shutdown(),
            None => Ok(()),
        }
    }

    fn memory(&mut self) -> Result<&mut MemoryLayers, String> {
        self.memory.as_mut().ok_or_else(|| {
            "Memorable memory is disabled; enable --memorable or attach MemoryLayers".into()
        })
    }

    fn submit(&mut self, request: MemorableRequest) -> Result<ToolResult, String> {
        let reply = self
            .memory()?
            .submit(request)
            .map_err(|error| error.to_string())?;
        self.reply(reply)
    }

    fn reply(&self, reply: MemoryReply) -> Result<ToolResult, String> {
        let cached = reply.cached;
        Ok(ToolResult {
            value: Arc::new(serde_json::to_value(reply).map_err(|error| error.to_string())?),
            cached,
            revision: self.engine.revision(),
        })
    }
}

impl<E: ToolExecutor> ToolExecutor for LayeredExecutor<E> {
    fn execute(&mut self, call: &ToolCall, speculative: bool) -> Result<ToolResult, String> {
        if speculative && !call.is_read() {
            return Err("speculative writes are forbidden".into());
        }
        match call {
            ToolCall::MemorableRecall { query } => self.submit(MemorableRequest::Recall {
                query: query.clone(),
            }),
            ToolCall::MemorableShow { slug } => {
                self.submit(MemorableRequest::Show { slug: slug.clone() })
            }
            ToolCall::MemorableChain { query } => self.submit(MemorableRequest::Chain {
                query: query.clone(),
            }),
            ToolCall::MemorableList { all } => self.submit(MemorableRequest::List { all: *all }),
            ToolCall::MemorableStatus => self.submit(MemorableRequest::Status),
            ToolCall::MemorableIngest { episode_id } => self.submit(MemorableRequest::Ingest {
                episode_id: episode_id.clone(),
            }),
            ToolCall::MemorablePoll { job_id } => {
                let reply = self
                    .memory()?
                    .poll(*job_id)
                    .map_err(|error| error.to_string())?;
                self.reply(reply)
            }
            ToolCall::MemorableInvalidate => {
                let generation = self.memory()?.invalidate();
                Ok(ToolResult {
                    value: Arc::new(json!({"generation": generation, "invalidated": true})),
                    cached: false,
                    revision: self.engine.revision(),
                })
            }
            ToolCall::MemorableDisable => {
                self.memory()?.disable();
                Ok(ToolResult {
                    value: Arc::new(json!({"enabled": false})),
                    cached: false,
                    revision: self.engine.revision(),
                })
            }
            _ => self.engine.execute(call, speculative),
        }
    }

    fn revision(&self) -> u64 {
        self.engine.revision()
    }
}
