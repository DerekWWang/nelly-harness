//! Bounded speculative memory reads over the cold Memorable CLI.
//!
//! `submit`, `poll`, and `invalidate` only touch bounded in-memory state and
//! nonblocking channels. One worker owns all process and filesystem operations.
//! These are low-latency control operations, not allocation-free audio callbacks.
//! Call `shutdown` on the cold path to drain every accepted operation.

use crate::memory::{read_consent_allowed, CliOutput, MemorableCli, MemoryStore};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum MemorableRequest {
    Recall {
        query: String,
    },
    Show {
        slug: String,
    },
    Chain {
        query: String,
    },
    List {
        #[serde(default)]
        all: bool,
    },
    Status,
    Ingest {
        episode_id: String,
    },
}

impl MemorableRequest {
    pub fn is_read(&self) -> bool {
        !matches!(self, Self::Ingest { .. })
    }

    fn cacheable(&self) -> bool {
        self.is_read() && !matches!(self, Self::Status)
    }

    fn validate(&self) -> io::Result<()> {
        match self {
            Self::Recall { query } | Self::Chain { query }
                if query.trim().is_empty()
                    || query.len() > 16 * 1024
                    || query.starts_with('-')
                    || query.contains('\0') =>
            {
                Err(invalid(
                    "query must contain 1..16384 bytes and cannot start with '-' or contain NUL",
                ))
            }
            Self::Show { slug }
                if !slug.starts_with("procedures/")
                    || slug.len() <= "procedures/".len()
                    || slug.len() > 512
                    || slug.contains("..")
                    || !slug
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b"/-_".contains(&c)) =>
            {
                Err(invalid("expected a procedures/<slug> identifier"))
            }
            Self::Ingest { episode_id }
                if episode_id.is_empty()
                    || episode_id.len() > 128
                    || !episode_id
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c)) =>
            {
                Err(invalid(
                    "episode id must contain 1..128 ASCII letters, digits, '_' or '-'",
                ))
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub queue_capacity: usize,
    pub max_jobs: usize,
    /// Maximum combined serialized bytes of retained payloads and errors.
    /// Cache references share job payloads and do not consume this budget twice.
    /// Caller-held reply Arcs and bounded worker temporaries are independent.
    pub max_result_bytes: usize,
    pub ttl: Duration,
    pub max_cache_entries: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 4,
            max_jobs: 32,
            max_result_bytes: 256 * 1024,
            ttl: Duration::from_secs(300),
            max_cache_entries: 32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    Ready,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryReply {
    pub job_id: u64,
    pub status: JobStatus,
    pub cached: bool,
    /// Read submission generation. Compare with `stats().generation` to detect
    /// a result completed after explicit invalidation or a successful ingest.
    pub generation: u64,
    pub data: Option<Arc<Value>>,
    pub error: Option<String>,
}

struct Job {
    request: MemorableRequest,
    reply: MemoryReply,
    bytes: usize,
}

struct CacheEntry {
    request: MemorableRequest,
    job_id: u64,
    expires: Instant,
}

struct Command {
    job_id: u64,
    request: MemorableRequest,
}

struct Completion {
    job_id: u64,
    result: Result<(Arc<Value>, usize), String>,
    invalidates: bool,
    read_denied: bool,
    finished_at: Instant,
}

/// Single-owner fast layer. Completed jobs are retained FIFO until job or byte
/// pressure evicts them; polling an expired/evicted job returns `NotFound`.
/// Pending jobs are never evicted. A full queue returns `WouldBlock` and rejects
/// the request, so callers can retry without an uncertain side effect.
///
/// Dropping detaches the worker, which still drains accepted requests while the
/// process remains alive. Use `shutdown` for an observable completion barrier.
pub struct MemoryLayers {
    config: MemoryConfig,
    sender: Option<mpsc::SyncSender<Command>>,
    receiver: mpsc::Receiver<Completion>,
    worker: Option<JoinHandle<()>>,
    jobs: HashMap<u64, Job>,
    pending: HashMap<MemorableRequest, u64>,
    completed: VecDeque<u64>,
    cache: VecDeque<CacheEntry>,
    generation: u64,
    next_job: u64,
    result_bytes: usize,
    disconnected: bool,
    first_failure: Option<String>,
    enabled: Arc<AtomicBool>,
}

impl MemoryLayers {
    pub fn new(
        store: MemoryStore,
        mut cli: MemorableCli,
        config: MemoryConfig,
    ) -> io::Result<Self> {
        if !(1..=1024).contains(&config.queue_capacity)
            || !(1..=1024).contains(&config.max_jobs)
            || !(256..=16 * 1024 * 1024).contains(&config.max_result_bytes)
            || config.max_cache_entries > config.max_jobs
            || cli.timeout.is_zero()
            || cli.max_output_bytes == 0
        {
            return Err(invalid("invalid memory limits: queue/jobs 1..1024, result bytes 256..16777216, cache <= jobs, positive CLI timeout/output limit"));
        }
        if Instant::now().checked_add(config.ttl).is_none() {
            return Err(invalid("memory cache TTL is too large"));
        }
        // Each stream and each completion has an independent upper bound; only
        // one completion can wait in the channel, plus the worker's current job.
        cli.max_output_bytes = cli
            .max_output_bytes
            .min(config.max_result_bytes)
            .min(1024 * 1024);
        let (sender, commands) = mpsc::sync_channel::<Command>(config.queue_capacity);
        let (finished, receiver) = mpsc::sync_channel(1);
        let max_result_bytes = config.max_result_bytes;
        let enabled = Arc::new(AtomicBool::new(true));
        let worker_enabled = Arc::clone(&enabled);
        let worker = std::thread::Builder::new()
            .name("nelly-memorable".into())
            .spawn(move || {
                while let Ok(command) = commands.recv() {
                    let result = if worker_enabled.load(Ordering::Acquire) {
                        execute(&store, &cli, &command.request)
                    } else {
                        Err(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            "local Memorable access disabled",
                        ))
                    };
                    // A successful mutation must invalidate even if its reply
                    // cannot fit the configured retained-result byte budget.
                    let invalidates = result.is_ok() && !command.request.is_read();
                    let read_denied = command.request.is_read()
                        && result
                            .as_ref()
                            .is_err_and(|error| error.kind() == io::ErrorKind::PermissionDenied);
                    if read_denied {
                        worker_enabled.store(false, Ordering::Release);
                    }
                    let result = result
                        .and_then(|value| {
                            let bytes = serialized_size(&value, max_result_bytes)?;
                            Ok((Arc::new(value), bytes))
                        })
                        .map_err(|error| {
                            bounded_error(&error.to_string(), max_result_bytes.min(1024))
                        });
                    // A dropped foreground owner must not cancel accepted
                    // ingests; keep draining commands if replies disconnect.
                    let _ = finished.send(Completion {
                        job_id: command.job_id,
                        result,
                        invalidates,
                        read_denied,
                        finished_at: Instant::now(),
                    });
                }
            })?;
        Ok(Self {
            config,
            sender: Some(sender),
            receiver,
            worker: Some(worker),
            jobs: HashMap::new(),
            pending: HashMap::new(),
            completed: VecDeque::new(),
            cache: VecDeque::new(),
            generation: 0,
            next_job: 1,
            result_bytes: 0,
            disconnected: false,
            first_failure: None,
            enabled,
        })
    }

    pub fn submit(&mut self, request: MemorableRequest) -> io::Result<MemoryReply> {
        request.validate()?;
        self.drain();
        if !self.is_enabled() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "local Memorable access disabled",
            ));
        }
        self.expire_cache();
        if request.cacheable() {
            if let Some(entry) = self.cache.iter().find(|entry| entry.request == request) {
                if let Some(job) = self.jobs.get(&entry.job_id) {
                    let mut reply = job.reply.clone();
                    reply.cached = true;
                    return Ok(reply);
                }
            }
        }
        if let Some(id) = self.pending.get(&request) {
            let job = &self.jobs[id];
            // After invalidation a fresh read must not attach to an old read.
            // Ingests remain single-flight across generations.
            if !request.is_read() || job.reply.generation == self.generation {
                return Ok(job.reply.clone());
            }
        }
        if self.disconnected {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "Memorable worker stopped",
            ));
        }
        while self.jobs.len() >= self.config.max_jobs {
            if !self.evict_oldest() {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "memory job limit reached; request was not accepted",
                ));
            }
        }
        let job_id = self.next_job;
        let next = job_id
            .checked_add(1)
            .ok_or_else(|| io::Error::other("memory job ids exhausted"))?;
        let command = Command {
            job_id,
            request: request.clone(),
        };
        match self
            .sender
            .as_ref()
            .expect("sender exists before shutdown")
            .try_send(command)
        {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(_)) => {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "memory queue full; request was not accepted",
                ))
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                self.mark_disconnected();
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "Memorable worker stopped",
                ));
            }
        }
        self.next_job = next;
        let reply = MemoryReply {
            job_id,
            status: JobStatus::Pending,
            cached: false,
            generation: self.generation,
            data: None,
            error: None,
        };
        self.pending.insert(request.clone(), job_id);
        self.jobs.insert(
            job_id,
            Job {
                request,
                reply: reply.clone(),
                bytes: 0,
            },
        );
        Ok(reply)
    }

    pub fn poll(&mut self, job_id: u64) -> io::Result<MemoryReply> {
        self.drain();
        self.jobs
            .get(&job_id)
            .map(|job| {
                let mut reply = job.reply.clone();
                if !self.is_enabled() {
                    reply.status = JobStatus::Failed;
                    reply.data = None;
                    reply
                        .error
                        .get_or_insert_with(|| "local Memorable access disabled".into());
                }
                reply
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "memory job is unknown or its retained result was evicted",
                )
            })
    }

    /// Existing job replies remain available with their original generation.
    /// Old pending results are never promoted into the new generation's cache.
    pub fn invalidate(&mut self) -> u64 {
        self.generation = self.generation.wrapping_add(1);
        self.cache.clear();
        self.generation
    }

    /// Revoke local access immediately, clear retained results, stop queued
    /// commands, and discard results of in-flight commands. An already-running
    /// ingest can still finish its external side effect. Previously returned
    /// caller-owned reply Arcs cannot be recalled. Create a new layer to enable
    /// access again; there is deliberately no model-callable enable operation.
    pub fn disable(&mut self) -> u64 {
        self.enabled.store(false, Ordering::Release);
        let generation = self.invalidate();
        while self.evict_oldest() {}
        generation
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    pub fn stats(&mut self) -> Value {
        self.drain();
        self.expire_cache();
        json!({
            "generation": self.generation,
            "jobs": self.jobs.len(),
            "pending": self.jobs.values().filter(|job| job.reply.status == JobStatus::Pending).count(),
            "cache_entries": self.cache.len(),
            "result_bytes": self.result_bytes,
            "max_result_bytes": self.config.max_result_bytes,
            "max_jobs": self.config.max_jobs,
            "worker_stopped": self.disconnected,
            "enabled": self.is_enabled(),
        })
    }

    /// Cold barrier: close the request queue, drain replies concurrently with
    /// the worker, then join. Draining prevents a full reply channel deadlock.
    /// Reports the first failed accepted job, even if its result was evicted.
    pub fn shutdown(mut self) -> io::Result<()> {
        drop(self.sender.take());
        while let Ok(completion) = self.receiver.recv() {
            self.complete(completion);
        }
        if self
            .worker
            .take()
            .expect("worker exists before shutdown")
            .join()
            .is_err()
        {
            return Err(io::Error::other("Memorable worker panicked"));
        }
        if self
            .jobs
            .values()
            .any(|job| job.reply.status == JobStatus::Pending)
        {
            return Err(io::Error::other(
                "Memorable worker stopped before completing accepted work",
            ));
        }
        if let Some(error) = self.first_failure.take() {
            return Err(io::Error::other(error));
        }
        Ok(())
    }

    fn drain(&mut self) {
        // A fixed iteration bound also bounds work if producer and consumer
        // interleave continuously on an unusually fast fake or embedded CLI.
        for _ in 0..self.config.max_jobs {
            match self.receiver.try_recv() {
                Ok(completion) => self.complete(completion),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.mark_disconnected();
                    break;
                }
            }
        }
    }

    fn complete(&mut self, mut completion: Completion) {
        if completion.read_denied {
            self.disable();
        }
        if !self.is_enabled() {
            if !completion.read_denied {
                completion.result = Err("local Memorable access disabled".into());
            }
            completion.invalidates = false;
        }
        let Some(job) = self.jobs.get(&completion.job_id) else {
            return;
        };
        let request = job.request.clone();
        let generation = job.reply.generation;
        if self.pending.get(&request) == Some(&completion.job_id) {
            self.pending.remove(&request);
        }
        let success = completion.result.is_ok();
        if completion.invalidates {
            self.invalidate();
        }
        let (data, error, bytes) = match completion.result {
            Ok((data, bytes)) => (Some(data), None, bytes),
            Err(error) => {
                self.first_failure.get_or_insert_with(|| error.clone());
                let bytes = error.len();
                (None, Some(error), bytes)
            }
        };
        while self.result_bytes + bytes > self.config.max_result_bytes {
            if !self.evict_oldest() {
                break;
            }
        }
        let job = self
            .jobs
            .get_mut(&completion.job_id)
            .expect("pending jobs are never evicted");
        job.reply.status = if success {
            JobStatus::Ready
        } else {
            JobStatus::Failed
        };
        job.reply.data = data;
        job.reply.error = error;
        if completion.invalidates {
            job.reply.generation = self.generation;
        }
        job.bytes = bytes;
        self.result_bytes += bytes;
        self.completed.push_back(completion.job_id);
        if success
            && request.cacheable()
            && generation == self.generation
            && self.config.max_cache_entries > 0
            && !self.config.ttl.is_zero()
        {
            let expires = completion
                .finished_at
                .checked_add(self.config.ttl)
                .unwrap_or(completion.finished_at);
            if expires <= Instant::now() {
                return;
            }
            self.expire_cache();
            while self.cache.len() >= self.config.max_cache_entries {
                self.cache.pop_front();
            }
            self.cache.push_back(CacheEntry {
                request,
                job_id: completion.job_id,
                expires,
            });
        }
    }

    fn expire_cache(&mut self) {
        let now = Instant::now();
        self.cache.retain(|entry| entry.expires > now);
    }

    fn evict_oldest(&mut self) -> bool {
        let Some(id) = self.completed.pop_front() else {
            return false;
        };
        if let Some(job) = self.jobs.remove(&id) {
            self.result_bytes -= job.bytes;
        }
        self.cache.retain(|entry| entry.job_id != id);
        true
    }

    fn mark_disconnected(&mut self) {
        if self.disconnected {
            return;
        }
        self.disconnected = true;
        let pending: Vec<_> = self
            .jobs
            .iter()
            .filter_map(|(&id, job)| (job.reply.status == JobStatus::Pending).then_some(id))
            .collect();
        for job_id in pending {
            self.complete(Completion {
                job_id,
                result: Err("Memorable worker stopped before completing this job".into()),
                invalidates: false,
                read_denied: false,
                finished_at: Instant::now(),
            });
        }
    }
}

fn execute(
    store: &MemoryStore,
    cli: &MemorableCli,
    request: &MemorableRequest,
) -> io::Result<Value> {
    let output = match request {
        MemorableRequest::Recall { query } => cli.recall(query)?,
        MemorableRequest::Show { slug } => cli.show(slug)?,
        MemorableRequest::Chain { query } => cli.chain(query)?,
        MemorableRequest::List { all } => cli.list(*all)?,
        MemorableRequest::Status => cli.status()?,
        MemorableRequest::Ingest { episode_id } => {
            let outcome = store.sync(episode_id, cli)?;
            return Ok(json!({
                "source": "memorable",
                "trusted": false,
                "acknowledged": true,
                "workflow_stored": null,
                "already_synced": outcome.already_synced,
                "text": outcome.stdout,
            }));
        }
    };
    if matches!(request, MemorableRequest::Status) && !read_consent_allowed(&output.stdout) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "Memorable status reports denied or unknown read consent",
        ));
    }
    if matches!(
        request,
        MemorableRequest::Chain { .. } | MemorableRequest::List { .. }
    ) {
        let parsed: Value = serde_json::from_str(&output.stdout).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Memorable returned malformed JSON",
            )
        })?;
        let shape_valid = match request {
            MemorableRequest::Chain { .. } => parsed.get("items").is_some_and(Value::is_array),
            MemorableRequest::List { .. } => parsed.is_array(),
            _ => unreachable!(),
        };
        if !shape_valid {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Memorable returned an unexpected JSON shape",
            ));
        }
        return Ok(
            json!({"source": "memorable", "trusted": false, "json": parsed, "stderr": output.stderr}),
        );
    }
    Ok(untrusted_output(output))
}

fn untrusted_output(output: CliOutput) -> Value {
    match serde_json::from_str::<Value>(&output.stdout) {
        Ok(value) => {
            json!({"source": "memorable", "trusted": false, "json": value, "stderr": output.stderr})
        }
        Err(_) => {
            json!({"source": "memorable", "trusted": false, "text": output.stdout, "stderr": output.stderr})
        }
    }
}

fn serialized_size(value: &Value, limit: usize) -> io::Result<usize> {
    struct Counter {
        bytes: usize,
        limit: usize,
    }
    impl Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > self.limit.saturating_sub(self.bytes) {
                return Err(io::Error::other(
                    "Memorable result exceeds memory result budget",
                ));
            }
            self.bytes += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter { bytes: 0, limit };
    serde_json::to_writer(&mut counter, value)?;
    Ok(counter.bytes)
}

fn bounded_error(message: &str, limit: usize) -> String {
    let mut end = limit.min(message.len());
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message[..end].to_owned()
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
