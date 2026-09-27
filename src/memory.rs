//! Durable local episodes and an explicitly invoked, cold Memorable adapter.
//!
//! Audio is copied with an 8 KiB buffer. JSONL records are individually bounded;
//! neither exports nor trace creation read an entire episode into memory.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const MEMORABLE_VERSION: &str = "0.5.30";
pub const MAX_STEP_BYTES: usize = 1_048_576;
/// The background queue is intended for compact control records, not waveforms.
pub const MAX_QUEUED_STEP_BYTES: usize = 64 * 1024;
const MAX_AUDIO_BYTES: u64 = 512 * 1024 * 1024;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeMetadata {
    pub schema_version: u32,
    pub id: String,
    pub task_description: String,
    pub created_at_ms: u64,
    pub finished_at_ms: Option<u64>,
    pub summary: Option<String>,
    pub step_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub name: String,
    pub input: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSource {
    pub path: PathBuf,
    /// Typically "observation" (user audio) or "response" (agent audio).
    pub role: String,
    pub media_type: String,
    #[serde(default)]
    pub sample_rate_hz: Option<u32>,
    #[serde(default)]
    pub channels: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioRef {
    /// Relative to the memory store root, so a moved store stays self-contained.
    pub path: String,
    pub role: String,
    pub media_type: String,
    pub bytes: u64,
    pub sample_rate_hz: Option<u32>,
    pub channels: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepInput {
    pub pre_state: Value,
    pub action: Action,
    #[serde(default)]
    pub result: Option<Value>,
    pub post_state: Value,
    /// Milliseconds on the caller's monotonic episode clock.
    pub started_ms: u64,
    pub ended_ms: u64,
    #[serde(default)]
    pub audio: Vec<AudioSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeStep {
    pub schema_version: u32,
    pub episode_id: String,
    pub sequence: u64,
    pub pre_state: Value,
    pub action: Action,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub result: Option<Value>,
    pub post_state: Value,
    pub started_ms: u64,
    pub ended_ms: u64,
    pub audio: Vec<AudioRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncOutcome {
    pub already_synced: bool,
    pub stdout: String,
}

#[derive(Debug, Clone)]
pub struct MemoryStore {
    root: PathBuf,
}

impl MemoryStore {
    pub fn open(root: impl AsRef<Path>) -> io::Result<Self> {
        crate::persistence::create_dir_all_durable(root.as_ref())?;
        Ok(Self {
            root: fs::canonicalize(root)?,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn begin(&self, id: &str, task: &str) -> io::Result<EpisodeMetadata> {
        validate_id(id)?;
        if task.is_empty() || task.len() > 64 * 1024 {
            return Err(invalid("task description must contain 1..65536 bytes"));
        }
        let directory = self.root.join(id);
        fs::create_dir(&directory)?;
        fs::create_dir(directory.join("audio"))?;
        let metadata = EpisodeMetadata {
            schema_version: 1,
            id: id.into(),
            task_description: task.into(),
            created_at_ms: now_ms(),
            finished_at_ms: None,
            summary: None,
            step_count: 0,
        };
        write_json_atomic(&directory.join("episode.json"), &metadata)?;
        File::create(directory.join("steps.jsonl"))?.sync_all()?;
        sync_directory(&directory)?;
        sync_directory(&self.root)?;
        Ok(metadata)
    }

    pub fn metadata(&self, id: &str) -> io::Result<EpisodeMetadata> {
        read_json(&self.episode_directory(id)?.join("episode.json"))
    }

    pub fn append(&self, id: &str, input: StepInput) -> io::Result<EpisodeStep> {
        let directory = self.episode_directory(id)?;
        let _lock = EpisodeLock::acquire(&directory)?;
        let metadata: EpisodeMetadata = read_json(&directory.join("episode.json"))?;
        if metadata.finished_at_ms.is_some() {
            return Err(invalid("episode is finished"));
        }
        if input.ended_ms < input.started_ms {
            return Err(invalid("ended_ms precedes started_ms"));
        }
        if input.action.name.is_empty() || input.action.name.len() > 256 {
            return Err(invalid("action name must contain 1..256 bytes"));
        }
        if input.audio.len() > 16 {
            return Err(invalid("at most 16 audio files per step"));
        }
        // Bound JSON before any audio copies or disk mutations.
        bounded_json(&input)?;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(directory.join("steps.jsonl"))?;
        let previous = recover_last_step(&mut file)?;
        let sequence = match previous {
            Some(step) => step
                .sequence
                .checked_add(1)
                .ok_or_else(|| invalid("episode sequence exhausted"))?,
            None => 0,
        };
        let original_length = file.metadata()?.len();
        let mut step = EpisodeStep {
            schema_version: 1,
            episode_id: id.into(),
            sequence,
            pre_state: input.pre_state,
            action: input.action,
            result: input.result,
            post_state: input.post_state,
            started_ms: input.started_ms,
            ended_ms: input.ended_ms,
            audio: Vec::new(),
        };
        let result = (|| {
            for (index, source) in input.audio.iter().enumerate() {
                step.audio
                    .push(self.copy_audio(id, &directory, sequence, index, source)?);
            }
            let mut bytes = bounded_json(&step)?;
            bytes.push(b'\n');
            file.seek(SeekFrom::End(0))?;
            file.write_all(&bytes)?;
            file.sync_data()
        })();
        if let Err(error) = result {
            // Keep audio if disk failure prevents rolling back its potential
            // JSONL reference. An orphan is preferable to dangling training data.
            if file
                .set_len(original_length)
                .and_then(|()| file.sync_data())
                .is_ok()
            {
                for audio in &step.audio {
                    let _ = fs::remove_file(self.root.join(&audio.path));
                }
            }
            return Err(error);
        }
        Ok(step)
    }

    pub fn finish(&self, id: &str, summary: &str) -> io::Result<EpisodeMetadata> {
        if summary.len() > 256 * 1024 {
            return Err(invalid("summary exceeds 256 KiB"));
        }
        let directory = self.episode_directory(id)?;
        let _lock = EpisodeLock::acquire(&directory)?;
        let mut metadata: EpisodeMetadata = read_json(&directory.join("episode.json"))?;
        if metadata.finished_at_ms.is_some() {
            return Ok(metadata);
        }
        let mut steps = OpenOptions::new()
            .read(true)
            .write(true)
            .open(directory.join("steps.jsonl"))?;
        recover_last_step(&mut steps)?;
        metadata.step_count = self.write_trace(&directory, &metadata)?;
        metadata.finished_at_ms = Some(now_ms());
        metadata.summary = Some(summary.into());
        write_json_atomic(&directory.join("episode.json"), &metadata)?;
        Ok(metadata)
    }

    /// Lossless local training records. Audio paths stay relative to `root()`.
    /// Metadata and final long-form summary are repeated per row intentionally.
    pub fn export_jsonl(&self, id: &str, output: &mut impl Write) -> io::Result<usize> {
        let directory = self.episode_directory(id)?;
        let _lock = EpisodeLock::acquire(&directory)?;
        let metadata: EpisodeMetadata = read_json(&directory.join("episode.json"))?;
        if metadata.finished_at_ms.is_none() {
            return Err(invalid("finish the episode before exporting"));
        }
        let mut count = 0;
        visit_steps(&directory.join("steps.jsonl"), |step| {
            #[derive(Serialize)]
            struct TrainingRow<'a> {
                episode: &'a EpisodeMetadata,
                step: &'a EpisodeStep,
            }
            serde_json::to_writer(
                &mut *output,
                &TrainingRow {
                    episode: &metadata,
                    step: &step,
                },
            )?;
            output.write_all(b"\n")?;
            count += 1;
            Ok(())
        })?;
        Ok(count)
    }

    /// Synchronous cold-path operation: never call from the audio callback.
    /// Successful CLI acknowledgement is recorded durably to avoid resubmission.
    pub fn sync(&self, id: &str, cli: &MemorableCli) -> io::Result<SyncOutcome> {
        let directory = self.episode_directory(id)?;
        let _lock = EpisodeLock::acquire(&directory)?;
        let marker = directory.join("memorable-synced.json");
        if marker.exists() {
            return Ok(SyncOutcome {
                already_synced: true,
                stdout: String::new(),
            });
        }
        let metadata: EpisodeMetadata = read_json(&directory.join("episode.json"))?;
        if metadata.finished_at_ms.is_none() {
            return Err(invalid("finish the episode before syncing"));
        }
        let trace = directory.join("memorable-trace.json");
        let output = cli.run([OsStr::new("ingest"), trace.as_os_str()])?;
        // Memorable can decline procedural extraction for read-only traces. The
        // acknowledgement means the CLI processed it, not necessarily stored it.
        write_json_atomic(
            &marker,
            &serde_json::json!({
                "acknowledged_at_ms": now_ms(), "cli_version": MEMORABLE_VERSION,
                "stdout": output.stdout, "stderr": output.stderr
            }),
        )?;
        Ok(SyncOutcome {
            already_synced: false,
            stdout: output.stdout,
        })
    }

    fn episode_directory(&self, id: &str) -> io::Result<PathBuf> {
        validate_id(id)?;
        let directory = self.root.join(id);
        if fs::symlink_metadata(&directory)?.file_type().is_symlink() {
            return Err(invalid("episode directories cannot be symbolic links"));
        }
        if !directory.is_dir() {
            return Err(invalid("episode is not a directory"));
        }
        for name in ["episode.json", "steps.jsonl"] {
            require_regular_file(&directory.join(name), false)?;
        }
        for name in ["memorable-trace.json", "memorable-synced.json", ".lock"] {
            require_regular_file(&directory.join(name), true)?;
        }
        let audio = fs::symlink_metadata(directory.join("audio"))?;
        if !audio.is_dir() || audio.file_type().is_symlink() {
            return Err(invalid("episode audio must be a real directory"));
        }
        Ok(directory)
    }

    fn copy_audio(
        &self,
        id: &str,
        directory: &Path,
        sequence: u64,
        index: usize,
        source: &AudioSource,
    ) -> io::Result<AudioRef> {
        if source.role.is_empty() || source.role.len() > 64 || source.media_type.len() > 128 {
            return Err(invalid("invalid audio role or media type"));
        }
        if source.sample_rate_hz == Some(0) || source.channels == Some(0) {
            return Err(invalid("sample rate and channel count must be positive"));
        }
        // Check before open as opening a FIFO can block forever. Explicit input
        // symlinks are allowed, but must resolve to a bounded regular file.
        let stat = fs::metadata(&source.path)?;
        if !stat.is_file() || stat.len() > MAX_AUDIO_BYTES {
            return Err(invalid("audio must be a regular file of at most 512 MiB"));
        }
        let mut input = File::open(&source.path)?;
        let stat = input.metadata()?;
        if !stat.is_file() || stat.len() > MAX_AUDIO_BYTES {
            return Err(invalid("audio must be a regular file of at most 512 MiB"));
        }
        let extension = source
            .path
            .extension()
            .and_then(OsStr::to_str)
            .filter(|s| s.len() <= 12 && s.bytes().all(|c| c.is_ascii_alphanumeric()))
            .unwrap_or("bin");
        // A unique suffix also makes a retried append safe after a partial copy.
        let name = format!("{sequence:08}-{index}-{}.{extension}", unique_suffix());
        let path = directory.join("audio").join(&name);
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let copied = (|| {
            let copied = io::copy(&mut (&mut input).take(MAX_AUDIO_BYTES + 1), &mut output)?;
            if copied > MAX_AUDIO_BYTES {
                return Err(invalid("audio grew beyond 512 MiB"));
            }
            output.sync_all()?;
            sync_directory(&directory.join("audio"))?;
            Ok(copied)
        })();
        let copied = match copied {
            Ok(bytes) => bytes,
            Err(error) => {
                let _ = fs::remove_file(&path);
                return Err(error);
            }
        };
        Ok(AudioRef {
            path: format!("{id}/audio/{name}"),
            role: source.role.clone(),
            media_type: source.media_type.clone(),
            bytes: copied,
            sample_rate_hz: source.sample_rate_hz,
            channels: source.channels,
        })
    }

    fn write_trace(&self, directory: &Path, metadata: &EpisodeMetadata) -> io::Result<u64> {
        let path = directory.join("memorable-trace.json");
        atomic_file(&path, |output| {
            output.write_all(b"{\"session_id\":")?;
            serde_json::to_writer(&mut *output, &metadata.id)?;
            output.write_all(b",\"task_description\":")?;
            serde_json::to_writer(&mut *output, &metadata.task_description)?;
            output.write_all(b",\"harness\":\"nelly-rust\",\"tool_calls\":[")?;
            let mut count = 0;
            visit_steps(&directory.join("steps.jsonl"), |step| {
                if count > 0 {
                    output.write_all(b",")?;
                }
                #[derive(Serialize)]
                struct ToolCall<'a> {
                    name: &'a str,
                    input: &'a Value,
                    #[serde(skip_serializing_if = "Option::is_none")]
                    result: Option<&'a Value>,
                }
                serde_json::to_writer(
                    &mut *output,
                    &ToolCall {
                        name: &step.action.name,
                        input: &step.action.input,
                        result: step.result.as_ref(),
                    },
                )?;
                count += 1;
                Ok(())
            })?;
            output.write_all(b"]}\n")?;
            Ok(count)
        })
    }
}

enum RecorderCommand {
    Append(StepInput),
    Flush(mpsc::Sender<()>),
    Finish(String, mpsc::Sender<EpisodeMetadata>),
}

struct RecorderFailure {
    kind: io::ErrorKind,
    message: String,
}

impl RecorderFailure {
    fn from_error(error: io::Error) -> Self {
        Self {
            kind: error.kind(),
            message: error.to_string(),
        }
    }

    fn to_error(&self) -> io::Error {
        io::Error::new(self.kind, self.message.clone())
    }
}

/// Optional bounded handoff for live inference threads. No audio files or
/// filesystem writes occur in `try_append`; accepted records become durable at
/// `flush`/`finish`. Keep audio source files unchanged until that barrier.
///
/// Dropping the recorder disconnects the queue and detaches its worker, which
/// drains accepted work but does not finish the episode. Call `finish` explicitly
/// before process shutdown to observe failures and guarantee durability.
pub struct BackgroundRecorder {
    sender: Option<mpsc::SyncSender<RecorderCommand>>,
    worker: Option<JoinHandle<()>>,
    failure: Arc<OnceLock<RecorderFailure>>,
}

impl BackgroundRecorder {
    pub fn start(store: MemoryStore, id: &str, capacity: usize) -> io::Result<Self> {
        if !(1..=16).contains(&capacity) {
            return Err(invalid("background recorder capacity must be 1..16"));
        }
        if store.metadata(id)?.finished_at_ms.is_some() {
            return Err(invalid("episode is finished"));
        }
        let id = id.to_owned();
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let failure = Arc::new(OnceLock::new());
        let worker_failure = Arc::clone(&failure);
        let worker = std::thread::Builder::new()
            .name("nelly-episode-writer".into())
            .spawn(move || {
                while let Ok(command) = receiver.recv() {
                    let result = match command {
                        RecorderCommand::Append(input) => store.append(&id, input).map(|_| ()),
                        RecorderCommand::Flush(reply) => {
                            let _ = reply.send(());
                            Ok(())
                        }
                        RecorderCommand::Finish(summary, reply) => {
                            match store.finish(&id, &summary) {
                                Ok(metadata) => {
                                    let _ = reply.send(metadata);
                                }
                                Err(error) => {
                                    let _ = worker_failure.set(RecorderFailure::from_error(error));
                                }
                            }
                            return;
                        }
                    };
                    if let Err(error) = result {
                        let _ = worker_failure.set(RecorderFailure::from_error(error));
                        return;
                    }
                }
            })?;
        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
            failure,
        })
    }

    /// Bounded nonblocking queue insertion. On any error the supplied record is
    /// rejected (not recorded); retain upstream state if retry is required.
    /// Serialization validates a 64 KiB cap and can allocate up to that bound.
    pub fn try_append(&self, input: StepInput) -> io::Result<()> {
        self.check_error()?;
        bounded_json_limit(&input, MAX_QUEUED_STEP_BYTES)?;
        match self
            .sender
            .as_ref()
            .expect("sender present until finish")
            .try_send(RecorderCommand::Append(input))
        {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(_)) => Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "episode queue full; record was not accepted",
            )),
            Err(mpsc::TrySendError::Disconnected(_)) => Err(self.disconnected_error()),
        }
    }

    /// Observe an asynchronous storage failure without waiting on the worker.
    pub fn check_error(&self) -> io::Result<()> {
        match self.failure.get() {
            Some(error) => Err(error.to_error()),
            None => Ok(()),
        }
    }

    /// Cold durability barrier for all previously accepted records.
    pub fn flush(&self) -> io::Result<()> {
        self.check_error()?;
        let (reply, received) = mpsc::channel();
        self.sender
            .as_ref()
            .expect("sender present until finish")
            .send(RecorderCommand::Flush(reply))
            .map_err(|_| self.disconnected_error())?;
        received.recv().map_err(|_| self.disconnected_error())?;
        self.check_error()
    }

    /// Drain accepted records, finish the episode, and join the storage worker.
    /// This can wait on disk and belongs outside the real-time inference loop.
    pub fn finish(mut self, summary: &str) -> io::Result<EpisodeMetadata> {
        let (reply, received) = mpsc::channel();
        let result = self
            .sender
            .take()
            .expect("sender present until finish")
            .send(RecorderCommand::Finish(summary.into(), reply))
            .map_err(|_| self.disconnected_error())
            .and_then(|()| received.recv().map_err(|_| self.disconnected_error()));
        if self
            .worker
            .take()
            .expect("worker present until finish")
            .join()
            .is_err()
        {
            return Err(io::Error::other("episode storage worker panicked"));
        }
        self.check_error()?;
        result
    }

    fn disconnected_error(&self) -> io::Error {
        self.failure.get().map_or_else(
            || io::Error::new(io::ErrorKind::BrokenPipe, "episode storage worker stopped"),
            RecorderFailure::to_error,
        )
    }
}

#[derive(Debug, Clone)]
pub struct MemorableCli {
    pub program: OsString,
    pub prefix_args: Vec<OsString>,
    pub timeout: Duration,
    /// Per stream, bounded independently for stdout and stderr.
    pub max_output_bytes: usize,
}

impl Default for MemorableCli {
    fn default() -> Self {
        Self {
            program: "npx".into(),
            prefix_args: vec![
                "--yes".into(),
                "--package".into(),
                format!("memorable-cli@{MEMORABLE_VERSION}").into(),
                "memorable".into(),
            ],
            timeout: Duration::from_secs(45),
            max_output_bytes: 64 * 1024,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliOutput {
    pub stdout: String,
    pub stderr: String,
}

impl MemorableCli {
    pub fn recall(&self, query: &str) -> io::Result<CliOutput> {
        if query.is_empty() || query.len() > 16 * 1024 || query.starts_with('-') {
            return Err(invalid(
                "recall query must be 1..16384 bytes and cannot start with '-'",
            ));
        }
        self.run(["recall", query])
    }

    pub fn show(&self, slug: &str) -> io::Result<CliOutput> {
        if !slug.starts_with("procedures/")
            || slug.len() > 512
            || slug.contains("..")
            || !slug
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"/-_".contains(&c))
        {
            return Err(invalid("expected a procedures/<slug> identifier"));
        }
        self.run(["show", slug])
    }

    fn run<I, S>(&self, args: I) -> io::Result<CliOutput>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        if self.max_output_bytes == 0
            || self.max_output_bytes > MAX_STEP_BYTES
            || self.timeout.is_zero()
        {
            return Err(invalid(
                "CLI needs a positive timeout and output limit of 1..1048576 bytes",
            ));
        }
        let mut command = Command::new(&self.program);
        command
            .args(&self.prefix_args)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let mut child = command.spawn()?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("missing stdout pipe"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| io::Error::other("missing stderr pipe"))?;
        let (sender, receiver) = mpsc::channel();
        for (index, mut reader) in [
            (0, Box::new(stdout) as Box<dyn Read + Send>),
            (1, Box::new(stderr) as Box<dyn Read + Send>),
        ] {
            let sender = sender.clone();
            let limit = self.max_output_bytes;
            std::thread::spawn(move || {
                let mut bytes = Vec::new();
                let result = (&mut reader).take(limit as u64 + 1).read_to_end(&mut bytes);
                let _ = sender.send((index, result.map(|_| bytes)));
            });
        }
        drop(sender);
        let start = Instant::now();
        let mut streams: [Option<Vec<u8>>; 2] = [None, None];
        let mut status = None;
        loop {
            while let Ok((index, result)) = receiver.try_recv() {
                let bytes = match result {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        terminate(&mut child);
                        return Err(error);
                    }
                };
                if bytes.len() > self.max_output_bytes {
                    terminate(&mut child);
                    return Err(io::Error::other("Memorable CLI exceeded output limit"));
                }
                streams[index] = Some(bytes);
            }
            if status.is_none() {
                status = match child.try_wait() {
                    Ok(status) => status,
                    Err(error) => {
                        terminate(&mut child);
                        return Err(error);
                    }
                };
            }
            if status.is_some() && streams.iter().all(Option::is_some) {
                break;
            }
            if start.elapsed() >= self.timeout {
                terminate(&mut child);
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Memorable CLI timed out; local episode remains available",
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let output = CliOutput {
            stdout: String::from_utf8_lossy(streams[0].as_deref().unwrap_or_default()).into_owned(),
            stderr: String::from_utf8_lossy(streams[1].as_deref().unwrap_or_default()).into_owned(),
        };
        if !status.expect("status checked above").success() {
            return Err(io::Error::other(format!(
                "Memorable CLI failed: {}",
                output.stderr.trim()
            )));
        }
        Ok(output)
    }
}

fn terminate(child: &mut Child) {
    #[cfg(unix)]
    {
        // Child started its own process group. Terminate npx and its subprocess,
        // otherwise a child holding stdout open can outlive the timeout.
        unsafe extern "C" {
            fn kill(pid: i32, signal: i32) -> i32;
        }
        unsafe {
            kill(-(child.id() as i32), 9);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

struct EpisodeLock {
    _file: File,
}

impl EpisodeLock {
    fn acquire(directory: &Path) -> io::Result<Self> {
        let path = directory.join(".lock");
        require_regular_file(&path, true)?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        fs2::FileExt::try_lock_exclusive(&file)?;
        Ok(Self { _file: file })
    }
}

fn validate_id(id: &str) -> io::Result<()> {
    if id.is_empty()
        || id.len() > 128
        || !id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
    {
        return Err(invalid(
            "episode id must contain 1..128 ASCII letters, digits, '_' or '-'",
        ));
    }
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
fn unique_suffix() -> String {
    format!(
        "{}-{}-{}",
        now_ms(),
        std::process::id(),
        TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn require_regular_file(path: &Path, optional: bool) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(invalid(
            "episode files must be regular files, not symlinks or special files",
        )),
        Err(error) if optional && error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn sync_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> io::Result<T> {
    let file = File::open(path)?;
    if file.metadata()?.len() > MAX_STEP_BYTES as u64 {
        return Err(invalid("metadata exceeds size limit"));
    }
    Ok(serde_json::from_reader(BufReader::new(file))?)
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> io::Result<()> {
    atomic_file(path, |file| {
        serde_json::to_writer(&mut *file, value)?;
        file.write_all(b"\n")
    })
}

fn atomic_file<T>(path: &Path, writer: impl FnOnce(&mut File) -> io::Result<T>) -> io::Result<T> {
    let temporary = path.with_extension(format!("{}.tmp", unique_suffix()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let value = writer(&mut file)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        sync_directory(path.parent().expect("file has parent"))?;
        Ok(value)
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

fn bounded_json(value: &impl Serialize) -> io::Result<Vec<u8>> {
    bounded_json_limit(value, MAX_STEP_BYTES)
}

fn bounded_json_limit(value: &impl Serialize, limit: usize) -> io::Result<Vec<u8>> {
    struct Limited {
        bytes: Vec<u8>,
        limit: usize,
    }
    impl Write for Limited {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
                return Err(invalid("step exceeds serialized size limit"));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut output = Limited {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut output, value)?;
    Ok(output.bytes)
}

/// Truncate only an interrupted final append. Completed records are immutable.
fn recover_last_step(file: &mut File) -> io::Result<Option<EpisodeStep>> {
    let length = file.metadata()?.len();
    if length == 0 {
        return Ok(None);
    }
    let offset = length.saturating_sub((2 * MAX_STEP_BYTES + 2) as u64);
    file.seek(SeekFrom::Start(offset))?;
    let mut tail = Vec::with_capacity((length - offset) as usize);
    (&mut *file)
        .take((2 * MAX_STEP_BYTES + 2) as u64)
        .read_to_end(&mut tail)?;
    let Some(end) = tail.iter().rposition(|&byte| byte == b'\n') else {
        if offset != 0 {
            return Err(invalid("unrecoverable oversized step"));
        }
        file.set_len(0)?;
        file.sync_data()?;
        return Ok(None);
    };
    if offset + end as u64 + 1 != length {
        file.set_len(offset + end as u64 + 1)?;
        file.sync_data()?;
    }
    let start = tail[..end]
        .iter()
        .rposition(|&byte| byte == b'\n')
        .map_or(0, |position| position + 1);
    if start == 0 && offset != 0 {
        return Err(invalid("step exceeds recovery bound"));
    }
    if end - start > MAX_STEP_BYTES {
        return Err(invalid("step exceeds 1 MiB"));
    }
    Ok(Some(serde_json::from_slice(&tail[start..end])?))
}

fn visit_steps(
    path: &Path,
    mut visitor: impl FnMut(EpisodeStep) -> io::Result<()>,
) -> io::Result<()> {
    let mut input = BufReader::new(File::open(path)?);
    let mut line = Vec::new();
    let mut expected = 0;
    loop {
        line.clear();
        let length = (&mut input)
            .take(MAX_STEP_BYTES as u64 + 2)
            .read_until(b'\n', &mut line)?;
        if length == 0 {
            break;
        }
        if length > MAX_STEP_BYTES + 1 || line.last() != Some(&b'\n') {
            return Err(invalid("oversized or incomplete step"));
        }
        let step: EpisodeStep = serde_json::from_slice(&line)?;
        if step.sequence != expected {
            return Err(invalid("episode sequence is not contiguous"));
        }
        expected += 1;
        visitor(step)?;
    }
    Ok(())
}

#[cfg(test)]
mod recorder_tests {
    use super::*;

    fn input() -> StepInput {
        StepInput {
            pre_state: Value::Null,
            action: Action {
                name: "notes_get".into(),
                input: Value::Null,
            },
            result: None,
            post_state: Value::Null,
            started_ms: 0,
            ended_ms: 1,
            audio: vec![],
        }
    }

    #[test]
    fn queue_has_explicit_backpressure_and_bounded_record_size() {
        // No worker consumes this queue, making overflow deterministic.
        let (sender, receiver) = mpsc::sync_channel(1);
        let recorder = BackgroundRecorder {
            sender: Some(sender),
            worker: None,
            failure: Arc::new(OnceLock::new()),
        };
        recorder.try_append(input()).unwrap();
        assert_eq!(
            recorder.try_append(input()).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
        assert!(matches!(
            receiver.try_recv().unwrap(),
            RecorderCommand::Append(_)
        ));
        assert!(receiver.try_recv().is_err());
        let mut oversized = input();
        oversized.pre_state = Value::String("x".repeat(MAX_QUEUED_STEP_BYTES));
        assert!(recorder.try_append(oversized).is_err());
        assert!(receiver.try_recv().is_err());
    }
}
