//! Append-only mutation journal. Reads never touch disk. Successful writes are
//! fsynced before acknowledgement; a process lock prevents concurrent owners.
use crate::{Harness, ToolCall, ToolResult};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, Seek, SeekFrom, Write},
    path::Path,
};

pub const MAX_REQUEST_BYTES: usize = 64 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u8,
    call: ToolCall,
}

pub struct DurableHarness {
    core: Harness,
    journal: File,
    _lock: File,
    poisoned: bool,
}

impl DurableHarness {
    pub fn open(directory: impl AsRef<Path>) -> io::Result<Self> {
        create_dir_all_durable(directory.as_ref())?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(directory.as_ref().join("owner.lock"))?;
        lock.try_lock_exclusive().map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("another harness owns this data directory: {e}"),
            )
        })?;
        let mut journal = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(directory.as_ref().join("mutations.jsonl"))?;
        journal.sync_all()?;
        #[cfg(unix)]
        File::open(directory.as_ref())?.sync_all()?;
        let core = replay(&mut journal)?;
        Ok(Self {
            core,
            journal,
            _lock: lock,
            poisoned: false,
        })
    }
    pub fn core(&self) -> &Harness {
        &self.core
    }
    pub fn execute(&mut self, call: &ToolCall, speculative: bool) -> Result<ToolResult, String> {
        if self.poisoned {
            return Err(
                "journal recovery failed; close this harness and repair the data directory".into(),
            );
        }
        if call.is_read() {
            return self.core.execute(call, speculative);
        }
        if speculative {
            return Err("speculative writes are forbidden".into());
        }
        let mut encoded = serde_json::to_vec(&Record {
            version: 1,
            call: call.clone(),
        })
        .map_err(|e| e.to_string())?;
        encoded.push(b'\n');
        if encoded.len() > MAX_REQUEST_BYTES {
            return Err("mutation exceeds journal record limit".into());
        }
        let offset = self
            .journal
            .seek(SeekFrom::End(0))
            .map_err(|e| e.to_string())?;
        let result = self.core.execute(call, false)?;
        if let Err(error) = self
            .journal
            .write_all(&encoded)
            .and_then(|_| self.journal.sync_data())
        {
            // Restore the last acknowledged state, including ID allocation and
            // cache revision. Fail closed if disk failure prevents recovery.
            let recovery = self
                .journal
                .set_len(offset)
                .and_then(|_| self.journal.sync_data())
                .and_then(|_| replay(&mut self.journal));
            match recovery {
                Ok(core) => self.core = core,
                Err(_) => self.poisoned = true,
            }
            return Err(format!("mutation not acknowledged: {error}"));
        }
        Ok(result)
    }
}

/// Persist newly created directory entries as well as files inside them.
pub(crate) fn create_dir_all_durable(path: &Path) -> io::Result<()> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut missing = Vec::new();
    let mut current = absolute.as_path();
    while !current.exists() {
        missing.push(current.to_path_buf());
        current = current.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "directory has no parent")
        })?;
    }
    for directory in missing.iter().rev() {
        match fs::create_dir(directory) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists && directory.is_dir() => {}
            Err(error) => return Err(error),
        }
        #[cfg(unix)]
        if let Some(parent) = directory.parent() {
            File::open(parent)?.sync_all()?;
        }
    }
    Ok(())
}

fn replay(journal: &mut File) -> io::Result<Harness> {
    journal.seek(SeekFrom::Start(0))?;
    let mut core = Harness::default();
    let mut reader = BufReader::new(journal.try_clone()?);
    let mut line = Vec::new();
    let mut valid_end = 0u64;
    loop {
        match bounded_line(&mut reader, &mut line, MAX_REQUEST_BYTES)? {
            Line::Eof => break,
            Line::Truncated => {
                // A torn final append was never acknowledged. Complete invalid
                // lines, including corruption in the middle, are never skipped.
                journal.set_len(valid_end)?;
                journal.sync_data()?;
                break;
            }
            Line::Complete => {
                let record: Record = serde_json::from_slice(&line).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid journal at byte {valid_end}: {e}"),
                    )
                })?;
                if record.version != 1 || record.call.is_read() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unsupported journal record",
                    ));
                }
                core.execute(&record.call, false)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                valid_end += line.len() as u64;
            }
        }
    }
    journal.seek(SeekFrom::End(0))?;
    Ok(core)
}

#[derive(Debug, PartialEq, Eq)]
pub enum Line {
    Eof,
    Complete,
    Truncated,
}

/// Unlike read_line/read_until, malicious input cannot force an unbounded
/// allocation. An oversized line is a protocol error and ends the connection.
pub fn bounded_line(reader: &mut impl BufRead, out: &mut Vec<u8>, max: usize) -> io::Result<Line> {
    out.clear();
    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            return Ok(if out.is_empty() {
                Line::Eof
            } else {
                Line::Truncated
            });
        }
        let end = buffer.iter().position(|b| *b == b'\n').map(|p| p + 1);
        let count = end.unwrap_or(buffer.len());
        if out.len().saturating_add(count) > max {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "line exceeds byte limit",
            ));
        }
        out.extend_from_slice(&buffer[..count]);
        reader.consume(count);
        if end.is_some() {
            return Ok(Line::Complete);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    fn temp() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "nelly-journal-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ))
    }
    #[test]
    fn durability_tail_recovery_and_single_owner() {
        let path = temp();
        let mut h = DurableHarness::open(&path).unwrap();
        assert!(DurableHarness::open(&path).is_err());
        h.execute(
            &ToolCall::NotesPut {
                topic: "x".into(),
                name: "y".into(),
                note: "remember".into(),
            },
            false,
        )
        .unwrap();
        let created = h
            .execute(
                &ToolCall::ScheduleCreate {
                    title: "meeting".into(),
                    start_minute: 10,
                    end_minute: 20,
                },
                false,
            )
            .unwrap();
        let id = created.value["id"].as_u64().unwrap();
        drop(h);
        OpenOptions::new()
            .append(true)
            .open(path.join("mutations.jsonl"))
            .unwrap()
            .write_all(b"{\"version\":")
            .unwrap();
        let mut h = DurableHarness::open(&path).unwrap();
        assert_eq!(h.core.notes().get("x", "y"), Some("remember"));
        assert_eq!(h.core.revision(), 2);
        assert_eq!(h.core.schedule().get(id).unwrap().title, "meeting");
        h.execute(&ToolCall::ScheduleDelete { id }, false).unwrap();
        drop(h);
        let h = DurableHarness::open(&path).unwrap();
        assert_eq!(h.core.revision(), 3);
        assert!(h.core.schedule().get(id).is_none());
        drop(h);
        fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn complete_corrupt_record_is_never_silently_skipped() {
        let path = temp();
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("mutations.jsonl"), b"garbage\n").unwrap();
        assert!(DurableHarness::open(&path).is_err());
        assert_eq!(
            fs::read(path.join("mutations.jsonl")).unwrap(),
            b"garbage\n"
        );
        fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn input_allocation_is_bounded() {
        let mut out = Vec::new();
        assert!(bounded_line(&mut &b"12345\n"[..], &mut out, 4).is_err());
        assert!(out.len() <= 4);
    }
}
