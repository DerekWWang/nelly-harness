//! Bounded-memory PCM16 mono WAV input. Audio is decoded a chunk at a time.

use std::{
    fs::File,
    io::{self, BufReader, Read, Seek, SeekFrom},
    path::Path,
};

/// A RIFF/WAVE reader with one fixed-size byte buffer and borrowed output PCM.
/// The complete chunk directory is validated before any samples are returned.
pub struct WavReader {
    input: BufReader<File>,
    sample_rate_hz: u32,
    remaining_bytes: u64,
}

impl WavReader {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let size = file.metadata()?.len();
        let (sample_rate_hz, data_offset, remaining_bytes) = parse_wave(&mut file, size)?;
        file.seek(SeekFrom::Start(data_offset))?;
        Ok(Self {
            input: BufReader::with_capacity(8192, file),
            sample_rate_hz,
            remaining_bytes,
        })
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    /// Decode into a caller-owned buffer, returning the number of actual
    /// samples. A short final chunk is zero-padded; the next call returns None.
    /// Empty output buffers are errors so they cannot cause non-progress loops.
    pub fn read_chunk(&mut self, pcm: &mut [f32]) -> io::Result<Option<usize>> {
        if pcm.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "PCM output buffer is empty",
            ));
        }
        if self.remaining_bytes == 0 {
            return Ok(None);
        }
        let count = (self.remaining_bytes / 2).min(pcm.len() as u64) as usize;
        let mut bytes = [0_u8; 8192];
        let mut decoded = 0;
        while decoded < count {
            let batch = (count - decoded).min(bytes.len() / 2);
            self.input.read_exact(&mut bytes[..batch * 2])?;
            for (output, pair) in pcm[decoded..decoded + batch]
                .iter_mut()
                .zip(bytes[..batch * 2].chunks_exact(2))
            {
                *output = f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32768.0;
            }
            decoded += batch;
            self.remaining_bytes -= (batch * 2) as u64;
        }
        pcm[count..].fill(0.0);
        Ok(Some(count))
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn parse_wave(input: &mut (impl Read + Seek), file_len: u64) -> io::Result<(u32, u64, u64)> {
    let mut header = [0_u8; 12];
    input.read_exact(&mut header)?;
    if &header[..4] != b"RIFF" || &header[8..] != b"WAVE" {
        return Err(invalid("expected a little-endian RIFF/WAVE file"));
    }
    let riff_end = u64::from(u32::from_le_bytes(header[4..8].try_into().unwrap())) + 8;
    if riff_end < 12 || riff_end > file_len {
        return Err(invalid("invalid or truncated RIFF length"));
    }
    let mut position = 12_u64;
    let mut sample_rate = None;
    let mut data = None;
    let mut chunk_count = 0;
    while position < riff_end {
        chunk_count += 1;
        if chunk_count > 65_536 {
            return Err(invalid("WAV contains too many header chunks"));
        }
        if riff_end - position < 8 {
            return Err(invalid("truncated WAV chunk header"));
        }
        input.seek(SeekFrom::Start(position))?;
        let mut chunk = [0_u8; 8];
        input.read_exact(&mut chunk)?;
        let length = u64::from(u32::from_le_bytes(chunk[4..].try_into().unwrap()));
        let payload = position + 8;
        let next = payload + length + (length & 1);
        if next > riff_end {
            return Err(invalid("WAV chunk payload or padding exceeds RIFF length"));
        }
        match &chunk[..4] {
            b"fmt " => {
                if sample_rate.is_some() {
                    return Err(invalid("duplicate WAV format chunk"));
                }
                if !(16..=4096).contains(&length) || length == 17 {
                    return Err(invalid("WAV format chunk must contain 16 to 4096 bytes"));
                }
                let mut format = [0_u8; 16];
                input.read_exact(&mut format)?;
                let encoding = u16::from_le_bytes(format[..2].try_into().unwrap());
                let channels = u16::from_le_bytes(format[2..4].try_into().unwrap());
                let rate = u32::from_le_bytes(format[4..8].try_into().unwrap());
                let byte_rate = u32::from_le_bytes(format[8..12].try_into().unwrap());
                let alignment = u16::from_le_bytes(format[12..14].try_into().unwrap());
                let bits = u16::from_le_bytes(format[14..16].try_into().unwrap());
                if encoding != 1 || channels != 1 || bits != 16 || alignment != 2 {
                    return Err(invalid("WAV must be uncompressed PCM16 mono"));
                }
                if !(1..=192_000).contains(&rate) || byte_rate != rate * 2 {
                    return Err(invalid("invalid WAV sample rate or byte rate"));
                }
                if length >= 18 {
                    let mut extension_size = [0_u8; 2];
                    input.read_exact(&mut extension_size)?;
                    if u64::from(u16::from_le_bytes(extension_size)) > length - 18 {
                        return Err(invalid("WAV format extension exceeds its chunk"));
                    }
                }
                sample_rate = Some(rate);
            }
            b"data" => {
                if data.is_some() {
                    return Err(invalid("multiple WAV data chunks are unsupported"));
                }
                if length & 1 != 0 {
                    return Err(invalid("PCM16 data must contain complete two-byte samples"));
                }
                data = Some((payload, length));
            }
            _ => {}
        }
        position = next;
    }
    let rate = sample_rate.ok_or_else(|| invalid("WAV is missing its format chunk"))?;
    let (offset, length) = data.ok_or_else(|| invalid("WAV is missing its data chunk"))?;
    Ok((rate, offset, length))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::Cursor,
        sync::atomic::{AtomicU64, Ordering},
    };

    fn format() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(1_u16.to_le_bytes());
        bytes.extend(1_u16.to_le_bytes());
        bytes.extend(16_000_u32.to_le_bytes());
        bytes.extend(32_000_u32.to_le_bytes());
        bytes.extend(2_u16.to_le_bytes());
        bytes.extend(16_u16.to_le_bytes());
        bytes
    }

    fn wave(chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        let mut bytes = b"RIFF\0\0\0\0WAVE".to_vec();
        for (name, data) in chunks {
            bytes.extend_from_slice(*name);
            bytes.extend((data.len() as u32).to_le_bytes());
            bytes.extend(*data);
            if data.len() & 1 != 0 {
                bytes.push(0);
            }
        }
        let length = bytes.len() as u32 - 8;
        bytes[4..8].copy_from_slice(&length.to_le_bytes());
        bytes
    }

    fn parse(bytes: &[u8]) -> io::Result<(u32, u64, u64)> {
        parse_wave(&mut Cursor::new(bytes), bytes.len() as u64)
    }

    struct TempWave(std::path::PathBuf);
    impl TempWave {
        fn new(bytes: &[u8]) -> Self {
            static SERIAL: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "nelly-audio-test-{}-{}.wav",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::write(&path, bytes).unwrap();
            Self(path)
        }
    }
    impl Drop for TempWave {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn streams_normalized_samples_and_pads_only_final_frame() {
        let samples: Vec<u8> = [-32768_i16, 0, 32767, 16384, -16384]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();
        let file = TempWave::new(&wave(&[
            (b"fmt ", &format()),
            (b"JUNK", &[1, 2, 3]),
            (b"data", &samples),
        ]));
        let mut wav = WavReader::open(&file.0).unwrap();
        assert_eq!(wav.sample_rate_hz(), 16_000);
        let mut output = [5.0; 3];
        assert_eq!(wav.read_chunk(&mut output).unwrap(), Some(3));
        assert_eq!(output, [-1.0, 0.0, 32767.0 / 32768.0]);
        assert_eq!(wav.read_chunk(&mut output).unwrap(), Some(2));
        assert_eq!(output, [0.5, -0.5, 0.0]);
        assert_eq!(wav.read_chunk(&mut output).unwrap(), None);
        assert!(wav.read_chunk(&mut []).is_err());
    }

    #[test]
    fn accepts_data_before_format_and_skips_padded_unknown_chunks() {
        let mut extended = format();
        extended.extend(0_u16.to_le_bytes());
        let bytes = wave(&[
            (b"JUNK", &[9]),
            (b"data", &[0, 0]),
            (b"fmt ", &extended),
            (b"LIST", &[1, 2, 3]),
        ]);
        let (rate, _, length) = parse(&bytes).unwrap();
        assert_eq!((rate, length), (16_000, 2));
        assert!(parse(&wave(&[(b"fmt ", &format()), (b"data", &[])])).is_ok());
    }

    #[test]
    fn rejects_truncation_invalid_formats_and_ambiguous_chunks() {
        let mut bytes = wave(&[(b"fmt ", &format()), (b"data", &[0, 0])]);
        bytes.pop();
        assert!(parse(&bytes).is_err());
        let mut bytes = wave(&[(b"fmt ", &format()), (b"data", &[0, 0])]);
        bytes[40..44].copy_from_slice(&100_u32.to_le_bytes());
        assert!(parse(&bytes).is_err());
        assert!(parse(&wave(&[(b"fmt ", &format()), (b"data", &[0])])).is_err());
        assert!(parse(&wave(&[
            (b"fmt ", &format()),
            (b"fmt ", &format()),
            (b"data", &[])
        ]))
        .is_err());
        assert!(parse(&wave(&[
            (b"fmt ", &format()),
            (b"data", &[]),
            (b"data", &[])
        ]))
        .is_err());
        assert!(parse(&wave(&[(b"data", &[])])).is_err());
        assert!(parse(&wave(&[(b"fmt ", &format())])).is_err());
        for (offset, value) in [(0, 3_u8), (2, 2), (12, 4), (14, 32), (9, 0)] {
            let mut bad = format();
            bad[offset] = value;
            assert!(parse(&wave(&[(b"fmt ", &bad), (b"data", &[])])).is_err());
        }
    }

    #[test]
    fn detects_audio_truncated_after_open() {
        let file = TempWave::new(&wave(&[(b"fmt ", &format()), (b"data", &[0; 100])]));
        let mut wav = WavReader::open(&file.0).unwrap();
        File::options()
            .write(true)
            .open(&file.0)
            .unwrap()
            .set_len(45)
            .unwrap();
        assert_eq!(
            wav.read_chunk(&mut [0.0; 50]).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }
}
