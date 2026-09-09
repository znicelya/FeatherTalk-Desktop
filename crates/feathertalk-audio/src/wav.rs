//! A reader for the one audio shape the feature extractor accepts.
//!
//! FeatherHuBERT consumes 16 kHz mono waveforms, and `normalize_media` already
//! writes exactly that file with `-ac 1 -ar 16000 -c:a pcm_s16le`. So this
//! module validates rather than converts: it walks the RIFF chunk list, refuses
//! anything that is not 16-bit PCM mono at 16 kHz, and scales the samples into
//! `f32`. There is no resampler and no channel mixer, because a file that needs
//! one did not come from our own normalisation step.

use std::{
    fs::{self, File},
    io::Read,
    path::Path,
};

use crate::AudioError;

/// The largest wav file the reader will open, 256 MiB.
///
/// That is roughly two hours of 16 kHz mono 16-bit audio. The whole file is
/// read into memory before the header walk, so this bound is what caps the
/// allocation a truncated or hostile file can ask for.
pub const MAX_WAV_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// The only sample rate the reader accepts.
pub const WAV_SAMPLE_RATE: u32 = 16_000;

/// `RIFF`, the declared size, and `WAVE`.
const RIFF_HEADER_BYTES: usize = 12;

/// A chunk identifier plus its declared size.
const CHUNK_HEADER_BYTES: usize = 8;

/// The 16 bytes every PCM `fmt ` chunk carries. Encoders may append more.
const MIN_FMT_BODY_BYTES: usize = 16;

/// The `wFormatTag` value for uncompressed PCM.
const WAVE_FORMAT_PCM: u16 = 1;

/// One 16-bit sample per frame.
const BLOCK_ALIGN: u16 = 2;

/// 16 kHz times two bytes per sample.
const BYTE_RATE: u32 = WAV_SAMPLE_RATE * 2;

/// Read a 16 kHz mono 16-bit wav file into `f32` samples in `[-1.0, 1.0]`.
///
/// The scaling divides by `32_768.0`, which is what the Python reference in
/// `data_utils/feather_hubert/feather_hubert.py` gets from `soundfile`, so
/// features extracted here and there start from identical numbers.
pub fn read_wav_16k_mono(path: impl AsRef<Path>) -> Result<Vec<f32>, AudioError> {
    let path = path.as_ref();
    let bytes = read_bounded(path)?;
    parse(&bytes)
}

/// Read the whole file once it is known to be a regular file within the bound.
fn read_bounded(path: &Path) -> Result<Vec<u8>, AudioError> {
    let metadata = fs::symlink_metadata(path).map_err(|source| io("metadata", path, source))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(AudioError::WavNotRegular {
            path: path.to_owned(),
        });
    }
    if metadata.len() > MAX_WAV_FILE_BYTES {
        return Err(AudioError::WavTooLarge {
            limit: MAX_WAV_FILE_BYTES,
            actual: metadata.len(),
        });
    }

    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(path)
        .map_err(|source| io("open", path, source))?
        .read_to_end(&mut bytes)
        .map_err(|source| io("read", path, source))?;
    Ok(bytes)
}

/// Walk the chunk list, then decode the payload the walk found.
fn parse(bytes: &[u8]) -> Result<Vec<f32>, AudioError> {
    if bytes.len() < RIFF_HEADER_BYTES
        || &bytes[..4] != b"RIFF"
        || &bytes[8..RIFF_HEADER_BYTES] != b"WAVE"
    {
        return Err(AudioError::InvalidRiffHeader);
    }

    let mut offset = RIFF_HEADER_BYTES;
    let mut format_seen = false;
    let mut data: Option<&[u8]> = None;
    while offset < bytes.len() {
        if bytes.len() - offset < CHUNK_HEADER_BYTES {
            return Err(AudioError::InvalidWavHeader {
                reason: format!("the chunk header at offset {offset} is truncated"),
            });
        }
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(
            bytes[offset + 4..offset + CHUNK_HEADER_BYTES]
                .try_into()
                .unwrap(),
        ) as usize;
        let body = offset + CHUNK_HEADER_BYTES;
        let available = bytes.len() - body;
        match id {
            b"fmt " => {
                if data.is_some() {
                    return Err(AudioError::InvalidWavHeader {
                        reason: "the fmt chunk follows the data chunk".to_owned(),
                    });
                }
                if available < size {
                    return Err(AudioError::InvalidWavHeader {
                        reason: format!(
                            "the fmt chunk declares {size} bytes, {available} are present"
                        ),
                    });
                }
                check_format(&bytes[body..body + size])?;
                format_seen = true;
            }
            b"data" => {
                if !size.is_multiple_of(2) {
                    return Err(AudioError::InvalidWavHeader {
                        reason: format!(
                            "the data chunk holds {size} bytes, which is not a whole number of 16-bit samples"
                        ),
                    });
                }
                if available < size {
                    return Err(AudioError::WavPayloadTruncated {
                        expected: size as u64,
                        actual: available as u64,
                    });
                }
                data = Some(&bytes[body..body + size]);
            }
            _ => {}
        }
        // Every RIFF chunk is padded to an even length, and the pad byte is not
        // counted by the declared size.
        offset = body + size + size % 2;
    }

    if !format_seen {
        return Err(AudioError::MissingWavChunk { chunk: "fmt " });
    }
    let Some(data) = data else {
        return Err(AudioError::MissingWavChunk { chunk: "data" });
    };
    if data.is_empty() {
        return Err(AudioError::EmptyWav);
    }

    Ok(data
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes(pair.try_into().unwrap())) / 32_768.0)
        .collect())
}

/// Refuse every `fmt ` chunk that is not 16 kHz mono 16-bit PCM.
fn check_format(body: &[u8]) -> Result<(), AudioError> {
    if body.len() < MIN_FMT_BODY_BYTES {
        return Err(AudioError::InvalidWavHeader {
            reason: format!(
                "the fmt chunk is {} bytes, expected at least {MIN_FMT_BODY_BYTES}",
                body.len()
            ),
        });
    }

    let code = u16::from_le_bytes(body[..2].try_into().unwrap());
    if code != WAVE_FORMAT_PCM {
        return Err(AudioError::UnsupportedWavFormat { code });
    }
    let channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
    if channels != 1 {
        return Err(AudioError::UnsupportedWavChannels { actual: channels });
    }
    let sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
    if sample_rate != WAV_SAMPLE_RATE {
        return Err(AudioError::UnsupportedWavSampleRate {
            actual: sample_rate,
            expected: WAV_SAMPLE_RATE,
        });
    }

    let byte_rate = u32::from_le_bytes(body[8..12].try_into().unwrap());
    let block_align = u16::from_le_bytes(body[12..14].try_into().unwrap());
    let bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
    // The bit depth is checked before the block align: a 24-bit file carries a
    // block align that is consistent with its own depth, and naming the depth
    // is the more useful diagnostic.
    if bits != 16 {
        return Err(AudioError::UnsupportedWavBitDepth { actual: bits });
    }
    if block_align != BLOCK_ALIGN {
        return Err(AudioError::InvalidWavHeader {
            reason: format!("block align {block_align} does not match 16-bit mono"),
        });
    }
    if byte_rate != BYTE_RATE {
        return Err(AudioError::InvalidWavHeader {
            reason: format!("byte rate {byte_rate} does not match 16 kHz 16-bit mono"),
        });
    }

    Ok(())
}

/// Name the operation an I/O failure came from.
fn io(operation: &'static str, path: &Path, source: std::io::Error) -> AudioError {
    AudioError::WavIo {
        operation,
        path: path.to_owned(),
        source,
    }
}
