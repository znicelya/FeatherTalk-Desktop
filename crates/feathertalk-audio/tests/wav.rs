//! The reader admits exactly one shape of file: 16 kHz, mono, 16-bit PCM.
//!
//! Every fixture here is assembled byte by byte, because what is under test is
//! the header walk, not a library's idea of a well-formed file.

use std::path::PathBuf;

use feathertalk_audio::{AudioError, MAX_WAV_FILE_BYTES, WAV_SAMPLE_RATE, read_wav_16k_mono};

fn fmt_body() -> Vec<u8> {
    format_body(1, 1, 16_000, 32_000, 2, 16)
}

fn format_body(
    code: u16,
    channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits: u16,
) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&code.to_le_bytes());
    body.extend_from_slice(&channels.to_le_bytes());
    body.extend_from_slice(&sample_rate.to_le_bytes());
    body.extend_from_slice(&byte_rate.to_le_bytes());
    body.extend_from_slice(&block_align.to_le_bytes());
    body.extend_from_slice(&bits.to_le_bytes());
    body
}

fn pcm(samples: &[i16]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect()
}

fn riff(chunks: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut payload = b"WAVE".to_vec();
    for (id, body) in chunks {
        payload.extend_from_slice(id.as_bytes());
        payload.extend_from_slice(&(body.len() as u32).to_le_bytes());
        payload.extend_from_slice(body);
        if !body.len().is_multiple_of(2) {
            payload.push(0);
        }
    }
    let mut bytes = b"RIFF".to_vec();
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&payload);
    bytes
}

fn wav_file(bytes: &[u8]) -> (tempfile::TempDir, PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("audio_16k_mono.wav");
    std::fs::write(&path, bytes).unwrap();
    (directory, path)
}

#[test]
fn canonical_wav_scales_samples_and_skips_unknown_chunks() {
    assert_eq!(MAX_WAV_FILE_BYTES, 256 * 1024 * 1024);
    assert_eq!(WAV_SAMPLE_RATE, 16_000);

    // The odd-length LIST body exercises the pad byte every RIFF chunk carries
    // when its size is odd.
    let bytes = riff(&[
        ("LIST", b"abc".to_vec()),
        ("fmt ", fmt_body()),
        ("data", pcm(&[0, 32_767, -32_768, -16_384])),
    ]);
    let (_directory, path) = wav_file(&bytes);

    let samples = read_wav_16k_mono(&path).unwrap();

    assert_eq!(samples, vec![0.0, 32_767.0 / 32_768.0, -1.0, -0.5]);
}

#[test]
fn a_longer_format_chunk_keeps_its_extension_out_of_the_way() {
    // An 18-byte fmt chunk -- the 16 PCM fields plus a zero `cbSize` -- is what
    // several encoders write, so the reader must read the fields it knows and
    // step over the rest.
    let mut body = fmt_body();
    body.extend_from_slice(&0_u16.to_le_bytes());
    let bytes = riff(&[("fmt ", body), ("data", pcm(&[1_024]))]);
    let (_directory, path) = wav_file(&bytes);

    let samples = read_wav_16k_mono(&path).unwrap();

    assert_eq!(samples, vec![0.031_25]);
}

#[test]
fn a_broken_container_is_rejected_before_any_chunk() {
    let valid = riff(&[("fmt ", fmt_body()), ("data", pcm(&[1]))]);

    let (_short_directory, short) = wav_file(b"RIFF");
    let error = read_wav_16k_mono(&short).unwrap_err();
    assert!(matches!(error, AudioError::InvalidRiffHeader), "{error:?}");

    let mut wrong_magic = valid.clone();
    wrong_magic[..4].copy_from_slice(b"RIFX");
    let (_magic_directory, magic) = wav_file(&wrong_magic);
    let error = read_wav_16k_mono(&magic).unwrap_err();
    assert!(matches!(error, AudioError::InvalidRiffHeader), "{error:?}");

    let mut wrong_form = valid.clone();
    wrong_form[8..12].copy_from_slice(b"AVI ");
    let (_form_directory, form) = wav_file(&wrong_form);
    let error = read_wav_16k_mono(&form).unwrap_err();
    assert!(matches!(error, AudioError::InvalidRiffHeader), "{error:?}");

    let mut trailing = valid.clone();
    trailing.extend_from_slice(b"data");
    let (_trailing_directory, path) = wav_file(&trailing);
    let error = read_wav_16k_mono(&path).unwrap_err();
    let AudioError::InvalidWavHeader { reason } = error else {
        panic!("expected an invalid header, got {error:?}");
    };
    assert!(reason.contains("truncated"), "{reason}");
}

#[test]
fn both_chunks_are_required_and_the_format_must_come_first() {
    let (_format_directory, format_only) = wav_file(&riff(&[("fmt ", fmt_body())]));
    let error = read_wav_16k_mono(&format_only).unwrap_err();
    assert!(
        matches!(error, AudioError::MissingWavChunk { chunk: "data" }),
        "{error:?}"
    );

    let (_data_directory, data_only) = wav_file(&riff(&[("data", pcm(&[1]))]));
    let error = read_wav_16k_mono(&data_only).unwrap_err();
    assert!(
        matches!(error, AudioError::MissingWavChunk { chunk: "fmt " }),
        "{error:?}"
    );

    let late = riff(&[("data", pcm(&[1])), ("fmt ", fmt_body())]);
    let (_late_directory, path) = wav_file(&late);
    let error = read_wav_16k_mono(&path).unwrap_err();
    let AudioError::InvalidWavHeader { reason } = error else {
        panic!("expected an invalid header, got {error:?}");
    };
    assert!(reason.contains("follows the data chunk"), "{reason}");
}

#[test]
fn every_unsupported_format_field_names_itself() {
    // `AudioError` has no `PartialEq`, so the expected variant is compared
    // through its `Display` text.
    let cases = [
        (
            format_body(3, 1, 16_000, 64_000, 4, 32),
            AudioError::UnsupportedWavFormat { code: 3 },
        ),
        (
            format_body(1, 2, 16_000, 64_000, 4, 16),
            AudioError::UnsupportedWavChannels { actual: 2 },
        ),
        (
            format_body(1, 1, 44_100, 88_200, 2, 16),
            AudioError::UnsupportedWavSampleRate {
                actual: 44_100,
                expected: 16_000,
            },
        ),
        (
            format_body(1, 1, 16_000, 48_000, 3, 24),
            AudioError::UnsupportedWavBitDepth { actual: 24 },
        ),
    ];

    for (body, expected) in cases {
        let (_directory, path) = wav_file(&riff(&[("fmt ", body), ("data", pcm(&[1]))]));
        let error = read_wav_16k_mono(&path).unwrap_err();
        assert_eq!(error.to_string(), expected.to_string());
    }
}

#[test]
fn an_inconsistent_format_chunk_is_rejected_with_a_reason() {
    let cases = [
        (format_body(1, 1, 16_000, 32_000, 4, 16), "block align 4"),
        (format_body(1, 1, 16_000, 16_000, 2, 16), "byte rate 16000"),
        (fmt_body()[..14].to_vec(), "14 bytes"),
    ];

    for (body, fragment) in cases {
        let (_directory, path) = wav_file(&riff(&[("fmt ", body), ("data", pcm(&[1]))]));
        let error = read_wav_16k_mono(&path).unwrap_err();
        let AudioError::InvalidWavHeader { reason } = error else {
            panic!("expected an invalid header, got {error:?}");
        };
        assert!(reason.contains(fragment), "{reason}");
    }
}

#[test]
fn an_unusable_data_chunk_is_rejected() {
    let empty = riff(&[("fmt ", fmt_body()), ("data", Vec::new())]);
    let (_empty_directory, path) = wav_file(&empty);
    let error = read_wav_16k_mono(&path).unwrap_err();
    assert!(matches!(error, AudioError::EmptyWav), "{error:?}");

    let odd = riff(&[("fmt ", fmt_body()), ("data", vec![1, 2, 3])]);
    let (_odd_directory, path) = wav_file(&odd);
    let error = read_wav_16k_mono(&path).unwrap_err();
    let AudioError::InvalidWavHeader { reason } = error else {
        panic!("expected an invalid header, got {error:?}");
    };
    assert!(reason.contains("3 bytes"), "{reason}");

    let mut truncated = riff(&[("fmt ", fmt_body()), ("data", pcm(&[1, 2, 3, 4]))]);
    truncated.truncate(truncated.len() - 4);
    let (_truncated_directory, path) = wav_file(&truncated);
    let error = read_wav_16k_mono(&path).unwrap_err();
    let AudioError::WavPayloadTruncated { expected, actual } = error else {
        panic!("expected a truncated payload, got {error:?}");
    };
    assert_eq!((expected, actual), (8, 4));
}

#[test]
fn a_path_that_is_not_a_regular_file_is_rejected_before_the_read() {
    let directory = tempfile::tempdir().unwrap();

    let error = read_wav_16k_mono(directory.path()).unwrap_err();
    assert!(
        matches!(error, AudioError::WavNotRegular { .. }),
        "{error:?}"
    );

    let error = read_wav_16k_mono(directory.path().join("missing.wav")).unwrap_err();
    let AudioError::WavIo { operation, .. } = error else {
        panic!("expected an I/O failure, got {error:?}");
    };
    assert_eq!(operation, "metadata");
}
