use std::path::PathBuf;

use feathertalk_pfld::{
    CropGeometry, MEAN_FACE, MeanFace, PFLD_OUTPUT_VALUE_COUNT, PfldError, decode_landmarks,
    decode_landmarks_with_default_mean_face, decode_landmarks_with_mean_face, read_mean_face,
};

fn crop() -> CropGeometry {
    CropGeometry {
        width: 100,
        height: 80,
        offset_x: 3,
        offset_y: -2,
    }
}

fn values() -> Vec<f32> {
    (0..PFLD_OUTPUT_VALUE_COUNT)
        .map(|index| index as f32 / 1000.0)
        .collect()
}

fn write_values(contents: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mean_face.txt");
    std::fs::write(&path, contents).unwrap();
    (dir, path)
}

#[test]
fn reads_repository_mean_face_fixture() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("data_utils")
        .join("mean_face.txt");
    let mean_face = read_mean_face(&path).unwrap();
    assert_eq!(mean_face.values().len(), PFLD_OUTPUT_VALUE_COUNT);
    assert_eq!(mean_face.values()[0], 0.07823661);
    let expected_last: f32 = "0.66389504".parse().unwrap();
    assert_eq!(
        mean_face.values()[PFLD_OUTPUT_VALUE_COUNT - 1],
        expected_last
    );
    assert_eq!(mean_face.values(), MEAN_FACE.values());
}

#[test]
fn accepts_whitespace_and_exposes_read_only_values() {
    let content = values()
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if index % 3 == 0 {
                format!("\t{value}\r\n")
            } else {
                format!(" {value} ")
            }
        })
        .collect::<String>();
    let (_dir, path) = write_values(&content);
    let mean_face = read_mean_face(&path).unwrap();
    let _: &[f32; PFLD_OUTPUT_VALUE_COUNT] = mean_face.values();
    assert_eq!(mean_face.values()[7], values()[7]);
}

#[test]
fn rejects_missing_utf8_malformed_non_finite_and_wrong_count_inputs() {
    let dir = tempfile::tempdir().unwrap();
    assert!(matches!(
        read_mean_face(&dir.path().join("missing.txt")),
        Err(PfldError::Io { .. })
    ));
    let invalid_utf8 = dir.path().join("invalid.txt");
    std::fs::write(&invalid_utf8, [0xff, 0xfe]).unwrap();
    assert!(matches!(
        read_mean_face(&invalid_utf8),
        Err(PfldError::InvalidUtf8 { .. })
    ));
    let (_dir, malformed) = write_values(&format!(
        "{} nope",
        values()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    ));
    assert!(matches!(
        read_mean_face(&malformed),
        Err(PfldError::InvalidMeanFaceToken { index: 220, .. })
    ));
    for token in ["NaN", "inf", "-inf"] {
        let (_dir, path) = write_values(&format!(
            "{} {token}",
            values()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        ));
        assert!(matches!(
            read_mean_face(&path),
            Err(PfldError::NonFiniteValue {
                field: "mean_face",
                index: 220
            })
        ));
    }
    for count in [0, PFLD_OUTPUT_VALUE_COUNT - 1, PFLD_OUTPUT_VALUE_COUNT + 1] {
        let content = values()[..count.min(PFLD_OUTPUT_VALUE_COUNT)]
            .iter()
            .map(ToString::to_string)
            .chain((count > PFLD_OUTPUT_VALUE_COUNT).then_some("0".to_owned()))
            .collect::<Vec<_>>()
            .join(" ");
        let (_dir, path) = write_values(&content);
        if count != PFLD_OUTPUT_VALUE_COUNT {
            assert!(matches!(
                read_mean_face(&path),
                Err(PfldError::InvalidMeanFaceCount { expected: PFLD_OUTPUT_VALUE_COUNT, actual, .. }) if actual == count
            ));
        }
    }
}

#[test]
fn typed_decoder_matches_slice_decoder() {
    let (_dir, path) = write_values(
        &values()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" "),
    );
    let mean_face = read_mean_face(&path).unwrap();
    let model_output = vec![0.25; PFLD_OUTPUT_VALUE_COUNT];
    let typed = decode_landmarks_with_mean_face(&model_output, &mean_face, crop()).unwrap();
    let sliced = decode_landmarks(&model_output, mean_face.values(), crop()).unwrap();
    assert_eq!(typed, sliced);
}

#[test]
fn mean_face_is_a_public_value_type() {
    let _: Option<MeanFace> = None;
}

#[test]
fn exposes_the_embedded_mean_face_constant() {
    assert_eq!(MEAN_FACE.values().len(), PFLD_OUTPUT_VALUE_COUNT);
    assert_eq!(MEAN_FACE.values()[0], 0.07823661);
    let expected_last: f32 = "0.66389504".parse().unwrap();
    assert_eq!(
        MEAN_FACE.values()[PFLD_OUTPUT_VALUE_COUNT - 1],
        expected_last
    );
}

#[test]
fn default_decoder_uses_the_embedded_mean_face() {
    let model_output = vec![0.0; PFLD_OUTPUT_VALUE_COUNT];
    let expected = decode_landmarks_with_mean_face(&model_output, &MEAN_FACE, crop()).unwrap();
    let actual = decode_landmarks_with_default_mean_face(&model_output, crop()).unwrap();
    assert_eq!(actual, expected);
}
