use std::fs;

use feathertalk_audio::{
    AudioError, FeatureMatrix, read_feature_file, write_feature_file, write_feature_file_no_clobber,
};

fn matrix() -> FeatureMatrix {
    FeatureMatrix::new(2, 4, vec![0.25; 8]).unwrap()
}

#[test]
fn feature_file_round_trips_with_header_and_hash() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("feather_hubert.f32");
    let artifact = write_feature_file(&path, &matrix()).unwrap();
    assert_eq!(artifact.tokens(), 2);
    assert_eq!(artifact.dims(), 4);
    assert_eq!(artifact.bytes(), fs::metadata(&path).unwrap().len());
    assert_eq!(artifact.sha256().len(), 64);
    assert_eq!(read_feature_file(&path).unwrap(), matrix());
}

#[test]
fn feature_reader_rejects_unknown_version_short_payload_trailing_bytes_and_symlink() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("feature.f32");
    write_feature_file(&path, &matrix()).unwrap();
    let mut bytes = fs::read(&path).unwrap();

    bytes[8] = 99;
    fs::write(&path, &bytes).unwrap();
    assert!(matches!(
        read_feature_file(&path),
        Err(AudioError::UnsupportedFeatureVersion { .. })
    ));

    let mut valid = Vec::new();
    write_feature_file(&path, &matrix()).unwrap();
    valid.extend_from_slice(&fs::read(&path).unwrap());
    fs::write(&path, &valid[..valid.len() - 1]).unwrap();
    assert!(matches!(
        read_feature_file(&path),
        Err(AudioError::FeaturePayloadTruncated { .. })
    ));

    write_feature_file(&path, &matrix()).unwrap();
    valid = fs::read(&path).unwrap();
    valid.push(0);
    fs::write(&path, &valid).unwrap();
    assert!(matches!(
        read_feature_file(&path),
        Err(AudioError::FeatureTrailingBytes { .. })
    ));

    let target = root.path().join("target.f32");
    fs::rename(&path, &target).unwrap();
    #[cfg(windows)]
    match std::os::windows::fs::symlink_file(&target, &path) {
        Ok(()) => {}
        Err(error) if error.raw_os_error() == Some(1314) => {
            eprintln!("skipping symlink reader assertion: Windows symlink privilege unavailable");
            return;
        }
        Err(error) => panic!("unable to create symlink fixture: {error}"),
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &path).unwrap();
    #[cfg(any(unix, windows))]
    assert!(matches!(
        read_feature_file(&path),
        Err(AudioError::FeatureNotRegular { .. })
    ));
}

#[test]
fn feature_header_contains_pair_width_two() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("feature.f32");
    write_feature_file(&path, &matrix()).unwrap();
    let bytes = fs::read(&path).unwrap();
    assert_eq!(u64::from_le_bytes(bytes[20..28].try_into().unwrap()), 2);
    assert_eq!(read_feature_file(&path).unwrap(), matrix());

    let mut invalid = bytes;
    invalid[20..28].copy_from_slice(&3_u64.to_le_bytes());
    fs::write(&path, invalid).unwrap();
    assert!(matches!(
        read_feature_file(&path),
        Err(AudioError::InvalidFeaturePairWidth { actual: 3 })
    ));
}

#[test]
fn feature_writer_rejects_existing_symlink_destination() {
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("target.f32");
    let path = root.path().join("feature.f32");
    fs::write(&target, b"sentinel").unwrap();
    #[cfg(windows)]
    let link_result = std::os::windows::fs::symlink_file(&target, &path);
    #[cfg(unix)]
    let link_result = std::os::unix::fs::symlink(&target, &path);
    if let Err(error) = link_result {
        #[cfg(windows)]
        if error.raw_os_error() == Some(1314) {
            eprintln!("skipping symlink writer test: Windows symlink privilege unavailable");
            return;
        }
        panic!("unable to create symlink fixture: {error}");
    }
    assert!(matches!(
        write_feature_file(&path, &matrix()),
        Err(AudioError::FeatureNotRegular { .. })
    ));
    assert_eq!(fs::read(&target).unwrap(), b"sentinel");
}

#[test]
fn no_clobber_writer_preserves_existing_destination() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("feature.f32");
    fs::write(&path, b"sentinel").unwrap();

    assert!(write_feature_file_no_clobber(&path, &matrix()).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"sentinel");
}
