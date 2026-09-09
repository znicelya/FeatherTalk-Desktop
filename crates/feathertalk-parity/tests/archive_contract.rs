use feathertalk_parity::archive::{FixtureError, GoldenArchive};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, fs::File, io::Write, path::Path};
use zip::{ZipWriter, write::SimpleFileOptions};

fn golden_archive() -> GoldenArchive {
    let root = env!("CARGO_MANIFEST_DIR");
    GoldenArchive::open(format!("{root}/../../tests/golden/burn-feasibility-v1.zip"))
        .expect("golden archive should open")
}

fn write_archive(path: &Path, entries: &[(&str, &[u8])]) {
    let file = File::create(path).unwrap();
    let mut writer = ZipWriter::new(file);
    for (name, bytes) in entries {
        writer
            .start_file(*name, SimpleFileOptions::default())
            .unwrap();
        writer.write_all(bytes).unwrap();
    }
    writer.finish().unwrap();
}

fn npy_with_shape_and_no_payload(elements: usize) -> Vec<u8> {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({elements},), }}")
            .into_bytes();
    let preamble_len = 10;
    let padding = (16 - ((preamble_len + header.len() + 1) % 16)) % 16;
    header.extend(std::iter::repeat_n(b' ', padding));
    header.push(b'\n');

    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes
}

#[test]
fn golden_archive_has_required_entries_and_valid_hash() {
    let archive = golden_archive();

    archive.verify_sidecar_sha256().expect("archive hash");
    for entry in [
        "manifest.json",
        "weights/tiny_direct.pth",
        "weights/tiny_nested.pth",
        "weights/tiny_missing.pth",
        "weights/tiny_unexpected.pth",
        "weights/feather_micro.pth",
        "weights/unet_production.pth",
        "weights/unet_micro_train.pth",
        "arrays/feather_input.npy",
        "arrays/feather_output.npy",
        "arrays/unet_image.npy",
        "arrays/unet_audio.npy",
        "arrays/unet_output.npy",
        "arrays/train_image.npy",
        "arrays/train_audio.npy",
        "arrays/train_target.npy",
        "arrays/train_expected.json",
        "arrays/train_parameter_00.npy",
        "arrays/train_parameter_01.npy",
        "arrays/train_parameter_02.npy",
        "arrays/train_batch_norm_00.npy",
        "arrays/train_batch_norm_01.npy",
        "arrays/train_batch_norm_02.npy",
        "arrays/train_batch_norm_03.npy",
    ] {
        assert!(archive.contains(entry), "missing {entry}");
    }
}

#[test]
fn feather_fixture_loads_named_arrays_with_expected_shapes() {
    let fixture = golden_archive()
        .load_fixture("feather_micro_eval")
        .expect("fixture should load");

    assert_eq!(fixture.id, "feather_micro_eval");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.fixture_set, "burn-feasibility-v1");
    assert_eq!(fixture.kind, "feather_hubert");
    assert_eq!(fixture.weights_entry, "weights/feather_micro.pth");
    assert_eq!(fixture.config["channels"], json!(32));
    assert_eq!(fixture.inputs["waveform"].shape(), &[1, 1360]);
    assert_eq!(fixture.expected["output"].shape(), &[1, 4, 64]);
    assert!(fixture.metrics["waveform_vs_zero_max_abs"] >= 1e-3);
}

#[test]
fn production_unet_fixture_exercises_both_input_branches() {
    let fixture = golden_archive()
        .load_fixture("unet_production_eval")
        .expect("fixture should load");

    assert!(fixture.metrics["image_branch_max_abs"] >= 1e-3);
    assert!(fixture.metrics["audio_branch_max_abs"] >= 1e-3);
}

#[test]
fn training_fixture_loads_inputs_scalars_and_updated_branch_parameters() {
    let fixture = golden_archive()
        .load_fixture("unet_micro_train_step")
        .expect("training fixture should load");

    assert_eq!(fixture.inputs["image"].shape(), &[1, 6, 160, 160]);
    assert_eq!(fixture.inputs["audio"].shape(), &[1, 16, 32, 32]);
    assert_eq!(fixture.inputs["target"].shape(), &[1, 3, 160, 160]);
    assert!(fixture.scalars["initial_loss"].is_finite());
    assert!(fixture.scalars["post_step_loss"].is_finite());
    assert_eq!(fixture.loss.as_deref(), Some("mean_absolute_error"));
    assert_eq!(fixture.expected_mode.as_deref(), Some("eval"));
    let optimizer = fixture.optimizer.as_ref().expect("optimizer metadata");
    assert_eq!(optimizer["type"], json!("adam"));
    assert_eq!(optimizer["learning_rate"], json!(1e-3));
    for metric in [
        "image_parameter_gradient_max_abs",
        "audio_parameter_gradient_max_abs",
        "output_parameter_gradient_max_abs",
        "image_parameter_update_max_abs",
        "audio_parameter_update_max_abs",
        "output_parameter_update_max_abs",
    ] {
        assert!(fixture.metrics[metric] >= 1e-6, "{metric}");
    }
    for name in [
        "inc.inconv.conv.0.weight",
        "audio_model.conv1.conv.0.weight",
        "outc.conv.weight",
        "inc.inconv.conv.1.running_mean",
        "inc.inconv.conv.1.running_var",
        "audio_model.conv1.conv.1.running_mean",
        "audio_model.conv1.conv.1.running_var",
    ] {
        assert!(fixture.expected.contains_key(name), "missing {name}");
    }
}

#[test]
fn extraction_rejects_parent_traversal() {
    let temp = tempfile::tempdir().unwrap();
    let archive_path = temp.path().join("malicious.zip");
    let file = File::create(&archive_path).unwrap();
    let mut writer = ZipWriter::new(file);
    writer
        .start_file("../escape.txt", SimpleFileOptions::default())
        .unwrap();
    writer.write_all(b"escaped").unwrap();
    writer.finish().unwrap();

    let destination = temp.path().join("unpacked");
    let archive = GoldenArchive::open(&archive_path).unwrap();
    assert!(archive.extract_to(&destination).is_err());
    assert!(!temp.path().join("escape.txt").exists());
}

#[test]
fn extraction_rejects_absolute_paths() {
    let temp = tempfile::tempdir().unwrap();
    let archive_path = temp.path().join("absolute.zip");
    write_archive(&archive_path, &[("/escape.txt", b"escaped")]);

    let destination = temp.path().join("unpacked");
    let archive = GoldenArchive::open(&archive_path).unwrap();
    assert!(matches!(
        archive.extract_to(&destination),
        Err(FixtureError::UnsafePath(_))
    ));
}

#[test]
fn golden_archive_extracts_into_a_new_directory() {
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("unpacked");

    golden_archive().extract_to(&destination).unwrap();

    assert!(destination.join("manifest.json").is_file());
    assert!(destination.join("weights/unet_production.pth").is_file());
    assert!(destination.join("arrays/train_expected.json").is_file());
}

#[test]
fn extraction_rejects_archive_symbolic_links() {
    let temp = tempfile::tempdir().unwrap();
    let archive_path = temp.path().join("symlink.zip");
    let file = File::create(&archive_path).unwrap();
    let mut writer = ZipWriter::new(file);
    writer
        .add_symlink("link", "../outside", SimpleFileOptions::default())
        .unwrap();
    writer.finish().unwrap();

    let destination = temp.path().join("unpacked");
    let error = GoldenArchive::open(archive_path)
        .unwrap()
        .extract_to(&destination)
        .unwrap_err();
    assert!(matches!(error, FixtureError::SymbolicLink(_)));
    assert!(!destination.exists());
}

#[test]
fn extraction_requires_a_new_destination_directory() {
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("existing");
    fs::create_dir(&destination).unwrap();

    let error = golden_archive().extract_to(&destination).unwrap_err();
    assert!(matches!(error, FixtureError::DestinationExists(_)));
    assert_eq!(fs::read_dir(destination).unwrap().count(), 0);
}

#[test]
fn extraction_rejects_an_existing_destination_symlink() {
    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("target");
    let destination = temp.path().join("destination");
    fs::create_dir(&target).unwrap();

    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &destination).unwrap();
    #[cfg(windows)]
    if let Err(error) = std::os::windows::fs::symlink_dir(&target, &destination) {
        // Windows without Developer Mode/SeCreateSymbolicLinkPrivilege reports
        // ERROR_PRIVILEGE_NOT_HELD (1314), which is not consistently classified
        // as ErrorKind::PermissionDenied across supported Rust toolchains.
        if error.kind() == std::io::ErrorKind::PermissionDenied
            || error.raw_os_error() == Some(1314)
        {
            return;
        }
        panic!("failed to create test symlink: {error}");
    }

    let error = golden_archive().extract_to(&destination).unwrap_err();
    assert!(matches!(error, FixtureError::DestinationExists(_)));
    assert_eq!(fs::read_dir(target).unwrap().count(), 0);
}

#[test]
fn malformed_npy_shape_is_rejected_before_array_allocation() {
    let temp = tempfile::tempdir().unwrap();
    let archive_path = temp.path().join("malformed.zip");
    let manifest = br#"{
        "schema_version": 1,
        "fixture_set": "burn-feasibility-v1",
        "fixtures": {
            "malformed": {
                "kind": "test",
                "weights": "weights/unused.pth",
                "config": {},
                "inputs": {"bomb": "arrays/bomb.npy"},
                "expected": {}
            }
        }
    }"#;
    let npy = npy_with_shape_and_no_payload(1024);
    write_archive(
        &archive_path,
        &[
            ("manifest.json", manifest),
            ("weights/unused.pth", b"unused"),
            ("arrays/bomb.npy", &npy),
        ],
    );

    let error = GoldenArchive::open(archive_path)
        .unwrap()
        .load_fixture("malformed")
        .unwrap_err();
    assert!(matches!(
        error,
        FixtureError::ArrayPayloadSizeMismatch { .. }
    ));
}

#[test]
fn oversized_npy_shape_is_rejected_before_array_allocation() {
    let temp = tempfile::tempdir().unwrap();
    let archive_path = temp.path().join("oversized.zip");
    let manifest = br#"{
        "schema_version": 1,
        "fixture_set": "burn-feasibility-v1",
        "fixtures": {
            "oversized": {
                "kind": "test",
                "weights": "weights/unused.pth",
                "config": {},
                "inputs": {"bomb": "arrays/bomb.npy"},
                "expected": {}
            }
        }
    }"#;
    let npy = npy_with_shape_and_no_payload(100_000_000);
    write_archive(
        &archive_path,
        &[
            ("manifest.json", manifest),
            ("weights/unused.pth", b"unused"),
            ("arrays/bomb.npy", &npy),
        ],
    );

    let error = GoldenArchive::open(archive_path)
        .unwrap()
        .load_fixture("oversized")
        .unwrap_err();
    assert!(matches!(error, FixtureError::ArrayTooLarge { .. }));
}

#[test]
fn archive_reads_remain_bound_to_the_opened_snapshot() {
    let temp = tempfile::tempdir().unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden");
    let archive_path = temp.path().join("snapshot.zip");
    let sidecar_path = temp.path().join("snapshot.sha256");
    fs::copy(root.join("burn-feasibility-v1.zip"), &archive_path).unwrap();
    fs::copy(root.join("burn-feasibility-v1.sha256"), &sidecar_path).unwrap();

    let archive = GoldenArchive::open(&archive_path).unwrap();
    fs::write(&archive_path, b"replaced after open").unwrap();

    archive.verify_sidecar_sha256().expect("snapshot hash");
    let fixture = archive
        .load_fixture("feather_micro_eval")
        .expect("snapshot fixture");
    assert_eq!(fixture.expected["output"].shape(), &[1, 4, 64]);

    let digest = hex::encode(Sha256::digest(b"replaced after open"));
    assert_ne!(fs::read_to_string(sidecar_path).unwrap().trim(), digest);
}

#[test]
fn archive_hash_mismatch_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden");
    let archive_path = temp.path().join("mismatch.zip");
    fs::copy(root.join("burn-feasibility-v1.zip"), &archive_path).unwrap();
    fs::write(
        temp.path().join("mismatch.sha256"),
        format!("{}\n", "0".repeat(64)),
    )
    .unwrap();

    let error = GoldenArchive::open(archive_path)
        .unwrap()
        .verify_sidecar_sha256()
        .unwrap_err();
    assert!(matches!(error, FixtureError::HashMismatch { .. }));
}

#[test]
fn archive_rejects_duplicate_entries() {
    let temp = tempfile::tempdir().unwrap();
    let archive_path = temp.path().join("duplicate.zip");
    let file = File::create(&archive_path).unwrap();
    let mut writer = ZipWriter::new(file);
    for (name, bytes) in [
        ("first.txt", b"first".as_slice()),
        ("other.txt", b"second".as_slice()),
    ] {
        writer
            .start_file(name, SimpleFileOptions::default())
            .unwrap();
        writer.write_all(bytes).unwrap();
    }
    writer.finish().unwrap();

    let mut bytes = fs::read(&archive_path).unwrap();
    let mut replacements = 0;
    for offset in 0..=bytes.len() - b"other.txt".len() {
        if &bytes[offset..offset + b"other.txt".len()] == b"other.txt" {
            bytes[offset..offset + b"first.txt".len()].copy_from_slice(b"first.txt");
            replacements += 1;
        }
    }
    assert_eq!(replacements, 2, "local and central ZIP names");
    fs::write(&archive_path, bytes).unwrap();

    let error = GoldenArchive::open(archive_path).unwrap_err();
    assert!(matches!(error, FixtureError::DuplicateEntry(_)));
}

#[test]
fn archive_rejects_excessive_entry_counts() {
    let temp = tempfile::tempdir().unwrap();
    let archive_path = temp.path().join("too-many.zip");
    let file = File::create(&archive_path).unwrap();
    let mut writer = ZipWriter::new(file);
    for index in 0..=4096 {
        writer
            .start_file(format!("entry-{index:04}"), SimpleFileOptions::default())
            .unwrap();
    }
    writer.finish().unwrap();

    let error = GoldenArchive::open(archive_path).unwrap_err();
    assert!(matches!(error, FixtureError::TooManyEntries { .. }));
}
