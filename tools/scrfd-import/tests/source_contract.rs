use std::path::{Path, PathBuf};

use feathertalk_scrfd_import::{generate_burn_files, inspect_source};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

#[test]
fn tracked_onnx_has_the_approved_graph_boundary() {
    let contract = inspect_source(&repo_root()).unwrap();
    assert_eq!(contract.opset, 12);
    assert_eq!(contract.input_name, "images");
    assert_eq!(contract.input_shape, vec![1, 3, 640, 640]);
    assert_eq!(contract.input_elem_type, 1);
    assert_eq!(
        contract
            .output_names
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec![
            "out0", "out1", "out2", "out3", "out4", "out5", "out6", "out7", "out8"
        ]
    );
    assert_eq!(
        contract.output_shapes,
        vec![
            vec![1, 12_800, 1],
            vec![1, 3_200, 1],
            vec![1, 800, 1],
            vec![1, 12_800, 4],
            vec![1, 3_200, 4],
            vec![1, 800, 4],
            vec![1, 12_800, 10],
            vec![1, 3_200, 10],
            vec![1, 800, 10],
        ]
    );
}

#[test]
#[ignore = "runs pinned Burn ONNX code generation"]
fn burn_generation_writes_only_reviewable_source_and_temporary_burnpack() {
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("raw");
    let generated = generate_burn_files(&repo_root(), &destination).unwrap();
    assert!(generated.source.is_file());
    assert!(generated.burnpack.is_file());
    let mut names = std::fs::read_dir(&destination)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["scrfd_2.5g_kps.bpk", "scrfd_2_5g.rs"]);
    let source = std::fs::read_to_string(generated.source).unwrap();
    assert!(source.contains("pub struct Model<B: Backend>"));
    assert!(source.contains("pub fn forward"));
    assert!(!source.contains("from_file"));
    assert!(!source.contains("from_bytes"));
}
