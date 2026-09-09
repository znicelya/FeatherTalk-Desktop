use std::{fs::File, io::Read, path::Path};

use burn::tensor::{Tensor, backend::Backend};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use feathertalk_models::{PFLD_GhostOne, PfldConfig, backend::CpuBackend};
use feathertalk_weights::{
    PfldImportManifest, PfldImportRequest, TensorAudit, TensorSummary, import_pfld_checkpoint,
};
use sha2::{Digest, Sha256};

fn sha256(path: &Path) -> String {
    let mut file = File::open(path).unwrap();
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).unwrap();
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    hex::encode(hasher.finalize())
}

fn assert_module_snapshots_equal<B: Backend, M: ModuleSnapshot<B>>(left: &M, right: &M) {
    let left = left.collect(None, None, false);
    let right = right.collect(None, None, false);
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right.iter()) {
        assert_eq!(left.full_path(), right.full_path());
        assert_eq!(left.shape, right.shape);
        assert_eq!(left.dtype, right.dtype);
        assert_eq!(left.to_data().unwrap(), right.to_data().unwrap());
    }
}

fn assert_sorted_unique(keys: &[String]) {
    assert!(keys.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn tracked_epoch_335_checkpoint_imports_and_round_trips() {
    let checkpoint = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../data_utils/checkpoint_epoch_335.pth.tar");
    assert!(
        checkpoint.is_file(),
        "tracked PFLD checkpoint is missing: {}",
        checkpoint.display()
    );
    assert_eq!(std::fs::metadata(&checkpoint).unwrap().len(), 5_039_598);
    assert_eq!(
        sha256(&checkpoint),
        "bada866661ad5fa1080a085f51fe9c016c69958c406951afa4afc7840f856de0"
    );

    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("pfld-import");
    let device = Default::default();
    let mut model = PFLD_GhostOne::<CpuBackend>::new(PfldConfig::production(), &device);
    let report = import_pfld_checkpoint::<CpuBackend, _>(
        &mut model,
        &PfldImportRequest {
            checkpoint: checkpoint.clone(),
            destination_dir: destination.clone(),
            ..PfldImportRequest::default()
        },
    )
    .unwrap();

    assert_eq!(report.destination_dir, destination);
    assert_eq!(report.manifest.schema_version, 1);
    assert_eq!(report.manifest.model_type, "pfld_ghost_one");
    assert_eq!(
        report.manifest.architecture_version,
        "burn-pfld-structure-v1"
    );
    assert_eq!(report.manifest.epoch, 335);
    assert_eq!(
        report.manifest.source.sha256,
        "bada866661ad5fa1080a085f51fe9c016c69958c406951afa4afc7840f856de0"
    );
    assert_eq!(
        report.manifest.backbone,
        TensorSummary {
            tensor_count: 2_090,
            total_elements: 913_663,
        }
    );
    assert_eq!(report.applied.len(), 1_735);
    assert_sorted_unique(&report.applied);
    assert_eq!(report.manifest.model.tensor_count, 1_735);
    assert_eq!(report.manifest.model.total_elements, 910_902);
    assert_eq!(
        report.manifest.ignored.batch_norm_counters.tensor_count,
        351
    );
    assert_eq!(
        report.manifest.ignored.batch_norm_counters.total_elements,
        351
    );
    assert_sorted_unique(&report.manifest.ignored.batch_norm_counters.keys);
    assert_eq!(
        report.manifest.ignored.localization,
        TensorAudit {
            tensor_count: 4,
            total_elements: 2_410,
            keys: vec![
                "localization.0.bias".to_owned(),
                "localization.0.weight".to_owned(),
                "localization.3.bias".to_owned(),
                "localization.3.weight".to_owned(),
            ],
        }
    );
    assert_sorted_unique(&report.manifest.ignored.localization.keys);
    let auxiliary = report.manifest.ignored.auxiliarynet.as_ref().unwrap();
    assert_eq!(auxiliary.tensor_count, 48);
    assert_eq!(auxiliary.total_elements, 137_036);
    assert!(
        auxiliary
            .keys
            .iter()
            .all(|key| key.starts_with("auxiliarynet."))
    );
    assert_sorted_unique(&auxiliary.keys);

    let mut entries = std::fs::read_dir(&destination)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    entries.sort();
    assert_eq!(
        entries,
        vec!["manifest.json".to_owned(), "model.safetensors".to_owned()]
    );

    let manifest: PfldImportManifest =
        serde_json::from_slice(&std::fs::read(destination.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest, report.manifest);
    assert_eq!(
        sha256(&destination.join("model.safetensors")),
        report.manifest.model.sha256
    );

    let mut reloaded = PFLD_GhostOne::<CpuBackend>::new(PfldConfig::production(), &device);
    let mut store = SafetensorsStore::from_file(destination.join("model.safetensors"));
    let result = reloaded.load_from(&mut store).unwrap();
    assert!(result.missing.is_empty());
    assert!(result.unused.is_empty());
    assert!(result.errors.is_empty());
    assert_module_snapshots_equal(&model, &reloaded);

    let input = Tensor::<CpuBackend, 4>::zeros([1, 3, 192, 192], &device);
    assert_eq!(reloaded.forward(input).dims(), [1, 220]);
}
