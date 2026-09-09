mod error;
mod model;
mod package;
mod persistence;
mod platform;

pub use error::ProjectError;
pub use model::{
    AssetManifest, AssetPackageState, FeatureType, ModelSelection, ProjectManifest,
    TaskHistoryEntry, TaskHistoryStatus, validate_relative_manifest_path,
};
pub use package::{AssetPackage, ValidatedProject, lock_asset_package, validate_project_dir};
pub use persistence::{
    read_asset_manifest, read_project_manifest, write_asset_manifest_atomic,
    write_project_manifest_atomic,
};
