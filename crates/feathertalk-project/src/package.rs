use crate::{
    AssetManifest, ProjectError, ProjectManifest, read_asset_manifest, read_project_manifest,
    write_asset_manifest_atomic,
};
use std::{
    fs,
    path::{Path, PathBuf},
};

const REQUIRED_FILES: &[&str] = &[
    "assets/video_25fps.mp4",
    "assets/audio_16k_mono.wav",
    "assets/features/feather_hubert.f32",
];
const REQUIRED_DIRS: &[&str] = &["assets/frames", "assets/landmarks"];

#[derive(Debug, Clone)]
pub struct AssetPackage {
    root: PathBuf,
    manifest: AssetManifest,
}
impl AssetPackage {
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn manifest(&self) -> &AssetManifest {
        &self.manifest
    }
}

#[derive(Debug, Clone)]
pub struct ValidatedProject {
    root: PathBuf,
    manifest: ProjectManifest,
    asset_package: AssetPackage,
}
impl ValidatedProject {
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn manifest(&self) -> &ProjectManifest {
        &self.manifest
    }
    pub fn asset_package(&self) -> &AssetPackage {
        &self.asset_package
    }
}

pub fn lock_asset_package(
    project_root: &Path,
    manifest: AssetManifest,
) -> Result<AssetPackage, ProjectError> {
    validate_root(project_root)?;
    manifest.validate_locked()?;
    validate_artifacts(project_root)?;
    let path = project_root.join("assets/assets.json");
    write_asset_manifest_atomic(&path, &manifest)?;
    Ok(AssetPackage {
        root: canonical_root(project_root)?,
        manifest,
    })
}

pub fn validate_project_dir(project_root: &Path) -> Result<ValidatedProject, ProjectError> {
    validate_root(project_root)?;
    let project_path = project_root.join("project.json");
    let manifest = read_project_manifest(&project_path)?;
    manifest.validate()?;
    let asset_path = project_root.join(&manifest.asset_package);
    let asset = read_asset_manifest(&asset_path)?;
    asset.validate_locked()?;
    validate_artifacts(project_root)?;
    let root = canonical_root(project_root)?;
    Ok(ValidatedProject {
        root: root.clone(),
        manifest,
        asset_package: AssetPackage {
            root,
            manifest: asset,
        },
    })
}

fn validate_root(root: &Path) -> Result<(), ProjectError> {
    let m = fs::symlink_metadata(root).map_err(|e| ProjectError::Io {
        operation: "stat",
        path: root.to_path_buf(),
        source: e,
    })?;
    if !m.is_dir() || m.file_type().is_symlink() {
        return Err(ProjectError::InvalidFilesystemEntry {
            path: root.to_path_buf(),
        });
    }
    Ok(())
}
fn validate_artifacts(root: &Path) -> Result<(), ProjectError> {
    for rel in REQUIRED_DIRS {
        let p = root.join(rel);
        reject_symlink_path(root, rel)?;
        let m = fs::symlink_metadata(&p)
            .map_err(|_| ProjectError::InvalidFilesystemEntry { path: p.clone() })?;
        if !m.is_dir() || m.file_type().is_symlink() {
            return Err(ProjectError::InvalidFilesystemEntry { path: p });
        }
    }
    for rel in REQUIRED_FILES {
        let p = root.join(rel);
        reject_symlink_path(root, rel)?;
        let m = fs::symlink_metadata(&p)
            .map_err(|_| ProjectError::InvalidFilesystemEntry { path: p.clone() })?;
        if !m.is_file() || m.file_type().is_symlink() {
            return Err(ProjectError::InvalidFilesystemEntry { path: p });
        }
        if m.len() == 0 {
            return Err(ProjectError::EmptyArtifact { path: p });
        }
    }
    Ok(())
}

fn reject_symlink_path(root: &Path, relative: &str) -> Result<(), ProjectError> {
    let mut current = root.to_path_buf();
    for component in relative.split('/') {
        current.push(component);
        if fs::symlink_metadata(&current).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
            return Err(ProjectError::Symlink { path: current });
        }
    }
    Ok(())
}

fn canonical_root(root: &Path) -> Result<PathBuf, ProjectError> {
    fs::canonicalize(root).map_err(|source| ProjectError::Io {
        operation: "canonicalize",
        path: root.to_path_buf(),
        source,
    })
}
