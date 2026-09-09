use crate::fixture::GoldenFixture;
use cap_std::{ambient_authority, fs::Dir};
use ndarray::ArrayD;
use ndarray_npy::{ReadNpyExt, npy::header::Header};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{self, BufReader, Cursor, Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    sync::{Mutex, MutexGuard},
};
use thiserror::Error;
use zip::ZipArchive;

const SUPPORTED_SCHEMA_VERSION: u32 = 1;
const EXPECTED_FIXTURE_SET: &str = "burn-feasibility-v1";
const MAX_ARCHIVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_ARCHIVE_ENTRIES: usize = 4096;
const MAX_ENTRY_BYTES: u64 = 512 * 1024 * 1024;
const MAX_EXPANDED_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_NPY_HEADER_BYTES: u64 = 1024 * 1024;
const MAX_ARRAY_BYTES: u64 = 256 * 1024 * 1024;
const MAX_FIXTURE_ARRAY_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum FixtureError {
    #[error("failed to access fixture file: {0}")]
    Io(#[from] io::Error),
    #[error("invalid ZIP archive: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("invalid fixture manifest: {0}")]
    Manifest(#[from] serde_json::Error),
    #[error("invalid NPY fixture: {0}")]
    Npy(#[from] ndarray_npy::ReadNpyError),
    #[error("archive exceeds {limit} bytes: {actual}")]
    ArchiveTooLarge { actual: u64, limit: u64 },
    #[error("archive contains too many entries: {actual} (limit {limit})")]
    TooManyEntries { actual: usize, limit: usize },
    #[error("archive entry exceeds {limit} bytes: {name} ({actual})")]
    EntryTooLarge {
        name: String,
        actual: u64,
        limit: u64,
    },
    #[error("archive expands beyond {limit} bytes")]
    ExpandedSizeExceeded { limit: u64 },
    #[error("unsafe archive entry path: {0}")]
    UnsafePath(String),
    #[error("archive entry is a symbolic link: {0}")]
    SymbolicLink(String),
    #[error("duplicate archive entry: {0}")]
    DuplicateEntry(String),
    #[error("invalid ZIP central directory: {0}")]
    InvalidCentralDirectory(String),
    #[error("multi-disk ZIP archives are not supported")]
    MultiDiskArchive,
    #[error("missing archive entry: {0}")]
    MissingEntry(String),
    #[error("unknown fixture id: {0}")]
    UnknownFixture(String),
    #[error("unsupported fixture schema version: {0}")]
    UnsupportedSchemaVersion(u32),
    #[error("unexpected fixture set: {0}")]
    UnexpectedFixtureSet(String),
    #[error("invalid SHA-256 sidecar")]
    InvalidHashSidecar,
    #[error("fixture SHA-256 mismatch: expected {expected}, actual {actual}")]
    HashMismatch { expected: String, actual: String },
    #[error("extraction destination already exists: {0}")]
    DestinationExists(PathBuf),
    #[error("invalid extraction destination: {0}")]
    InvalidDestination(PathBuf),
    #[error("fixture snapshot lock is poisoned")]
    SnapshotLockPoisoned,
    #[error("invalid NPY header for {name}: {message}")]
    InvalidArrayHeader { name: String, message: String },
    #[error("NPY header exceeds {limit} bytes: {name} ({actual})")]
    ArrayHeaderTooLarge {
        name: String,
        actual: u64,
        limit: u64,
    },
    #[error("array exceeds {limit} bytes: {name} ({actual})")]
    ArrayTooLarge {
        name: String,
        actual: u64,
        limit: u64,
    },
    #[error("array payload size mismatch for {name}: expected {expected}, actual {actual}")]
    ArrayPayloadSizeMismatch {
        name: String,
        expected: u64,
        actual: u64,
    },
    #[error("fixture arrays exceed aggregate allocation budget of {limit} bytes")]
    FixtureArrayBudgetExceeded { limit: u64 },
    #[error("declared shape does not match array {name}: expected {expected:?}, actual {actual:?}")]
    ArrayShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("duplicate expected array name: {0}")]
    DuplicateExpectedArray(String),
}

#[derive(Debug)]
pub struct GoldenArchive {
    sidecar_path: PathBuf,
    snapshot: Mutex<File>,
    entries: BTreeSet<String>,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: u32,
    fixture_set: String,
    #[serde(default, rename = "seed")]
    _seed: Option<u64>,
    #[serde(default, rename = "generator")]
    generator: Option<BTreeMap<String, Value>>,
    fixtures: BTreeMap<String, FixtureManifest>,
}

#[derive(Debug, Deserialize)]
struct FixtureManifest {
    kind: String,
    weights: String,
    config: BTreeMap<String, Value>,
    #[serde(default)]
    optimizer: Option<BTreeMap<String, Value>>,
    #[serde(default)]
    loss: Option<String>,
    #[serde(default)]
    inputs: BTreeMap<String, String>,
    #[serde(default)]
    expected: BTreeMap<String, String>,
    #[serde(default)]
    expected_json: Option<String>,
    #[serde(default)]
    metrics: BTreeMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct TrainingExpected {
    initial_loss: f64,
    post_step_loss: f64,
    post_step_mode: String,
    parameters: BTreeMap<String, ArrayReference>,
    batch_norm_state: BTreeMap<String, ArrayReference>,
}

#[derive(Debug, Deserialize)]
struct ArrayReference {
    path: String,
    shape: Vec<usize>,
}

#[derive(Debug, Default)]
struct ArrayBudget {
    used: u64,
}

impl ArrayBudget {
    fn charge(&mut self, name: &str, bytes: u64) -> Result<(), FixtureError> {
        if bytes > MAX_ARRAY_BYTES {
            return Err(FixtureError::ArrayTooLarge {
                name: name.to_owned(),
                actual: bytes,
                limit: MAX_ARRAY_BYTES,
            });
        }
        self.used =
            self.used
                .checked_add(bytes)
                .ok_or(FixtureError::FixtureArrayBudgetExceeded {
                    limit: MAX_FIXTURE_ARRAY_BYTES,
                })?;
        if self.used > MAX_FIXTURE_ARRAY_BYTES {
            return Err(FixtureError::FixtureArrayBudgetExceeded {
                limit: MAX_FIXTURE_ARRAY_BYTES,
            });
        }
        Ok(())
    }
}

impl GoldenArchive {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, FixtureError> {
        let path = path.into();
        let mut source = File::open(&path)?;
        let mut snapshot = tempfile::tempfile()?;
        let copied = io::copy(
            &mut Read::by_ref(&mut source).take(MAX_ARCHIVE_BYTES + 1),
            &mut snapshot,
        )?;
        if copied > MAX_ARCHIVE_BYTES {
            return Err(FixtureError::ArchiveTooLarge {
                actual: copied,
                limit: MAX_ARCHIVE_BYTES,
            });
        }
        snapshot.flush()?;
        snapshot.seek(SeekFrom::Start(0))?;
        let declared_entries = read_declared_entry_count(&mut snapshot)?;
        if declared_entries > MAX_ARCHIVE_ENTRIES {
            return Err(FixtureError::TooManyEntries {
                actual: declared_entries,
                limit: MAX_ARCHIVE_ENTRIES,
            });
        }
        snapshot.seek(SeekFrom::Start(0))?;

        let entries = {
            let mut archive = ZipArchive::new(BufReader::new(&mut snapshot))?;
            if archive.len() != declared_entries {
                return Err(FixtureError::DuplicateEntry(
                    "central directory contains duplicate names".to_owned(),
                ));
            }
            let mut entries = BTreeSet::new();
            for index in 0..archive.len() {
                let name = archive.by_index(index)?.name().to_owned();
                if !entries.insert(name.clone()) {
                    return Err(FixtureError::DuplicateEntry(name));
                }
            }
            entries
        };
        snapshot.seek(SeekFrom::Start(0))?;

        Ok(Self {
            sidecar_path: path.with_extension("sha256"),
            snapshot: Mutex::new(snapshot),
            entries,
        })
    }

    pub fn contains(&self, entry: &str) -> bool {
        self.entries.contains(entry)
    }

    pub fn verify_sidecar_sha256(&self) -> Result<(), FixtureError> {
        let sidecar = File::open(&self.sidecar_path)?;
        let mut expected = String::new();
        sidecar.take(129).read_to_string(&mut expected)?;
        let expected = expected.trim();
        if expected.len() != 64
            || !expected
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(FixtureError::InvalidHashSidecar);
        }

        let mut snapshot = self.lock_snapshot()?;
        snapshot.seek(SeekFrom::Start(0))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = snapshot.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let actual = hex::encode(hasher.finalize());
        if actual != expected {
            return Err(FixtureError::HashMismatch {
                expected: expected.to_owned(),
                actual,
            });
        }
        Ok(())
    }

    pub fn extract_to(&self, directory: &Path) -> Result<(), FixtureError> {
        self.preflight_extraction()?;
        let parent_path = directory
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let directory_name = directory
            .file_name()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| FixtureError::InvalidDestination(directory.to_path_buf()))?;
        let parent = Dir::open_ambient_dir(parent_path, ambient_authority())?;
        match parent.create_dir(directory_name) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                return Err(FixtureError::DestinationExists(directory.to_path_buf()));
            }
            Err(error) => return Err(error.into()),
        }
        let root = parent.open_dir(directory_name)?;

        self.with_zip(|archive| {
            let mut expanded = 0_u64;
            for index in 0..archive.len() {
                let mut entry = archive.by_index(index)?;
                let relative = validate_relative_path(entry.name())?;
                if entry.is_dir() {
                    root.create_dir_all(&relative)?;
                    continue;
                }

                if let Some(parent) = relative.parent()
                    && !parent.as_os_str().is_empty()
                {
                    root.create_dir_all(parent)?;
                }
                let mut options = cap_std::fs::OpenOptions::new();
                options.write(true).create_new(true);
                let mut destination = root.open_with(&relative, &options)?;
                let remaining = MAX_EXPANDED_BYTES.saturating_sub(expanded);
                let copy_limit = MAX_ENTRY_BYTES.min(remaining).saturating_add(1);
                let written = io::copy(&mut (&mut entry).take(copy_limit), &mut destination)?;
                if written > MAX_ENTRY_BYTES {
                    return Err(FixtureError::EntryTooLarge {
                        name: entry.name().to_owned(),
                        actual: written,
                        limit: MAX_ENTRY_BYTES,
                    });
                }
                expanded =
                    expanded
                        .checked_add(written)
                        .ok_or(FixtureError::ExpandedSizeExceeded {
                            limit: MAX_EXPANDED_BYTES,
                        })?;
                if expanded > MAX_EXPANDED_BYTES {
                    return Err(FixtureError::ExpandedSizeExceeded {
                        limit: MAX_EXPANDED_BYTES,
                    });
                }
            }
            Ok(())
        })
    }

    pub fn load_fixture(&self, id: &str) -> Result<GoldenFixture, FixtureError> {
        let manifest_bytes = self.read_entry("manifest.json", MAX_MANIFEST_BYTES)?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
        if manifest.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(FixtureError::UnsupportedSchemaVersion(
                manifest.schema_version,
            ));
        }
        if manifest.fixture_set != EXPECTED_FIXTURE_SET {
            return Err(FixtureError::UnexpectedFixtureSet(manifest.fixture_set));
        }
        let fixture = manifest
            .fixtures
            .get(id)
            .ok_or_else(|| FixtureError::UnknownFixture(id.to_owned()))?;
        validate_relative_path(&fixture.weights)?;
        if !self.contains(&fixture.weights) {
            return Err(FixtureError::MissingEntry(fixture.weights.clone()));
        }

        let mut budget = ArrayBudget::default();
        let inputs = self.load_arrays(&fixture.inputs, &mut budget)?;
        let mut expected = self.load_arrays(&fixture.expected, &mut budget)?;
        let mut scalars = BTreeMap::new();
        let mut expected_mode = None;

        if let Some(path) = &fixture.expected_json {
            validate_relative_path(path)?;
            let bytes = self.read_entry(path, MAX_MANIFEST_BYTES)?;
            let training: TrainingExpected = serde_json::from_slice(&bytes)?;
            scalars.insert("initial_loss".to_owned(), training.initial_loss);
            scalars.insert("post_step_loss".to_owned(), training.post_step_loss);
            expected_mode = Some(training.post_step_mode);
            self.load_referenced_arrays(&training.parameters, &mut expected, &mut budget)?;
            self.load_referenced_arrays(&training.batch_norm_state, &mut expected, &mut budget)?;
        }

        Ok(GoldenFixture {
            id: id.to_owned(),
            schema_version: manifest.schema_version,
            fixture_set: manifest.fixture_set,
            generator: manifest.generator,
            kind: fixture.kind.clone(),
            weights_entry: fixture.weights.clone(),
            config: fixture.config.clone(),
            optimizer: fixture.optimizer.clone(),
            loss: fixture.loss.clone(),
            expected_mode,
            inputs,
            expected,
            metrics: fixture.metrics.clone(),
            scalars,
        })
    }

    fn load_arrays(
        &self,
        entries: &BTreeMap<String, String>,
        budget: &mut ArrayBudget,
    ) -> Result<BTreeMap<String, ArrayD<f32>>, FixtureError> {
        let mut arrays = BTreeMap::new();
        for (name, path) in entries {
            arrays.insert(name.clone(), self.load_array(name, path, budget)?);
        }
        Ok(arrays)
    }

    fn load_referenced_arrays(
        &self,
        references: &BTreeMap<String, ArrayReference>,
        arrays: &mut BTreeMap<String, ArrayD<f32>>,
        budget: &mut ArrayBudget,
    ) -> Result<(), FixtureError> {
        for (name, reference) in references {
            if arrays.contains_key(name) {
                return Err(FixtureError::DuplicateExpectedArray(name.clone()));
            }
            let array = self.load_array(name, &reference.path, budget)?;
            if array.shape() != reference.shape {
                return Err(FixtureError::ArrayShapeMismatch {
                    name: name.clone(),
                    expected: reference.shape.clone(),
                    actual: array.shape().to_vec(),
                });
            }
            arrays.insert(name.clone(), array);
        }
        Ok(())
    }

    fn load_array(
        &self,
        name: &str,
        path: &str,
        budget: &mut ArrayBudget,
    ) -> Result<ArrayD<f32>, FixtureError> {
        validate_relative_path(path)?;
        let bytes = self.read_entry(path, MAX_ARRAY_BYTES + MAX_NPY_HEADER_BYTES)?;
        let payload_bytes = validate_npy_before_allocation(name, &bytes)?;
        budget.charge(name, payload_bytes)?;
        Ok(ArrayD::<f32>::read_npy(Cursor::new(bytes))?)
    }

    fn read_entry(&self, name: &str, limit: u64) -> Result<Vec<u8>, FixtureError> {
        self.with_zip(|archive| {
            let mut entry = archive.by_name(name).map_err(|error| match error {
                zip::result::ZipError::FileNotFound => FixtureError::MissingEntry(name.to_owned()),
                other => FixtureError::Zip(other),
            })?;
            if entry.size() > limit {
                return Err(FixtureError::EntryTooLarge {
                    name: name.to_owned(),
                    actual: entry.size(),
                    limit,
                });
            }
            let capacity = usize::try_from(entry.size()).unwrap_or(usize::MAX);
            let mut bytes = Vec::with_capacity(capacity.min(1024 * 1024));
            (&mut entry).take(limit + 1).read_to_end(&mut bytes)?;
            if bytes.len() as u64 > limit {
                return Err(FixtureError::EntryTooLarge {
                    name: name.to_owned(),
                    actual: bytes.len() as u64,
                    limit,
                });
            }
            Ok(bytes)
        })
    }

    fn preflight_extraction(&self) -> Result<(), FixtureError> {
        self.with_zip(|archive| {
            let mut expanded = 0_u64;
            let mut normalized_paths = BTreeSet::new();
            for index in 0..archive.len() {
                let entry = archive.by_index(index)?;
                let relative = validate_relative_path(entry.name())?;
                if !normalized_paths.insert(relative) {
                    return Err(FixtureError::DuplicateEntry(entry.name().to_owned()));
                }
                if is_symbolic_link(entry.unix_mode()) {
                    return Err(FixtureError::SymbolicLink(entry.name().to_owned()));
                }
                if entry.size() > MAX_ENTRY_BYTES {
                    return Err(FixtureError::EntryTooLarge {
                        name: entry.name().to_owned(),
                        actual: entry.size(),
                        limit: MAX_ENTRY_BYTES,
                    });
                }
                expanded = expanded.checked_add(entry.size()).ok_or(
                    FixtureError::ExpandedSizeExceeded {
                        limit: MAX_EXPANDED_BYTES,
                    },
                )?;
                if expanded > MAX_EXPANDED_BYTES {
                    return Err(FixtureError::ExpandedSizeExceeded {
                        limit: MAX_EXPANDED_BYTES,
                    });
                }
            }
            Ok(())
        })
    }

    fn with_zip<T>(
        &self,
        operation: impl FnOnce(&mut ZipArchive<BufReader<&mut File>>) -> Result<T, FixtureError>,
    ) -> Result<T, FixtureError> {
        let mut snapshot = self.lock_snapshot()?;
        snapshot.seek(SeekFrom::Start(0))?;
        let mut archive = ZipArchive::new(BufReader::new(&mut *snapshot))?;
        operation(&mut archive)
    }

    fn lock_snapshot(&self) -> Result<MutexGuard<'_, File>, FixtureError> {
        self.snapshot
            .lock()
            .map_err(|_| FixtureError::SnapshotLockPoisoned)
    }
}

fn validate_relative_path(path: &str) -> Result<PathBuf, FixtureError> {
    if path.is_empty()
        || path.starts_with('/')
        || path.starts_with('\\')
        || path.contains('\\')
        || path
            .as_bytes()
            .get(1)
            .is_some_and(|character| *character == b':')
        || path.split('/').any(|component| component == "..")
    {
        return Err(FixtureError::UnsafePath(path.to_owned()));
    }
    let path = Path::new(path);
    if path.components().any(|component| {
        matches!(
            component,
            Component::Prefix(_) | Component::RootDir | Component::ParentDir
        )
    }) {
        return Err(FixtureError::UnsafePath(path.display().to_string()));
    }
    Ok(path.to_path_buf())
}

fn validate_npy_before_allocation(name: &str, bytes: &[u8]) -> Result<u64, FixtureError> {
    if let Some(header_bytes) = declared_npy_header_bytes(bytes)
        && header_bytes > MAX_NPY_HEADER_BYTES
    {
        return Err(FixtureError::ArrayHeaderTooLarge {
            name: name.to_owned(),
            actual: header_bytes,
            limit: MAX_NPY_HEADER_BYTES,
        });
    }

    let mut cursor = Cursor::new(bytes);
    let header =
        Header::from_reader(&mut cursor).map_err(|error| FixtureError::InvalidArrayHeader {
            name: name.to_owned(),
            message: error.to_string(),
        })?;
    if cursor.position() > MAX_NPY_HEADER_BYTES {
        return Err(FixtureError::ArrayHeaderTooLarge {
            name: name.to_owned(),
            actual: cursor.position(),
            limit: MAX_NPY_HEADER_BYTES,
        });
    }
    let elements = header.shape.iter().try_fold(1_u64, |count, dimension| {
        count.checked_mul(*dimension as u64)
    });
    let payload_bytes = elements
        .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
        .ok_or_else(|| FixtureError::ArrayTooLarge {
            name: name.to_owned(),
            actual: u64::MAX,
            limit: MAX_ARRAY_BYTES,
        })?;
    if payload_bytes > MAX_ARRAY_BYTES {
        return Err(FixtureError::ArrayTooLarge {
            name: name.to_owned(),
            actual: payload_bytes,
            limit: MAX_ARRAY_BYTES,
        });
    }
    let actual = (bytes.len() as u64).saturating_sub(cursor.position());
    if actual != payload_bytes {
        return Err(FixtureError::ArrayPayloadSizeMismatch {
            name: name.to_owned(),
            expected: payload_bytes,
            actual,
        });
    }
    Ok(payload_bytes)
}

fn declared_npy_header_bytes(bytes: &[u8]) -> Option<u64> {
    if bytes.get(..6)? != b"\x93NUMPY" {
        return None;
    }
    match *bytes.get(6)? {
        1 => {
            let length = u16::from_le_bytes(bytes.get(8..10)?.try_into().ok()?);
            Some(10 + u64::from(length))
        }
        2 | 3 => {
            let length = u32::from_le_bytes(bytes.get(8..12)?.try_into().ok()?);
            Some(12 + u64::from(length))
        }
        _ => None,
    }
}

fn read_declared_entry_count(file: &mut File) -> Result<usize, FixtureError> {
    const EOCD_SIGNATURE: &[u8; 4] = b"PK\x05\x06";
    const ZIP64_LOCATOR_SIGNATURE: &[u8; 4] = b"PK\x06\x07";
    const ZIP64_EOCD_SIGNATURE: &[u8; 4] = b"PK\x06\x06";
    const EOCD_FIXED_BYTES: usize = 22;
    const MAX_ZIP_COMMENT_BYTES: u64 = u16::MAX as u64;

    let file_len = file.seek(SeekFrom::End(0))?;
    let tail_len = file_len.min(EOCD_FIXED_BYTES as u64 + MAX_ZIP_COMMENT_BYTES);
    file.seek(SeekFrom::Start(file_len - tail_len))?;
    let mut tail = vec![0_u8; tail_len as usize];
    file.read_exact(&mut tail)?;

    let eocd_offset = (0..=tail.len().saturating_sub(EOCD_FIXED_BYTES))
        .rev()
        .find(|offset| {
            tail.get(*offset..*offset + 4) == Some(EOCD_SIGNATURE)
                && read_u16(&tail, *offset + 20).is_some_and(|comment_len| {
                    *offset + EOCD_FIXED_BYTES + comment_len as usize == tail.len()
                })
        })
        .ok_or_else(|| FixtureError::InvalidCentralDirectory("missing end record".to_owned()))?;

    let disk_number = read_u16(&tail, eocd_offset + 4).unwrap();
    let central_disk = read_u16(&tail, eocd_offset + 6).unwrap();
    let entries_on_disk = read_u16(&tail, eocd_offset + 8).unwrap();
    let total_entries = read_u16(&tail, eocd_offset + 10).unwrap();
    if disk_number != 0 || central_disk != 0 || entries_on_disk != total_entries {
        return Err(FixtureError::MultiDiskArchive);
    }
    if total_entries != u16::MAX {
        return Ok(total_entries as usize);
    }

    let absolute_eocd = file_len - tail_len + eocd_offset as u64;
    if absolute_eocd < 20 {
        return Err(FixtureError::InvalidCentralDirectory(
            "missing ZIP64 locator".to_owned(),
        ));
    }
    file.seek(SeekFrom::Start(absolute_eocd - 20))?;
    let mut locator = [0_u8; 20];
    file.read_exact(&mut locator)?;
    if locator.get(..4) != Some(ZIP64_LOCATOR_SIGNATURE) {
        return Err(FixtureError::InvalidCentralDirectory(
            "missing ZIP64 locator".to_owned(),
        ));
    }
    if read_u32(&locator, 4) != Some(0) || read_u32(&locator, 16) != Some(1) {
        return Err(FixtureError::MultiDiskArchive);
    }
    let zip64_offset = read_u64(&locator, 8).unwrap();
    file.seek(SeekFrom::Start(zip64_offset))?;
    let mut zip64 = [0_u8; 56];
    file.read_exact(&mut zip64)?;
    if zip64.get(..4) != Some(ZIP64_EOCD_SIGNATURE) {
        return Err(FixtureError::InvalidCentralDirectory(
            "invalid ZIP64 end record".to_owned(),
        ));
    }
    if read_u32(&zip64, 16) != Some(0) || read_u32(&zip64, 20) != Some(0) {
        return Err(FixtureError::MultiDiskArchive);
    }
    let entries_on_disk = read_u64(&zip64, 24).unwrap();
    let total_entries = read_u64(&zip64, 32).unwrap();
    if entries_on_disk != total_entries {
        return Err(FixtureError::MultiDiskArchive);
    }
    usize::try_from(total_entries).map_err(|_| FixtureError::TooManyEntries {
        actual: usize::MAX,
        limit: MAX_ARCHIVE_ENTRIES,
    })
}

fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        bytes.get(offset..offset + 2)?.try_into().ok()?,
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

fn is_symbolic_link(mode: Option<u32>) -> bool {
    mode.is_some_and(|mode| mode & 0o170000 == 0o120000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_array_budget_is_enforced_without_allocating() {
        let mut budget = ArrayBudget::default();
        budget.charge("first", MAX_ARRAY_BYTES).unwrap();
        budget.charge("second", MAX_ARRAY_BYTES).unwrap();
        assert!(matches!(
            budget.charge("third", size_of::<f32>() as u64),
            Err(FixtureError::FixtureArrayBudgetExceeded { .. })
        ));
    }
}
