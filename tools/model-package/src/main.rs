use std::{
    fs,
    path::{Path, PathBuf}};

use clap::{Parser, Subcommand, ValueEnum};
use feathertalk_export::{
    FeatherHubertPackageRequest, MANIFEST_FILE_NAME, MAX_MODEL_BYTES, ModelConfiguration,
    ModelDescription, ModelPackageManifest, build_feather_hubert_package,
    export_feather_hubert_onnx, export_mobileone_unet_onnx, export_original_unet_onnx,
    load_model_package,
    onnx::{ONNX_OPSET_VERSION, OnnxModelKind, validate_model_contract},
    publish_onnx_model};
use feathertalk_models::unet::{
        MobileOneUnet, MobileOneUnetConfig, MobileOneUnetInference, OriginalUnet,
        OriginalUnetConfig};
use feathertalk_weights::{
    LegacyImportRequest, LegacyModelKind, import_into, load_feather_hubert_checkpoint};
use serde::Serialize;
use sha2::{Digest, Sha256};

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

mod migrate;

use migrate::{ModelMigrationKind, ModelMigrationRequest, migrate_features, migrate_model};

#[derive(Debug, Parser)]
#[command(
    name = "feathertalk-model-package",
    about = "Build auditable FeatherTalk model packages"
)]
struct Arguments {
    #[command(subcommand)]
    command: Command}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(name = "feather-hubert")]
    FeatherHubert {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        licenses: PathBuf,
        #[arg(long)]
        destination: PathBuf,
        #[arg(long)]
        created_at: String,
        #[arg(long)]
        minimum_app_version: String},
    #[command(subcommand)]
    Onnx(OnnxCommand),
    #[command(subcommand)]
    Migrate(MigrateCommand)}

#[derive(Debug, Subcommand)]
enum MigrateCommand {
    Model {
        #[arg(long, value_enum)]
        kind: MigratableModelKind,
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        licenses: PathBuf,
        #[arg(long)]
        destination: PathBuf,
        #[arg(long)]
        created_at: String,
        #[arg(long)]
        minimum_app_version: String},
    Features {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        destination: PathBuf}}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MigratableModelKind {
    FeatherHubert,
    OriginalUnet}

impl From<MigratableModelKind> for ModelMigrationKind {
    fn from(value: MigratableModelKind) -> Self {
        match value {
            MigratableModelKind::FeatherHubert => Self::FeatherHubert,
            MigratableModelKind::OriginalUnet => Self::OriginalUnet}
    }
}

#[derive(Debug, Subcommand)]
enum OnnxCommand {
    FeatherHubert {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        destination: PathBuf},
    Unet {
        #[arg(long)]
        source: PathBuf,
        #[arg(long, value_enum)]
        variant: UnetVariant,
        #[arg(long)]
        destination: PathBuf},
    Validate {
        #[arg(long)]
        source: PathBuf,
        #[arg(long, value_enum)]
        kind: OnnxKind}}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum UnetVariant {
    Original,
    Mobileone}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OnnxKind {
    FeatherHubert,
    OriginalUnet,
    MobileoneUnet}

impl From<OnnxKind> for OnnxModelKind {
    fn from(value: OnnxKind) -> Self {
        match value {
            OnnxKind::FeatherHubert => Self::FeatherHubert,
            OnnxKind::OriginalUnet => Self::OriginalUnet,
            OnnxKind::MobileoneUnet => Self::MobileOneUnet}
    }
}

#[derive(Debug, Serialize)]
struct OnnxReport {
    model_kind: &'static str,
    opset: i64,
    bytes: usize,
    sha256: String}

fn main() -> CliResult<()> {
    match Arguments::parse().command {
        Command::FeatherHubert {
            source,
            licenses,
            destination,
            created_at,
            minimum_app_version} => package_feather_hubert(
            source,
            licenses,
            destination,
            created_at,
            minimum_app_version,
        ),
        Command::Onnx(command) => run_onnx(command),
        Command::Migrate(command) => run_migrate(command)}
}

fn run_migrate(command: MigrateCommand) -> CliResult<()> {
    match command {
        MigrateCommand::Model {
            kind,
            source,
            licenses,
            destination,
            created_at,
            minimum_app_version} => {
            ensure_destination_absent(&destination)?;
            reject_protected_source(&source)?;
            let manifest = migrate_model(&ModelMigrationRequest {
                kind: kind.into(),
                source,
                licenses,
                destination: destination.clone(),
                created_at,
                minimum_app_version})?;
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "kind": "model",
                    "model_kind": manifest.model_type,
                    "destination": destination,
                    "source_sha256": manifest.source.sha256,
                    "model_sha256": manifest.model.sha256,
                    "tensor_count": manifest.tensors.tensor_count,
                    "total_elements": manifest.tensors.total_elements}))?
            );
            Ok(())
        }
        MigrateCommand::Features {
            source,
            destination} => {
            ensure_destination_absent(&destination)?;
            reject_protected_source(&source)?;
            let report = migrate_features(&source, &destination)?;
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "kind": "features",
                    "source_shape": report.source_shape,
                    "destination": destination,
                    "tokens": report.artifact.tokens(),
                    "dims": report.artifact.dims(),
                    "bytes": report.artifact.bytes(),
                    "sha256": report.artifact.sha256()}))?
            );
            Ok(())
        }
    }
}

fn package_feather_hubert(
    source: PathBuf,
    licenses: PathBuf,
    destination: PathBuf,
    created_at: String,
    minimum_app_version: String,
) -> CliResult<()> {
    let report = build_feather_hubert_package(&FeatherHubertPackageRequest {
        source,
        licenses,
        destination: destination.clone(),
        created_at,
        minimum_app_version})?;
    println!("destination={}", destination.display());
    println!("source_sha256={}", report.manifest.source.sha256);
    println!("model_sha256={}", report.manifest.model.sha256);
    println!("tensor_count={}", report.manifest.tensors.tensor_count);
    println!("total_elements={}", report.manifest.tensors.total_elements);
    if let ModelConfiguration::FeatherHubert {
        channels,
        expansion,
        num_blocks,
        output_dim,
        dropout} = report.manifest.configuration
    {
        println!(
            "configuration=channels:{channels},expansion:{expansion},num_blocks:{num_blocks},output_dim:{output_dim},dropout:{dropout}"
        );
    }
    Ok(())
}

fn run_onnx(command: OnnxCommand) -> CliResult<()> {
    match command {
        OnnxCommand::FeatherHubert {
            source,
            destination} => {
            ensure_destination_absent(&destination)?;
            reject_protected_source(&source)?;
            let device = Default::default();
            let (model, checkpoint) =
                load_feather_hubert_checkpoint(&source, &device)?;
            let bytes = export_feather_hubert_onnx(&model, checkpoint.config())?;
            publish_onnx(&destination, OnnxModelKind::FeatherHubert, &bytes)
        }
        OnnxCommand::Unet {
            source,
            variant,
            destination} => {
            ensure_destination_absent(&destination)?;
            reject_protected_source(&source)?;
            let (kind, bytes) = export_unet(&source, variant)?;
            publish_onnx(&destination, kind, &bytes)
        }
        OnnxCommand::Validate { source, kind } => {
            reject_protected_source(&source)?;
            let kind: OnnxModelKind = kind.into();
            let bytes = read_bounded(&source, MAX_MODEL_BYTES)?;
            validate_model_contract(&bytes, &kind.public_contract())?;
            print_report(kind, &bytes)
        }
    }
}

fn export_unet(source: &Path, variant: UnetVariant) -> CliResult<(OnnxModelKind, Vec<u8>)> {
    let device = Default::default();
    match variant {
        UnetVariant::Original => {
            if source.is_dir() {
                let manifest = read_package_manifest(source)?;
                let ModelConfiguration::OriginalUnet { channels } = manifest.configuration else {
                    return Err("package is not an Original UNet model".into());
                };
                let config = OriginalUnetConfig { channels };
                let expected = ModelDescription::original_unet(config.clone());
                let (model, _) = load_model_package::<OriginalUnet, _>(
                    source,
                    &expected,
                    &device,
                    |device| config.init(device),
                )?;
                Ok((
                    OnnxModelKind::OriginalUnet,
                    export_original_unet_onnx(&model, &config)?,
                ))
            } else {
                let config = OriginalUnetConfig::production();
                let mut model = config.init(&device);
                import_into::<_>(
                    &mut model,
                    &LegacyImportRequest {
                        path: source.to_owned(),
                        kind: LegacyModelKind::OriginalUnet,
                        ..Default::default()
                    },
                )?;
                Ok((
                    OnnxModelKind::OriginalUnet,
                    export_original_unet_onnx(&model, &config)?,
                ))
            }
        }
        UnetVariant::Mobileone => export_mobileone_package(source)}
}

fn export_mobileone_package(source: &Path) -> CliResult<(OnnxModelKind, Vec<u8>)> {
    if !source.is_dir() {
        return Err("MobileOne ONNX export requires a standard model package".into());
    }
    let manifest = read_package_manifest(source)?;
    let ModelConfiguration::MobileOneUnet {
        channels,
        num_conv_branches,
        reparameterized} = manifest.configuration
    else {
        return Err("package is not a MobileOne UNet model".into());
    };
    let config = MobileOneUnetConfig {
        channels,
        num_conv_branches};
    let device = Default::default();
    let inference = if reparameterized {
        let expected = ModelDescription::mobileone_unet(config.clone(), true);
        load_model_package::<MobileOneUnetInference, _>(
            source,
            &expected,
            &device,
            |device| config.init(device).reparameterize(),
        )?
        .0
    } else {
        let expected = ModelDescription::mobileone_unet(config.clone(), false);
        let (training, _) = load_model_package::<MobileOneUnet, _>(
            source,
            &expected,
            &device,
            |device| config.init(device),
        )?;
        training.reparameterize()
    };
    Ok((
        OnnxModelKind::MobileOneUnet,
        export_mobileone_unet_onnx(&inference, &config)?,
    ))
}

fn read_package_manifest(directory: &Path) -> CliResult<ModelPackageManifest> {
    let bytes = read_bounded(&directory.join(MANIFEST_FILE_NAME), 64 * 1024)?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// Validates and writes the graph, then reports it. The publication itself lives
/// in `feathertalk-export` so the worker's `export_onnx` command and this tool
/// cannot drift apart.
fn publish_onnx(destination: &Path, kind: OnnxModelKind, bytes: &[u8]) -> CliResult<()> {
    publish_onnx_model(destination, kind, bytes)?;
    print_report(kind, bytes)
}

fn print_report(kind: OnnxModelKind, bytes: &[u8]) -> CliResult<()> {
    let report = OnnxReport {
        model_kind: match kind {
            OnnxModelKind::FeatherHubert => "feather_hubert",
            OnnxModelKind::OriginalUnet => "original_unet",
            OnnxModelKind::MobileOneUnet => "mobileone_unet"},
        opset: ONNX_OPSET_VERSION,
        bytes: bytes.len(),
        sha256: hex::encode(Sha256::digest(bytes))};
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

fn ensure_destination_absent(destination: &Path) -> CliResult<()> {
    if destination.exists() {
        return Err(format!("destination already exists: {}", destination.display()).into());
    }
    Ok(())
}

fn reject_protected_source(source: &Path) -> CliResult<()> {
    let normalized = source
        .to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase();
    if normalized.ends_with(
        "demo/kanghui_training_video_featherhubert_188_latest/kanghui_training_video.mov",
    ) {
        return Err("protected demo MOV cannot be used as a model source".into());
    }
    Ok(())
}

fn read_bounded(path: &Path, limit: u64) -> CliResult<Vec<u8>> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() > limit {
        return Err(format!("invalid or oversized file: {}", path.display()).into());
    }
    Ok(fs::read(path)?)
}
