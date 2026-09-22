use burn::tensor::Device;
use feathertalk_export::{
    ModelConfiguration, ModelDescription, PackageError, load_model_package, read_package_manifest};
use feathertalk_models::feather_hubert::{BurnFeatherHubertEncoder, FeatherHubertConfig, FeatherHubertEncoder};

use crate::FeatureToolchain;

/// A FeatherHuBERT encoder loaded from a strict model package, paired with the
/// weight hash the package declares.
#[derive(Debug)]
pub struct FeatureModel {
    encoder: BurnFeatherHubertEncoder,
    model_sha256: String}

impl FeatureModel{
    pub fn load(features: &FeatureToolchain) -> Result<Self, PackageError> {
        Self::load_on(features, Default::default())
    }
}

impl FeatureModel {
    /// Loads the encoder from the directory the FeatherHuBERT toolchain resolved.
    ///
    /// The five hyperparameters come from the package manifest, never from
    /// `FeatherHubertConfig::default()`: the shipped model is 256/2/8/1024/0.0
    /// while the default is 512/2/12/1024/0.05.
    pub fn load_on(features: &FeatureToolchain, device: Device) -> Result<Self, PackageError> {
        let directory = features.hubert_dir();
        let manifest = read_package_manifest(directory)?;
        let config = feather_hubert_config(&manifest.configuration)?;
        let (model, _) = load_model_package::<FeatherHubertEncoder, _>(
            directory,
            &ModelDescription::feather_hubert(config.clone()),
            &device,
            |device| config.init(device),
        )?;
        Ok(Self {
            encoder: BurnFeatherHubertEncoder::from_model(model, &device),
            model_sha256: manifest.model.sha256})
    }

    /// Splits the loaded model into the two pieces the command needs.
    pub fn into_parts(self) -> (BurnFeatherHubertEncoder, String) {
        (self.encoder, self.model_sha256)
    }
}

/// Copies the five FeatherHuBERT hyperparameters out of a package configuration.
pub(crate) fn feather_hubert_config(
    configuration: &ModelConfiguration,
) -> Result<FeatherHubertConfig, PackageError> {
    match configuration {
        ModelConfiguration::FeatherHubert {
            channels,
            expansion,
            num_blocks,
            output_dim,
            dropout} => Ok(FeatherHubertConfig {
            channels: *channels,
            expansion: *expansion,
            num_blocks: *num_blocks,
            output_dim: *output_dim,
            dropout: *dropout}),
        other => Err(PackageError::InvalidManifest(format!(
            "expected a feather_hubert configuration, got {}",
            other.model_type()
        )))}
}
