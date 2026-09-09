//! SCRFD Burn model artifact and raw inference contract.

mod artifact;
mod error;
mod generated;
mod manifest;
mod model;
mod output;

pub use artifact::ScrfdArtifactPaths;
pub use error::ScrfdError;
pub use manifest::{
    SCRFD_ANCHORS, SCRFD_ARCHITECTURE_VERSION, SCRFD_INPUT_SHAPE, SCRFD_MODEL_KIND,
    SCRFD_SCHEMA_VERSION, SCRFD_SOURCE_ONNX_BYTES, SCRFD_SOURCE_ONNX_SHA256, SCRFD_SOURCE_OPSET,
    SCRFD_STRIDES, ScrfdArtifactManifest, ScrfdFileManifest, ScrfdGeneratorManifest,
    ScrfdInputManifest, ScrfdLevelManifest, ScrfdLicenseManifest, ScrfdOutputManifest,
    ScrfdSourceManifest, ScrfdWeightManifest,
};
pub use model::ScrfdModel;
pub use output::{ScrfdLevelOutput, ScrfdRawOutput};
