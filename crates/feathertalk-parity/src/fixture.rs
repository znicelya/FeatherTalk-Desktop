use ndarray::ArrayD;
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

use burn::tensor::{Tensor, TensorData};
use burn::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::ElementConversion,
};
use feathertalk_models::{
    backend::{CpuAutodiffBackend, CpuBackend, GpuAutodiffBackend, GpuBackend},
    feather_hubert::FeatherHubertConfig,
    train_step::{adam_train_step, l1_loss},
    unet::OriginalUnetConfig,
};
use feathertalk_weights::{
    LegacyImportRequest, LegacyModelKind, import_into, load_feather_hubert_checkpoint,
};

use crate::{
    archive::{FixtureError, GoldenArchive},
    metrics::{ParityError, ParityMetrics, compare_f32},
};

#[derive(Debug)]
pub struct GoldenFixture {
    pub id: String,
    pub schema_version: u32,
    pub fixture_set: String,
    pub generator: Option<BTreeMap<String, Value>>,
    pub kind: String,
    pub weights_entry: String,
    pub config: BTreeMap<String, Value>,
    pub optimizer: Option<BTreeMap<String, Value>>,
    pub loss: Option<String>,
    pub expected_mode: Option<String>,
    pub inputs: BTreeMap<String, ArrayD<f32>>,
    pub expected: BTreeMap<String, ArrayD<f32>>,
    pub metrics: BTreeMap<String, f64>,
    pub scalars: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardCase {
    FeatherMicro,
    UnetProduction,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct FeatherForwardConfig {
    channels: usize,
    expansion: usize,
    num_blocks: usize,
    output_dim: usize,
    dropout: f64,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct UnetForwardConfig {
    channels: [usize; 5],
    mode: UnetMode,
    n_channels: usize,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum UnetMode {
    Hubert,
}

#[derive(Debug, Clone, Copy)]
struct ArrayContract {
    name: &'static str,
    shape: &'static [usize],
}

const FEATHER_WAVEFORM_SHAPE: &[usize] = &[1, 1360];
const FEATHER_OUTPUT_SHAPE: &[usize] = &[1, 4, 64];
const UNET_AUDIO_SHAPE: &[usize] = &[1, 16, 32, 32];
const UNET_IMAGE_SHAPE: &[usize] = &[1, 6, 160, 160];
const UNET_OUTPUT_SHAPE: &[usize] = &[1, 3, 160, 160];

const FEATHER_INPUTS: &[ArrayContract] = &[ArrayContract {
    name: "waveform",
    shape: FEATHER_WAVEFORM_SHAPE,
}];
const FEATHER_OUTPUTS: &[ArrayContract] = &[ArrayContract {
    name: "output",
    shape: FEATHER_OUTPUT_SHAPE,
}];
const UNET_INPUTS: &[ArrayContract] = &[
    ArrayContract {
        name: "audio",
        shape: UNET_AUDIO_SHAPE,
    },
    ArrayContract {
        name: "image",
        shape: UNET_IMAGE_SHAPE,
    },
];
const UNET_OUTPUTS: &[ArrayContract] = &[ArrayContract {
    name: "output",
    shape: UNET_OUTPUT_SHAPE,
}];

impl ForwardCase {
    const fn name(self) -> &'static str {
        match self {
            Self::FeatherMicro => "FeatherMicro",
            Self::UnetProduction => "UnetProduction",
        }
    }

    const fn fixture_kind(self) -> &'static str {
        match self {
            Self::FeatherMicro => "feather_hubert",
            Self::UnetProduction => "original_unet",
        }
    }

    const fn fixture_id(self) -> &'static str {
        match self {
            Self::FeatherMicro => "feather_micro_eval",
            Self::UnetProduction => "unet_production_eval",
        }
    }

    const fn weights_entry(self) -> &'static str {
        match self {
            Self::FeatherMicro => "weights/feather_micro.pth",
            Self::UnetProduction => "weights/unet_production.pth",
        }
    }

    const fn weight_kind(self) -> LegacyModelKind {
        match self {
            Self::FeatherMicro => LegacyModelKind::FeatherHubert,
            Self::UnetProduction => LegacyModelKind::OriginalUnet,
        }
    }

    const fn arrays(self) -> (&'static [ArrayContract], &'static [ArrayContract]) {
        match self {
            Self::FeatherMicro => (FEATHER_INPUTS, FEATHER_OUTPUTS),
            Self::UnetProduction => (UNET_INPUTS, UNET_OUTPUTS),
        }
    }
}

pub fn validate_forward_fixture(
    fixture: &GoldenFixture,
    case: ForwardCase,
) -> Result<(), ParityError> {
    if fixture.id != case.fixture_id() {
        return Err(ParityError::FixtureContract {
            case: case.name(),
            field: "fixture_id",
            expected: case.fixture_id().to_owned(),
            actual: fixture.id.clone(),
        });
    }
    if fixture.kind != case.fixture_kind() {
        return Err(ParityError::FixtureContract {
            case: case.name(),
            field: "kind",
            expected: case.fixture_kind().to_owned(),
            actual: fixture.kind.clone(),
        });
    }
    if fixture.weights_entry != case.weights_entry() {
        return Err(ParityError::FixtureContract {
            case: case.name(),
            field: "weights_entry",
            expected: case.weights_entry().to_owned(),
            actual: fixture.weights_entry.clone(),
        });
    }
    if case == ForwardCase::FeatherMicro {
        let expected = FeatherForwardConfig {
            channels: 32,
            expansion: 2,
            num_blocks: 2,
            output_dim: 64,
            dropout: 0.0,
        };
        let actual = parse_config::<FeatherForwardConfig>(fixture, case, &expected)?;
        if actual != expected {
            return Err(config_mismatch(case, &expected, &actual));
        }
    } else {
        let expected = UnetForwardConfig {
            channels: [32, 64, 128, 256, 512],
            mode: UnetMode::Hubert,
            n_channels: 6,
        };
        let actual = parse_config::<UnetForwardConfig>(fixture, case, &expected)?;
        if actual != expected {
            return Err(config_mismatch(case, &expected, &actual));
        }
    }
    let (inputs, outputs) = case.arrays();
    validate_array_map(case, "input", &fixture.inputs, inputs)?;
    validate_array_map(case, "expected", &fixture.expected, outputs)?;
    Ok(())
}

fn validate_array_map(
    case: ForwardCase,
    role: &'static str,
    arrays: &BTreeMap<String, ArrayD<f32>>,
    contracts: &[ArrayContract],
) -> Result<(), ParityError> {
    let expected_names = contracts
        .iter()
        .map(|contract| contract.name)
        .collect::<BTreeSet<_>>();
    let actual_names = arrays.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if actual_names != expected_names {
        return Err(ParityError::FixtureArraySet {
            case: case.name(),
            role,
            expected: expected_names.into_iter().map(str::to_owned).collect(),
            actual: actual_names.into_iter().map(str::to_owned).collect(),
        });
    }

    for contract in contracts {
        let actual = &arrays[contract.name];
        if actual.shape() != contract.shape {
            return Err(ParityError::FixtureArrayShape {
                case: case.name(),
                role,
                name: contract.name,
                expected: contract.shape.to_vec(),
                actual: actual.shape().to_vec(),
            });
        }
    }
    Ok(())
}

fn parse_config<T>(
    fixture: &GoldenFixture,
    case: ForwardCase,
    expected: &T,
) -> Result<T, ParityError>
where
    T: serde::de::DeserializeOwned + std::fmt::Debug,
{
    let value = Value::Object(fixture.config.clone().into_iter().collect());
    serde_json::from_value(value).map_err(|error| ParityError::FixtureContract {
        case: case.name(),
        field: "config",
        expected: format!("{expected:?}"),
        actual: format!("invalid structured config: {error}"),
    })
}

fn config_mismatch(
    case: ForwardCase,
    expected: &impl std::fmt::Debug,
    actual: &impl std::fmt::Debug,
) -> ParityError {
    ParityError::FixtureContract {
        case: case.name(),
        field: "config",
        expected: format!("{expected:?}"),
        actual: format!("{actual:?}"),
    }
}

pub fn run_cpu_forward(
    archive: &GoldenArchive,
    case: ForwardCase,
) -> Result<ParityMetrics, ParityError> {
    archive
        .verify_sidecar_sha256()
        .map_err(ParityError::ArchiveVerification)?;
    let fixture = archive.load_fixture(case.fixture_id())?;
    validate_forward_fixture(&fixture, case)?;
    let expected = fixture
        .expected
        .get("output")
        .ok_or_else(|| ParityError::MissingArray("output".to_owned()))?;

    let temp = tempfile::tempdir().map_err(FixtureError::from)?;
    let extracted = temp.path().join("fixture");
    archive.extract_to(&extracted)?;
    let weights_path = extracted.join(case.weights_entry());
    let device = Default::default();

    let actual = match case {
        ForwardCase::FeatherMicro => {
            let (model, checkpoint) =
                load_feather_hubert_checkpoint::<CpuBackend>(&weights_path, &device)?;
            if checkpoint.config().output_dim != FEATHER_OUTPUT_SHAPE[2] {
                return Err(config_mismatch(
                    case,
                    &FEATHER_OUTPUT_SHAPE[2],
                    &checkpoint.config().output_dim,
                ));
            }
            let input = fixture
                .inputs
                .get("waveform")
                .ok_or_else(|| ParityError::MissingArray("waveform".to_owned()))?;
            let input = tensor_from_array::<2>("waveform", input, FEATHER_WAVEFORM_SHAPE, &device)?;
            array_from_tensor(model.forward(input))?
        }
        ForwardCase::UnetProduction => {
            let request = LegacyImportRequest {
                path: weights_path,
                kind: case.weight_kind(),
                ..Default::default()
            };
            let mut model = OriginalUnetConfig::production().init::<CpuBackend>(&device);
            import_into::<CpuBackend, _>(&mut model, &request)?;
            let image = fixture
                .inputs
                .get("image")
                .ok_or_else(|| ParityError::MissingArray("image".to_owned()))?;
            let audio = fixture
                .inputs
                .get("audio")
                .ok_or_else(|| ParityError::MissingArray("audio".to_owned()))?;
            let image = tensor_from_array::<4>("image", image, UNET_IMAGE_SHAPE, &device)?;
            let audio = tensor_from_array::<4>("audio", audio, UNET_AUDIO_SHAPE, &device)?;
            array_from_tensor(model.forward(image, audio))?
        }
    };

    compare_f32(actual.view(), expected.view())
}

fn tensor_from_array<const D: usize>(
    name: &'static str,
    array: &ArrayD<f32>,
    expected_shape: &[usize],
    device: &burn::tensor::Device<CpuBackend>,
) -> Result<Tensor<CpuBackend, D>, ParityError> {
    if array.ndim() != D || array.shape() != expected_shape {
        return Err(ParityError::TensorShape {
            name,
            expected: expected_shape.to_vec(),
            actual: array.shape().to_vec(),
        });
    }
    Ok(Tensor::from_data(
        TensorData::new(
            array.iter().copied().collect::<Vec<_>>(),
            array.shape().to_vec(),
        ),
        device,
    ))
}

fn array_from_tensor<const D: usize>(
    tensor: Tensor<CpuBackend, D>,
) -> Result<ArrayD<f32>, ParityError> {
    let shape = tensor.dims().to_vec();
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| ParityError::TensorData(error.to_string()))?;
    ArrayD::from_shape_vec(shape, values).map_err(|error| ParityError::Array(error.to_string()))
}

fn tensor_from_gpu_array<const D: usize>(
    name: &'static str,
    array: &ArrayD<f32>,
    expected_shape: &[usize],
    device: &burn::tensor::Device<GpuBackend>,
) -> Result<Tensor<GpuBackend, D>, ParityError> {
    if array.ndim() != D || array.shape() != expected_shape {
        return Err(ParityError::TensorShape {
            name,
            expected: expected_shape.to_vec(),
            actual: array.shape().to_vec(),
        });
    }
    Ok(Tensor::from_data(
        TensorData::new(
            array.iter().copied().collect::<Vec<_>>(),
            array.shape().to_vec(),
        ),
        device,
    ))
}

fn tensor_from_gpu_ad_array<const D: usize>(
    name: &'static str,
    array: &ArrayD<f32>,
    expected_shape: &[usize],
    device: &burn::tensor::Device<GpuAutodiffBackend>,
) -> Result<Tensor<GpuAutodiffBackend, D>, ParityError> {
    if array.ndim() != D || array.shape() != expected_shape {
        return Err(ParityError::TensorShape {
            name,
            expected: expected_shape.to_vec(),
            actual: array.shape().to_vec(),
        });
    }
    Ok(Tensor::from_data(
        TensorData::new(
            array.iter().copied().collect::<Vec<_>>(),
            array.shape().to_vec(),
        ),
        device,
    ))
}

fn array_from_gpu_tensor<const D: usize>(
    tensor: Tensor<GpuBackend, D>,
) -> Result<ArrayD<f32>, ParityError> {
    let shape = tensor.dims().to_vec();
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| ParityError::TensorData(error.to_string()))?;
    ArrayD::from_shape_vec(shape, values).map_err(|error| ParityError::Array(error.to_string()))
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct AdamOptimizerConfig {
    #[serde(rename = "type")]
    optimizer_type: OptimizerType,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum OptimizerType {
    Adam,
}

#[derive(Debug)]
pub struct TrainStepParity {
    pub initial_loss_relative: f32,
    pub post_step_loss_relative: f32,
    pub selected_parameter_relative: BTreeMap<String, f32>,
    pub batch_norm_state_relative: BTreeMap<String, f32>,
}

#[derive(Debug, serde::Serialize)]
pub struct WgpuForwardResult {
    pub execution: crate::probe::ExecutionEvidence,
    pub metrics: ParityMetrics,
}

#[derive(Debug, serde::Serialize)]
pub struct WgpuTrainStepResult {
    pub execution: crate::probe::ExecutionEvidence,
    pub initial_loss: f32,
    pub gradient_norm: f32,
    pub output_weight_changed: bool,
}

pub fn run_wgpu_forward(
    archive: &GoldenArchive,
    case: ForwardCase,
    graphics: crate::probe::GraphicsSelection,
) -> Result<WgpuForwardResult, ParityError> {
    let graphics = graphics.resolved();
    graphics.validate_for_target()?;
    let (metrics, execution) = match graphics {
        crate::probe::GraphicsSelection::Auto => run_wgpu_forward_with::<
            burn::backend::wgpu::graphics::AutoGraphicsApi,
        >(archive, case, "auto")?,
        crate::probe::GraphicsSelection::Dx12 => {
            run_wgpu_forward_with::<burn::backend::wgpu::graphics::Dx12>(archive, case, "dx12")?
        }
        crate::probe::GraphicsSelection::Metal => {
            run_wgpu_forward_with::<burn::backend::wgpu::graphics::Metal>(archive, case, "metal")?
        }
        crate::probe::GraphicsSelection::Vulkan => {
            run_wgpu_forward_with::<burn::backend::wgpu::graphics::Vulkan>(archive, case, "vulkan")?
        }
    };
    Ok(WgpuForwardResult { execution, metrics })
}

pub fn run_wgpu_train_step(
    archive: &GoldenArchive,
    graphics: crate::probe::GraphicsSelection,
    full_production_model: bool,
) -> Result<WgpuTrainStepResult, ParityError> {
    let graphics = graphics.resolved();
    graphics.validate_for_target()?;
    match graphics {
        crate::probe::GraphicsSelection::Auto => run_wgpu_train_step_with::<
            burn::backend::wgpu::graphics::AutoGraphicsApi,
        >(archive, "auto", full_production_model),
        crate::probe::GraphicsSelection::Dx12 => run_wgpu_train_step_with::<
            burn::backend::wgpu::graphics::Dx12,
        >(archive, "dx12", full_production_model),
        crate::probe::GraphicsSelection::Metal => run_wgpu_train_step_with::<
            burn::backend::wgpu::graphics::Metal,
        >(archive, "metal", full_production_model),
        crate::probe::GraphicsSelection::Vulkan => run_wgpu_train_step_with::<
            burn::backend::wgpu::graphics::Vulkan,
        >(
            archive, "vulkan", full_production_model
        ),
    }
}

pub(crate) fn probe_wgpu_with<G: burn::backend::wgpu::graphics::GraphicsApi>(
    requested_graphics: &str,
) -> Result<crate::probe::ExecutionEvidence, ParityError> {
    let (device, execution) = init_wgpu::<G>(requested_graphics)?;
    let value = Tensor::<GpuBackend, 1>::from_data(TensorData::from([1.0_f32, 2.0]), &device)
        .sum()
        .into_scalar()
        .elem::<f32>();
    if !value.is_finite() || (value - 3.0).abs() > f32::EPSILON {
        return Err(ParityError::Backend(format!(
            "WGPU execution probe returned {value}"
        )));
    }
    Ok(execution)
}

fn init_wgpu<G: burn::backend::wgpu::graphics::GraphicsApi>(
    requested_graphics: &str,
) -> Result<
    (
        burn::tensor::Device<GpuBackend>,
        crate::probe::ExecutionEvidence,
    ),
    ParityError,
> {
    static RUNTIME: OnceLock<
        Result<
            (
                burn::backend::wgpu::WgpuDevice,
                crate::probe::ExecutionEvidence,
            ),
            String,
        >,
    > = OnceLock::new();
    let result = RUNTIME.get_or_init(|| {
        let device: burn::tensor::Device<GpuBackend> = Default::default();
        let setup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            burn::backend::wgpu::init_setup::<G>(&device, Default::default())
        }))
        .map_err(panic_message)?;
        let info = setup.adapter.get_info();
        let used_cpu_fallback = format!("{:?}", info.device_type) == "Cpu";
        if used_cpu_fallback {
            return Err(format!(
                "WGPU selected CPU adapter {} instead of a GPU",
                info.name
            ));
        }
        let graphics = format!("{:?}", setup.backend).to_ascii_lowercase();
        Ok((
            device,
            crate::probe::ExecutionEvidence {
                backend: "wgpu".to_owned(),
                graphics: if graphics.is_empty() {
                    requested_graphics.to_owned()
                } else {
                    graphics
                },
                device: format!("{} ({:?})", info.name, info.device_type),
                used_cpu_fallback,
            },
        ))
    });
    let (device, evidence) = result
        .as_ref()
        .map_err(|error| ParityError::Backend(error.clone()))?;
    if requested_graphics != "auto" && evidence.graphics != requested_graphics {
        return Err(ParityError::Backend(format!(
            "WGPU runtime already initialized with {}, requested {}",
            evidence.graphics, requested_graphics
        )));
    }
    Ok((device.clone(), evidence.clone()))
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|value| (*value).to_owned())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown WGPU initialization panic".to_owned())
}

fn run_wgpu_forward_with<G: burn::backend::wgpu::graphics::GraphicsApi>(
    archive: &GoldenArchive,
    case: ForwardCase,
    requested_graphics: &str,
) -> Result<(ParityMetrics, crate::probe::ExecutionEvidence), ParityError> {
    archive
        .verify_sidecar_sha256()
        .map_err(ParityError::ArchiveVerification)?;
    let fixture = archive.load_fixture(case.fixture_id())?;
    validate_forward_fixture(&fixture, case)?;
    let expected = fixture
        .expected
        .get("output")
        .ok_or_else(|| ParityError::MissingArray("output".to_owned()))?;
    let temp = tempfile::tempdir().map_err(FixtureError::from)?;
    let extracted = temp.path().join("fixture");
    archive.extract_to(&extracted)?;
    let request = LegacyImportRequest {
        path: extracted.join(case.weights_entry()),
        kind: case.weight_kind(),
        ..Default::default()
    };
    let (device, execution) = init_wgpu::<G>(requested_graphics)?;
    let actual = match case {
        ForwardCase::FeatherMicro => {
            let mut model = FeatherHubertConfig::parity_micro().init::<GpuBackend>(&device);
            import_into::<GpuBackend, _>(&mut model, &request)?;
            let input = tensor_from_gpu_array::<2>(
                "waveform",
                &fixture.inputs["waveform"],
                FEATHER_WAVEFORM_SHAPE,
                &device,
            )?;
            array_from_gpu_tensor(model.forward(input))?
        }
        ForwardCase::UnetProduction => {
            let mut model = OriginalUnetConfig::production().init::<GpuBackend>(&device);
            import_into::<GpuBackend, _>(&mut model, &request)?;
            let image = tensor_from_gpu_array::<4>(
                "image",
                &fixture.inputs["image"],
                UNET_IMAGE_SHAPE,
                &device,
            )?;
            let audio = tensor_from_gpu_array::<4>(
                "audio",
                &fixture.inputs["audio"],
                UNET_AUDIO_SHAPE,
                &device,
            )?;
            array_from_gpu_tensor(model.forward(image, audio))?
        }
    };
    Ok((compare_f32(actual.view(), expected.view())?, execution))
}

fn run_wgpu_train_step_with<G: burn::backend::wgpu::graphics::GraphicsApi>(
    archive: &GoldenArchive,
    requested_graphics: &str,
    full_production_model: bool,
) -> Result<WgpuTrainStepResult, ParityError> {
    archive
        .verify_sidecar_sha256()
        .map_err(ParityError::ArchiveVerification)?;
    let _ = full_production_model;
    let case = ForwardCase::UnetProduction;
    let fixture = archive.load_fixture(case.fixture_id())?;
    validate_forward_fixture(&fixture, case)?;
    let temp = tempfile::tempdir().map_err(FixtureError::from)?;
    let extracted = temp.path().join("fixture");
    archive.extract_to(&extracted)?;
    let request = LegacyImportRequest {
        path: extracted.join(case.weights_entry()),
        kind: case.weight_kind(),
        ..Default::default()
    };
    let (device, execution) = init_wgpu::<G>(requested_graphics)?;
    let mut model = OriginalUnetConfig::production().init::<GpuAutodiffBackend>(&device);
    import_into::<GpuAutodiffBackend, _>(&mut model, &request)?;
    let image = tensor_from_gpu_ad_array::<4>(
        "image",
        &fixture.inputs["image"],
        UNET_IMAGE_SHAPE,
        &device,
    )?;
    let audio = tensor_from_gpu_ad_array::<4>(
        "audio",
        &fixture.inputs["audio"],
        UNET_AUDIO_SHAPE,
        &device,
    )?;
    let target = Tensor::<GpuAutodiffBackend, 4>::zeros([1, 3, 160, 160], &device);
    let prediction = model.forward(image, audio);
    let loss = l1_loss(prediction, target);
    let initial_loss = loss.clone().into_scalar().elem::<f32>();
    let raw_gradients = loss.backward();
    let gradient = model
        .outc
        .conv
        .weight
        .grad(&raw_gradients)
        .ok_or_else(|| ParityError::Backend("output weight gradient is missing".to_owned()))?;
    let gradient_norm = gradient.abs().mean().into_scalar().elem::<f32>();
    let gradients = GradientsParams::from_grads(raw_gradients, &model);
    let before = model.outc.conv.weight.val().into_data();
    let mut optimizer = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .init();
    let model = optimizer.step(1e-3, model, gradients);
    let after = model.outc.conv.weight.val().into_data();
    let before = before
        .to_vec::<f32>()
        .map_err(|error| ParityError::TensorData(error.to_string()))?;
    let after = after
        .to_vec::<f32>()
        .map_err(|error| ParityError::TensorData(error.to_string()))?;
    let output_weight_changed = before.iter().zip(after.iter()).any(|(a, b)| a != b);
    Ok(WgpuTrainStepResult {
        execution,
        initial_loss,
        gradient_norm,
        output_weight_changed,
    })
}

const TRAIN_CASE: &str = "UnetMicroTrainStep";
const TRAIN_FIXTURE_ID: &str = "unet_micro_train_step";
const TRAIN_FIXTURE_KIND: &str = "original_unet_train_step";
const TRAIN_WEIGHTS_ENTRY: &str = "weights/unet_micro_train.pth";

const TRAIN_INPUTS: &[ArrayContract] = &[
    ArrayContract {
        name: "audio",
        shape: UNET_AUDIO_SHAPE,
    },
    ArrayContract {
        name: "image",
        shape: UNET_IMAGE_SHAPE,
    },
    ArrayContract {
        name: "target",
        shape: UNET_OUTPUT_SHAPE,
    },
];
const TRAIN_PARAMETERS: &[ArrayContract] = &[
    ArrayContract {
        name: "inc.inconv.conv.0.weight",
        shape: &[12, 6, 1, 1],
    },
    ArrayContract {
        name: "audio_model.conv1.conv.0.weight",
        shape: &[32, 16, 1, 1],
    },
    ArrayContract {
        name: "outc.conv.weight",
        shape: &[3, 2, 1, 1],
    },
];
const TRAIN_BATCH_NORM_STATE: &[ArrayContract] = &[
    ArrayContract {
        name: "inc.inconv.conv.1.running_mean",
        shape: &[12],
    },
    ArrayContract {
        name: "inc.inconv.conv.1.running_var",
        shape: &[12],
    },
    ArrayContract {
        name: "audio_model.conv1.conv.1.running_mean",
        shape: &[32],
    },
    ArrayContract {
        name: "audio_model.conv1.conv.1.running_var",
        shape: &[32],
    },
];
const TRAIN_L1_RESIDUAL_MARGIN: f64 = 1e-3;
const TRAIN_TARGET_ELEMENTS: f64 = (3 * 160 * 160) as f64;
const TRAIN_INPUT_SEED: u64 = 3;
const TRAIN_INPUT_DTYPE: &str = "float32";

pub fn validate_train_step_fixture(fixture: &GoldenFixture) -> Result<(), ParityError> {
    validate_contract_field(
        fixture.id == TRAIN_FIXTURE_ID,
        "fixture_id",
        TRAIN_FIXTURE_ID,
        &fixture.id,
    )?;
    validate_contract_field(
        fixture.kind == TRAIN_FIXTURE_KIND,
        "kind",
        TRAIN_FIXTURE_KIND,
        &fixture.kind,
    )?;
    validate_contract_field(
        fixture.weights_entry == TRAIN_WEIGHTS_ENTRY,
        "weights_entry",
        TRAIN_WEIGHTS_ENTRY,
        &fixture.weights_entry,
    )?;

    let expected_config = UnetForwardConfig {
        channels: [2, 4, 8, 16, 32],
        mode: UnetMode::Hubert,
        n_channels: 6,
    };
    let actual_config = parse_structured_map("config", &fixture.config, &expected_config)?;
    if actual_config != expected_config {
        return Err(train_contract_mismatch(
            "config",
            &expected_config,
            &actual_config,
        ));
    }

    let expected_optimizer = AdamOptimizerConfig {
        optimizer_type: OptimizerType::Adam,
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weight_decay: 0.0,
    };
    let optimizer = fixture
        .optimizer
        .as_ref()
        .ok_or_else(|| train_contract_mismatch("optimizer", &expected_optimizer, &"missing"))?;
    let actual_optimizer = parse_structured_map("optimizer", optimizer, &expected_optimizer)?;
    if actual_optimizer != expected_optimizer {
        return Err(train_contract_mismatch(
            "optimizer",
            &expected_optimizer,
            &actual_optimizer,
        ));
    }

    let expected_loss = "mean_absolute_error";
    let actual_loss = fixture.loss.as_deref().unwrap_or("missing");
    if actual_loss != expected_loss {
        return Err(train_contract_mismatch(
            "loss",
            &expected_loss,
            &actual_loss,
        ));
    }

    let expected_mode = "eval";
    let actual_mode = fixture.expected_mode.as_deref().unwrap_or("missing");
    if actual_mode != expected_mode {
        return Err(train_contract_mismatch(
            "expected_mode",
            &expected_mode,
            &actual_mode,
        ));
    }

    validate_train_inputs(fixture)?;
    validate_array_subset(
        TRAIN_CASE,
        "selected_parameter",
        &fixture.expected,
        TRAIN_PARAMETERS,
    )?;
    validate_array_subset(
        TRAIN_CASE,
        "batch_norm_state",
        &fixture.expected,
        TRAIN_BATCH_NORM_STATE,
    )?;
    validate_expected_array_set(fixture)?;
    validate_training_scalars(fixture)?;
    validate_training_metrics(fixture)?;
    validate_training_provenance(fixture)?;
    Ok(())
}

fn validate_contract_field(
    valid: bool,
    field: &'static str,
    expected: &str,
    actual: &str,
) -> Result<(), ParityError> {
    if valid {
        Ok(())
    } else {
        Err(ParityError::FixtureContract {
            case: TRAIN_CASE,
            field,
            expected: expected.to_owned(),
            actual: actual.to_owned(),
        })
    }
}

fn train_contract_mismatch(
    field: &'static str,
    expected: &impl std::fmt::Debug,
    actual: &impl std::fmt::Debug,
) -> ParityError {
    ParityError::FixtureContract {
        case: TRAIN_CASE,
        field,
        expected: format!("{expected:?}"),
        actual: format!("{actual:?}"),
    }
}

fn parse_structured_map<T>(
    field: &'static str,
    map: &BTreeMap<String, Value>,
    expected: &T,
) -> Result<T, ParityError>
where
    T: serde::de::DeserializeOwned + std::fmt::Debug,
{
    let value = Value::Object(map.clone().into_iter().collect());
    serde_json::from_value(value).map_err(|error| ParityError::FixtureContract {
        case: TRAIN_CASE,
        field,
        expected: format!("{expected:?}"),
        actual: format!("invalid structured metadata: {error}"),
    })
}

fn validate_train_inputs(fixture: &GoldenFixture) -> Result<(), ParityError> {
    let expected_names = TRAIN_INPUTS.iter().map(|c| c.name).collect::<BTreeSet<_>>();
    let actual_names = fixture
        .inputs
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if actual_names != expected_names {
        return Err(ParityError::FixtureArraySet {
            case: TRAIN_CASE,
            role: "input",
            expected: expected_names.into_iter().map(str::to_owned).collect(),
            actual: actual_names.into_iter().map(str::to_owned).collect(),
        });
    }
    for contract in TRAIN_INPUTS {
        let actual = &fixture.inputs[contract.name];
        if actual.shape() != contract.shape {
            return Err(ParityError::FixtureArrayShape {
                case: TRAIN_CASE,
                role: "input",
                name: contract.name,
                expected: contract.shape.to_vec(),
                actual: actual.shape().to_vec(),
            });
        }
    }
    Ok(())
}

fn validate_array_subset(
    case: &'static str,
    role: &'static str,
    arrays: &BTreeMap<String, ArrayD<f32>>,
    contracts: &[ArrayContract],
) -> Result<(), ParityError> {
    let expected_names = contracts.iter().map(|c| c.name).collect::<BTreeSet<_>>();
    let actual_names = arrays
        .keys()
        .map(String::as_str)
        .filter(|n| expected_names.contains(n))
        .collect::<BTreeSet<_>>();
    if actual_names != expected_names {
        return Err(ParityError::FixtureArraySet {
            case,
            role,
            expected: expected_names.into_iter().map(str::to_owned).collect(),
            actual: actual_names.into_iter().map(str::to_owned).collect(),
        });
    }
    for contract in contracts {
        let actual = &arrays[contract.name];
        if actual.shape() != contract.shape {
            return Err(ParityError::FixtureArrayShape {
                case,
                role,
                name: contract.name,
                expected: contract.shape.to_vec(),
                actual: actual.shape().to_vec(),
            });
        }
    }
    Ok(())
}

fn validate_expected_array_set(fixture: &GoldenFixture) -> Result<(), ParityError> {
    let expected = TRAIN_PARAMETERS
        .iter()
        .chain(TRAIN_BATCH_NORM_STATE.iter())
        .map(|c| c.name)
        .collect::<BTreeSet<_>>();
    let actual = fixture
        .expected
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if actual == expected {
        Ok(())
    } else {
        Err(ParityError::FixtureArraySet {
            case: TRAIN_CASE,
            role: "expected",
            expected: expected.into_iter().map(str::to_owned).collect(),
            actual: actual.into_iter().map(str::to_owned).collect(),
        })
    }
}

fn validate_training_scalars(fixture: &GoldenFixture) -> Result<(), ParityError> {
    let expected = BTreeSet::from(["initial_loss", "post_step_loss"]);
    let actual = fixture
        .scalars
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if actual != expected || fixture.scalars.values().any(|v| !v.is_finite()) {
        return Err(ParityError::FixtureContract {
            case: TRAIN_CASE,
            field: "scalars",
            expected: "finite initial_loss and post_step_loss".to_owned(),
            actual: format!("{:?}", fixture.scalars),
        });
    }
    Ok(())
}

fn validate_training_metrics(fixture: &GoldenFixture) -> Result<(), ParityError> {
    let residual = fixture
        .metrics
        .get("initial_l1_residual_min_abs")
        .copied()
        .unwrap_or(f64::NAN);
    let adjusted = fixture
        .metrics
        .get("l1_cusp_adjusted_elements")
        .copied()
        .unwrap_or(f64::NAN);
    let valid_residual = residual.is_finite() && residual >= TRAIN_L1_RESIDUAL_MARGIN;
    let valid_adjusted = adjusted.is_finite()
        && adjusted.fract() == 0.0
        && adjusted > 0.0
        && adjusted <= TRAIN_TARGET_ELEMENTS;
    if !valid_residual || !valid_adjusted {
        return Err(ParityError::FixtureContract {
            case: TRAIN_CASE,
            field: "training_metrics",
            expected: format!(
                "finite initial_l1_residual_min_abs >= {TRAIN_L1_RESIDUAL_MARGIN} and positive integral l1_cusp_adjusted_elements <= {TRAIN_TARGET_ELEMENTS}"
            ),
            actual: format!(
                "initial_l1_residual_min_abs={residual}, l1_cusp_adjusted_elements={adjusted}"
            ),
        });
    }
    Ok(())
}

fn validate_training_provenance(fixture: &GoldenFixture) -> Result<(), ParityError> {
    let generator = fixture.generator.as_ref();
    let seed = generator
        .and_then(|metadata| metadata.get("train_input_seed"))
        .and_then(Value::as_u64);
    let dtype = generator
        .and_then(|metadata| metadata.get("train_input_dtype"))
        .and_then(Value::as_str);
    if seed != Some(TRAIN_INPUT_SEED) || dtype != Some(TRAIN_INPUT_DTYPE) {
        return Err(ParityError::FixtureContract {
            case: TRAIN_CASE,
            field: "generator",
            expected: format!(
                "train_input_seed={TRAIN_INPUT_SEED} and train_input_dtype={TRAIN_INPUT_DTYPE}"
            ),
            actual: format!("train_input_seed={seed:?}, train_input_dtype={dtype:?}"),
        });
    }
    Ok(())
}

pub fn run_cpu_train_step(archive: &GoldenArchive) -> Result<TrainStepParity, ParityError> {
    use burn::module::AutodiffModule;
    use burn::optim::AdamConfig;
    archive
        .verify_sidecar_sha256()
        .map_err(ParityError::ArchiveVerification)?;
    let fixture = archive.load_fixture(TRAIN_FIXTURE_ID)?;
    validate_train_step_fixture(&fixture)?;

    let temp = tempfile::tempdir().map_err(FixtureError::from)?;
    let extracted = temp.path().join("fixture");
    archive.extract_to(&extracted)?;
    let request = LegacyImportRequest {
        path: extracted.join(TRAIN_WEIGHTS_ENTRY),
        kind: LegacyModelKind::OriginalUnet,
        ..Default::default()
    };
    let device = Default::default();
    let mut model = OriginalUnetConfig::parity_micro().init::<CpuAutodiffBackend>(&device);
    import_into::<CpuAutodiffBackend, _>(&mut model, &request)?;

    let image = train_tensor_from_fixture(&fixture, "image", UNET_IMAGE_SHAPE, &device)?;
    let audio = train_tensor_from_fixture(&fixture, "audio", UNET_AUDIO_SHAPE, &device)?;
    let target = train_tensor_from_fixture(&fixture, "target", UNET_OUTPUT_SHAPE, &device)?;

    let mut optimizer = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .init();
    let (model, initial_loss) = adam_train_step(
        model,
        &mut optimizer,
        image.clone(),
        audio.clone(),
        target.clone(),
        1e-3,
    );

    let model = model.valid();
    let post_step_loss =
        l1_loss(model.forward(image.inner(), audio.inner()), target.inner()).into_scalar();

    let mut parameters = BTreeMap::new();
    parameters.insert(
        "inc.inconv.conv.0.weight".to_owned(),
        array_from_tensor(model.inc.inconv.expand_conv.weight.val())?,
    );
    parameters.insert(
        "audio_model.conv1.conv.0.weight".to_owned(),
        array_from_tensor(model.audio_model.conv1.expand_conv.weight.val())?,
    );
    parameters.insert(
        "outc.conv.weight".to_owned(),
        array_from_tensor(model.outc.conv.weight.val())?,
    );

    let mut batch_norm_state = BTreeMap::new();
    batch_norm_state.insert(
        "inc.inconv.conv.1.running_mean".to_owned(),
        array_from_tensor(model.inc.inconv.expand_bn.running_mean.value_sync())?,
    );
    batch_norm_state.insert(
        "inc.inconv.conv.1.running_var".to_owned(),
        array_from_tensor(model.inc.inconv.expand_bn.running_var.value_sync())?,
    );
    batch_norm_state.insert(
        "audio_model.conv1.conv.1.running_mean".to_owned(),
        array_from_tensor(model.audio_model.conv1.expand_bn.running_mean.value_sync())?,
    );
    batch_norm_state.insert(
        "audio_model.conv1.conv.1.running_var".to_owned(),
        array_from_tensor(model.audio_model.conv1.expand_bn.running_var.value_sync())?,
    );

    Ok(TrainStepParity {
        initial_loss_relative: compare_scalar(
            initial_loss,
            fixture.scalars["initial_loss"] as f32,
        )?,
        post_step_loss_relative: compare_scalar(
            post_step_loss,
            fixture.scalars["post_step_loss"] as f32,
        )?,
        selected_parameter_relative: compare_named_arrays(
            parameters,
            &fixture.expected,
            TRAIN_PARAMETERS,
        )?,
        batch_norm_state_relative: compare_named_arrays(
            batch_norm_state,
            &fixture.expected,
            TRAIN_BATCH_NORM_STATE,
        )?,
    })
}

fn train_tensor_from_fixture(
    fixture: &GoldenFixture,
    name: &'static str,
    expected_shape: &[usize],
    device: &burn::tensor::Device<CpuAutodiffBackend>,
) -> Result<Tensor<CpuAutodiffBackend, 4>, ParityError> {
    let array = fixture
        .inputs
        .get(name)
        .ok_or_else(|| ParityError::MissingArray(name.to_owned()))?;
    if array.ndim() != 4 || array.shape() != expected_shape {
        return Err(ParityError::TensorShape {
            name,
            expected: expected_shape.to_vec(),
            actual: array.shape().to_vec(),
        });
    }
    Ok(Tensor::from_data(
        TensorData::new(
            array.iter().copied().collect::<Vec<_>>(),
            array.shape().to_vec(),
        ),
        device,
    ))
}

fn compare_scalar(actual: f32, expected: f32) -> Result<f32, ParityError> {
    let actual = ndarray::arr0(actual).into_dyn();
    let expected = ndarray::arr0(expected).into_dyn();
    Ok(compare_f32(actual.view(), expected.view())?.max_relative)
}

fn compare_named_arrays(
    actual: BTreeMap<String, ArrayD<f32>>,
    expected: &BTreeMap<String, ArrayD<f32>>,
    contracts: &[ArrayContract],
) -> Result<BTreeMap<String, f32>, ParityError> {
    contracts
        .iter()
        .map(|contract| {
            let actual = actual
                .get(contract.name)
                .ok_or_else(|| ParityError::MissingArray(contract.name.to_owned()))?;
            let expected = expected
                .get(contract.name)
                .ok_or_else(|| ParityError::MissingArray(contract.name.to_owned()))?;
            Ok((
                contract.name.to_owned(),
                compare_f32(actual.view(), expected.view())?.max_relative,
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    #[test]
    fn fixed_rank_tensor_conversion_rejects_wrong_rank() {
        let device = Default::default();
        let waveform = ArrayD::zeros(IxDyn(&[1360]));

        assert!(matches!(
            tensor_from_array::<2>("waveform", &waveform, FEATHER_WAVEFORM_SHAPE, &device),
            Err(ParityError::TensorShape {
                name: "waveform",
                ..
            })
        ));
    }

    #[test]
    fn fixed_rank_tensor_conversion_rejects_wrong_shape() {
        let device = Default::default();
        let waveform = ArrayD::zeros(IxDyn(&[1, 1359]));

        assert!(matches!(
            tensor_from_array::<2>("waveform", &waveform, FEATHER_WAVEFORM_SHAPE, &device),
            Err(ParityError::TensorShape {
                name: "waveform",
                ..
            })
        ));
    }
}
