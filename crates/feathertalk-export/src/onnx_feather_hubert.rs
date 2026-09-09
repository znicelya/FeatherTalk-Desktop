use burn::tensor::backend::Backend;
use burn_store::ModuleSnapshot;
use feathertalk_models::feather_hubert::{FeatherHubertConfig, FeatherHubertEncoder};

use crate::onnx::{
    InitializerSet, ONNX_FLOAT_DATA_TYPE, OnnxAttributeProto, OnnxExportError, OnnxGraph,
    OnnxModel, OnnxModelKind, OnnxNodeProto, OnnxTensorProto, OnnxValue, add_snapshot_initializers,
    serialize_model, validate_graph_integrity, validate_model_contract,
};

const FRONTEND_KERNELS: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
const FRONTEND_STRIDES: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];
const FRONTEND_CHANNELS: [usize; 7] = [64, 128, 256, 384, 0, 0, 0];

pub fn export_feather_hubert_onnx<B: Backend>(
    model: &FeatherHubertEncoder<B>,
    config: &FeatherHubertConfig,
) -> Result<Vec<u8>, OnnxExportError> {
    validate_config(config)?;
    let snapshots = model.collect(None, None, false);
    let mut initializers = InitializerSet::new();
    add_snapshot_initializers(&mut initializers, snapshots.iter())?;
    let mut nodes = Vec::new();

    let waveform_shape = "const.waveform.reshape";
    add_i64_initializer(&mut initializers, waveform_shape, &[1, 1, -1]);
    nodes.push(node(
        "frontend.input.reshape",
        "Reshape",
        ["waveform", waveform_shape],
        ["frontend.input"],
        Vec::new(),
    ));
    let mut current = "frontend.input".to_owned();

    let mut channels = FRONTEND_CHANNELS;
    channels[4] = config.channels;
    channels[5] = config.channels;
    channels[6] = config.channels;
    for (index, ((channels_out, kernel), stride)) in channels
        .into_iter()
        .zip(FRONTEND_KERNELS)
        .zip(FRONTEND_STRIDES)
        .enumerate()
    {
        let prefix = format!("frontend.layers.{index}");
        let conv_output = format!("{prefix}.conv_out");
        nodes.push(conv_node(
            &format!("{prefix}.conv"),
            &current,
            &format!("frontend.layers.{index}.conv.weight"),
            None,
            &conv_output,
            kernel,
            stride,
            1,
            1,
            0,
        ));
        let norm_output = format!("{prefix}.norm_out");
        emit_group_norm(
            &mut nodes,
            &mut initializers,
            &conv_output,
            &norm_output,
            &format!("{prefix}.norm"),
            channels_out,
            group_count(channels_out),
            &format!("frontend.layers.{index}.norm.gamma"),
            &format!("frontend.layers.{index}.norm.beta"),
        )?;
        let gelu_output = format!("{prefix}.gelu_out");
        emit_gelu(
            &mut nodes,
            &mut initializers,
            &norm_output,
            &gelu_output,
            &format!("{prefix}.gelu"),
        );
        current = gelu_output;
    }

    let hidden_channels = config
        .channels
        .checked_mul(config.expansion)
        .ok_or_else(|| {
            OnnxExportError::UnsupportedConfiguration("channels * expansion overflowed".to_owned())
        })?;
    for index in 0..config.num_blocks {
        let prefix = format!("encoder.{index}");
        let norm_output = format!("{prefix}.norm_out");
        emit_group_norm(
            &mut nodes,
            &mut initializers,
            &current,
            &norm_output,
            &format!("{prefix}.norm"),
            config.channels,
            group_count(config.channels),
            &format!("encoder.{index}.norm.gamma"),
            &format!("encoder.{index}.norm.beta"),
        )?;
        let expanded = format!("{prefix}.pw_expand_out");
        nodes.push(conv_node(
            &format!("{prefix}.pw_expand"),
            &norm_output,
            &format!("encoder.{index}.pw_expand.weight"),
            None,
            &expanded,
            1,
            1,
            1,
            1,
            0,
        ));
        let depthwise = format!("{prefix}.dw_conv_out");
        let dilation = [1, 2, 4, 8][index % 4];
        nodes.push(conv_node(
            &format!("{prefix}.dw_conv"),
            &expanded,
            &format!("encoder.{index}.dw_conv.weight"),
            None,
            &depthwise,
            5,
            1,
            dilation,
            hidden_channels,
            2 * dilation,
        ));
        let gelu_output = format!("{prefix}.gelu_out");
        emit_gelu(
            &mut nodes,
            &mut initializers,
            &depthwise,
            &gelu_output,
            &format!("{prefix}.gelu"),
        );
        let projected = format!("{prefix}.pw_project_out");
        nodes.push(conv_node(
            &format!("{prefix}.pw_project"),
            &gelu_output,
            &format!("encoder.{index}.pw_project.weight"),
            None,
            &projected,
            1,
            1,
            1,
            1,
            0,
        ));
        let residual = format!("{prefix}.residual_add");
        nodes.push(node(
            &format!("{prefix}.residual"),
            "Add",
            [&current, &projected],
            [&residual],
            Vec::new(),
        ));
        current = residual;
    }

    let final_norm = "final_norm.out";
    emit_group_norm(
        &mut nodes,
        &mut initializers,
        &current,
        final_norm,
        "final_norm",
        config.channels,
        group_count(config.channels),
        "final_norm.gamma",
        "final_norm.beta",
    )?;
    let final_gelu = "final_norm.gelu_out";
    emit_gelu(
        &mut nodes,
        &mut initializers,
        final_norm,
        final_gelu,
        "final_norm.gelu",
    );
    nodes.push(conv_node(
        "proj",
        final_gelu,
        "proj.weight",
        Some("proj.bias"),
        "proj.out",
        1,
        1,
        1,
        1,
        0,
    ));
    nodes.push(node(
        "output.transpose",
        "Transpose",
        ["proj.out"],
        ["hidden"],
        vec![attribute_ints("perm", &[0, 2, 1])],
    ));

    let onnx = OnnxModel {
        ir_version: crate::onnx::ONNX_IR_VERSION,
        opset_version: crate::onnx::ONNX_OPSET_VERSION,
        graph_present: true,
        graph: OnnxGraph {
            name: "feathertalk.feather_hubert".to_owned(),
            inputs: vec![OnnxValue {
                name: "waveform".to_owned(),
                shape: vec![1, -1],
                dtype: ONNX_FLOAT_DATA_TYPE,
            }],
            outputs: vec![OnnxValue {
                name: "hidden".to_owned(),
                shape: vec![1, -1, 1024],
                dtype: ONNX_FLOAT_DATA_TYPE,
            }],
            nodes,
            initializers,
        },
    };
    validate_graph_integrity(&onnx)
        .map_err(|error| OnnxExportError::InvalidGraph(error.to_string()))?;
    let expected = OnnxModelKind::FeatherHubert;
    let bytes = serialize_model(&onnx)?;
    let contract = crate::onnx::OnnxModelContract::new(
        expected,
        vec![crate::onnx::OnnxTensorContract::new(
            "waveform",
            vec![1, -1],
        )],
        vec![crate::onnx::OnnxTensorContract::new(
            "hidden",
            vec![1, -1, 1024],
        )],
    );
    validate_model_contract(&bytes, &contract)
        .map_err(|error| OnnxExportError::InvalidGraph(error.to_string()))?;
    Ok(bytes)
}

fn validate_config(config: &FeatherHubertConfig) -> Result<(), OnnxExportError> {
    if config.channels == 0 || config.expansion == 0 || config.num_blocks == 0 {
        return Err(OnnxExportError::UnsupportedConfiguration(
            "channels, expansion, and num_blocks must be positive".to_owned(),
        ));
    }
    if config.output_dim != 1024 {
        return Err(OnnxExportError::UnsupportedConfiguration(format!(
            "FeatherHuBERT output_dim must be 1024, got {}",
            config.output_dim
        )));
    }
    if !config.dropout.is_finite() || !(0.0..1.0).contains(&config.dropout) {
        return Err(OnnxExportError::UnsupportedConfiguration(
            "dropout must be finite and in [0,1)".to_owned(),
        ));
    }
    Ok(())
}

fn group_count(channels: usize) -> usize {
    for groups in [32, 16, 8, 4, 2] {
        if channels.is_multiple_of(groups) {
            return groups;
        }
    }
    1
}

#[allow(clippy::too_many_arguments)]
fn emit_group_norm(
    nodes: &mut Vec<OnnxNodeProto>,
    initializers: &mut InitializerSet,
    input: &str,
    output: &str,
    prefix: &str,
    channels: usize,
    groups: usize,
    weight: &str,
    bias: &str,
) -> Result<(), OnnxExportError> {
    let reshape_in_shape = format!("{prefix}.shape.in");
    let reshape_out_shape = format!("{prefix}.shape.out");
    let affine_shape = format!("{prefix}.shape.affine");
    add_i64_initializer(initializers, &reshape_in_shape, &[0, groups as i64, -1]);
    add_i64_initializer(initializers, &reshape_out_shape, &[0, channels as i64, -1]);
    add_i64_initializer(initializers, &affine_shape, &[1, channels as i64, 1]);
    let instance_scale = format!("{prefix}.instance_scale");
    let instance_bias = format!("{prefix}.instance_bias");
    add_f32_initializer(
        initializers,
        &instance_scale,
        vec![1.0; groups],
        vec![groups as i64],
    );
    add_f32_initializer(
        initializers,
        &instance_bias,
        vec![0.0; groups],
        vec![groups as i64],
    );

    let reshaped = format!("{prefix}.reshape");
    nodes.push(node(
        &format!("{prefix}.reshape_in"),
        "Reshape",
        [input, reshape_in_shape.as_str()],
        [reshaped.as_str()],
        Vec::new(),
    ));
    let normalized = format!("{prefix}.instance_normalized");
    nodes.push(node(
        &format!("{prefix}.instance_normalization"),
        "InstanceNormalization",
        [
            reshaped.as_str(),
            instance_scale.as_str(),
            instance_bias.as_str(),
        ],
        [normalized.as_str()],
        vec![attribute_float("epsilon", 1e-5)],
    ));
    let restored = format!("{prefix}.restored");
    nodes.push(node(
        &format!("{prefix}.reshape_out"),
        "Reshape",
        [normalized.as_str(), reshape_out_shape.as_str()],
        [restored.as_str()],
        Vec::new(),
    ));
    let weight_reshaped = format!("{prefix}.weight_reshaped");
    nodes.push(node(
        &format!("{prefix}.weight_reshape"),
        "Reshape",
        [weight, affine_shape.as_str()],
        [weight_reshaped.as_str()],
        Vec::new(),
    ));
    let bias_reshaped = format!("{prefix}.bias_reshaped");
    nodes.push(node(
        &format!("{prefix}.bias_reshape"),
        "Reshape",
        [bias, affine_shape.as_str()],
        [bias_reshaped.as_str()],
        Vec::new(),
    ));
    let scaled = format!("{prefix}.scaled");
    nodes.push(node(
        &format!("{prefix}.scale"),
        "Mul",
        [restored.as_str(), weight_reshaped.as_str()],
        [scaled.as_str()],
        Vec::new(),
    ));
    nodes.push(node(
        &format!("{prefix}.affine"),
        "Add",
        [scaled.as_str(), bias_reshaped.as_str()],
        [output],
        Vec::new(),
    ));
    Ok(())
}

fn emit_gelu(
    nodes: &mut Vec<OnnxNodeProto>,
    initializers: &mut InitializerSet,
    input: &str,
    output: &str,
    prefix: &str,
) {
    let sqrt_two = format!("{prefix}.sqrt_two_inverse");
    let one = format!("{prefix}.one");
    let half = format!("{prefix}.half");
    add_f32_initializer(initializers, &sqrt_two, vec![0.707_106_77], Vec::new());
    add_f32_initializer(initializers, &one, vec![1.0], Vec::new());
    add_f32_initializer(initializers, &half, vec![0.5], Vec::new());
    let scaled = format!("{prefix}.scaled");
    let erf = format!("{prefix}.erf");
    let shifted = format!("{prefix}.shifted");
    let half_shifted = format!("{prefix}.half_shifted");
    nodes.push(node(
        &format!("{prefix}.scale"),
        "Mul",
        [input, sqrt_two.as_str()],
        [scaled.as_str()],
        Vec::new(),
    ));
    nodes.push(node(
        &format!("{prefix}.erf"),
        "Erf",
        [scaled.as_str()],
        [erf.as_str()],
        Vec::new(),
    ));
    nodes.push(node(
        &format!("{prefix}.add_one"),
        "Add",
        [erf.as_str(), one.as_str()],
        [shifted.as_str()],
        Vec::new(),
    ));
    nodes.push(node(
        &format!("{prefix}.half"),
        "Mul",
        [shifted.as_str(), half.as_str()],
        [half_shifted.as_str()],
        Vec::new(),
    ));
    nodes.push(node(
        &format!("{prefix}.output"),
        "Mul",
        [input, half_shifted.as_str()],
        [output],
        Vec::new(),
    ));
}

#[allow(clippy::too_many_arguments)]
fn conv_node(
    name: &str,
    input: &str,
    weight: &str,
    bias: Option<&str>,
    output: &str,
    kernel: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    padding: usize,
) -> OnnxNodeProto {
    let mut inputs = vec![input.to_owned(), weight.to_owned()];
    if let Some(bias) = bias {
        inputs.push(bias.to_owned());
    }
    node(
        name,
        "Conv",
        inputs,
        [output],
        vec![
            attribute_ints("kernel_shape", &[kernel as i64]),
            attribute_ints("strides", &[stride as i64]),
            attribute_ints("dilations", &[dilation as i64]),
            attribute_ints("pads", &[padding as i64, padding as i64]),
            attribute_int("group", groups as i64),
        ],
    )
}

fn node<I, O>(
    name: &str,
    op_type: &str,
    inputs: I,
    outputs: O,
    attribute: Vec<OnnxAttributeProto>,
) -> OnnxNodeProto
where
    I: IntoIterator,
    I::Item: AsRef<str>,
    O: IntoIterator,
    O::Item: AsRef<str>,
{
    OnnxNodeProto {
        input: inputs
            .into_iter()
            .map(|value| value.as_ref().to_owned())
            .collect(),
        output: outputs
            .into_iter()
            .map(|value| value.as_ref().to_owned())
            .collect(),
        name: name.to_owned(),
        op_type: op_type.to_owned(),
        attribute,
        doc_string: String::new(),
        domain: String::new(),
    }
}

fn attribute_int(name: &str, value: i64) -> OnnxAttributeProto {
    OnnxAttributeProto {
        name: name.to_owned(),
        i: value,
        r#type: 2,
        ..Default::default()
    }
}

fn attribute_ints(name: &str, values: &[i64]) -> OnnxAttributeProto {
    OnnxAttributeProto {
        name: name.to_owned(),
        ints: values.to_vec(),
        r#type: 7,
        ..Default::default()
    }
}

fn attribute_float(name: &str, value: f32) -> OnnxAttributeProto {
    OnnxAttributeProto {
        name: name.to_owned(),
        f: value,
        r#type: 1,
        ..Default::default()
    }
}

fn add_f32_initializer(
    initializers: &mut InitializerSet,
    name: &str,
    values: Vec<f32>,
    dims: Vec<i64>,
) {
    let mut raw_data = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        raw_data.extend_from_slice(&value.to_le_bytes());
    }
    initializers
        .insert_generated(OnnxTensorProto {
            dims,
            data_type: ONNX_FLOAT_DATA_TYPE,
            name: name.to_owned(),
            raw_data,
            doc_string: String::new(),
        })
        .expect("generated ONNX initializer names are unique");
}

fn add_i64_initializer(initializers: &mut InitializerSet, name: &str, values: &[i64]) {
    let mut raw_data = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        raw_data.extend_from_slice(&value.to_le_bytes());
    }
    initializers
        .insert_generated(OnnxTensorProto {
            dims: vec![values.len() as i64],
            data_type: 7,
            name: name.to_owned(),
            raw_data,
            doc_string: String::new(),
        })
        .expect("generated ONNX initializer names are unique");
}
