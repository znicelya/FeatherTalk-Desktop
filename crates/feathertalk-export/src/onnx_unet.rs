use burn::tensor::backend::Backend;
use burn_store::ModuleSnapshot;
use feathertalk_models::unet::{
    MobileOneUnetConfig, MobileOneUnetInference, OriginalUnet, OriginalUnetConfig,
};

use crate::onnx::{
    InitializerSet, ONNX_FLOAT_DATA_TYPE, OnnxAttributeProto, OnnxExportError, OnnxGraph,
    OnnxModel, OnnxModelContract, OnnxModelKind, OnnxNodeProto, OnnxTensorContract,
    OnnxTensorProto, OnnxValue, add_snapshot_initializers, serialize_model,
    validate_graph_integrity, validate_model_contract,
};

pub fn export_original_unet_onnx<B: Backend>(
    model: &OriginalUnet<B>,
    config: &OriginalUnetConfig,
) -> Result<Vec<u8>, OnnxExportError> {
    validate_channels(&config.channels)?;
    let snapshots = model.collect(None, None, false);
    let mut builder = Builder::new(snapshots.iter())?;
    let x1 = builder.original_inverted(
        "inc.inconv",
        "input",
        config.channels[0],
        6,
        config.channels[0],
        2,
        1,
    )?;
    let x2 = builder.original_double(
        "down1.maxpool_conv",
        &x1,
        config.channels[0],
        config.channels[1],
        2,
    )?;
    let x3 = builder.original_double(
        "down2.maxpool_conv",
        &x2,
        config.channels[1],
        config.channels[2],
        2,
    )?;
    let x4 = builder.original_double(
        "down3.maxpool_conv",
        &x3,
        config.channels[2],
        config.channels[3],
        2,
    )?;
    let x5_image = builder.original_double(
        "down4.maxpool_conv",
        &x4,
        config.channels[3],
        config.channels[4],
        2,
    )?;
    let audio = builder.original_audio(&config.channels)?;
    let fused = builder.concat("bottleneck.concat", [x5_image.as_str(), audio.as_str()], 1);
    let fused = builder.original_double(
        "fuse_first",
        &fused,
        config.channels[4] * 2,
        config.channels[4],
        1,
    )?;
    let fused = builder.original_double(
        "fuse_second",
        &fused,
        config.channels[4],
        config.channels[3],
        1,
    )?;
    let up1 = builder.original_up(
        "up1",
        &fused,
        &x4,
        config.channels[4],
        config.channels[3] / 2,
    )?;
    let up2 = builder.original_up(
        "up2",
        &up1,
        &x3,
        config.channels[3] / 2 + config.channels[2],
        config.channels[2] / 2,
    )?;
    let up3 = builder.original_up(
        "up3",
        &up2,
        &x2,
        config.channels[2] / 2 + config.channels[1],
        config.channels[1] / 2,
    )?;
    let up4 = builder.original_up(
        "up4",
        &up3,
        &x1,
        config.channels[1] / 2 + config.channels[0],
        config.channels[0],
    )?;
    builder.final_output("outc.conv", &up4, "output")?;
    finish(OnnxModelKind::OriginalUnet, builder)
}

pub fn export_mobileone_unet_onnx<B: Backend>(
    model: &MobileOneUnetInference<B>,
    config: &MobileOneUnetConfig,
) -> Result<Vec<u8>, OnnxExportError> {
    validate_channels(&config.channels)?;
    if config.num_conv_branches == 0 {
        return Err(OnnxExportError::UnsupportedConfiguration(
            "num_conv_branches must be positive".to_owned(),
        ));
    }
    let snapshots = model.collect(None, None, false);
    let mut builder = Builder::new(snapshots.iter())?;
    let x1 = builder.mobile_sep("inc", "input", 6, config.channels[0], 1, false)?;
    let x2 = builder.mobile_double(
        "down1.maxpool_conv",
        &x1,
        config.channels[0],
        config.channels[1],
        2,
    )?;
    let x3 = builder.mobile_double(
        "down2.maxpool_conv",
        &x2,
        config.channels[1],
        config.channels[2],
        2,
    )?;
    let x4 = builder.mobile_double(
        "down3.maxpool_conv",
        &x3,
        config.channels[2],
        config.channels[3],
        2,
    )?;
    let x5_image = builder.mobile_double(
        "down4.maxpool_conv",
        &x4,
        config.channels[3],
        config.channels[4],
        2,
    )?;
    let audio = builder.mobile_audio(&config.channels)?;
    let fused = builder.concat("bottleneck.concat", [x5_image.as_str(), audio.as_str()], 1);
    let fused = builder.mobile_double(
        "fuse_first",
        &fused,
        config.channels[4] * 2,
        config.channels[4],
        1,
    )?;
    let fused = builder.mobile_double(
        "fuse_second",
        &fused,
        config.channels[4],
        config.channels[3],
        1,
    )?;
    let up1 = builder.mobile_up(
        "up1",
        &fused,
        &x4,
        config.channels[4],
        config.channels[3] / 2,
    )?;
    let up2 = builder.mobile_up("up2", &up1, &x3, config.channels[3], config.channels[2] / 2)?;
    let up3 = builder.mobile_up("up3", &up2, &x2, config.channels[2], config.channels[1] / 2)?;
    let up4 = builder.mobile_up("up4", &up3, &x1, config.channels[1], config.channels[0])?;
    builder.final_output("outc.conv", &up4, "output")?;
    finish(OnnxModelKind::MobileOneUnet, builder)
}

struct Builder {
    nodes: Vec<OnnxNodeProto>,
    initializers: InitializerSet,
}

impl Builder {
    fn new<'a>(
        snapshots: impl IntoIterator<Item = &'a burn_store::TensorSnapshot>,
    ) -> Result<Self, OnnxExportError> {
        let mut initializers = InitializerSet::new();
        add_snapshot_initializers(&mut initializers, snapshots)?;
        Ok(Self {
            nodes: Vec::new(),
            initializers,
        })
    }

    fn original_audio(&mut self, channels: &[usize; 5]) -> Result<String, OnnxExportError> {
        let mut current =
            self.original_inverted("audio_model.conv1", "audio", 16, channels[1], 2, 1, 1)?;
        current = self.original_inverted(
            "audio_model.conv2",
            &current,
            channels[1],
            channels[2],
            2,
            1,
            1,
        )?;
        current = self.original_conv_bn_relu(
            "audio_model.conv3",
            &current,
            channels[2],
            channels[3],
            3,
            2,
            1,
            true,
            "audio_model.bn3",
        )?;
        current = self.original_inverted(
            "audio_model.conv4",
            &current,
            channels[3],
            channels[3],
            6,
            1,
            1,
        )?;
        current = self.original_conv_bn_relu(
            "audio_model.conv5",
            &current,
            channels[3],
            channels[4],
            3,
            2,
            3,
            true,
            "audio_model.bn5",
        )?;
        current = self.original_inverted(
            "audio_model.conv6",
            &current,
            channels[4],
            channels[4],
            6,
            1,
            1,
        )?;
        self.original_inverted(
            "audio_model.conv7",
            &current,
            channels[4],
            channels[4],
            6,
            1,
            1,
        )
    }

    fn original_double(
        &mut self,
        prefix: &str,
        input: &str,
        inp: usize,
        oup: usize,
        stride: usize,
    ) -> Result<String, OnnxExportError> {
        let first =
            self.original_inverted(&format!("{prefix}.first"), input, inp, oup, 2, stride, 1)?;
        self.original_inverted(&format!("{prefix}.second"), &first, oup, oup, 2, 1, 1)
    }

    #[allow(clippy::too_many_arguments)]
    fn original_inverted(
        &mut self,
        prefix: &str,
        input: &str,
        inp: usize,
        oup: usize,
        expansion: usize,
        stride: usize,
        _padding: usize,
    ) -> Result<String, OnnxExportError> {
        let hidden = inp.checked_mul(expansion).ok_or_else(|| {
            OnnxExportError::UnsupportedConfiguration(
                "inverted residual channel overflow".to_owned(),
            )
        })?;
        let expanded = format!("{prefix}.expand");
        self.nodes.push(conv_node(
            &format!("{prefix}.expand_conv"),
            input,
            &format!("{prefix}.expand_conv.weight"),
            None,
            &expanded,
            1,
            1,
            0,
            1,
        ));
        self.nodes.push(batch_norm_node(
            &format!("{prefix}.expand_bn"),
            &expanded,
            &format!("{prefix}.expand_bn"),
            &format!("{prefix}.expand_bn.out"),
        ));
        self.nodes.push(relu_node(
            &format!("{prefix}.expand_relu"),
            &format!("{prefix}.expand_bn.out"),
            &format!("{prefix}.expand_relu.out"),
        ));
        let depth = format!("{prefix}.depthwise");
        self.nodes.push(conv_node_2d(
            &format!("{prefix}.depthwise_conv"),
            &format!("{prefix}.expand_relu.out"),
            &format!("{prefix}.depthwise_conv.weight"),
            "",
            &depth,
            3,
            [stride, stride],
            1,
            hidden,
        ));
        self.nodes.push(batch_norm_node(
            &format!("{prefix}.depthwise_bn"),
            &depth,
            &format!("{prefix}.depthwise_bn"),
            &format!("{prefix}.depthwise_bn.out"),
        ));
        self.nodes.push(relu_node(
            &format!("{prefix}.depthwise_relu"),
            &format!("{prefix}.depthwise_bn.out"),
            &format!("{prefix}.depthwise_relu.out"),
        ));
        let projected = format!("{prefix}.project");
        self.nodes.push(conv_node(
            &format!("{prefix}.project_conv"),
            &format!("{prefix}.depthwise_relu.out"),
            &format!("{prefix}.project_conv.weight"),
            None,
            &projected,
            1,
            1,
            0,
            1,
        ));
        self.nodes.push(batch_norm_node(
            &format!("{prefix}.project_bn"),
            &projected,
            &format!("{prefix}.project_bn"),
            &format!("{prefix}.project_bn.out"),
        ));
        let output = format!("{prefix}.out");
        if stride == 1 && inp == oup {
            self.nodes.push(node(
                &format!("{prefix}.residual"),
                "Add",
                [input, &format!("{prefix}.project_bn.out")],
                [&output],
                Vec::new(),
            ));
        } else {
            self.nodes.push(node(
                &format!("{prefix}.output"),
                "Identity",
                [&format!("{prefix}.project_bn.out")],
                [&output],
                Vec::new(),
            ));
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn original_conv_bn_relu(
        &mut self,
        prefix: &str,
        input: &str,
        _inp: usize,
        _out: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        has_bias: bool,
        bn_prefix: &str,
    ) -> Result<String, OnnxExportError> {
        let conv_output = format!("{prefix}.conv");
        self.nodes.push(conv_node(
            &format!("{prefix}.conv_node"),
            input,
            &format!("{prefix}.weight"),
            has_bias.then_some(format!("{prefix}.bias")).as_deref(),
            &conv_output,
            kernel,
            stride,
            padding,
            1,
        ));
        self.nodes.push(batch_norm_node(
            &format!("{prefix}.batch_norm"),
            &conv_output,
            bn_prefix,
            &format!("{prefix}.bn_out"),
        ));
        let output = format!("{prefix}.relu");
        self.nodes.push(relu_node(
            &format!("{prefix}.relu_node"),
            &format!("{prefix}.bn_out"),
            &output,
        ));
        Ok(output)
    }

    fn original_up(
        &mut self,
        prefix: &str,
        input: &str,
        skip: &str,
        inp: usize,
        oup: usize,
    ) -> Result<String, OnnxExportError> {
        let resized = self.resize(prefix, input);
        let concat = self.concat(&format!("{prefix}.concat"), [&resized, skip], 1);
        self.original_double(&format!("{prefix}.conv"), &concat, inp, oup, 1)
    }

    fn mobile_audio(&mut self, channels: &[usize; 5]) -> Result<String, OnnxExportError> {
        let mut current =
            self.mobile_sep("audio_model.conv1", "audio", 16, channels[1], 1, false)?;
        current = self.mobile_sep(
            "audio_model.conv2",
            &current,
            channels[1],
            channels[2],
            1,
            false,
        )?;
        current = self.mobile_block(
            "audio_model.conv3",
            &current,
            channels[2],
            channels[3],
            3,
            [2, 2],
            1,
            false,
            true,
        )?;
        current = self.mobile_sep(
            "audio_model.conv4",
            &current,
            channels[3],
            channels[3],
            1,
            true,
        )?;
        current = self.mobile_conv5(&current, channels[3], channels[4])?;
        current = self.mobile_sep(
            "audio_model.conv6",
            &current,
            channels[4],
            channels[4],
            1,
            true,
        )?;
        self.mobile_sep(
            "audio_model.conv7",
            &current,
            channels[4],
            channels[4],
            1,
            true,
        )
    }

    fn mobile_double(
        &mut self,
        prefix: &str,
        input: &str,
        inp: usize,
        oup: usize,
        stride: usize,
    ) -> Result<String, OnnxExportError> {
        let first = self.mobile_sep(&format!("{prefix}.first"), input, inp, oup, stride, false)?;
        self.mobile_sep(&format!("{prefix}.second"), &first, oup, oup, 1, true)
    }

    fn mobile_sep(
        &mut self,
        prefix: &str,
        input: &str,
        inp: usize,
        oup: usize,
        stride: usize,
        residual: bool,
    ) -> Result<String, OnnxExportError> {
        let depth = self.mobile_block(
            &format!("{prefix}.depthwise"),
            input,
            inp,
            inp,
            3,
            [stride, stride],
            inp,
            false,
            true,
        )?;
        self.mobile_block(
            &format!("{prefix}.pointwise"),
            &depth,
            inp,
            oup,
            1,
            [1, 1],
            1,
            residual,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn mobile_block(
        &mut self,
        prefix: &str,
        input: &str,
        _inp: usize,
        _out: usize,
        kernel: usize,
        stride: [usize; 2],
        groups: usize,
        residual: bool,
        activation: bool,
    ) -> Result<String, OnnxExportError> {
        let conv = format!("{prefix}.conv_out");
        let bias = format!("{prefix}.conv.bias");
        if !self.initializers.iter().any(|tensor| tensor.name == bias) {
            add_f32_initializer(
                &mut self.initializers,
                &bias,
                vec![0.0; _out],
                vec![_out as i64],
            );
        }
        self.nodes.push(conv_node_2d(
            &format!("{prefix}.conv"),
            input,
            &format!("{prefix}.conv.weight"),
            &bias,
            &conv,
            kernel,
            stride,
            kernel / 2,
            groups,
        ));
        let mut output = conv;
        if activation {
            let relu = format!("{prefix}.relu_out");
            self.nodes
                .push(relu_node(&format!("{prefix}.relu"), &output, &relu));
            output = relu;
        }
        if residual {
            let add = format!("{prefix}.residual_out");
            self.nodes.push(node(
                &format!("{prefix}.residual"),
                "Add",
                [input, &output],
                [&add],
                Vec::new(),
            ));
            output = add;
        }
        Ok(output)
    }

    fn mobile_conv5(
        &mut self,
        input: &str,
        _inp: usize,
        out: usize,
    ) -> Result<String, OnnxExportError> {
        let conv = "audio_model.conv5.conv.out";
        let zero_bias = "audio_model.conv5.conv.generated_bias";
        add_f32_initializer(
            &mut self.initializers,
            zero_bias,
            vec![0.0; out],
            vec![out as i64],
        );
        self.nodes.push(conv_node_2d(
            "audio_model.conv5.conv",
            input,
            "audio_model.conv5.conv.weight",
            zero_bias,
            conv,
            3,
            [2, 2],
            3,
            1,
        ));
        let shape = "audio_model.conv5.bn.shape";
        add_i64_initializer(&mut self.initializers, shape, &[1, out as i64, 1, 1]);
        let mean = "audio_model.conv5.batch_norm.running_mean";
        let var = "audio_model.conv5.batch_norm.running_var";
        let gamma = "audio_model.conv5.batch_norm.gamma";
        let beta = "audio_model.conv5.batch_norm.beta";
        let mean_r = "audio_model.conv5.bn.mean_r";
        let var_r = "audio_model.conv5.bn.var_r";
        let gamma_r = "audio_model.conv5.bn.gamma_r";
        let beta_r = "audio_model.conv5.bn.beta_r";
        self.nodes.push(node(
            "audio_model.conv5.bn.mean_reshape",
            "Reshape",
            [mean, shape],
            [mean_r],
            Vec::new(),
        ));
        self.nodes.push(node(
            "audio_model.conv5.bn.var_reshape",
            "Reshape",
            [var, shape],
            [var_r],
            Vec::new(),
        ));
        self.nodes.push(node(
            "audio_model.conv5.bn.gamma_reshape",
            "Reshape",
            [gamma, shape],
            [gamma_r],
            Vec::new(),
        ));
        self.nodes.push(node(
            "audio_model.conv5.bn.beta_reshape",
            "Reshape",
            [beta, shape],
            [beta_r],
            Vec::new(),
        ));
        add_f32_initializer(
            &mut self.initializers,
            "audio_model.conv5.bn.epsilon",
            vec![1e-5],
            Vec::new(),
        );
        self.nodes.push(node(
            "audio_model.conv5.bn.center",
            "Sub",
            [conv, mean_r],
            ["audio_model.conv5.bn.centered"],
            Vec::new(),
        ));
        self.nodes.push(node(
            "audio_model.conv5.bn.var_eps",
            "Add",
            [var_r, "audio_model.conv5.bn.epsilon"],
            ["audio_model.conv5.bn.var_eps_out"],
            Vec::new(),
        ));
        self.nodes.push(node(
            "audio_model.conv5.bn.sqrt",
            "Sqrt",
            ["audio_model.conv5.bn.var_eps_out"],
            ["audio_model.conv5.bn.std"],
            Vec::new(),
        ));
        self.nodes.push(node(
            "audio_model.conv5.bn.normalize",
            "Div",
            ["audio_model.conv5.bn.centered", "audio_model.conv5.bn.std"],
            ["audio_model.conv5.bn.normalized"],
            Vec::new(),
        ));
        self.nodes.push(node(
            "audio_model.conv5.bn.gamma_mul",
            "Mul",
            ["audio_model.conv5.bn.normalized", gamma_r],
            ["audio_model.conv5.bn.scaled"],
            Vec::new(),
        ));
        self.nodes.push(node(
            "audio_model.conv5.bn.shift",
            "Add",
            ["audio_model.conv5.bn.scaled", beta_r],
            ["audio_model.conv5.bn.out"],
            Vec::new(),
        ));
        let output = "audio_model.conv5.relu";
        self.nodes.push(relu_node(
            "audio_model.conv5.relu_node",
            "audio_model.conv5.bn.out",
            output,
        ));
        Ok(output.to_owned())
    }

    fn mobile_up(
        &mut self,
        prefix: &str,
        input: &str,
        skip: &str,
        inp: usize,
        oup: usize,
    ) -> Result<String, OnnxExportError> {
        let resized = self.resize(prefix, input);
        let concat = self.concat(&format!("{prefix}.concat"), [&resized, skip], 1);
        self.mobile_double(&format!("{prefix}.conv"), &concat, inp, oup, 1)
    }

    fn resize(&mut self, prefix: &str, input: &str) -> String {
        let scales = format!("{prefix}.resize.scales");
        add_f32_initializer(
            &mut self.initializers,
            &scales,
            vec![1.0, 1.0, 2.0, 2.0],
            vec![4],
        );
        let output = format!("{prefix}.resize.out");
        self.nodes.push(node(
            &format!("{prefix}.resize"),
            "Resize",
            [input, "", &scales],
            [&output],
            vec![
                attribute_string("mode", "linear"),
                attribute_string("coordinate_transformation_mode", "align_corners"),
            ],
        ));
        output
    }

    fn concat<'a, I>(&mut self, name: &str, inputs: I, axis: i64) -> String
    where
        I: IntoIterator<Item = &'a str>,
    {
        let output = format!("{name}.out");
        self.nodes.push(node(
            name,
            "Concat",
            inputs,
            [&output],
            vec![attribute_int("axis", axis)],
        ));
        output
    }

    fn final_output(
        &mut self,
        prefix: &str,
        input: &str,
        output: &str,
    ) -> Result<(), OnnxExportError> {
        self.nodes.push(conv_node_2d(
            prefix,
            input,
            &format!("{prefix}.weight"),
            &format!("{prefix}.bias"),
            "output.logits",
            1,
            [1, 1],
            0,
            1,
        ));
        self.nodes.push(node(
            "output.sigmoid",
            "Sigmoid",
            ["output.logits"],
            [output],
            Vec::new(),
        ));
        Ok(())
    }
}

fn finish(kind: OnnxModelKind, builder: Builder) -> Result<Vec<u8>, OnnxExportError> {
    let (inputs, outputs, name) = match kind {
        OnnxModelKind::OriginalUnet => (
            vec![
                OnnxValue {
                    name: "input".to_owned(),
                    shape: vec![1, 6, 160, 160],
                    dtype: ONNX_FLOAT_DATA_TYPE,
                },
                OnnxValue {
                    name: "audio".to_owned(),
                    shape: vec![1, 16, 32, 32],
                    dtype: ONNX_FLOAT_DATA_TYPE,
                },
            ],
            vec![OnnxValue {
                name: "output".to_owned(),
                shape: vec![1, 3, 160, 160],
                dtype: ONNX_FLOAT_DATA_TYPE,
            }],
            "feathertalk.original_unet",
        ),
        OnnxModelKind::MobileOneUnet => (
            vec![
                OnnxValue {
                    name: "input".to_owned(),
                    shape: vec![1, 6, 160, 160],
                    dtype: ONNX_FLOAT_DATA_TYPE,
                },
                OnnxValue {
                    name: "audio".to_owned(),
                    shape: vec![1, 16, 32, 32],
                    dtype: ONNX_FLOAT_DATA_TYPE,
                },
            ],
            vec![OnnxValue {
                name: "output".to_owned(),
                shape: vec![1, 3, 160, 160],
                dtype: ONNX_FLOAT_DATA_TYPE,
            }],
            "feathertalk.mobileone_unet.reparameterized",
        ),
        OnnxModelKind::FeatherHubert => unreachable!(),
    };
    let model = OnnxModel {
        ir_version: crate::onnx::ONNX_IR_VERSION,
        opset_version: crate::onnx::ONNX_OPSET_VERSION,
        graph_present: true,
        graph: OnnxGraph {
            name: name.to_owned(),
            inputs,
            outputs,
            nodes: builder.nodes,
            initializers: builder.initializers,
        },
    };
    validate_graph_integrity(&model)
        .map_err(|error| OnnxExportError::InvalidGraph(error.to_string()))?;
    let bytes = serialize_model(&model)?;
    let contract = OnnxModelContract::new(
        kind,
        vec![
            OnnxTensorContract::new("input", vec![1, 6, 160, 160]),
            OnnxTensorContract::new("audio", vec![1, 16, 32, 32]),
        ],
        vec![OnnxTensorContract::new("output", vec![1, 3, 160, 160])],
    );
    validate_model_contract(&bytes, &contract)
        .map_err(|error| OnnxExportError::InvalidGraph(error.to_string()))?;
    Ok(bytes)
}

fn validate_channels(channels: &[usize; 5]) -> Result<(), OnnxExportError> {
    if channels.contains(&0)
        || !channels[1].is_multiple_of(2)
        || !channels[2].is_multiple_of(2)
        || !channels[3].is_multiple_of(2)
    {
        return Err(OnnxExportError::UnsupportedConfiguration(
            "UNet channels must be positive and decoder channels must be even".to_owned(),
        ));
    }
    Ok(())
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
    padding: usize,
    _dilation: usize,
) -> OnnxNodeProto {
    conv_node_2d(
        name,
        input,
        weight,
        bias.unwrap_or(""),
        output,
        kernel,
        [stride, stride],
        padding,
        1,
    )
}

#[allow(clippy::too_many_arguments)]
fn conv_node_2d(
    name: &str,
    input: &str,
    weight: &str,
    bias: &str,
    output: &str,
    kernel: usize,
    stride: [usize; 2],
    padding: usize,
    groups: usize,
) -> OnnxNodeProto {
    let mut inputs = vec![input.to_owned(), weight.to_owned()];
    if !bias.is_empty() {
        inputs.push(bias.to_owned());
    }
    node(
        name,
        "Conv",
        inputs,
        [output],
        vec![
            attribute_ints("kernel_shape", &[kernel as i64, kernel as i64]),
            attribute_ints("strides", &[stride[0] as i64, stride[1] as i64]),
            attribute_ints("pads", &[padding as i64; 4]),
            attribute_ints("dilations", &[1, 1]),
            attribute_int("group", groups as i64),
        ],
    )
}

fn batch_norm_node(name: &str, input: &str, prefix: &str, output: &str) -> OnnxNodeProto {
    node(
        name,
        "BatchNormalization",
        [
            input,
            &format!("{prefix}.gamma"),
            &format!("{prefix}.beta"),
            &format!("{prefix}.running_mean"),
            &format!("{prefix}.running_var"),
        ],
        [output],
        vec![
            attribute_float("epsilon", 1e-5),
            attribute_float("momentum", 0.1),
        ],
    )
}

fn relu_node(name: &str, input: &str, output: &str) -> OnnxNodeProto {
    node(name, "Relu", [input], [output], Vec::new())
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
fn attribute_string(name: &str, value: &str) -> OnnxAttributeProto {
    OnnxAttributeProto {
        name: name.to_owned(),
        s: value.as_bytes().to_vec(),
        r#type: 3,
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
    if !initializers.iter().any(|tensor| tensor.name == name) {
        initializers
            .insert_generated(OnnxTensorProto {
                dims,
                data_type: ONNX_FLOAT_DATA_TYPE,
                name: name.to_owned(),
                raw_data,
                doc_string: String::new(),
            })
            .expect("generated initializer names are unique");
    }
}

fn add_i64_initializer(initializers: &mut InitializerSet, name: &str, values: &[i64]) {
    let mut raw_data = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        raw_data.extend_from_slice(&value.to_le_bytes());
    }
    if !initializers.iter().any(|tensor| tensor.name == name) {
        initializers
            .insert_generated(OnnxTensorProto {
                dims: vec![values.len() as i64],
                data_type: 7,
                name: name.to_owned(),
                raw_data,
                doc_string: String::new(),
            })
            .expect("generated initializer names are unique");
    }
}
