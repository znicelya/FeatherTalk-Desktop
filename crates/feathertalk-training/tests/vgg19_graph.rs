use burn::{
    module::{Module, Param},
    nn::conv::Conv2d,
    tensor::{Tensor}};
use feathertalk_training::Vgg19Conv3_3;

type CpuBackend = burn::backend::Flex;

fn assert_module<M: Module>() {}

#[test]
fn vgg19_conv3_3_is_a_burn_module() {
    assert_module::<Vgg19Conv3_3>();
}

#[test]
fn vgg19_conv3_3_maps_sixteen_to_four() {
    let device = Default::default();
    let model = Vgg19Conv3_3::new_for_import(&device);

    let output = model.forward(Tensor::zeros([1, 3, 16, 16], &device));

    assert_eq!(output.dims(), [1, 256, 4, 4]);
}

#[test]
fn no_grad_clears_require_grad_on_every_conv_weight() {
    let device = Default::default();
    let model = Vgg19Conv3_3::new_for_import(&device).no_grad();
    for conv in [
        &model.conv1_1,
        &model.conv1_2,
        &model.conv2_1,
        &model.conv2_2,
        &model.conv3_1,
        &model.conv3_2,
        &model.conv3_3,
    ] {
        assert!(!conv.weight.val().is_require_grad());
    }
}

#[test]
fn conv3_3_output_is_not_post_relu() {
    let device = Default::default();
    let mut model = Vgg19Conv3_3::new_for_import(&device);

    zero_conv(&mut model.conv1_1, &device);
    zero_conv(&mut model.conv1_2, &device);
    zero_conv(&mut model.conv2_1, &device);
    zero_conv(&mut model.conv2_2, &device);
    zero_conv(&mut model.conv3_1, &device);
    zero_conv(&mut model.conv3_2, &device);
    zero_conv(&mut model.conv3_3, &device);

    let bias_dims = model.conv3_3.bias.as_ref().unwrap().val().dims();
    model.conv3_3.bias = Some(Param::from_tensor(Tensor::full(bias_dims, -1.0, &device)));

    let output = model
        .forward(Tensor::zeros([1, 3, 4, 4], &device))
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();

    assert_eq!(output, vec![-1.0; 256]);
}

#[test]
#[should_panic(expected = "VGG19 input batch must be non-zero")]
fn vgg19_rejects_empty_batches_at_the_graph_boundary() {
    let device = Default::default();
    let model = Vgg19Conv3_3::new_for_import(&device);

    let _ = model.forward(Tensor::zeros([0, 3, 4, 4], &device));
}

#[test]
#[should_panic(expected = "VGG19 input must have exactly 3 channels")]
fn vgg19_rejects_non_bgr_channel_counts_at_the_graph_boundary() {
    let device = Default::default();
    let model = Vgg19Conv3_3::new_for_import(&device);

    let _ = model.forward(Tensor::zeros([1, 1, 4, 4], &device));
}

#[test]
#[should_panic(expected = "VGG19 input spatial dimensions must both be at least 4")]
fn vgg19_rejects_spatial_dimensions_smaller_than_two_pooling_stages() {
    let device = Default::default();
    let model = Vgg19Conv3_3::new_for_import(&device);

    let _ = model.forward(Tensor::zeros([1, 3, 3, 4], &device));
}

fn zero_conv(conv: &mut Conv2d, device: &burn::tensor::Device) {
    let weight_dims = conv.weight.val().dims();
    conv.weight = Param::from_tensor(Tensor::zeros(weight_dims, device));

    let bias_dims = conv.bias.as_ref().unwrap().val().dims();
    conv.bias = Some(Param::from_tensor(Tensor::zeros(bias_dims, device)));
}
