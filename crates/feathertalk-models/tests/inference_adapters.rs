use burn::tensor::Tensor;
use feathertalk_models::{
    backend::CpuBackend,
    unet::{
        MobileOneUnetConfig, MobileOneUnetInference, OriginalUnet, OriginalUnetConfig,
        TalkingHeadModel,
    },
};

fn assert_talking_head_model<M: TalkingHeadModel<CpuBackend>>() {}

#[test]
fn original_and_reparameterized_mobileone_implement_the_public_inference_trait() {
    assert_talking_head_model::<OriginalUnet<CpuBackend>>();
    assert_talking_head_model::<MobileOneUnetInference<CpuBackend>>();
}

#[test]
fn trait_forward_preserves_the_fixed_unet_contract() {
    let device = Default::default();
    let image = Tensor::<CpuBackend, 4>::zeros([1, 6, 160, 160], &device);
    let audio = Tensor::<CpuBackend, 4>::zeros([1, 16, 32, 32], &device);

    let original = OriginalUnetConfig::parity_micro().init::<CpuBackend>(&device);
    assert_eq!(
        original
            .forward_talking_head(image.clone(), audio.clone())
            .dims(),
        [1, 3, 160, 160]
    );

    let mobile = MobileOneUnetConfig::parity_micro()
        .init::<CpuBackend>(&device)
        .reparameterize();
    assert_eq!(
        mobile.forward_talking_head(image, audio).dims(),
        [1, 3, 160, 160]
    );
}
