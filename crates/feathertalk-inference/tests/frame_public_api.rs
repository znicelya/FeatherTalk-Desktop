use feathertalk_inference::{
    BgrFrame, InferenceError, RenderGeometry, UnetImageInput, apply_unet_prediction,
    build_unet_image_input, crop_bgr, paste_bgr, render_frame, resize_bilinear,
};
use feathertalk_preprocess::FaceBoundingBox;

#[test]
fn frame_kernel_is_available_from_crate_root() {
    let mut destination = BgrFrame::new(2, 2, vec![0; 12]).unwrap();
    let source = BgrFrame::new(1, 1, vec![1, 2, 3]).unwrap();
    let bbox = FaceBoundingBox {
        xmin: 0,
        ymin: 0,
        xmax: 1,
        ymax: 1,
    };
    let _cropped: BgrFrame = crop_bgr(&destination, &bbox).unwrap();
    let _resized: BgrFrame = resize_bilinear(&source, 2, 2).unwrap();
    paste_bgr(&mut destination, &source, 1, 1).unwrap();

    let crop = BgrFrame::new(168, 168, vec![0; 168 * 168 * 3]).unwrap();
    let geometry = RenderGeometry::standard();
    let input: UnetImageInput = build_unet_image_input(&crop, &geometry).unwrap();
    assert_eq!(input.shape(), [1, 6, 160, 160]);
    let mut prediction = vec![0.0; 3 * 160 * 160];
    prediction[0] = 1.0;
    let mut processed = crop.clone();
    apply_unet_prediction(&mut processed, &prediction, &geometry).unwrap();
    let _rendered = render_frame(
        &BgrFrame::new(1, 1, vec![0; 3]).unwrap(),
        &FaceBoundingBox {
            xmin: 0,
            ymin: 0,
            xmax: 1,
            ymax: 1,
        },
        &vec![0.0; 3 * 160 * 160],
        &geometry,
    )
    .unwrap();
    let _ = InferenceError::ArithmeticOverflow;
}
