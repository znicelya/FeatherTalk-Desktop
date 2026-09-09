use feathertalk_face::{
    Detection, DetectionConfig, FaceCropGeometry, ImageSize, RectI, ResizeTransform,
    compute_face_crop_geometry, decode_level, generate_anchor_centers, non_max_suppression,
    resize_with_padding,
};

#[test]
fn crate_root_exposes_schema_one_postprocess_api() {
    let _: fn(ImageSize, [f32; 4]) -> Result<FaceCropGeometry, _> = compute_face_crop_geometry;
    let _: RectI = RectI {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
    };
    let transform: ResizeTransform = resize_with_padding(ImageSize {
        width: 640,
        height: 640,
    })
    .unwrap();
    let anchors = generate_anchor_centers(
        ImageSize {
            width: 640,
            height: 640,
        },
        8,
        2,
    )
    .unwrap();
    let detections: Vec<Detection> = decode_level(
        0,
        8,
        &anchors[..1],
        &[0.9],
        &[[1.0, 1.0, 1.0, 1.0]],
        &[[0.0; 10]],
        &transform,
    )
    .unwrap();
    let _: Vec<usize> = non_max_suppression(&detections, &DetectionConfig::default()).unwrap();
}
