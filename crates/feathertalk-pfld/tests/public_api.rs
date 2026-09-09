use feathertalk_pfld::{
    CropGeometry, LandmarkPoint, MEAN_FACE, MeanFace, PFLD_LANDMARK_COUNT, PFLD_OUTPUT_VALUE_COUNT,
    PFLDLandmarks, PfldError, decode_landmarks, decode_landmarks_with_default_mean_face,
    decode_landmarks_with_mean_face, read_mean_face,
};

#[test]
fn crate_root_exposes_pfld_contract() {
    let _: usize = PFLD_OUTPUT_VALUE_COUNT;
    let _: usize = PFLD_LANDMARK_COUNT;
    let _: LandmarkPoint = LandmarkPoint { x: 0, y: 0 };
    let _: &MeanFace = &MEAN_FACE;
    let _: fn(&std::path::Path) -> Result<MeanFace, PfldError> = read_mean_face;
    let output = decode_landmarks(
        &[0.0; PFLD_OUTPUT_VALUE_COUNT],
        &[0.0; PFLD_OUTPUT_VALUE_COUNT],
        CropGeometry {
            width: 1,
            height: 1,
            offset_x: 0,
            offset_y: 0,
        },
    )
    .unwrap();
    let _: &PFLDLandmarks = &output;
    let _: Result<PFLDLandmarks, PfldError> = Ok(output);
    let _ = decode_landmarks_with_mean_face;
    let _ = decode_landmarks_with_default_mean_face;
}
