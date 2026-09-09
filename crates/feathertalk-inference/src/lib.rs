mod burn;
mod command;
mod error;
mod executor;
mod frame;
mod frame_reader;
mod plan;
mod publish;
mod raw_sink;
mod render;
mod sequence;

pub use burn::{
    UnetAudioInput, build_unet_audio_input, build_unet_audio_window, render_planned_frame,
    run_unet_prediction,
};
pub use command::{CommandSpec, raw_video_command};
pub use error::InferenceError;
pub use executor::{OfflineRenderRequest, OfflineRenderResult, execute_offline_render};
pub use frame::{
    BgrFrame, InnerImagePlanes, MouthMasking, UnetImageInput, apply_unet_prediction,
    build_face_crop, build_inner_image_planes, build_unet_image_input, crop_bgr, paste_bgr,
    render_frame, resize_bilinear,
};
pub use frame_reader::{DEFAULT_MAX_FRAME_PIXELS, FrameReader, JpegFrameReader};
pub use plan::{InferenceFramePlan, RenderPlan};
pub use raw_sink::{RawVideoSink, RawVideoSinkFactory, SystemRawVideoSinkFactory};
pub use render::{
    RawFrameRenderSpec, RenderGeometry, staging_output_path, validate_output_destination,
};
pub use sequence::PingPongFrames;
