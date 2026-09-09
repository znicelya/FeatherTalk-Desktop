use std::path::{Path, PathBuf};

use burn::tensor::{Tensor, TensorData};
use feathertalk_inference::{
    BgrFrame, CommandSpec, FrameReader, InferenceError, InferenceFramePlan, JpegFrameReader,
    OfflineRenderRequest, OfflineRenderResult, PingPongFrames, RawFrameRenderSpec, RawVideoSink,
    RawVideoSinkFactory, RenderGeometry, RenderPlan, SystemRawVideoSinkFactory,
    execute_offline_render, raw_video_command, staging_output_path, validate_output_destination,
};

#[test]
fn crate_root_exposes_read_only_inference_contract() {
    let mut picker = PingPongFrames::new(2).unwrap();
    let _: usize = picker.next();
    let plan = RenderPlan::new(2, 4, Some(2)).unwrap();
    let frame: InferenceFramePlan = plan.frame(0).unwrap();
    assert_eq!(frame.reference_frame_index, frame.source_frame_index);

    let geometry = RenderGeometry::standard();
    assert_eq!(geometry.replacement_offset(), (4, 4));

    let audio = Path::new("audio.wav");
    let output = Path::new("output.mp4");
    let spec = RawFrameRenderSpec::new(640, 480, audio, output).unwrap();
    let _: &Path = spec.audio_path();
    let _: &Path = spec.output_path();
    assert_eq!(spec.fps(), 25);

    let command: CommandSpec = raw_video_command(Path::new("C:/ffmpeg.exe"), &spec).unwrap();
    assert!(
        command
            .arguments()
            .windows(2)
            .any(|pair| { pair[0] == "-framerate" && pair[1] == "25" })
    );
    assert!(command.arguments().iter().any(|arg| arg == "-shortest"));

    let _ = PathBuf::from(spec.output_path());
    let _ = validate_output_destination;
    let _ = staging_output_path;
    let _ = InferenceError::EmptyFeatures;

    let _reader = JpegFrameReader::default();
    let _factory = SystemRawVideoSinkFactory::new();
    type RequestConstructor = fn(
        PathBuf,
        PathBuf,
        PathBuf,
        PathBuf,
        PathBuf,
        PathBuf,
        String,
        usize,
        Option<usize>,
    ) -> Result<OfflineRenderRequest, InferenceError>;
    let _request_new: RequestConstructor = OfflineRenderRequest::new;
    let _execute = execute_offline_render::<
        feathertalk_models::backend::CpuBackend,
        DummyModel,
        JpegFrameReader,
        SystemRawVideoSinkFactory,
    >;
    let _result_accessors: fn(&OfflineRenderResult) -> (&Path, usize, u32, u32) = |result| {
        (
            result.output_path(),
            result.frame_count(),
            result.width(),
            result.height(),
        )
    };
    fn assert_traits<R: FrameReader, S: RawVideoSink, F: RawVideoSinkFactory>() {}
    let _ = assert_traits::<JpegFrameReader, DummySink, SystemRawVideoSinkFactory>;
    let _frame = BgrFrame::new(1, 1, vec![0, 0, 0]).unwrap();
}

struct DummySink;

struct DummyModel;

impl feathertalk_models::unet::TalkingHeadModel<feathertalk_models::backend::CpuBackend>
    for DummyModel
{
    fn forward_talking_head(
        &self,
        image: Tensor<feathertalk_models::backend::CpuBackend, 4>,
        _audio: Tensor<feathertalk_models::backend::CpuBackend, 4>,
    ) -> Tensor<feathertalk_models::backend::CpuBackend, 4> {
        let device = image.device();
        Tensor::from_data(
            TensorData::new(vec![0.0; 3 * 160 * 160], [1, 3, 160, 160]),
            &device,
        )
    }
}

impl RawVideoSink for DummySink {
    fn write_frame(&mut self, _frame: &BgrFrame) -> Result<(), InferenceError> {
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), InferenceError> {
        Ok(())
    }
}
