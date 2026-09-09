use std::path::Path;

use feathertalk_media::{AudioMetadata, MediaArtifact, NormalizedMedia, VideoMetadata};
use serde_json::{Value, json};

use crate::probe_to_json;

/// Shapes a normalization as the JSON object a `completed` event carries.
///
/// Unlike a probe, the payload names the files: the caller asked for a
/// directory and the worker chose the file names, so a later task would
/// otherwise have to guess them. The paths are the canonical ones the media
/// crate committed to, reported as produced rather than prettified.
pub fn normalize_to_json(media: &NormalizedMedia) -> Value {
    json!({
        "output_dir": path_text(media.layout().output_dir()),
        "video": media
            .video()
            .map(|video| video_json(media.layout().video_path(), video, media.video_artifact())),
        "audio": media
            .audio()
            .map(|audio| audio_json(media.layout().audio_path(), audio, media.audio_artifact())),
        "source": probe_to_json(media.source()),
    })
}

fn video_json(path: &Path, video: &VideoMetadata, artifact: &MediaArtifact) -> Value {
    json!({
        "path": path_text(path),
        "bytes": artifact.bytes(),
        "sha256": artifact.sha256(),
        "codec_name": video.codec_name(),
        "pixel_format": video.pixel_format(),
        "width": video.width(),
        "height": video.height(),
        "frame_rate": {
            "numerator": video.frame_rate().numerator(),
            "denominator": video.frame_rate().denominator(),
        },
        "frame_count": video.frame_count(),
        "duration_seconds": video.duration_seconds(),
    })
}

fn audio_json(path: &Path, audio: &AudioMetadata, artifact: &MediaArtifact) -> Value {
    json!({
        "path": path_text(path),
        "bytes": artifact.bytes(),
        "sha256": artifact.sha256(),
        "codec_name": audio.codec_name(),
        "sample_format": audio.sample_format(),
        "sample_rate": audio.sample_rate(),
        "channels": audio.channels(),
        "sample_count": audio.sample_count(),
        "duration_seconds": audio.duration_seconds(),
    })
}

fn path_text(path: &Path) -> String {
    path.display().to_string()
}
