use feathertalk_media::MediaProbe;
use serde_json::{Value, json};

/// Shapes a probe as the JSON object a `completed` event carries.
///
/// The input path is deliberately absent: the desktop already knows which file
/// it asked about, and the event stream is written to logs.
pub fn probe_to_json(probe: &MediaProbe) -> Value {
    json!({
        "format": {
            "format_name": probe.format().format_name(),
            "duration_seconds": probe.format().duration_seconds(),
        },
        "video": probe.video().map(|video| {
            json!({
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
        }),
        "audio": probe.audio().map(|audio| {
            json!({
                "codec_name": audio.codec_name(),
                "sample_format": audio.sample_format(),
                "sample_rate": audio.sample_rate(),
                "channels": audio.channels(),
                "sample_count": audio.sample_count(),
                "duration_seconds": audio.duration_seconds(),
            })
        }),
    })
}
