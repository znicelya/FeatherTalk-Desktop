use feathertalk_media::{MediaError, parse_probe_json};

fn valid_json() -> String {
    r#"
    {
      "format": {"format_name":"mov,mp4","duration":"2.000"},
      "streams": [
        {
          "codec_type":"video", "codec_name":"h264", "pix_fmt":"yuv420p",
          "width":640, "height":480, "avg_frame_rate":"25/1",
          "nb_read_frames":"50", "duration":"2.000"
        },
        {
          "codec_type":"audio", "codec_name":"aac", "sample_fmt":"fltp",
          "sample_rate":"48000", "channels":2, "duration":"2.000"
        }
      ]
    }
    "#
    .to_owned()
}

#[test]
fn parses_exactly_one_video_and_audio_stream() {
    let probe = parse_probe_json(valid_json().as_bytes()).unwrap();
    assert_eq!(probe.format().format_name(), "mov,mp4");
    assert_eq!(probe.format().duration_seconds(), 2.0);
    let video = probe.video().unwrap();
    assert_eq!(video.codec_name(), "h264");
    assert_eq!(video.pixel_format(), "yuv420p");
    assert_eq!((video.width(), video.height()), (640, 480));
    assert_eq!(video.frame_rate().numerator(), 25);
    assert_eq!(video.frame_rate().denominator(), 1);
    assert_eq!(video.frame_count(), 50);
    let audio = probe.audio().unwrap();
    assert_eq!(audio.codec_name(), "aac");
    assert_eq!(audio.sample_format(), "fltp");
    assert_eq!(audio.sample_rate(), 48_000);
    assert_eq!(audio.channels(), 2);
    assert_eq!(audio.sample_count(), 96_000);
}

#[test]
fn accepts_unknown_metadata_fields() {
    let mut value: serde_json::Value = serde_json::from_str(&valid_json()).unwrap();
    value["future_field"] = serde_json::json!({"nested": true});
    assert!(parse_probe_json(&serde_json::to_vec(&value).unwrap()).is_ok());
}

#[test]
fn rejects_missing_and_duplicate_required_streams() {
    let no_audio = valid_json().replace(
        r#",
        {
          "codec_type":"audio", "codec_name":"aac", "sample_fmt":"fltp",
          "sample_rate":"48000", "channels":2, "duration":"2.000"
        }"#,
        "",
    );
    assert!(matches!(
        parse_probe_json(no_audio.as_bytes()),
        Err(MediaError::MissingStream { stream: "audio" })
    ));

    let duplicate = valid_json().replace(
        r#"        {
          "codec_type":"audio", "codec_name":"aac", "sample_fmt":"fltp",
          "sample_rate":"48000", "channels":2, "duration":"2.000"
        }"#,
        r#"        {
          "codec_type":"audio", "codec_name":"aac", "sample_fmt":"fltp",
          "sample_rate":"48000", "channels":2, "duration":"2.000"
        },
        {
          "codec_type":"audio", "codec_name":"aac", "sample_fmt":"fltp",
          "sample_rate":"48000", "channels":2, "duration":"2.000"
        }"#,
    );
    assert!(matches!(
        parse_probe_json(duplicate.as_bytes()),
        Err(MediaError::DuplicateStream { stream: "audio" })
    ));
}

#[test]
fn rejects_numeric_contract_boundaries() {
    for (needle, replacement) in [
        ("\"width\":640", "\"width\":0"),
        ("\"width\":640", "\"width\":16385"),
        ("\"avg_frame_rate\":\"25/1\"", "\"avg_frame_rate\":\"0/1\""),
        ("\"duration\":\"2.000\"", "\"duration\":\"NaN\""),
        (
            "\"nb_read_frames\":\"50\"",
            "\"nb_read_frames\":\"100000000001\"",
        ),
        ("\"channels\":2", "\"channels\":0"),
    ] {
        let json = valid_json().replacen(needle, replacement, 1);
        assert!(
            matches!(
                parse_probe_json(json.as_bytes()),
                Err(MediaError::ProbeContract { .. })
            ),
            "replacement {replacement} unexpectedly accepted"
        );
    }
}

#[test]
fn rejects_invalid_json_and_oversized_probe() {
    assert!(matches!(
        parse_probe_json(b"not json"),
        Err(MediaError::ProbeJson { .. })
    ));
    let oversized = vec![b' '; 1_048_577];
    assert!(matches!(
        parse_probe_json(&oversized),
        Err(MediaError::ProbeTooLarge {
            limit: 1_048_576,
            ..
        })
    ));
}
