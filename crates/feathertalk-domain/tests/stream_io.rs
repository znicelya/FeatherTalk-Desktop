use std::io::Cursor;

use feathertalk_domain::{
    DomainError, FrameReader, FrameWriter, MAX_FRAME_BYTES, PROTOCOL_VERSION, RejectedFrame,
    ServerFrame,
};

fn rejected(reason: &str) -> ServerFrame {
    ServerFrame::Rejected(RejectedFrame {
        protocol_version: PROTOCOL_VERSION,
        reason: reason.to_owned(),
    })
}

#[test]
fn frames_written_then_read_back_survive_the_trip() {
    let mut writer = FrameWriter::new(Vec::new());
    writer.write_frame(&rejected("first")).unwrap();
    writer.write_frame(&rejected("second")).unwrap();
    let bytes = writer.into_inner();
    assert_eq!(bytes.iter().filter(|byte| **byte == b'\n').count(), 2);

    let mut reader = FrameReader::new(Cursor::new(bytes));
    assert_eq!(
        reader.read_frame::<ServerFrame>().unwrap().unwrap(),
        rejected("first")
    );
    assert_eq!(
        reader.read_frame::<ServerFrame>().unwrap().unwrap(),
        rejected("second")
    );
    assert!(reader.read_frame::<ServerFrame>().is_none());
}

#[test]
fn blank_lines_between_frames_are_skipped() {
    let line = feathertalk_domain::encode_line(&rejected("only")).unwrap();
    let input = format!("\n\n{line}\n   \n");
    let mut reader = FrameReader::new(Cursor::new(input));
    assert_eq!(
        reader.read_frame::<ServerFrame>().unwrap().unwrap(),
        rejected("only")
    );
    assert!(reader.read_frame::<ServerFrame>().is_none());
}

#[test]
fn a_final_line_without_a_newline_is_still_delivered() {
    let line = feathertalk_domain::encode_line(&rejected("unterminated")).unwrap();
    let mut reader = FrameReader::new(Cursor::new(line));
    assert_eq!(
        reader.read_frame::<ServerFrame>().unwrap().unwrap(),
        rejected("unterminated")
    );
    assert!(reader.read_frame::<ServerFrame>().is_none());
}

#[test]
fn a_bad_line_is_reported_and_the_reader_keeps_going() {
    let good = feathertalk_domain::encode_line(&rejected("after")).unwrap();
    let input = format!("not json\n{good}\n");
    let mut reader = FrameReader::new(Cursor::new(input));
    assert!(matches!(
        reader.read_frame::<ServerFrame>().unwrap(),
        Err(DomainError::MalformedFrame { .. })
    ));
    assert_eq!(
        reader.read_frame::<ServerFrame>().unwrap().unwrap(),
        rejected("after")
    );
}

#[test]
fn an_oversized_line_is_refused() {
    let input = format!("{}\n", "x".repeat(MAX_FRAME_BYTES + 1_024));
    let mut reader = FrameReader::new(Cursor::new(input));
    assert!(matches!(
        reader.read_frame::<ServerFrame>().unwrap(),
        Err(DomainError::FrameTooLong { .. })
    ));
}

#[test]
fn an_oversized_line_is_discarded_before_the_next_frame() {
    let oversized = format!("{}\n", "x".repeat(MAX_FRAME_BYTES + 1_024));
    let good = feathertalk_domain::encode_line(&rejected("after oversized")).unwrap();
    let input = format!("{oversized}{good}\n");
    let mut reader = FrameReader::new(Cursor::new(input));
    assert!(matches!(
        reader.read_frame::<ServerFrame>().unwrap(),
        Err(DomainError::FrameTooLong { .. })
    ));
    assert_eq!(
        reader.read_frame::<ServerFrame>().unwrap().unwrap(),
        rejected("after oversized")
    );
}

#[test]
fn an_oversized_unterminated_line_is_reported_once_at_eof() {
    let input = "x".repeat(MAX_FRAME_BYTES + 1_024);
    let mut reader = FrameReader::new(Cursor::new(input));
    assert!(matches!(
        reader.read_frame::<ServerFrame>().unwrap(),
        Err(DomainError::FrameTooLong { .. })
    ));
    assert!(reader.read_frame::<ServerFrame>().is_none());
}

fn rejected_with_exact_json_size() -> ServerFrame {
    let empty = rejected("");
    let overhead = feathertalk_domain::encode_line(&empty).unwrap().len();
    rejected(&"x".repeat(MAX_FRAME_BYTES - overhead))
}

#[test]
fn a_max_sized_json_frame_round_trips_with_its_newline_delimiter() {
    let frame = rejected_with_exact_json_size();
    let line = feathertalk_domain::encode_line(&frame).unwrap();
    assert_eq!(line.len(), MAX_FRAME_BYTES);

    let mut writer = FrameWriter::new(Vec::new());
    writer.write_frame(&frame).unwrap();
    let bytes = writer.into_inner();
    assert_eq!(bytes.len(), MAX_FRAME_BYTES + 1);
    assert_eq!(bytes.last(), Some(&b'\n'));

    let mut reader = FrameReader::new(Cursor::new(bytes));
    assert_eq!(reader.read_frame::<ServerFrame>().unwrap().unwrap(), frame);
}

#[test]
fn decode_line_accepts_a_max_sized_json_line_with_a_trailing_newline() {
    let frame = rejected_with_exact_json_size();
    let line = feathertalk_domain::encode_line(&frame).unwrap();
    assert_eq!(line.len(), MAX_FRAME_BYTES);
    let with_newline = format!("{line}\n");
    assert_eq!(
        feathertalk_domain::decode_line::<ServerFrame>(&with_newline).unwrap(),
        frame
    );
}

#[test]
fn invalid_utf8_is_reported_as_a_malformed_frame() {
    let input: Vec<u8> = vec![0xff, 0xfe, b'\n'];
    let mut reader = FrameReader::new(Cursor::new(input));
    assert!(matches!(
        reader.read_frame::<ServerFrame>().unwrap(),
        Err(DomainError::MalformedFrame { .. })
    ));
}

#[test]
fn writing_an_oversized_frame_leaves_the_stream_untouched() {
    let mut writer = FrameWriter::new(Vec::new());
    let huge = ServerFrame::Rejected(RejectedFrame {
        protocol_version: PROTOCOL_VERSION,
        reason: "x".repeat(MAX_FRAME_BYTES),
    });
    assert!(matches!(
        writer.write_frame(&huge),
        Err(DomainError::FrameTooLong { .. })
    ));
    writer
        .write_frame(&ServerFrame::Rejected(RejectedFrame {
            protocol_version: PROTOCOL_VERSION,
            reason: "small".to_owned(),
        }))
        .unwrap();
    let bytes = writer.into_inner();
    assert_eq!(bytes.iter().filter(|byte| **byte == b'\n').count(), 1);
}
