use feathertalk_domain::{DomainError, PROTOCOL_VERSION, TaskId, TaskKind, TaskStatus};

#[test]
fn protocol_version_is_two() {
    assert_eq!(PROTOCOL_VERSION, 2);
}

#[test]
fn task_id_accepts_the_canonical_shape_and_orders_by_time() {
    let older = TaskId::parse("1787900000000-0000000a").unwrap();
    let newer = TaskId::parse("1787900000001-0000000a").unwrap();
    assert_eq!(older.as_str(), "1787900000000-0000000a");
    assert!(older < newer);
}

#[test]
fn task_id_rejects_every_off_contract_shape() {
    for bad in [
        "",
        "1787900000000",
        "178790000000-0000000a",
        "17879000000000-0000000a",
        "1787900000000-0000000A",
        "1787900000000-0000000",
        "1787900000000_0000000a",
        "abcdefghijklm-0000000a",
    ] {
        assert!(
            matches!(TaskId::parse(bad), Err(DomainError::InvalidTaskId { .. })),
            "expected rejection for {bad:?}"
        );
    }
}

#[test]
fn task_id_rejects_non_ascii_without_panicking() {
    let result = std::panic::catch_unwind(|| TaskId::parse("123456789012é12345678"));
    assert!(result.is_ok(), "task id validation must not panic");
    assert!(matches!(
        result.unwrap(),
        Err(DomainError::InvalidTaskId { .. })
    ));
}

#[test]
fn task_id_deserialization_rejects_malformed_wire_values() {
    let result = serde_json::from_str::<TaskId>(r#""not-a-task-id""#);
    assert!(
        result.is_err(),
        "wire TaskId values must go through validation"
    );
}

#[test]
fn task_kind_slugs_match_their_serde_form_and_are_all_distinct() {
    let mut seen = std::collections::BTreeSet::new();
    for kind in TaskKind::ALL {
        let slug = kind.as_slug();
        assert_eq!(serde_json::to_string(&kind).unwrap(), format!("\"{slug}\""));
        assert_eq!(TaskKind::from_slug(slug), Some(kind));
        assert!(seen.insert(slug), "duplicate slug {slug}");
    }
    assert_eq!(seen.len(), 13);
    assert_eq!(TaskKind::from_slug("no_such_command"), None);
}

#[test]
fn only_queued_and_running_are_incomplete() {
    assert_eq!(TaskStatus::ALL.len(), 5);
    for status in TaskStatus::ALL {
        let expected = matches!(status, TaskStatus::Queued | TaskStatus::Running);
        assert_eq!(status.is_incomplete(), expected, "{status:?}");
    }
}
