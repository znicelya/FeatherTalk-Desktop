use feathertalk_client::generate_task_id;
use feathertalk_domain::TaskId;

#[test]
fn a_generated_task_id_parses_as_a_domain_task_id() {
    let generated = generate_task_id().unwrap();
    let reparsed = TaskId::parse(generated.as_str()).unwrap();
    assert_eq!(reparsed.as_str(), generated.as_str());
    assert_eq!(generated.as_str().len(), 22);
    let (millis, suffix) = generated.as_str().split_once('-').unwrap();
    assert_eq!(millis.len(), 13);
    assert!(millis.bytes().all(|byte| byte.is_ascii_digit()));
    assert_eq!(suffix.len(), 8);
    assert!(
        suffix
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    );
}

#[test]
fn two_task_ids_generated_in_the_same_millisecond_differ() {
    let first = generate_task_id().unwrap();
    let second = generate_task_id().unwrap();
    assert_ne!(first.as_str(), second.as_str());
}
