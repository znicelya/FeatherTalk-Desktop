use feathertalk_domain::TaskId;
use feathertalk_worker::{AdapterLockError, AdapterLocks, CPU_ADAPTER_ID};

fn task(suffix: &str) -> TaskId {
    TaskId::parse(&format!("1787900000000-{suffix}")).unwrap()
}

fn cpu_locks() -> AdapterLocks {
    AdapterLocks::new([CPU_ADAPTER_ID.to_owned()])
}

#[test]
fn a_fresh_table_reports_every_known_adapter_as_free() {
    let locks = cpu_locks();
    assert!(locks.is_free(CPU_ADAPTER_ID));
    assert_eq!(locks.holder(CPU_ADAPTER_ID), None);
}

#[test]
fn an_unknown_adapter_cannot_be_locked_or_released() {
    let mut locks = cpu_locks();
    assert!(matches!(
        locks.acquire("gpu-9", task("0000000a")),
        Err(AdapterLockError::Unknown(_))
    ));
    assert!(matches!(
        locks.release("gpu-9"),
        Err(AdapterLockError::Unknown(_))
    ));
    assert!(!locks.is_free("gpu-9"));
}

#[test]
fn a_locked_adapter_refuses_a_second_task_and_names_the_holder() {
    let mut locks = cpu_locks();
    let first = task("0000000a");
    locks.acquire(CPU_ADAPTER_ID, first.clone()).unwrap();
    assert!(!locks.is_free(CPU_ADAPTER_ID));
    assert_eq!(locks.holder(CPU_ADAPTER_ID), Some(&first));

    match locks.acquire(CPU_ADAPTER_ID, task("0000000b")) {
        Err(AdapterLockError::Occupied { adapter_id, holder }) => {
            assert_eq!(adapter_id, CPU_ADAPTER_ID);
            assert_eq!(holder, first);
        }
        other => panic!("expected an occupied adapter, got {other:?}"),
    }
}

#[test]
fn releasing_frees_the_adapter_for_the_next_task() {
    let mut locks = cpu_locks();
    locks.acquire(CPU_ADAPTER_ID, task("0000000a")).unwrap();
    locks.release(CPU_ADAPTER_ID).unwrap();
    assert!(locks.is_free(CPU_ADAPTER_ID));
    locks.acquire(CPU_ADAPTER_ID, task("0000000b")).unwrap();
    assert_eq!(locks.holder(CPU_ADAPTER_ID), Some(&task("0000000b")));
}

#[test]
fn releasing_a_free_adapter_is_an_error() {
    let mut locks = cpu_locks();
    assert!(matches!(
        locks.release(CPU_ADAPTER_ID),
        Err(AdapterLockError::NotHeld(_))
    ));
}

#[test]
fn adapters_are_locked_independently() {
    let mut locks = AdapterLocks::new(["gpu-a".to_owned(), "gpu-b".to_owned()]);
    let first = task("0000000a");
    let second = task("0000000b");
    locks.acquire("gpu-a", first.clone()).unwrap();
    locks.acquire("gpu-b", second.clone()).unwrap();
    assert_eq!(locks.holder("gpu-a"), Some(&first));
    assert_eq!(locks.holder("gpu-b"), Some(&second));
    locks.release("gpu-a").unwrap();
    assert!(locks.is_free("gpu-a"));
    assert_eq!(locks.holder("gpu-b"), Some(&second));
}
