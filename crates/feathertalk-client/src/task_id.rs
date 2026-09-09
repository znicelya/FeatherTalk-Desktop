use std::sync::atomic::{AtomicU32, Ordering};

use feathertalk_domain::{DomainError, TaskId};
use time::OffsetDateTime;

/// Bumped once per generated id so two ids minted in the same millisecond
/// cannot collide.
static COUNTER: AtomicU32 = AtomicU32::new(0);

/// 10^13, the exclusive bound of the 13-digit millisecond field.
const MILLIS_MODULUS: i128 = 10_000_000_000_000;

/// Mint a task id in the domain's wire format: thirteen decimal digits of Unix
/// milliseconds, `-`, then eight lowercase hex digits.
///
/// The generator lives here rather than in `feathertalk-domain` because the
/// format is domain-owned but the *policy* — which clock, how uniqueness is
/// obtained — is a client concern.
pub fn generate_task_id() -> Result<TaskId, DomainError> {
    let millis =
        (OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000).rem_euclid(MILLIS_MODULUS);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let suffix = std::process::id() ^ counter.rotate_left(16);
    TaskId::parse(&format!("{millis:013}-{suffix:08x}"))
}
