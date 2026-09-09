//! The client's fake worker, compiled again as a binary of this crate.
//!
//! `include!` rather than a copy: one source of truth for the scenarios, and a
//! distinct binary name so the two crates cannot collide in the target
//! directory. A `[[bin]]` cannot use dev-dependencies, which is why the included
//! file only needs `feathertalk-domain` and `serde_json` — both are ordinary
//! dependencies of this crate.

include!("../../../feathertalk-client/tests/support/fake_worker.rs");
