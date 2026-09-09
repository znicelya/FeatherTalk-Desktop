//! One `name：value` line, and what its value is made of.
//!
//! Three panels paint this shape -- the asset package, the training metrics and
//! the training configuration -- and they all face the same question: is this
//! value something a worker measured, or a word the user reads? A measured number
//! is shown as it is, because translating a frame count is not a thing; a state
//! or a mode is a catalog key. Keeping the pair here means the answer is given
//! once instead of once per page.

/// One line of a fact card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fact {
    /// The catalog key of the label.
    pub label: &'static str,
    /// What to show after it.
    pub value: FactValue,
}

/// A fact's value: either something a worker measured, or copy to translate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactValue {
    Text(String),
    Key(&'static str),
}
