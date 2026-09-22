use burn::tensor::{TensorData, Transaction};
use feathertalk_training::{LossBreakdown, TrainingError};

/// Scalar view of a `LossBreakdown`, detached from the autodiff graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossValues {
    pub total: f64,
    pub full: f64,
    pub perceptual: f64,
    pub mouth: Option<f64>,
    pub temporal: Option<f64>,
    pub temporal_mouth: Option<f64>}

impl LossValues {
    pub fn from_breakdown(breakdown: &LossBreakdown) -> Self {
        // Read all metrics at the same boundary. Separate into_scalar calls
        // force a GPU submission and host wait for every individual loss.
        let mut transaction = Transaction::default()
            .register(breakdown.total.clone())
            .register(breakdown.full.clone())
            .register(breakdown.perceptual.clone());
        for loss in [
            &breakdown.mouth,
            &breakdown.temporal,
            &breakdown.temporal_mouth,
        ]
        .into_iter()
        .flatten()
        {
            transaction = transaction.register(loss.clone());
        }
        let mut values = transaction.execute().into_iter();
        let mut next = || scalar(values.next().expect("one readback per registered loss"));
        Self {
            total: next(),
            full: next(),
            perceptual: next(),
            mouth: breakdown.mouth.as_ref().map(|_| next()),
            temporal: breakdown.temporal.as_ref().map(|_| next()),
            temporal_mouth: breakdown.temporal_mouth.as_ref().map(|_| next())}
    }

    pub fn require_finite(&self) -> Result<(), TrainingError> {
        check("total", Some(self.total))?;
        check("full", Some(self.full))?;
        check("perceptual", Some(self.perceptual))?;
        check("mouth", self.mouth)?;
        check("temporal", self.temporal)?;
        check("temporal_mouth", self.temporal_mouth)
    }
}

fn scalar(value: TensorData) -> f64 {
    let mut elements = value.iter::<f64>();
    let scalar = elements.next().expect("loss tensors contain one scalar");
    assert!(elements.next().is_none(), "loss tensors contain one scalar");
    scalar
}

fn check(field: &str, value: Option<f64>) -> Result<(), TrainingError> {
    match value {
        Some(value) if !value.is_finite() => {
            let message = format!("training loss {field} is not finite: {value}");
            Err(TrainingError::InvalidInput(message))
        }
        _ => Ok(())}
}
