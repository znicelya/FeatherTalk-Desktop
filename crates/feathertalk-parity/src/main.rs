use clap::{Parser, Subcommand, ValueEnum};
use feathertalk_parity::{
    archive::GoldenArchive,
    fixture::{
        ForwardCase, run_cpu_forward, run_cpu_train_step, run_wgpu_forward, run_wgpu_train_step,
    },
    probe::{ExecutionEvidence, GraphicsSelection, run_wgpu_probe},
};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "feathertalk-parity",
    about = "FeatherTalk Burn parity acceptance tooling"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Probe {
        #[arg(long, value_enum, default_value_t = GraphicsSelection::Auto)]
        graphics: GraphicsSelection,
    },
    Forward {
        #[arg(long, value_enum)]
        model: ModelSelection,
        #[arg(long, value_enum)]
        backend: BackendSelection,
        #[arg(long)]
        fixture: PathBuf,
    },
    TrainStep {
        #[arg(long, value_enum)]
        backend: BackendSelection,
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        full: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelSelection {
    Feather,
    Unet,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendSelection {
    Cpu,
    Wgpu,
}

#[derive(Debug, Serialize)]
struct Success<T: Serialize> {
    status: &'static str,
    #[serde(flatten)]
    value: T,
}

#[derive(Debug, Serialize)]
struct CpuForwardOutput {
    backend: &'static str,
    graphics: Option<&'static str>,
    metrics: feathertalk_parity::metrics::ParityMetrics,
}

#[derive(Debug, Serialize)]
struct CpuTrainOutput {
    backend: &'static str,
    initial_loss_relative: f32,
    post_step_loss_relative: f32,
    selected_parameter_relative: std::collections::BTreeMap<String, f32>,
    batch_norm_state_relative: std::collections::BTreeMap<String, f32>,
}

fn main() {
    let result = std::thread::Builder::new()
        .name("feathertalk-parity".to_owned())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| run().map_err(|error| error.to_string()))
        .expect("parity worker thread should start")
        .join()
        .unwrap_or_else(|panic| Err(format!("parity worker panicked: {panic:?}")));
    if let Err(error) = result {
        println!(
            "{}",
            serde_json::json!({"status": "failed", "error": error.to_string()})
        );
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Probe { graphics } => {
            let evidence = run_wgpu_probe(graphics)?;
            print_json(&Success {
                status: "passed",
                value: evidence,
            })?;
        }
        Command::Forward {
            model,
            backend,
            fixture,
        } => {
            let archive = GoldenArchive::open(fixture)?;
            let case = match model {
                ModelSelection::Feather => ForwardCase::FeatherMicro,
                ModelSelection::Unet => ForwardCase::UnetProduction,
            };
            match backend {
                BackendSelection::Cpu => {
                    let metrics = run_cpu_forward(&archive, case)?;
                    ensure_forward_tolerance(metrics.max_abs)?;
                    print_json(&Success {
                        status: "passed",
                        value: CpuForwardOutput {
                            backend: "cpu",
                            graphics: None,
                            metrics,
                        },
                    })?;
                }
                BackendSelection::Wgpu => {
                    let result = run_wgpu_forward(&archive, case, GraphicsSelection::Auto)?;
                    ensure_forward_tolerance(result.metrics.max_abs)?;
                    print_json(&Success {
                        status: "passed",
                        value: result,
                    })?;
                }
            }
        }
        Command::TrainStep {
            backend,
            fixture,
            full,
        } => {
            let archive = GoldenArchive::open(fixture)?;
            match backend {
                BackendSelection::Cpu => {
                    let result = run_cpu_train_step(&archive)?;
                    print_json(&Success {
                        status: "passed",
                        value: CpuTrainOutput {
                            backend: "cpu",
                            initial_loss_relative: result.initial_loss_relative,
                            post_step_loss_relative: result.post_step_loss_relative,
                            selected_parameter_relative: result.selected_parameter_relative,
                            batch_norm_state_relative: result.batch_norm_state_relative,
                        },
                    })?;
                }
                BackendSelection::Wgpu => {
                    let result = run_wgpu_train_step(&archive, GraphicsSelection::Auto, full)?;
                    if !result.initial_loss.is_finite()
                        || !result.gradient_norm.is_finite()
                        || result.gradient_norm <= 0.0
                        || !result.output_weight_changed
                        || result.execution.used_cpu_fallback
                    {
                        return Err("WGPU train-step acceptance failed".into());
                    }
                    print_json(&Success {
                        status: "passed",
                        value: result,
                    })?;
                }
            }
        }
    }
    Ok(())
}

fn ensure_forward_tolerance(max_abs: f32) -> Result<(), Box<dyn std::error::Error>> {
    if max_abs <= 1e-3 {
        Ok(())
    } else {
        Err(format!("forward max_abs {max_abs} exceeds 1e-3").into())
    }
}

fn print_json<T: Serialize>(value: &T) -> Result<(), serde_json::Error> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

#[allow(dead_code)]
fn _evidence_type_check(_: ExecutionEvidence) {}
