use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2dConfig;
use burn::tensor::{Distribution, Tensor};
use feathertalk_models::unet::{OriginalUnetConfig, TrainableTalkingHead};
use std::env;
use std::time::Instant;

struct ProbeArgs {
    backend: String,
    batches: Vec<usize>,
    iters: usize,
    skip_full: bool,
}

fn parse_args() -> ProbeArgs {
    let mut backend = "wgpu".to_string();
    let mut batches = vec![1usize];
    let mut iters = 5usize;
    let mut skip_full = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => {
                backend = args
                    .next()
                    .unwrap_or_else(|| usage("missing --backend value"));
            }
            "--batch" => {
                let raw = args
                    .next()
                    .unwrap_or_else(|| usage("missing --batch value"));
                batches = raw
                    .split(',')
                    .map(|part| {
                        part.trim()
                            .parse::<usize>()
                            .unwrap_or_else(|_| usage(&format!("invalid batch size {part}")))
                    })
                    .collect();
            }
            "--iters" => {
                let raw = args
                    .next()
                    .unwrap_or_else(|| usage("missing --iters value"));
                iters = raw
                    .parse::<usize>()
                    .unwrap_or_else(|_| usage(&format!("invalid iters {raw}")));
            }
            "--skip-full" => skip_full = true,
            "--help" | "-h" => usage(""),
            other => usage(&format!("unknown argument {other}")),
        }
    }
    if batches.is_empty() || batches.iter().any(|batch| *batch == 0) || iters == 0 {
        usage("batch and iters must be positive");
    }
    ProbeArgs { backend, batches, iters, skip_full }
}

fn usage(message: &str) -> ! {
    if !message.is_empty() {
        eprintln!("{message}");
    }
    eprintln!("usage: bwd_probe [--backend wgpu|cuda] [--batch 1,4] [--iters 5] [--skip-full]");
    std::process::exit(2);
}

fn sync(device: &burn::tensor::Device) {
    let _ = device.sync();
}

fn time_fb<F>(device: &burn::tensor::Device, label: &str, iters: usize, mut f: F)
where
    F: FnMut() -> (f64, f64),
{
    let _ = f();
    sync(device);
    let mut fwd = 0.0;
    let mut bwd = 0.0;
    for _ in 0..iters {
        let (fw, bw) = f();
        fwd += fw;
        bwd += bw;
    }
    println!(
        "{label:40} fwd={:.4}s bwd={:.4}s  bwd/fwd={:.1}x  (avg over {iters})",
        fwd / iters as f64,
        bwd / iters as f64,
        (bwd / iters as f64) / (fwd / iters as f64).max(1e-9)
    );
}

fn probe(backend: &str, device: burn::tensor::Device, batches: &[usize], iters: usize, skip_full: bool) {
    println!("== backend={backend} iters={iters} batches={batches:?} ==");
    for &batch in batches {
        println!("-- batch={batch} --");
        if !skip_full {
            let model = OriginalUnetConfig::production().init(&device);
            time_fb(
                &device,
                &format!("original_unet full b={batch}"),
                iters,
                || {
                    let image = Tensor::<4>::random(
                        [batch, 6, 160, 160],
                        Distribution::Default,
                        &device,
                    )
                    .require_grad();
                    let audio =
                        Tensor::<4>::random([batch, 16, 32, 32], Distribution::Default, &device);
                    let t0 = Instant::now();
                    let out = model.forward_training(image, audio);
                    let loss = out.abs().mean();
                    sync(&device);
                    let fw = t0.elapsed().as_secs_f64();
                    let t1 = Instant::now();
                    let _ = loss.backward();
                    sync(&device);
                    (fw, t1.elapsed().as_secs_f64())
                },
            );
        }

        for (ch, hw) in [
            (32usize, 160usize),
            (64, 80),
            (128, 40),
            (256, 20),
            (512, 10),
        ] {
            let conv = Conv2dConfig::new([ch, ch], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_groups(ch)
                .with_bias(false)
                .init(&device);
            time_fb(
                &device,
                &format!("depthwise conv c={ch} hw={hw} b={batch}"),
                iters,
                || {
                    let x =
                        Tensor::<4>::random([batch, ch, hw, hw], Distribution::Default, &device)
                            .require_grad();
                    let t0 = Instant::now();
                    let out = conv.forward(x);
                    let loss = out.abs().mean();
                    sync(&device);
                    let fw = t0.elapsed().as_secs_f64();
                    let t1 = Instant::now();
                    let _ = loss.backward();
                    sync(&device);
                    (fw, t1.elapsed().as_secs_f64())
                },
            );
        }

        for (ci, co, hw) in [(32usize, 32usize, 160usize), (256, 256, 20)] {
            let conv = Conv2dConfig::new([ci, co], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false)
                .init(&device);
            time_fb(
                &device,
                &format!("dense conv {ci}->{co} hw={hw} b={batch}"),
                iters,
                || {
                    let x =
                        Tensor::<4>::random([batch, ci, hw, hw], Distribution::Default, &device)
                            .require_grad();
                    let t0 = Instant::now();
                    let out = conv.forward(x);
                    let loss = out.abs().mean();
                    sync(&device);
                    let fw = t0.elapsed().as_secs_f64();
                    let t1 = Instant::now();
                    let _ = loss.backward();
                    sync(&device);
                    (fw, t1.elapsed().as_secs_f64())
                },
            );
        }
    }
}

fn main() {
    let args = parse_args();
    match args.backend.as_str() {
        "wgpu" => probe(
            "wgpu",
            burn::tensor::Device::default().autodiff(),
            &args.batches,
            args.iters,
            args.skip_full,
        ),
        "cuda" => {
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            {
                probe(
                    "cuda",
                    burn::tensor::Device::default().autodiff(),
                    &args.batches,
                    args.iters,
                    args.skip_full,
                );
            }
            #[cfg(not(any(target_os = "windows", target_os = "linux")))]
            {
                usage("cuda backend is only available on windows/linux");
            }
        }
        other => usage(&format!("unsupported backend {other}")),
    }
}