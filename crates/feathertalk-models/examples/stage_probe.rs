// Per-stage forward/backward profiler for the real OriginalUnet, to see where
// the backward time actually concentrates. Each stage is measured in isolation
// on a detached copy of its real intermediate tensor, so shapes and volumes
// match the live model while the timing stays attributable to one stage.
use burn::tensor::{Distribution, Tensor};
use feathertalk_models::unet::{OriginalUnetConfig, TrainableTalkingHead};
use std::env;
use std::time::Instant;

fn sync(device: &burn::tensor::Device) {
    let _ = device.sync();
}

fn time_stage<F>(device: &burn::tensor::Device, label: &str, iters: usize, mut f: F)
where
    F: FnMut() -> (f64, f64),
{
    let _ = f();
    sync(device);
    let (mut fwd, mut bwd) = (0.0, 0.0);
    for _ in 0..iters {
        let (fw, bw) = f();
        fwd += fw;
        bwd += bw;
    }
    let fwd = fwd / iters as f64;
    let bwd = bwd / iters as f64;
    println!(
        "{label:28} fwd={:.4}s bwd={:.4}s  bwd/fwd={:.1}x",
        fwd,
        bwd,
        bwd / fwd.max(1e-9)
    );
}

fn main() {
    let mut batch = 4usize;
    let mut iters = 20usize;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--batch" => batch = args.next().and_then(|v| v.parse().ok()).unwrap_or(4),
            "--iters" => iters = args.next().and_then(|v| v.parse().ok()).unwrap_or(20),
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let device = burn::tensor::Device::default().autodiff();
    let model = OriginalUnetConfig::production().init(&device);
    println!("== stage profile batch={batch} iters={iters} ==");

    let image = Tensor::<4>::random([batch, 6, 160, 160], Distribution::Default, &device);
    let audio = Tensor::<4>::random([batch, 16, 32, 32], Distribution::Default, &device);

    // One real forward to capture concrete intermediates; detach so each stage
    // measurement starts from a fresh leaf of the correct shape.
    let x1 = model.inc.forward(image.clone());
    let x2 = model.down1.forward(x1.clone());
    let x3 = model.down2.forward(x2.clone());
    let x4 = model.down3.forward(x3.clone());
    let x5 = model.down4.forward(x4.clone());
    let a = model.audio_model.forward(audio.clone());
    let x5c = Tensor::cat(vec![x5.clone(), a.clone()], 1);
    let f1 = model.fuse_first.forward(x5c.clone());
    let f2 = model.fuse_second.forward(f1.clone());
    let o1 = model.up1.forward(f2.clone(), x4.clone());
    let o2 = model.up2.forward(o1.clone(), x3.clone());
    let o3 = model.up3.forward(o2.clone(), x2.clone());
    let o4 = model.up4.forward(o3.clone(), x1.clone());

    let image = image.detach();
    let audio = audio.detach();
    let x1d = x1.detach();
    let x2d = x2.detach();
    let x3d = x3.detach();
    let x4d = x4.detach();
    let x5d = x5.detach();
    let ad = a.detach();
    let x5cd = x5c.detach();
    let f1d = f1.detach();
    let f2d = f2.detach();
    let o1d = o1.detach();
    let o2d = o2.detach();
    let o3d = o3.detach();
    let o4d = o4.detach();

    macro_rules! stage {
        ($label:expr, $body:expr) => {
            time_stage(&device, $label, iters, || {
                let t0 = Instant::now();
                let out: Tensor<4> = $body;
                let loss = out.abs().mean();
                sync(&device);
                let fw = t0.elapsed().as_secs_f64();
                let t1 = Instant::now();
                let _ = loss.backward();
                sync(&device);
                (fw, t1.elapsed().as_secs_f64())
            });
        };
    }

    stage!("inc            @160", model.inc.forward(image.clone().require_grad()));
    stage!("down1  32->64  @160", model.down1.forward(x1d.clone().require_grad()));
    stage!("down2  64->128 @80 ", model.down2.forward(x2d.clone().require_grad()));
    stage!("down3 128->256 @40 ", model.down3.forward(x3d.clone().require_grad()));
    stage!("down4 256->512 @20 ", model.down4.forward(x4d.clone().require_grad()));
    stage!("audio_model    @32 ", model.audio_model.forward(audio.clone().require_grad()));
    stage!("fuse_first     @10 ", model.fuse_first.forward(x5cd.clone().require_grad()));
    stage!("fuse_second    @10 ", model.fuse_second.forward(f1d.clone().require_grad()));
    stage!("up1            @10>20", model.up1.forward(f2d.clone().require_grad(), x4d.clone()));
    stage!("up2            @20>40", model.up2.forward(o1d.clone().require_grad(), x3d.clone()));
    stage!("up3            @40>80", model.up3.forward(o2d.clone().require_grad(), x2d.clone()));
    stage!("up4            @80>160", model.up4.forward(o3d.clone().require_grad(), x1d.clone()));
    stage!("outc           @160", model.outc.forward(o4d.clone().require_grad()));

    // Whole-model reference.
    time_stage(&device, "FULL MODEL", iters, || {
        let image = Tensor::<4>::random([batch, 6, 160, 160], Distribution::Default, &device)
            .require_grad();
        let audio = Tensor::<4>::random([batch, 16, 32, 32], Distribution::Default, &device);
        let t0 = Instant::now();
        let out = model.forward_training(image, audio);
        let loss = out.abs().mean();
        sync(&device);
        let fw = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let _ = loss.backward();
        sync(&device);
        (fw, t1.elapsed().as_secs_f64())
    });
}