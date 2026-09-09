#[cfg(scrfd_generated)]
#[derive(clap::Parser)]
struct Args {
    #[arg(long)]
    burnpack: std::path::PathBuf,
    #[arg(long)]
    safetensors: std::path::PathBuf,
}

#[cfg(scrfd_generated)]
fn main() {
    use burn::backend::NdArray;
    use clap::Parser;
    use feathertalk_scrfd_import::{ToolError, ensure_destination_absent};

    let args = Args::parse();
    if let Err(error) = ensure_destination_absent(&args.safetensors).and_then(|()| {
        feathertalk_scrfd_import::convert::convert_burnpack::<NdArray<f32>>(
            &args.burnpack,
            &args.safetensors,
        )
    }) {
        eprintln!(
            "SCRFD conversion failed for {}: {error}",
            args.burnpack.display()
        );
        std::process::exit(1);
    }
}

#[cfg(not(scrfd_generated))]
fn main() {
    eprintln!("SCRFD_GENERATED_RS is required");
    std::process::exit(2);
}
