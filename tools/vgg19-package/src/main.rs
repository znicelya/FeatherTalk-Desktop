use std::path::PathBuf;

use clap::Parser;
use feathertalk_vgg19_package::{Vgg19PackageRequest, build_vgg19_package};

#[derive(Debug, Parser)]
#[command(name = "feathertalk-vgg19-package")]
struct Arguments {
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    licenses: PathBuf,
    #[arg(long)]
    destination: PathBuf,
}

fn main() -> Result<(), feathertalk_vgg19_package::PackageError> {
    let arguments = Arguments::parse();
    let report = build_vgg19_package(&Vgg19PackageRequest {
        source: arguments.source,
        licenses: arguments.licenses,
        destination: arguments.destination.clone(),
    })?;
    println!("destination={}", arguments.destination.display());
    println!("source_sha256={}", report.manifest.source.sha256);
    println!("model_sha256={}", report.manifest.model.sha256);
    println!("tensor_count={}", report.manifest.tensor_count);
    Ok(())
}
