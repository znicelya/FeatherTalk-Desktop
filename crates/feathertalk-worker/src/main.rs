use std::io::{self, BufReader};

use feathertalk_worker::{WorkerConfig, serve};

fn main() {
    let config = WorkerConfig::from_env();
    // stdout carries the protocol; every diagnostic goes to stderr.
    if let Err(error) = serve(BufReader::new(io::stdin()), io::stdout(), &config) {
        eprintln!("feathertalk-worker: {error}");
        std::process::exit(1);
    }
}
