use clap::Parser;
use clap::error::ErrorKind;

use feathertalk_cli::{Cli, EXIT_SESSION_ERROR};

fn main() {
    match Cli::try_parse() {
        Ok(cli) => std::process::exit(feathertalk_cli::run(cli)),
        Err(error) => {
            // `--help` and `--version` are requests that succeeded.
            if matches!(
                error.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            ) {
                print!("{error}");
                std::process::exit(EXIT_COMPLETED_ON_HELP);
            }
            // Clap's own default for a usage error is 2, which this CLI has
            // already spent on "cancelled". A misused command line is a session
            // error, so it exits 3.
            eprint!("{error}");
            std::process::exit(EXIT_SESSION_ERROR);
        }
    }
}

/// Help and version are output, not failure.
const EXIT_COMPLETED_ON_HELP: i32 = 0;
