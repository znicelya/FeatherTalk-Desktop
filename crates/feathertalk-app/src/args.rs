//! The launch options the shell accepts on the command line.
//!
//! One flag, `--project <directory>`. It exists because the task page needs a
//! project before it can do anything -- the task history, the crash logs and the
//! only command this slice submits all live under one -- and the picker that will
//! replace it belongs to the asset page slice. Arguments arrive as `OsString`:
//! `std::env::args` panics on an argument that is not valid Unicode, and a
//! desktop application must not die because of the name of a directory.

use std::ffi::OsString;
use std::path::PathBuf;

use thiserror::Error;

/// The flag that names the project directory.
pub const PROJECT_FLAG: &str = "--project";

/// What one launch was asked to do.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LaunchOptions {
    /// The project this session works on.
    pub project_dir: Option<PathBuf>,
}

/// Why a command line could not be understood.
///
/// English, like every other diagnostic in this crate that reaches a terminal
/// rather than a window: by the time this is printed there is no window yet.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ArgsError {
    #[error("{flag} needs a directory")]
    MissingValue { flag: &'static str },
    #[error("{flag} was given twice")]
    Repeated { flag: &'static str },
    #[error("unknown argument: {argument}")]
    Unknown { argument: String },
}

/// Parse the arguments after the program name.
pub fn parse<I>(arguments: I) -> Result<LaunchOptions, ArgsError>
where
    I: IntoIterator<Item = OsString>,
{
    let mut options = LaunchOptions::default();
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        let Some(value) = project_value(&argument, &mut arguments)? else {
            return Err(ArgsError::Unknown {
                // Lossy only in the error message: the argument is already
                // rejected, and a mangled character reads better than none.
                argument: argument.to_string_lossy().into_owned(),
            });
        };
        if options.project_dir.is_some() {
            return Err(ArgsError::Repeated { flag: PROJECT_FLAG });
        }
        if value.is_empty() {
            return Err(ArgsError::MissingValue { flag: PROJECT_FLAG });
        }
        options.project_dir = Some(PathBuf::from(value));
    }
    Ok(options)
}

/// The value `argument` carries, or `None` when it is not the project flag.
///
/// `--project <dir>` keeps the value exactly as the operating system gave it;
/// `--project=<dir>` needs the argument itself to be valid Unicode, because the
/// split happens inside the string.
fn project_value<I>(argument: &OsString, rest: &mut I) -> Result<Option<OsString>, ArgsError>
where
    I: Iterator<Item = OsString>,
{
    if argument == PROJECT_FLAG {
        return match rest.next() {
            Some(value) => Ok(Some(value)),
            None => Err(ArgsError::MissingValue { flag: PROJECT_FLAG }),
        };
    }
    let Some(text) = argument.to_str() else {
        return Ok(None);
    };
    match text
        .strip_prefix(PROJECT_FLAG)
        .and_then(|rest| rest.strip_prefix('='))
    {
        Some(value) => Ok(Some(OsString::from(value))),
        None => Ok(None),
    }
}
