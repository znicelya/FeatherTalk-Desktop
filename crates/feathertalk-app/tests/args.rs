use std::ffi::OsString;
use std::path::PathBuf;

use feathertalk_app::args::{ArgsError, LaunchOptions, parse};

fn arguments(values: &[&str]) -> Vec<OsString> {
    values.iter().map(OsString::from).collect()
}

#[test]
fn no_arguments_means_no_project() {
    assert_eq!(parse(arguments(&[])), Ok(LaunchOptions::default()));
}

#[test]
fn a_separate_value_sets_the_project_directory() {
    let options =
        parse(arguments(&["--project", "C:/projects/demo"])).expect("a valid command line");

    assert_eq!(options.project_dir, Some(PathBuf::from("C:/projects/demo")));
}

#[test]
fn an_inline_value_sets_the_project_directory() {
    let options = parse(arguments(&["--project=C:/projects/demo"])).expect("a valid command line");

    assert_eq!(options.project_dir, Some(PathBuf::from("C:/projects/demo")));
}

#[test]
fn a_flag_without_a_value_is_rejected() {
    assert_eq!(
        parse(arguments(&["--project"])),
        Err(ArgsError::MissingValue { flag: "--project" })
    );
}

#[test]
fn an_empty_value_is_rejected() {
    assert_eq!(
        parse(arguments(&["--project="])),
        Err(ArgsError::MissingValue { flag: "--project" })
    );
}

#[test]
fn a_repeated_flag_is_rejected() {
    assert_eq!(
        parse(arguments(&["--project", "one", "--project", "two"])),
        Err(ArgsError::Repeated { flag: "--project" })
    );
}

#[test]
fn an_unknown_argument_is_rejected() {
    assert_eq!(
        parse(arguments(&["--verbose"])),
        Err(ArgsError::Unknown {
            argument: "--verbose".to_owned()
        })
    );
}
