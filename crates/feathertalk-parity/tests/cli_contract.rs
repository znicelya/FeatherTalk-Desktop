use std::process::Command;

#[test]
fn help_exposes_probe_forward_and_train_step() {
    let output = Command::new(env!("CARGO_BIN_EXE_feathertalk-parity"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    for command in ["probe", "forward", "train-step"] {
        assert!(
            stdout.contains(command),
            "missing command {command}: {stdout}"
        );
    }
}
