fn main() {
    println!("cargo:rustc-check-cfg=cfg(scrfd_generated)");
    println!("cargo:rerun-if-env-changed=SCRFD_GENERATED_RS");
    if let Ok(source) = std::env::var("SCRFD_GENERATED_RS") {
        println!("cargo:rerun-if-changed={source}");
        let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap())
            .join("scrfd_generated.rs");
        std::fs::copy(source, out).unwrap();
        println!("cargo:rustc-cfg=scrfd_generated");
    }
}
