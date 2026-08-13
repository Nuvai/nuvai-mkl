//! Generates raw FFI bindings from the oneMKL headers that `nuvai-mkl-src`
//! has acquired (and cached), via `nuvai_mkl_src::locate()`.
//!
//! On `aarch64-apple-darwin` Intel ships no oneMKL, so the bindgen pass is
//! skipped entirely and the crate compiles the hand-written Accelerate FFI
//! surface in `src/aarch64.rs` instead. The Intel x86_64 path is unchanged.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use std::env;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=MKLROOT");
    println!("cargo:rerun-if-changed=src/aarch64.rs");

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // No Intel oneMKL on Apple Silicon: the FFI surface is hand-written
        // (src/aarch64.rs). Surface the selected backend to downstream crates.
        let backend = nuvai_mkl_src::backend();
        println!(
            "[nuvai-mkl-sys] aarch64-apple-darwin: hand-written FFI surface ({})",
            nuvai_mkl_src::backend_tag(backend)
        );
        println!("cargo:metadata=BACKEND={}", nuvai_mkl_src::backend_tag(backend));
        return;
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let info = nuvai_mkl_src::locate();
        eprintln!(
            "[nuvai-mkl-sys] binding oneMKL {} from {}",
            nuvai_mkl_src::MKL_VERSION,
            info.include_dir.display()
        );

        let out_file = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("bindings.rs");

        let bindings = bindgen::Builder::default()
            .header("wrapper.h")
            .clang_arg(format!("-I{}", info.include_dir.display()))
            .size_t_is_usize(true)
            .generate_comments(false)
            .prepend_enum_name(false)
            .generate()
            .expect("bindgen failed to parse oneMKL headers");

        bindings
            .write_to_file(&out_file)
            .expect("failed to write bindings");
    }
}
