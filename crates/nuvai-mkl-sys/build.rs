//! Generates raw FFI bindings from the oneMKL headers that `nuvai-mkl-src`
//! has acquired (and cached), via `nuvai_mkl_src::locate()`.
//!
//! On the aarch64 targets where Intel ships no oneMKL, the bindgen pass is
//! skipped entirely and the crate compiles a hand-written FFI surface instead:
//! `src/aarch64.rs` (Accelerate) on `aarch64-apple-darwin` and
//! `src/linux_aarch64.rs` (OpenBLAS) on `aarch64-unknown-linux-gnu`. The Intel
//! x86_64 path is unchanged.
//!
//! # Host vs target
//!
//! Build scripts compile for and run on the *host*, so `#[cfg(...)]` here
//! describes the host (and Cargo resolves `[target.'cfg'.build-dependencies]`
//! against the host too — the bindgen build-dep is present exactly when the
//! host is an Intel x86_64 non-macOS platform). The *backend* belongs to the
//! target being built, which Cargo exposes as `CARGO_CFG_TARGET_*`. Dispatch on
//! those so cross-compiling `--target aarch64-unknown-linux-gnu` from an
//! x86_64 host selects the hand-written OpenBLAS surface instead of running the
//! Intel bindgen path (`nuvai_mkl_src::locate()` is unavailable on an aarch64
//! host and would also acquire the wrong-architecture headers).
//!
//! # docs.rs
//!
//! The docs.rs build has no network and no MKL. docs.rs sets the `DOCS_RS`
//! environment variable, so this script returns before `nuvai_mkl_src::locate()`
//! / bindgen on every target — mirroring the guard in `nuvai-mkl-src/build.rs`.

#![allow(unused_imports)] // `PathBuf` is used only on Intel host builds

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=MKLROOT");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-changed=src/aarch64.rs");
    println!("cargo:rerun-if-changed=src/linux_aarch64.rs");
    println!("cargo:rerun-if-changed=src/netlib_abi.rs");

    // docs.rs has no network and no MKL; skip locate()/bindgen there.
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS is set");
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH is set");
    let target_env = env::var("CARGO_CFG_TARGET_ENV").ok();

    match (target_os.as_str(), target_arch.as_str()) {
        // No Intel oneMKL on any aarch64 target: the FFI surface is hand-written
        // (src/aarch64.rs on macOS, src/linux_aarch64.rs on Linux). Surface the
        // selected backend as a diagnostic; `cargo:metadata` is not emitted here
        // because this crate declares no `links` key, so Cargo would drop it —
        // the backend already propagates via `nuvai-mkl-src` (`DEP_MKL_BACKEND`).
        ("macos", "aarch64") | ("linux", "aarch64") => {
            let backend = nuvai_mkl_src::backend_for_target(
                &target_os,
                &target_arch,
                target_env.as_deref(),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            eprintln!(
                "[nuvai-mkl-sys] {target_os}-{target_arch}: hand-written FFI surface ({})",
                nuvai_mkl_src::backend_tag(backend)
            );
        }
        // Intel targets (x86_64 Linux/Windows): generate bindings from the
        // oneMKL headers `nuvai-mkl-src` acquired. The bindgen reference is
        // gated by the *host* cfg — the exact cfg Cargo uses to install the
        // bindgen build-dep — so it is only compiled where `bindgen` exists.
        ("linux", "x86_64") | ("windows", "x86_64") => {
            #[cfg(all(not(target_os = "macos"), not(target_arch = "aarch64")))]
            {
                let info = nuvai_mkl_src::locate();
                eprintln!(
                    "[nuvai-mkl-sys] binding oneMKL {} from {}",
                    nuvai_mkl_src::MKL_VERSION,
                    info.include_dir.display()
                );

                let out_file =
                    PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("bindings.rs");

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
        other => panic!("nuvai-mkl-sys: unsupported target {other:?}"),
    }
}
