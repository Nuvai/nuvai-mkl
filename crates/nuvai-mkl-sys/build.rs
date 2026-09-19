//! Generates raw FFI bindings from the oneMKL headers that `nuvai-mkl-src`
//! has acquired (and cached), via `nuvai_mkl_src::MklInfo::from_build_metadata()`.
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
//! Intel bindgen path (which would also bind the wrong-architecture headers).
//!
//! The reverse direction is not symmetric: an Intel *target* on a host that has
//! no bindgen (macOS, or any aarch64 host) has no FFI surface to compile, since
//! the bindings are generated rather than hand-written. That combination now
//! panics with the host, the target and the build-dep gate named (#34); before,
//! it fell through silently and surfaced as a missing `bindings.rs`.
//!
//! # docs.rs
//!
//! The docs.rs build has no network and no MKL. docs.rs sets the `DOCS_RS`
//! environment variable, so this script returns before bindgen on every target —
//! mirroring the guard in `nuvai-mkl-src/build.rs`.

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

    // docs.rs has no network and no MKL; skip bindgen there.
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
        // because this crate declares no `links` key, so Cargo would drop it.
        // The backend reaches the final link through `nuvai-mkl-src`, which is
        // the sole `links = "mkl"` provider (ADR-0003, decision 1).
        ("macos", "aarch64") | ("linux", "aarch64") => {
            let backend =
                nuvai_mkl_src::backend_for_target(&target_os, &target_arch, target_env.as_deref())
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
        //
        // A target that needs bindgen on a host that has no bindgen therefore
        // has no arm to run, and the two cfgs below are complements covering
        // exactly that split (#34). The `cfg(any(...))` arm must stay the exact
        // negation of the `cfg(all(...))` arm *and* of the build-dep gate in
        // `Cargo.toml`; all three describe one condition, "this host is an Intel
        // x86_64 non-macOS platform". Falling through instead — which is what
        // happened before — left `include!(concat!(env!("OUT_DIR"),
        // "/bindings.rs"))` in `src/lib.rs` to report a bare "couldn't find file
        // .../bindings.rs" that names neither the host/target split nor the
        // build-dep gate that caused it.
        ("linux", "x86_64") | ("windows", "x86_64") => {
            #[cfg(any(target_os = "macos", target_arch = "aarch64"))]
            {
                panic!(
                    "nuvai-mkl-sys: cannot build for {target_os}-{target_arch} from a {host} \
                     host — bindgen cannot be called here. bindgen is a *host*-resolved \
                     build-dependency, so Cargo installs it only when the host is an Intel \
                     x86_64 non-macOS platform (see the `[target.'cfg(...)'.build-dependencies]` \
                     table in this crate's Cargo.toml), and generation for an x86_64 \
                     Linux/Windows target is impossible without it. Nothing was written to \
                     OUT_DIR, so the build would otherwise fail later with a bare \
                     \"couldn't find file .../bindings.rs\". Build this target on an x86_64 \
                     Linux/Windows host — the `x86_64-linux` and `x86_64-windows` CI jobs do \
                     — rather than cross-compiling to it from {host}.",
                    host = env::var("HOST").unwrap_or_else(|_| "<unknown>".to_string()),
                );
            }
            #[cfg(all(not(target_os = "macos"), not(target_arch = "aarch64")))]
            {
                // The paths its build script acquired and published, not a
                // second acquisition of our own (#24) — `locate()` is no longer
                // in the library target, so this is the only way to reach them.
                let info = nuvai_mkl_src::MklInfo::from_build_metadata().expect(
                    "nuvai-mkl-src published no DEP_MKL_* metadata — it must stay in this \
                     crate's [build-dependencies] for Cargo to forward it",
                );
                // The published paths are a snapshot that nothing re-derives
                // when `~/.cache/nuvai-mkl` is cleared, so check rather than let
                // bindgen report the missing header as a parse failure.
                if !info.include_dir.is_dir() {
                    panic!(
                        "the MKL include directory nuvai-mkl-src published, {}, no longer \
                         exists. If the MKL cache was cleared, force acquisition to re-run \
                         with `cargo clean -p nuvai-mkl-src` and rebuild.",
                        info.include_dir.display()
                    );
                }
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
