//! Build script for `nuvai-mkl-src`: locate oneMKL 2026.1.0 (Intel x86_64) or
//! emit the fallback linker directives (Accelerate / OpenBLAS).
//!
//! On `aarch64-apple-darwin`, Intel ships no oneMKL, so this script never calls
//! [`locate`]; it emits `-framework Accelerate` (default) or `-lopenblas`
//! (`openblas` feature). On `aarch64-unknown-linux-gnu` it emits `-lopenblas`.
//! The Intel x86_64 path is byte-identical to the pre-fallback behaviour and is
//! selected by [`backend_for_target`].
//!
//! Build scripts compile for and run on the *host*, so `#[cfg(...)]` here would
//! describe the host, not the crate being built. The backend is therefore
//! selected from the target triple Cargo exposes as `CARGO_CFG_TARGET_OS` /
//! `CARGO_CFG_TARGET_ARCH` / `CARGO_CFG_TARGET_ENV` — without this,
//! cross-compiling `--target aarch64-unknown-linux-gnu` from an x86_64 host
//! would select `IntelMkl` and emit x86_64 MKL directives for an ARM target.

include!("src/acquire.rs");
include!("src/backend.rs");

fn main() {
    println!("cargo:rerun-if-env-changed=MKLROOT");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/acquire.rs");
    println!("cargo:rerun-if-changed=src/backend.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_OPENBLAS");
    println!("cargo:rerun-if-env-changed=OPENBLAS_ROOT");

    // docs.rs has no network and no MKL; skip linking there.
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    // The backend is a property of the *target* being built, not of the host
    // this build script runs on (see the module docs above).
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS is set");
    let target_arch =
        std::env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH is set");
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").ok();

    let backend = backend_for_target(&target_os, &target_arch, target_env.as_deref())
        .unwrap_or_else(|e| panic!("{e}"));

    match backend {
        Backend::IntelMkl => emit_intel_mkl(&target_os),
        Backend::Accelerate => {
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:metadata=BACKEND=accelerate");
        }
        Backend::OpenBlas => {
            println!("cargo:rustc-link-lib=dylib=openblas");
            println!("cargo:metadata=BACKEND=openblas");
            // macOS: OpenBLAS replaces only BLAS/LAPACK (vecLib); FFT (vDSP),
            // VML (vForce) and the sparse solvers (Sparse/SparseSolve) still
            // call Accelerate, so both must be linked on this path.
            if target_os == "macos" {
                println!("cargo:rustc-link-lib=framework=Accelerate");
            }
            // Linux-aarch64: OpenBLAS is the only backend and covers only
            // BLAS/LAPACK, so there is no Accelerate to link. A distro
            // `libopenblas-dev` lives in the system search path; an explicit
            // OPENBLAS_ROOT (e.g. a conda/pip install) contributes its `lib`
            // dir to the link search path. The runtime *rpath* is deliberately
            // not emitted here: `cargo:rustc-link-arg` only applies to the
            // emitting package's own targets, and this crate owns no binaries —
            // the crate that owns the test/example binaries (`nuvai-mkl/build.rs`)
            // emits the rpath instead.
            if target_os == "linux"
                && target_arch == "aarch64"
                && let Ok(root) = std::env::var("OPENBLAS_ROOT")
                && !root.trim().is_empty()
            {
                println!("cargo:rustc-link-search=native={root}/lib");
            }
        }
    }
}

/// Emit the Intel oneMKL linker directives (x86_64 Linux/Windows).
fn emit_intel_mkl(target_os: &str) {
    let info = locate();

    println!("cargo:rustc-link-search=native={}", info.lib_dir.display());

    match target_os {
        "linux" => {
            println!("cargo:rustc-link-lib=dylib=mkl_rt");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=m");
            if let Some(omp) = &info.omp_lib_dir {
                println!("cargo:rustc-link-search=native={}", omp.display());
                println!("cargo:rustc-link-lib=dylib=iomp5");
            }
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", info.lib_dir.display());
        }
        "windows" => {
            // conda win-64 `mkl` ships 26 DLLs but zero import libs; `mkl-devel`
            // ships `mkl_rt.lib` (which embeds `mkl_rt.3.dll`). Link that import
            // lib. `user32` is a dependency of the MKL DLLs on Windows.
            println!("cargo:rustc-link-lib=dylib=mkl_rt");
            println!("cargo:rustc-link-lib=dylib=user32");
        }
        _ => panic!("Intel oneMKL is only acquired for x86_64 Linux/Windows targets"),
    }

    // Informational metadata, surfaced to downstream build scripts as
    // `DEP_MKL_*` (this crate declares `links = "mkl"`).
    println!("cargo:metadata=INCLUDE_DIR={}", info.include_dir.display());
    println!("cargo:metadata=LIB_DIR={}", info.lib_dir.display());
    for dll_dir in &info.dll_dirs {
        println!("cargo:metadata=DLL_DIR={}", dll_dir.display());
    }
    println!("cargo:metadata=VERSION={}", MKL_VERSION);
}
