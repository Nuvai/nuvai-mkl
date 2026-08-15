//! Build script for `nuvai-mkl-src`: locate oneMKL 2026.1.0 (Intel x86_64) or
//! emit the Apple Silicon fallback linker directives (Accelerate / OpenBLAS).
//!
//! On `aarch64-apple-darwin`, Intel ships no oneMKL, so this script never calls
//! [`locate`]; it emits `-framework Accelerate` (default) or `-lopenblas`
//! (`openblas` feature). The Intel x86_64 path is byte-identical to the
//! pre-fallback behaviour and is selected by [`backend`].

include!("src/acquire.rs");
include!("src/backend.rs");

fn main() {
    println!("cargo:rerun-if-env-changed=MKLROOT");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/acquire.rs");
    println!("cargo:rerun-if-changed=src/backend.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_OPENBLAS");

    // docs.rs has no network and no MKL; skip linking there.
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    match backend() {
        Backend::IntelMkl => emit_intel_mkl(),
        Backend::Accelerate => {
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:metadata=BACKEND=accelerate");
        }
        Backend::OpenBlas => {
            // OpenBLAS replaces only BLAS/LAPACK (vecLib); FFT (vDSP), VML
            // (vForce) and the sparse solvers (Sparse/SparseSolve) still call
            // Accelerate, so both must be linked on this path.
            println!("cargo:rustc-link-lib=dylib=openblas");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:metadata=BACKEND=openblas");
        }
    }
}

/// Emit the Intel oneMKL linker directives (x86_64 Linux/Windows).
fn emit_intel_mkl() {
    let info = locate();

    println!("cargo:rustc-link-search=native={}", info.lib_dir.display());

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=dylib=mkl_rt");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", info.lib_dir.display());
    }
    #[cfg(target_os = "windows")]
    {
        // conda win-64 mkl ships the import lib `mkl_rt.2.lib` + `mkl_rt.2.dll`.
        println!("cargo:rustc-link-lib=dylib=mkl_rt.2");
        println!("cargo:rustc-link-lib=dylib=user32");
    }

    // Informational metadata (also surfaced to downstream build scripts).
    println!("cargo:metadata=INCLUDE_DIR={}", info.include_dir.display());
    println!("cargo:metadata=LIB_DIR={}", info.lib_dir.display());
    println!("cargo:metadata=VERSION={}", MKL_VERSION);
}
