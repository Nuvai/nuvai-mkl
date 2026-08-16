//! Linker flags for `nuvai-mkl`'s own test/example/bench binaries.
//!
//! `rustc-link-arg` is scoped to the targets of the package that emits it, so
//! the runtime rpath (and the `--no-as-needed` around libm) must be emitted
//! here — `nuvai-mkl-src` emits the *propagating* directives (`rustc-link-lib`
//! / `rustc-link-search`), which reach every downstream link, but its
//! `rustc-link-arg` would only ever apply to its own (nonexistent) binaries.
//!
//! The aarch64 fallbacks carry no Intel MKL, so `locate()` (which panics on
//! an aarch64 host) is only ever called for the x86_64 Linux/Windows targets.

fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    // Build scripts compile for and run on the *host*, so `#[cfg(...)]` here
    // would describe the host, not the crate being built. Dispatch on the
    // target triple Cargo exposes as `CARGO_CFG_TARGET_*` instead (the same
    // target-aware selection `nuvai-mkl-src` uses) so cross-compiling
    // `--target aarch64-unknown-linux-gnu` from an x86_64 host emits the
    // OpenBLAS rpath rather than Intel x86_64 directives.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS is set");
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH is set");

    match (target_os.as_str(), target_arch.as_str()) {
        // aarch64 fallbacks: no Intel MKL, no conda shared objects, so no
        // --no-as-needed. Accelerate lives in the SDK and a system OpenBLAS is
        // linked by name — but an OpenBLAS in a non-default prefix (OPENBLAS_ROOT,
        // e.g. conda/pip) is found at link time via the search path emitted by
        // nuvai-mkl-src, and the loader needs an rpath at run time to find it.
        ("linux", "aarch64") => {
            if let Ok(root) = std::env::var("OPENBLAS_ROOT")
                && !root.trim().is_empty()
            {
                // nuvai-mkl owns the test/example/bench binaries, so this
                // rustc-link-arg — unlike `nuvai-mkl-src`'s, which cannot
                // propagate — reaches exactly the links that need the rpath.
                println!("cargo:rustc-link-arg=-Wl,-rpath,{root}/lib");
            }
        }
        ("macos", "aarch64") => {}
        // Intel x86_64 Linux: keep libm in the final link (conda's
        // `libmkl_core.so.3` references `log`/`exp`/`sin`/… without a
        // DT_NEEDED) and add the runtime rpath to the conda MKL shared objects.
        ("linux", "x86_64") => {
            let info = nuvai_mkl_src::locate();
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-lm,--as-needed");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", info.lib_dir.display());
            if let Some(omp) = &info.omp_lib_dir {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", omp.display());
            }
        }
        // Intel x86_64 Windows: no rpath; the loader resolves `mkl_rt.3.dll`
        // (and `libiomp5md.dll` / `tbb12.dll`) at process start from PATH (or
        // the exe's directory). Surface the runtime DLL directories so CI can
        // prepend them to PATH and local dev knows where to add them.
        ("windows", "x86_64") => {
            let info = nuvai_mkl_src::locate();
            for dll_dir in info.dll_dirs() {
                println!(
                    "cargo:warning=MKL runtime DLLs in {} — add to PATH (or copy beside the exe) before cargo run/test",
                    dll_dir.display()
                );
            }
        }
        other => panic!("nuvai-mkl: unsupported target {other:?}"),
    }
}
