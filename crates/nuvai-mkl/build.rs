//! Linker flags for `nuvai-mkl`'s own test/example/bench binaries.
//!
//! `rustc-link-arg` is scoped to the targets of the package that emits it, so
//! the runtime rpath (and the `--no-as-needed` around libm) must be emitted
//! here — `nuvai-mkl-src` emits the *propagating* directives (`rustc-link-lib`
//! / `rustc-link-search`), which reach every downstream link, but its
//! `rustc-link-arg` would only ever apply to its own (nonexistent) binaries.
//!
//! On `aarch64-apple-darwin` there is no Intel MKL and no rpath to add: the
//! Accelerate framework lives in the SDK and OpenBLAS is linked by name, so
//! `locate()` (which panics on aarch64) is never called.

fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let backend = nuvai_mkl_src::backend();
        println!("cargo:metadata=BACKEND={}", nuvai_mkl_src::backend_tag(backend));
    }

    #[cfg(not(target_os = "macos"))]
    {
        let info = nuvai_mkl_src::locate();
        // Only used for the Linux rpath directives; Windows has no rpath and
        // instead surfaces the runtime DLL directory via `dll_dir()`.
        #[cfg(not(target_os = "windows"))]
        let lib_dir = info.lib_dir.display();

        #[cfg(target_os = "linux")]
        {
            // conda-forge's `libmkl_core.so.3` references `log`/`exp`/`sin`/… but
            // does not declare a DT_NEEDED on libm, so keep libm in the final link.
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-lm,--as-needed");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
        }
        #[cfg(target_os = "windows")]
        {
            // Windows has no rpath: the loader resolves `mkl_rt.3.dll` (and its
            // runtime dependencies `libiomp5md.dll` / `tbb12.dll`) at process
            // start from PATH (or the exe's directory). Surface the runtime DLL
            // directories so CI can prepend them to PATH and local dev knows
            // where to add them (or deploy the DLLs beside the executable).
            for dll_dir in info.dll_dirs() {
                println!(
                    "cargo:warning=MKL runtime DLLs in {} — add to PATH (or copy beside the exe) before cargo run/test",
                    dll_dir.display()
                );
                println!("cargo:metadata=DLL_DIR={}", dll_dir.display());
            }
        }
    }
}
