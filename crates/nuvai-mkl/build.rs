//! Linker flags for `nuvai-mkl`'s own test/example/bench binaries.
//!
//! `rustc-link-arg` is scoped to the targets of the package that emits it, so
//! the runtime rpath (and the `--no-as-needed` around libm) must be emitted
//! here — `nuvai-mkl-src` emits the *propagating* directives (`rustc-link-lib`
//! / `rustc-link-search`), which reach every downstream link, but its
//! `rustc-link-arg` would only ever apply to its own (nonexistent) binaries.

fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    let info = nuvai_mkl_src::locate();
    let lib_dir = info.lib_dir.display();

    #[cfg(target_os = "linux")]
    {
        // conda-forge's `libmkl_core.so.3` references `log`/`exp`/`sin`/… but
        // does not declare a DT_NEEDED on libm, so keep libm in the final link.
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-lm,--as-needed");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
}
