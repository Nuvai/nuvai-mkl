//! Build script for `nuvai-mkl-src`: locate oneMKL 2026.1.0 and emit the
//! linker directives needed by any crate (and final binary) that depends on
//! this one.

include!("src/acquire.rs");

fn main() {
    println!("cargo:rerun-if-env-changed=MKLROOT");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/acquire.rs");

    // docs.rs has no network and no MKL; skip linking there.
    if env::var("DOCS_RS").is_ok() {
        return;
    }

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
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=dylib=mkl_rt");
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
