//! Generates raw FFI bindings from the oneMKL headers that `nuvai-mkl-src`
//! has acquired (and cached), via `nuvai_mkl_src::locate()`.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=MKLROOT");

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
