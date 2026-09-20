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

// `mkl_info.rs` first: `acquire.rs` relies on the `env`/`Path`/`PathBuf`
// imports it declares, since both share this script's module.
include!("src/mkl_info.rs");
include!("src/acquire.rs");
include!("src/backend.rs");

fn main() {
    println!("cargo:rerun-if-env-changed=MKLROOT");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/mkl_info.rs");
    println!("cargo:rerun-if-changed=src/acquire.rs");
    println!("cargo:rerun-if-changed=src/backend.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_OPENBLAS");
    println!("cargo:rerun-if-env-changed=OPENBLAS_ROOT");
    // Changing the mirror changes what is fetched, so the script must re-run.
    println!("cargo:rerun-if-env-changed=NUVAI_MKL_CONDA_BASE");

    // docs.rs has no network and no MKL; skip linking there.
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    // The backend is a property of the *target* being built, not of the host
    // this build script runs on (see the module docs above).
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS is set");
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH is set");
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").ok();

    let backend = backend_for_target(&target_os, &target_arch, target_env.as_deref())
        .unwrap_or_else(|e| panic!("{e}"));

    // `static` (ADR-0005) is x86_64-unknown-linux-gnu only: no static-archive
    // conda package exists for Windows, and Accelerate/OpenBLAS have no static
    // form. The library-target `compile_error!` in `backend.rs` catches this
    // for the crate itself, but a build script is host-compiled and cannot
    // `compile_error!` for a *different* target — so cross-compiling
    // `--target x86_64-pc-windows-msvc --features static` from any host must be
    // caught here instead, loudly, rather than silently linking dynamically.
    if wants_static_link() && !(target_os == "linux" && target_arch == "x86_64") {
        panic!(
            "nuvai-mkl-src: the `static` feature is only supported on \
             x86_64-unknown-linux-gnu (target is {target_os}-{target_arch}) — Windows has \
             no static-archive conda package, and Accelerate/OpenBLAS (the aarch64 \
             fallbacks) have no static form to switch to. Disable `static` for this target."
        );
    }

    match backend {
        Backend::IntelMkl => emit_intel_mkl(&target_os),
        Backend::Accelerate => {
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo::metadata=BACKEND=accelerate");
        }
        Backend::OpenBlas => {
            println!("cargo:rustc-link-lib=dylib=openblas");
            println!("cargo::metadata=BACKEND=openblas");
            // macOS: OpenBLAS replaces only BLAS/LAPACK (vecLib); FFT (vDSP),
            // VML (vForce) and the sparse solvers (Sparse/SparseSolve) still
            // call Accelerate, so both must be linked on this path.
            if target_os == "macos" {
                println!("cargo:rustc-link-lib=framework=Accelerate");
            }
            // Both aarch64 fallbacks need the search path, for different reasons.
            //
            // Linux-aarch64: OpenBLAS is the only backend and covers only
            // BLAS/LAPACK, so there is no Accelerate to link. A distro
            // `libopenblas-dev` lives in the system search path; an explicit
            // OPENBLAS_ROOT (e.g. a conda/pip install) contributes its `lib`
            // dir to the link search path.
            //
            // macOS: Homebrew's `openblas` is keg-only, so it is *not* symlinked
            // into `/opt/homebrew/lib` and a bare `-lopenblas` cannot resolve —
            // the explicit prefix is the only way to find it. This is not
            // optional polish: because this crate is a `[build-dependencies]`
            // entry of `nuvai-mkl` and `nuvai-mkl-sys`, the propagating
            // `rustc-link-lib` above also lands on *their build-script*
            // binaries, and `RUSTFLAGS` is not applied to host units — so a
            // caller-side `-L` (which is what CI passed) never reaches the link
            // that fails. Before this, `--features openblas` on
            // aarch64-apple-darwin died with `ld: library 'openblas' not found`
            // while linking the build script.
            //
            // The runtime *rpath* is deliberately not emitted here:
            // `cargo:rustc-link-arg` only applies to the emitting package's own
            // targets, and this crate owns no binaries — the crate that owns the
            // test/example binaries (`nuvai-mkl/build.rs`) emits the rpath
            // instead. macOS needs none anyway: the Homebrew dylib carries an
            // absolute install_name.
            let aarch64_fallback =
                (target_os == "linux" && target_arch == "aarch64") || target_os == "macos";
            if aarch64_fallback
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
        "linux" if info.static_link => emit_intel_mkl_static(),
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

    // What the build scripts of `nuvai-mkl` and `nuvai-mkl-sys` read back
    // through `MklInfo::from_build_metadata()` (#24), so the acquisition runs
    // once here rather than once per dependent. This crate declares
    // `links = "mkl"`, which is what makes Cargo forward them as `DEP_MKL_*`.
    //
    // The directive is `cargo::metadata=` — two colons. The single-colon form
    // this used before is not a directive at all: Cargo read it as *legacy*
    // metadata under the literal key `metadata`, so every line below was
    // recorded as `DEP_MKL_METADATA="BACKEND=…"` and none of these variables
    // existed. Nothing consumed them, which is why it went unnoticed.
    //
    // CI does not read `DLL_DIR_*` either, and should not: it prepends the
    // directories to `PATH` for the test process, which a build script cannot do
    // for anyone — hence the PowerShell step that globs the cache layout.
    println!("cargo::metadata=INCLUDE_DIR={}", info.include_dir.display());
    println!("cargo::metadata=LIB_DIR={}", info.lib_dir.display());
    if let Some(omp) = &info.omp_lib_dir {
        println!("cargo::metadata=OMP_LIB_DIR={}", omp.display());
    }
    // Indexed, not repeated: Cargo keeps only the *last* value for a repeated
    // metadata key, so the loop this replaces published exactly one directory —
    // whichever happened to be extracted last.
    //
    // The count is published alongside the indices and required by the reader.
    // Indices alone cannot distinguish "n directories" from "the first k of n",
    // and a list that is silently short is a missing `PATH` entry, which fails
    // as a DLL that will not load at run time rather than as a build error.
    println!("cargo::metadata=DLL_DIR_COUNT={}", info.dll_dirs.len());
    for (i, dll_dir) in info.dll_dirs.iter().enumerate() {
        println!("cargo::metadata=DLL_DIR_{i}={}", dll_dir.display());
    }
    println!("cargo::metadata=VERSION={}", MKL_VERSION);
    // Read back by `nuvai-mkl-sys`/`nuvai-mkl`'s build scripts to skip the
    // dynamic-only linker tricks (rpath, `force_runtime.c`) that a static
    // archive link neither needs nor has any use for (see
    // `emit_intel_mkl_static`'s doc comment).
    println!(
        "cargo::metadata=STATIC_LINK={}",
        if info.static_link { "1" } else { "0" }
    );
}

/// Statically link Intel oneMKL on `x86_64-unknown-linux-gnu` (`static`
/// feature — ADR-0005).
///
/// Three archives, in the order Intel's own link-line advisor specifies:
/// `mkl_intel_lp64` (the LP64 C interface, what `nuvai-mkl-sys`'s bindgen
/// output calls into), `mkl_sequential` (the single-threaded threading layer —
/// chosen over `mkl_intel_thread`/`mkl_tbb_thread` specifically because it
/// needs no OpenMP or TBB runtime, so the static link pulls in nothing beyond
/// libc/libm/libpthread/libdl, which every Linux target already has), and
/// `mkl_core` (the computational kernels both other layers call into).
///
/// The three resolve symbols *circularly* — `mkl_core` calls back into the
/// interface and threading layers as well as being called by them — which a
/// linear, single-pass linker command cannot resolve no matter what order the
/// archives are listed in: whichever one is listed last still has unresolved
/// references into one listed earlier. `--start-group`/`--end-group` (or the
/// equivalent repeated-pass behaviour) tells the linker to keep re-scanning
/// the archives inside the group until nothing new is pulled in, which is
/// what actually resolves them. This is not a Cargo `rustc-link-lib=static`
/// concern to route around — those three directives alone, in any order,
/// leave the group unresolved — so the group is emitted as a raw linker
/// argument instead.
///
/// No rpath and no `force_runtime.c`-style `--no-as-needed` trick: both exist
/// only for the *dynamic* path, where a `.so` can leave symbols undefined for
/// the loader to resolve later. A static archive has no "later" — every
/// symbol a linked object needs must resolve at link time or the link fails
/// outright — so an incomplete static link surfaces immediately as an
/// `undefined reference` rather than as a load-time abort, and there is no
/// runtime search path to set because nothing is loaded at run time.
fn emit_intel_mkl_static() {
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    println!("cargo:rustc-link-arg=-lmkl_intel_lp64");
    println!("cargo:rustc-link-arg=-lmkl_sequential");
    println!("cargo:rustc-link-arg=-lmkl_core");
    println!("cargo:rustc-link-arg=-Wl,--end-group");
    // Not part of the group: plain external dependencies the archives above
    // reference (pthread creation, dlopen for MKL's own runtime CPU dispatch,
    // libm transcendentals), each resolved by one ordinary shared library with
    // no circularity of its own.
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=m");
}
