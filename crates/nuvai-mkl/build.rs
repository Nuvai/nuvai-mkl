//! Linker flags (and one compiled object) for `nuvai-mkl`'s own
//! test/example/bench binaries.
//!
//! Cargo's build-script linker output splits in two, and this file exists
//! because of which half this package can reach. `cargo:rustc-link-lib` /
//! `cargo:rustc-link-search` are applied to every link that depends on the
//! emitting package, so all of oneMKL's propagate from `nuvai-mkl-src`, the
//! sole `links = "mkl"` provider — including, since #70, the static group link,
//! which is a linker script reached through `-l`. `cargo:rustc-link-arg` is
//! scoped to the emitting package's *own* targets, so the arguments that cannot
//! be spelled as a library or a search path — an executable's runtime rpath,
//! `--no-as-needed`, and the object that keeps the OpenMP runtime in DT_NEEDED
//! against mold (#44) — have to come from whoever owns the binary.
//!
//! For the workspace's binaries that owner is this crate; for a downstream
//! crate's it is that crate's own build script. Both call the same
//! implementation, `nuvai_mkl_src::emit_binary_link_args`, so the arguments
//! cannot drift apart — before #70 this file hand-rolled them, and a consumer
//! that merely depended on `nuvai-mkl` received none of them: it linked (the
//! search path and `-lmkl_rt` did propagate) and then failed at load time with
//! `libmkl_rt.so.3: cannot open shared object file`.
//!
//! The aarch64 fallbacks carry no Intel MKL and need no OpenMP retention
//! object, so `MklInfo` is only ever read for the x86_64 Linux/Windows targets
//! — which are also the only ones where `nuvai-mkl-src`'s build script
//! publishes the metadata it reads.

fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    // Build scripts compile for and run on the *host*, so `#[cfg(...)]` here
    // would describe the host, not the crate being built. Dispatch on the
    // target triple Cargo exposes as `CARGO_CFG_TARGET_*` instead (the same
    // target-aware selection `nuvai-mkl-src` uses) so cross-compiling
    // `--target aarch64-unknown-linux-gnu` from an x86_64 host does not emit
    // Intel x86_64 directives.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS is set");
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH is set");

    // An invariant of *this* manifest, not of a consumer's: without the
    // dependency Cargo forwards no `DEP_MKL_*` — the metadata travels through a
    // normal `[dependencies]` edge only, never through `[build-dependencies]`,
    // which is what #70 established — so the acquisition and every propagating
    // directive would be missing too, and the symptom would be a test binary
    // that cannot load `libmkl_rt.so.3`, which reads as a cache problem. A
    // downstream crate may legitimately call the same helper with no such
    // dependency (the static path needs none), which is why the helper warns
    // there and this crate fails loudly here.
    if matches!(target_os.as_str(), "linux" | "windows")
        && target_arch == "x86_64"
        && nuvai_mkl_src::MklInfo::from_build_metadata().is_none()
    {
        panic!(
            "nuvai-mkl-src published no DEP_MKL_* metadata — it must stay in this crate's \
             [dependencies] for Cargo to forward it (#70)"
        );
    }
    nuvai_mkl_src::emit_binary_link_args();

    // aarch64-apple-darwin is the one target with something this crate's
    // binaries need that is not a link argument at all.
    if target_os == "macos" && target_arch == "aarch64" {
        // The Accelerate interleaved vDSP DFT (`vDSP_DFT_Interleaved_*`) used
        // by the FFT backend is API_AVAILABLE(macos(12.0)); rustc defaults
        // the aarch64-apple-darwin deployment target to 11.0, which would
        // strong-link those symbols and abort at load time on macOS 10.15/11
        // with "Symbol not found". Pin the minimum so nuvai-mkl's own
        // test/example/bench binaries load only on macOS 12.0+. `rustc-env`
        // only reaches *this* crate's own rustc invocation — it cannot set
        // a downstream consumer's deployment target, so a `cargo:warning`
        // (surfaced by Cargo for every crate the build touches, including
        // this one) is the only channel that reaches them: they must set
        // MACOSX_DEPLOYMENT_TARGET=12.0 themselves before linking (#32).
        println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET=12.0");
        println!(
            "cargo:warning=nuvai-mkl's FFT backend on Apple Silicon calls a macOS \
             12.0+ Accelerate symbol (vDSP_DFT_Interleaved_*). Binaries linking \
             nuvai-mkl must set MACOSX_DEPLOYMENT_TARGET=12.0 (or higher) themselves, \
             or they will abort at load time with \"Symbol not found\" on macOS 10.15/11."
        );
    }
}
