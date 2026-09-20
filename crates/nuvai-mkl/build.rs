//! Linker flags (and one compiled object) for `nuvai-mkl`'s own
//! test/example/bench binaries.
//!
//! `rustc-link-arg` is scoped to the targets of the package that emits it, so
//! the runtime rpath (and the `--no-as-needed` around libm) must be emitted
//! here — `nuvai-mkl-src` emits the *propagating* directives (`rustc-link-lib`
//! / `rustc-link-search`), which reach every downstream link, but its
//! `rustc-link-arg` would only ever apply to its own (nonexistent) binaries.
//!
//! The x86_64 Linux arm additionally compiles `build/force_runtime.c` and
//! links the resulting object into those binaries. MKL's shared objects need
//! `omp_*` and `log`/`exp`/… in the process-global scope but declare no
//! DT_NEEDED for them, and no linker flag reliably forces them in — see that
//! file for the full reasoning.
//!
//! `static` (ADR-0005, x86_64 Linux only) takes the same "this crate owns the
//! binaries" fact and applies it in the opposite direction: instead of
//! working around what a dynamic `.so` leaves unresolved, `emit_intel_mkl_static`
//! emits the actual `-l`/group-link arguments for the static archives —
//! `nuvai-mkl-src` cannot, for exactly the `rustc-link-arg`-scoping reason
//! above.
//!
//! The aarch64 fallbacks carry no Intel MKL, so [`mkl_info`] is only ever
//! called for the x86_64 Linux/Windows targets — which are also the only ones
//! where `nuvai-mkl-src`'s build script publishes the metadata it reads.

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
        ("macos", "aarch64") => {
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
        // Intel x86_64 Linux: keep libm and the OpenMP runtime in the final
        // link (conda's `libmkl_core.so.3` references `log`/`exp`/`sin`/… and
        // `libmkl_intel_thread.so.3` references `omp_*`, both without a
        // DT_NEEDED of their own), and add the runtime rpath to the conda MKL
        // shared objects.
        //
        // Two mechanisms, because no single one covers every linker. The flags
        // below are what GNU ld and lld honour; mold records `-l` inputs by
        // *resolved file* and discards a repeat mention — however it is spelled
        // (`-liomp5`, `-l:libiomp5.so`, or a bare path) — before it ever
        // consults `--no-as-needed`, then prunes the library as unreferenced
        // (issue #44: the test binary aborted at load time with
        // `undefined symbol: omp_in_parallel`). What mold does honour is an
        // undefined symbol from a regular object file, which is what
        // `build/force_runtime.c` supplies.
        ("linux", "x86_64") => {
            let info = mkl_info();
            // Static linking (`static` feature — ADR-0005) needs none of the
            // dynamic-path workarounds below: every one of them exists because
            // a *dynamic* `.so` can leave a symbol undefined for the loader to
            // resolve later (no DT_NEEDED for libm/OpenMP), which has no
            // equivalent failure mode for a static archive — every symbol
            // resolves at link time or the link fails outright. There is also
            // no shared object to add an rpath to: the code is in the binary.
            // It needs its own, different linking, done in full by
            // `emit_intel_mkl_static` below.
            if info.static_link {
                emit_intel_mkl_static(&info);
                return;
            }
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-lm,--as-needed");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", info.lib_dir.display());
            if let Some(omp) = &info.omp_lib_dir {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", omp.display());
                // `libmkl_intel_thread.so.3` references `omp_*` symbols without
                // a DT_NEEDED on the OpenMP runtime, so keep `libiomp5.so` in
                // the final link the same way libm is kept (see above): the
                // test/example objects never reference it directly, so plain
                // `--as-needed` would drop it from DT_NEEDED.
                println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-liomp5,--as-needed");
            }

            // Pass the compiled objects to the linker directly rather than via
            // `rustc-link-lib=static`: an archive member is only pulled in when
            // something references a symbol it *defines*, and nothing does —
            // this object exists for the symbols it leaves undefined.
            let force_c = std::path::PathBuf::from(
                std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"),
            )
            .join("build/force_runtime.c");
            println!("cargo:rerun-if-changed={}", force_c.display());
            for obj in cc::Build::new().file(&force_c).compile_intermediates() {
                println!("cargo:rustc-link-arg={}", obj.display());
            }
        }
        // Intel x86_64 Windows: no rpath; the loader resolves `mkl_rt.3.dll`
        // (and `libiomp5md.dll` / `tbb12.dll`) at process start from PATH (or
        // the exe's directory). Surface the runtime DLL directories so CI can
        // prepend them to PATH and local dev knows where to add them.
        ("windows", "x86_64") => {
            let info = mkl_info();
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

/// Statically link Intel oneMKL on `x86_64-unknown-linux-gnu` (`static`
/// feature — ADR-0005), into `nuvai-mkl`'s own test/example/bench binaries.
///
/// This lives here, in the crate that owns those binaries, and not in
/// `nuvai-mkl-src` — even though `nuvai-mkl-src` is the crate that acquires
/// the archives and is the sole `links = "mkl"` provider for everything else.
/// `rustc-link-arg` is scoped to the *emitting* package's own targets (see the
/// module doc above and the OpenBLAS aarch64 rpath note in
/// `nuvai-mkl-src/build.rs`), and the group-link directives below are
/// `rustc-link-arg`, not the propagating `rustc-link-lib`/`rustc-link-search`
/// — so emitting them from `nuvai-mkl-src`, which owns no binaries, would
/// silently reach nothing. (An earlier revision of this feature did exactly
/// that: the build succeeded because acquisition and search-path setup both
/// worked, but every real link against `nuvai-mkl`'s own binaries failed with
/// undefined MKL symbols, because the group-link arguments never reached
/// their link command at all.) `nuvai-mkl-sys` has no binaries either — it
/// too would need a downstream owner, but this crate is that owner for
/// both, since it depends on `nuvai-mkl-sys` and both link into the same
/// binaries.
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
/// `pthread`/`dl`/`m` are appended as plain `-l` flags in this *same*
/// `rustc-link-arg` invocation, immediately after `--end-group`, rather than
/// as separate `cargo:rustc-link-lib=dylib=…` directives: Cargo does not
/// promise those land after the group on the final link line (they are a
/// different directive kind, collected and ordered independently), and `ld`/
/// `mold` resolve a library's symbols only against references already seen
/// earlier on the command line — MKL's static archives call libm's
/// `log`/`sqrt`/`ceil`/… internally, so libm must follow them, not merely
/// appear somewhere in the link. Keeping all of it in one `-Wl,`-prefixed
/// argument guarantees the order Cargo passes to the linker matches the order
/// written here.
fn emit_intel_mkl_static(info: &nuvai_mkl_src::MklInfo) {
    println!(
        "cargo:rustc-link-arg=-Wl,--start-group,-l:libmkl_intel_lp64.a,-l:libmkl_sequential.a,\
         -l:libmkl_core.a,--end-group,-lpthread,-ldl,-lm"
    );
    // `rustc-link-arg` is a raw argument, not a search path — the linker still
    // needs to be told where `libmkl_*.a` live. `nuvai-mkl-src` emits this
    // `native=` search directory too via the propagating `rustc-link-search`
    // (unlike `rustc-link-arg`, that one does reach downstream crates), but
    // repeating it here is what makes this function's own `-l:libmkl_*.a`
    // arguments resolvable without relying on an emission order between two
    // different crates' build scripts.
    println!("cargo:rustc-link-search=native={}", info.lib_dir.display());
}

/// The MKL install `nuvai-mkl-src` acquired and published to this build script
/// through `DEP_MKL_*` (#24).
///
/// This script used to call `nuvai_mkl_src::locate()` and re-derive the paths
/// itself. Acquisition now lives entirely in `nuvai-mkl-src`'s build script —
/// which runs first, and publishes what it found — so re-running it here would
/// repeat a conda-forge download that has already happened, once per dependent.
fn mkl_info() -> nuvai_mkl_src::MklInfo {
    let info = nuvai_mkl_src::MklInfo::from_build_metadata().expect(
        "nuvai-mkl-src published no DEP_MKL_* metadata — it must stay in this crate's \
         [build-dependencies] for Cargo to forward it",
    );
    // Checked here rather than in `from_build_metadata`, which must stay a pure
    // parse of what was published.
    //
    // The paths are a *snapshot*: `nuvai-mkl-src`'s build script records its
    // fingerprint over its own sources and the env vars, not over
    // `~/.cache/nuvai-mkl`, so clearing the cache does not re-run the
    // acquisition. Re-deriving the paths here (which is what calling `locate()`
    // used to do) was self-healing for that case; reading them back is not, so
    // the failure is moved from the linker's `cannot find -lmkl_rt` to a message
    // that says what to do.
    if !info.lib_dir.is_dir() {
        panic!(
            "the MKL library directory nuvai-mkl-src published, {}, no longer exists. \
             If the MKL cache was cleared, force acquisition to re-run with \
             `cargo clean -p nuvai-mkl-src` and rebuild.",
            info.lib_dir.display()
        );
    }
    info
}
