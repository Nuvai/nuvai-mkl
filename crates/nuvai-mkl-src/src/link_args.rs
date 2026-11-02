// The linker arguments a *binary* needs that Cargo cannot propagate from a
// build script, and the reader that rebuilds the MKL paths needed to emit them
// (#70).
//
// Cargo's build-script linker output is two things, and only one of them
// travels. `cargo:rustc-link-lib` / `cargo:rustc-link-search` are collected
// from every build script in the graph and applied to each final link, so a
// directive expressed as a library or a search path reaches a *downstream*
// binary for free — which is why `nuvai-mkl-src`, the sole `links = "mkl"`
// provider, owns all of them (`build.rs`). `cargo:rustc-link-arg` is scoped to
// the *emitting package's* own targets, so a link argument emitted by a crate
// that owns no binaries — or by one whose binaries are not the consumer's —
// reaches nothing at all. Before #70 every oneMKL directive was a link-arg
// from `nuvai-mkl/build.rs`, so a consumer that merely depended on `nuvai-mkl`
// linked (the search path and `-lmkl_rt` did propagate) but could not *run*:
// its own executable never received the runtime rpath, and neither it nor the
// static group link ever appeared on the consumer's link line.
//
// What is left for a binary owner after that split is genuinely per-binary
// rather than merely misplaced: an executable's DT_RUNPATH cannot be expressed
// as a library or a search path, and neither can `--no-as-needed` around a
// library that has to be kept precisely because nothing references it. So this
// module is a *library* entry point a build script calls, not a directive:
// `nuvai-mkl/build.rs` calls it for the workspace's test/example/bench
// binaries, and a downstream crate calls it from its own `build.rs`.
//
// Which brings in the second Cargo rule this module exists to absorb: the
// `DEP_MKL_*` metadata it reads arrives **only through a normal
// `[dependencies]` edge**, never through `[build-dependencies]`. A build script
// that depends on a `links` provider receives nothing — verified directly
// (#70): a package with `nuvai-mkl-src` in `[build-dependencies]` alone reads
// `DEP_MKL_LIB_DIR` as unset, and the same package with it in `[dependencies]`
// reads it. A downstream crate therefore needs it in both tables — the
// dependency for the metadata, the build-dependency for this function's code —
// which is what the module doc's example and the warning
// [`emit_binary_link_args`] prints when the metadata is missing both describe.
// (`nuvai-mkl` and `nuvai-mkl-sys` already carry it in both, for their own
// reasons: the propagating link directives reach the target graph through the
// normal edge, and their build scripts read the same metadata.)
//
// ```toml
// [dependencies]
// nuvai-mkl-src = { git = "…", rev = "…" }   # the revision nuvai-mkl resolves to
//
// [build-dependencies]
// nuvai-mkl-src = { git = "…", rev = "…" }
// ```
//
// ```no_run
// #![allow(clippy::needless_doctest_main)] // a build.rs really does have a `main`
// fn main() {
//     nuvai_mkl_src::emit_binary_link_args();
// }
// ```
//
// Static linking (ADR-0005) needs none of it — the code is in the binary, so
// there is no rpath to point at a shared object and no
// `libmkl_intel_thread.so.3` leaving `omp_*` undefined — and [`link_args_for`]
// returns nothing on that path. Since #72 that is the *default* on
// `x86_64-unknown-linux-gnu`, so the two manifest entries and the one line
// below are the price of the `dynamic` opt-in (MKL's own threading), not of
// using this crate at all; see the README's "Linking from another crate".
//
// Like `src/mkl_info.rs`, this file is `include!`d into the crate root by
// `lib.rs`, so the imports that file declares (`env`, `Path`, `PathBuf`) are
// already in scope here and must not be declared again — a second
// `use std::env;` in the same module is E0252.

/// Linker arguments a binary that links the selected backend must be given,
/// which no propagating directive can express (#70).
///
/// Returns the raw arguments to pass on as `cargo:rustc-link-arg=…`, in the
/// order they must appear on the link line. It is pure — the two
/// environment-derived inputs are parameters, not reads — so the decision is
/// testable on every platform, whatever the host happens to be; see the tests
/// below. [`emit_binary_link_args`] is the build-script-facing wrapper that
/// supplies them from the build environment.
///
/// * `info` — the install `nuvai-mkl-src`'s build script published, or `None`
///   when it could not be read. The Intel targets are the only ones that
///   acquire anything, so `None` there means the build script calling this has
///   no `[dependencies]` entry on `nuvai-mkl-src` — the edge the metadata
///   travels through — and [`emit_binary_link_args`] reports exactly that
///   rather than producing link arguments from nothing.
/// * `openblas_root` — `$OPENBLAS_ROOT`, for the aarch64-Linux fallback whose
///   OpenBLAS may live outside the system search path. Ignored everywhere else.
/// * `target_os` / `target_arch` — the *target* triple's components
///   (`CARGO_CFG_TARGET_OS` / `CARGO_CFG_TARGET_ARCH` in a build script). The
///   target, not the host: a build script is host-compiled and would otherwise
///   emit the host's directives for a cross-compiled target (the same split
///   `backend_for_target` documents).
pub fn link_args_for(
    info: Option<&MklInfo>,
    openblas_root: Option<&str>,
    target_os: &str,
    target_arch: &str,
) -> Vec<String> {
    match (target_os, target_arch) {
        // Intel oneMKL, statically linked (ADR-0005). Nothing to emit: the
        // archives are pulled in by the linker script `nuvai-mkl-src/build.rs`
        // publishes as a propagating `-l` input, and a static link has no
        // shared object to point an rpath at and no OpenMP runtime to keep
        // (the `mkl_sequential` threading layer needs none). This arm is what
        // makes a dependency-only consumer work with no build script of its
        // own — the case #70 is about, and since #72 the default one: a
        // consumer on this target reaches it by declaring nothing.
        ("linux", "x86_64") if info.is_some_and(|i| i.static_link) => Vec::new(),
        // Intel oneMKL, dynamically linked (`mkl_rt`). Everything Cargo can
        // propagate — the search paths, `-lmkl_rt`, `-liomp5`, `-ldl`,
        // `-lpthread`, `-lm` — is emitted by `nuvai-mkl-src/build.rs`; what
        // remains is the runtime rpath and the two ways a library that nothing
        // references is kept in DT_NEEDED.
        ("linux", "x86_64") => match info {
            None => Vec::new(), // reported by `emit_binary_link_args`
            Some(info) => {
                let mut args = vec![
                    // `libmkl_core.so.3` calls `log`/`exp`/`sin`/… without a
                    // DT_NEEDED on libm, so libm has to be in the executable's
                    // own DT_NEEDED. Nothing in the Rust objects references it
                    // (Rust's `f64` math is lowered to LLVM intrinsics), so
                    // plain `--as-needed` would drop it.
                    "-Wl,--no-as-needed,-lm,--as-needed".to_string(),
                    // The executable must find `libmkl_rt.so.3` at load time.
                    // A build script cannot put a DT_RUNPATH on a *dependent's*
                    // binary — the one directive that could is a link-arg, and
                    // link-args do not propagate — so this has to be emitted by
                    // whoever owns the binary (the module doc above).
                    format!("-Wl,-rpath,{}", info.lib_dir.display()),
                ];
                if let Some(omp) = &info.omp_lib_dir {
                    args.push(format!("-Wl,-rpath,{}", omp.display()));
                    // `libmkl_intel_thread.so.3` references `omp_*` without a
                    // DT_NEEDED on the OpenMP runtime, so `libiomp5.so` has to
                    // be in the executable's DT_NEEDED as well. See
                    // `build/force_runtime.c` for why the object below — not
                    // this flag — is what actually keeps it there under mold
                    // (#44). Both are emitted: the flag is what GNU ld and lld
                    // honour.
                    args.push("-Wl,--no-as-needed,-liomp5,--as-needed".to_string());
                }
                // Passed to the linker as a regular object rather than through
                // `-l`: an archive member is pulled in only when something
                // references a symbol it *defines*, and this object exists for
                // the symbols it leaves undefined. Compiled by
                // `nuvai-mkl-src/build.rs` and published as `DEP_MKL_FORCE_OBJ`,
                // since only the `links = "mkl"` provider can publish metadata
                // and every dependent — including a downstream consumer's build
                // script — needs the same object.
                if let Some(obj) = &info.force_obj {
                    args.push(obj.display().to_string());
                }
                args
            }
        },
        // aarch64-Linux fallback: OpenBLAS is the only backend. A distro
        // `libopenblas-dev` is on the system search path and needs nothing,
        // but one under a non-default prefix (a conda/pip install named by
        // `OPENBLAS_ROOT`) is found at link time only through the search path
        // `nuvai-mkl-src/build.rs` propagates, and needs this rpath to be
        // found at run time. The trim mirrors that build script's check: an
        // empty-but-set variable is not a prefix.
        ("linux", "aarch64") => match openblas_root {
            Some(root) if !root.trim().is_empty() => vec![format!("-Wl,-rpath,{root}/lib")],
            _ => Vec::new(),
        },
        // aarch64-apple-darwin: Accelerate is a framework in the SDK, and
        // Homebrew's OpenBLAS carries an absolute install_name, so its own
        // `-lopenblas` needs no rpath. Nothing per-binary remains.
        ("macos", "aarch64") => Vec::new(),
        // Intel oneMKL on Windows: no rpath — the loader resolves
        // `mkl_rt.3.dll` (and `libiomp5md.dll`/`tbb12.dll`) at process start
        // from `PATH` or beside the executable, neither of which a linker
        // argument can supply. The DLL directories are surfaced as
        // `cargo:warning`s by `emit_binary_link_args` instead.
        ("windows", "x86_64") => Vec::new(),
        // Reachable only from a target with no backend at all, which every
        // `compile_error!`/panic in `backend.rs` and the build scripts already
        // rejects. Empty rather than a panic: this is a *library* function, and
        // a caller that has a target-specific reason to ask gets no arguments
        // rather than a build failure from a helper.
        _ => Vec::new(),
    }
}

/// Whether this target acquires an MKL install whose paths a build script can
/// only read from `DEP_MKL_*` (i.e. it must depend on `nuvai-mkl-src` to have
/// them). The Intel targets are exactly those; every aarch64 target uses a
/// system/SDK framework or a name-resolved OpenBLAS.
fn acquires_mkl(target_os: &str, target_arch: &str) -> bool {
    matches!(
        (target_os, target_arch),
        ("linux", "x86_64") | ("windows", "x86_64")
    )
}

/// Emit the per-binary linker directives for the crate being built, from a
/// **build script** (#70).
///
/// This is the whole of what a downstream crate has to do — plus the pair of
/// manifest entries the module doc explains, since Cargo forwards a `links`
/// provider's `DEP_*` metadata only through a normal `[dependencies]` edge
/// while a build script can only *use* code from `[build-dependencies]`. With
/// `nuvai-mkl-src` in both, at the revision your `nuvai-mkl` dependency
/// resolves to, a `build.rs` is one call:
///
/// ```no_run
/// #![allow(clippy::needless_doctest_main)] // a build.rs really does have a `main`
/// fn main() {
///     nuvai_mkl_src::emit_binary_link_args();
/// }
/// ```
///
/// It reads the target from `CARGO_CFG_TARGET_OS`/`CARGO_CFG_TARGET_ARCH` and
/// the resolved install from `DEP_MKL_*`. Under the static link — the default
/// on `x86_64-unknown-linux-gnu` since #72 — it emits nothing, because a static
/// link needs no per-binary arguments: so this call, and both manifest entries
/// above, exist for the `dynamic` opt-in and can be dropped on the default
/// path.
///
/// # Panics
///
/// If the MKL library directory the metadata names no longer exists — the
/// published paths are a snapshot, and clearing `~/.cache/nuvai-mkl` does not
/// re-run acquisition. The message names the fix. Panicking here, in the
/// consumer's own build script, is deliberate: the alternative is a linker
/// error naming a missing `-lmkl_rt` that says nothing about the cache.
pub fn emit_binary_link_args() {
    // Build scripts compile for and run on the *host*, so the target has to
    // come from these rather than from `cfg!` — see `backend_for_target`.
    let target_os = env::var("CARGO_CFG_TARGET_OS")
        .expect("CARGO_CFG_TARGET_OS is set in a build script; emit_binary_link_args is for build.rs");
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")
        .expect("CARGO_CFG_TARGET_ARCH is set in a build script; emit_binary_link_args is for build.rs");

    let info = MklInfo::from_build_metadata();
    if let Some(info) = &info
        && !info.lib_dir.is_dir()
    {
        panic!(
            "the MKL library directory nuvai-mkl-src published, {}, no longer exists. \
             If the MKL cache was cleared, force acquisition to re-run with \
             `cargo clean -p nuvai-mkl-src` and rebuild.",
            info.lib_dir.display()
        );
    }

    let openblas_root = env::var("OPENBLAS_ROOT").ok();
    let args = link_args_for(
        info.as_ref(),
        openblas_root.as_deref(),
        &target_os,
        &target_arch,
    );
    for arg in args {
        println!("cargo:rustc-link-arg={arg}");
    }

    // The Intel targets link `mkl_rt`, whose shared objects live in the
    // acquisition cache: an executable that cannot be told where they are will
    // link and then fail at load time with `libmkl_rt.so.3: cannot open shared
    // object file`. That is silent until the binary runs, so say it here — and
    // name the *dependency* table, not the build-dependency one, because that
    // is the edge Cargo forwards the metadata through (see the module doc).
    if info.is_none() && acquires_mkl(&target_os, &target_arch) {
        println!(
            "cargo:warning=nuvai-mkl: this build script called \
             nuvai_mkl_src::emit_binary_link_args() but no DEP_MKL_* metadata is \
             available, so the runtime rpath and the OpenMP-retention object cannot be \
             emitted — the binary will link and then fail at start-up with \
             `libmkl_rt.so.3: cannot open shared object file`. Add `nuvai-mkl-src` to \
             [dependencies] (the only edge Cargo forwards a links provider's metadata \
             through; [build-dependencies] alone does not) at the revision your \
             nuvai-mkl dependency resolves to. On the default static link (no `dynamic` \
             feature) no per-binary link arguments are needed, and this call can be \
             removed."
        );
    }

    // Windows has no rpath: the loader resolves `mkl_rt.3.dll` (and
    // `libiomp5md.dll`/`tbb12.dll`) at process start from `PATH`, or from
    // beside the executable. Surface the directories so a consumer knows what
    // to prepend — the same warning `nuvai-mkl`'s own binaries get.
    if target_os == "windows"
        && let Some(info) = &info
    {
        for dll_dir in info.dll_dirs() {
            println!(
                "cargo:warning=MKL runtime DLLs in {} — add to PATH (or copy beside the exe) before cargo run/test",
                dll_dir.display()
            );
        }
    }
}

// Named rather than a plain `mod tests`: this file, `mkl_info.rs` and
// `backend.rs` are all included into the same module (`lib.rs`'s crate root),
// so a second `mod tests` would be E0428 there.
#[cfg(test)]
mod link_args_tests {
    use super::*;

    /// An `MklInfo` for the dynamic Intel path with the given extras. Fields
    /// this module never reads are filled with a placeholder.
    fn dynamic_info(omp_lib_dir: Option<&str>, force_obj: Option<&str>) -> MklInfo {
        MklInfo {
            include_dir: PathBuf::from("/mkl/include"),
            lib_dir: PathBuf::from("/mkl/lib"),
            omp_lib_dir: omp_lib_dir.map(PathBuf::from),
            dll_dirs: Vec::new(),
            static_link: false,
            force_obj: force_obj.map(PathBuf::from),
        }
    }

    /// #70's headline: a *static* consumer needs no per-binary link arguments
    /// at all, which is what lets a plain dependency-only crate link and run
    /// with no `build.rs` of its own. Asserted with every optional input
    /// present, so the arm cannot pass by having nothing to work with.
    #[test]
    fn static_linking_needs_no_per_binary_arguments() {
        let mut info = dynamic_info(Some("/omp/lib"), Some("/out/force_runtime.o"));
        info.static_link = true;

        assert!(
            link_args_for(Some(&info), Some("/openblas"), "linux", "x86_64").is_empty(),
            "a static link has no shared object to rpath and no OpenMP runtime to keep"
        );
    }

    /// The dynamic path still needs everything it needed before #70 — this
    /// half did *not* become a propagating directive, because none of these
    /// can be spelled as one.
    #[test]
    fn dynamic_linking_emits_rpath_libm_iomp5_and_the_force_object() {
        let info = dynamic_info(Some("/omp/lib"), Some("/out/force_runtime.o"));

        assert_eq!(
            link_args_for(Some(&info), None, "linux", "x86_64"),
            [
                "-Wl,--no-as-needed,-lm,--as-needed",
                "-Wl,-rpath,/mkl/lib",
                "-Wl,-rpath,/omp/lib",
                "-Wl,--no-as-needed,-liomp5,--as-needed",
                "/out/force_runtime.o",
            ]
        );
    }

    /// A system oneAPI install keeps its OpenMP runtime beside MKL and
    /// publishes no `omp_lib_dir`, so the two OpenMP-related arguments are
    /// absent — but the loader still has to find `libmkl_rt.so.3`.
    #[test]
    fn dynamic_linking_without_a_published_openmp_directory() {
        let info = dynamic_info(None, None);

        assert_eq!(
            link_args_for(Some(&info), None, "linux", "x86_64"),
            ["-Wl,--no-as-needed,-lm,--as-needed", "-Wl,-rpath,/mkl/lib"]
        );
    }

    /// No metadata (this crate absent from `[build-dependencies]`) must not
    /// invent an rpath from a `"."` or a placeholder: `emit_binary_link_args`
    /// warns instead, and the warning is the caller's only symptom.
    #[test]
    fn no_metadata_emits_nothing() {
        assert!(link_args_for(None, None, "linux", "x86_64").is_empty());
        assert!(link_args_for(None, Some("/openblas"), "linux", "x86_64").is_empty());
    }

    /// The aarch64-Linux fallback: an OpenBLAS in a non-default prefix needs
    /// the runtime rpath an environment variable names, and only then.
    #[test]
    fn linux_aarch64_rpaths_only_a_named_openblas_prefix() {
        assert_eq!(
            link_args_for(None, Some("/opt/openblas"), "linux", "aarch64"),
            ["-Wl,-rpath,/opt/openblas/lib"]
        );
        // Unset and set-but-empty are the same non-answer, matching the search
        // path emitted by `nuvai-mkl-src/build.rs`.
        assert!(link_args_for(None, None, "linux", "aarch64").is_empty());
        assert!(link_args_for(None, Some("   "), "linux", "aarch64").is_empty());
    }

    /// Every other supported target has nothing per-binary: Accelerate and a
    /// Homebrew OpenBLAS resolve at load time without help, and Windows has no
    /// rpath at all. Named explicitly so an arm added later cannot quietly
    /// start emitting arguments for them.
    #[test]
    fn the_remaining_targets_emit_nothing() {
        let info = dynamic_info(Some("/omp/lib"), Some("/out/force_runtime.o"));

        assert!(link_args_for(Some(&info), None, "macos", "aarch64").is_empty());
        assert!(link_args_for(Some(&info), None, "windows", "x86_64").is_empty());
    }

    /// The metadata is only readable on the targets that acquire an install;
    /// this is what decides whether a missing one is worth a warning.
    #[test]
    fn only_the_intel_targets_need_the_published_paths() {
        assert!(acquires_mkl("linux", "x86_64"));
        assert!(acquires_mkl("windows", "x86_64"));
        assert!(!acquires_mkl("linux", "aarch64"));
        assert!(!acquires_mkl("macos", "aarch64"));
    }
}
