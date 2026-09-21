//! Build script for `nuvai-mkl-src`: locate oneMKL 2026.1.0 (Intel x86_64) or
//! emit the fallback linker directives (Accelerate / OpenBLAS).
//!
//! On `aarch64-apple-darwin`, Intel ships no oneMKL, so this script never calls
//! [`locate`]; it emits `-framework Accelerate` (default) or `-lopenblas`
//! (`openblas` feature). On `aarch64-unknown-linux-gnu` it emits `-lopenblas`.
//! The Intel x86_64 path is byte-identical to the pre-fallback behaviour and is
//! selected by [`backend_for_target`].
//!
//! This crate owns no binaries, which decides what it can emit. Cargo applies
//! `cargo:rustc-link-lib` / `cargo:rustc-link-search` to every link that
//! depends on the emitting package, but scopes `cargo:rustc-link-arg` to that
//! package's *own* targets — so everything a downstream binary needs has to be
//! expressed here as a library or a search path, and that is now true of every
//! oneMKL directive including the static group link, which is a linker script
//! reached through `-l` (#70). The arguments that genuinely cannot be
//! expressed that way — an executable's runtime rpath, and the object that
//! keeps the OpenMP runtime in its DT_NEEDED against mold (#44) — are emitted
//! by whoever owns the binary, through `nuvai_mkl_src::emit_binary_link_args`
//! (see `src/link_args.rs`). The one exception here is this crate's own
//! `tests/`, whose binaries *are* this package's targets: they get their
//! OpenBLAS rpath directly from the aarch64 arm below.
//!
//! Build scripts compile for and run on the *host*, so `#[cfg(...)]` here would
//! describe the host, not the crate being built. The backend is therefore
//! selected from the target triple Cargo exposes as `CARGO_CFG_TARGET_OS` /
//! `CARGO_CFG_TARGET_ARCH` / `CARGO_CFG_TARGET_ENV` — without this,
//! cross-compiling `--target aarch64-unknown-linux-gnu` from an x86_64 host
//! would select `IntelMkl` and emit x86_64 MKL directives for an ARM target.

// `mkl_info.rs` first: `acquire.rs` relies on the `env`/`Path`/`PathBuf`
// imports it declares, since both share this script's module — which is also
// where `fs` (from `acquire.rs`) comes from.
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

    // `static` (ADR-0005) is x86_64-unknown-linux-gnu **glibc** only: no
    // static-archive conda package exists for Windows, Accelerate/OpenBLAS have
    // no static form, and the conda linux-64 archives are glibc's. The
    // library-target `compile_error!` in `backend.rs` catches that for the crate
    // itself, but a build script is host-compiled and cannot `compile_error!`
    // for a *different* target — so a cross-compile must be caught here instead,
    // loudly, rather than silently linking dynamically. Which cross-compiles
    // those are, and why `HOST` is part of the test, is below.
    //
    // The `target_env` test is not redundant with `backend.rs`'s, which has
    // always required gnu: this check used to accept any `x86_64-unknown-linux-*`
    // and would then report "only supported on x86_64-unknown-linux-gnu" *for*
    // an `x86_64-unknown-linux-musl` target — a message that reads as
    // nonsense just before the `compile_error!` says the same thing
    // understandably.
    //
    // The `HOST` term answers the question #72 made this check unable to answer
    // alone: *who* asked. `static` now also arrives without the caller asking —
    // from the wrapper crates' `[build-dependencies]` edge, which Cargo matches
    // against the **host** — so on the supported host cross-compiling to a
    // target that cannot link statically (say `aarch64-unknown-linux-gnu`) this
    // unit carries the feature while no one requested it.
    //
    // Requiring the host to be *not* the supported triple makes the fire
    // condition exact rather than a heuristic, because it is then the only way
    // the feature can have been requested at all:
    //
    // * the `[dependencies]` edge that also carries `static` is matched against
    //   the *target*, and requests it only for the one target this check
    //   accepts — so it can never be the source of a panic here, and
    // * the `[build-dependencies]` edge can only fire on the supported host.
    //
    // So "host is not the supported triple and the target is not either" means
    // an explicit request, which is precisely what this must reject. On the
    // supported host every unsupported target is still rejected, one step later
    // and by the unit that can describe it truthfully: `backend.rs`'s
    // `compile_error!`, which the *target* unit hits because its `cfg` describes
    // the target rather than the host.
    let static_host = env::var("HOST").is_ok_and(|host| host == "x86_64-unknown-linux-gnu");
    let target_supports_static =
        target_os == "linux" && target_arch == "x86_64" && target_env.as_deref() == Some("gnu");
    if wants_static_link() && !static_host && !target_supports_static {
        // The libc component is part of the message because it is part of the
        // test: naming only `linux-x86_64` at an `x86_64-unknown-linux-musl`
        // target is the kind of message that sends a reader looking for a bug in
        // the wrong place.
        let target = match target_env.as_deref() {
            Some(env) => format!("{target_os}-{target_arch}-{env}"),
            None => format!("{target_os}-{target_arch}"),
        };
        panic!(
            "nuvai-mkl-src: `static` is only supported on x86_64-unknown-linux-gnu (this \
             build's target is {target}) — Windows has no static-archive conda package, the \
             conda linux-64 archives are glibc's, and Accelerate/OpenBLAS (the aarch64 \
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
            // targets, and this crate owns no binaries — whoever owns the
            // binary emits it, through
            // `nuvai_mkl_src::emit_binary_link_args` (see `src/link_args.rs`,
            // #70). macOS needs none anyway: the Homebrew dylib carries an
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

    // The one build artifact a *dependent's* build script needs a path to
    // rather than a copy of: compiled below for the dynamic Linux arm, and
    // published as `DEP_MKL_FORCE_OBJ` at the end of this function.
    let mut force_obj: Option<PathBuf> = None;

    match target_os {
        // Static linking (`static` feature — ADR-0005) has nothing to express
        // as a `-l` per archive: `libmkl_intel_lp64.a`, `libmkl_sequential.a`
        // and `libmkl_core.a` resolve symbols *circularly* — `mkl_core` calls
        // back into the interface and threading layers — so no linear order of
        // three separate libraries resolves, which is what a single-pass
        // linker command is. `--start-group`/`--end-group` is the documented
        // remedy, and it cannot be spelled as `rustc-link-lib`…
        //
        // …but a *linker script* can, and a linker script is just a `-l`
        // input: `GROUP ( a b c )` is the same repeated rescan semantics
        // written into a file, so `-lnuvai_mkl_static_group` carries it
        // through the propagating directive that reaches every dependent
        // (#70). Before this, the group was a `rustc-link-arg` emitted by
        // `nuvai-mkl` — which owns binaries, so it worked for the workspace's
        // own tests, and reached nothing in a crate that merely depended on
        // `nuvai-mkl`: acquisition and the search path both succeeded, and
        // every downstream link failed with undefined MKL symbols.
        //
        // `pthread`/`dl`/`m` — which `mkl_core` calls into — are deliberately
        // *not* named here. `rustc` already emits them on the tail of every
        // Linux link line, after the libraries, which is the order static
        // resolution requires; naming them again would only guarantee a
        // second, differently-ordered mention.
        "linux" if info.static_link => {
            // Written into `OUT_DIR`, which is stable for the whole build and
            // is where a build script may write. The `rustc-link-search` below
            // is a propagating directive, so a dependent's linker finds the
            // script at the same path this crate's own build does.
            let dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set"));
            write_static_group_script(&info, &dir);
            println!("cargo:rustc-link-search=native={}", dir.display());
            // `-bundle` is load-bearing, not a style choice. A `-l static=…`
            // directive defaults to *bundle*, which asks rustc to fold the
            // library into this crate's rlib so dependents need not find it
            // themselves — and rustc does that by opening the file and reading
            // it as an archive. A linker script is not one, and the build dies
            // at that point with "failed to add native library
            // …/libnuvai_mkl_static_group.a: Unsupported archive identifier".
            // (`+verbatim`, which skips resolving the name against the search
            // path, does not help — the read is for bundling, not for the
            // lookup.) With bundling off, rustc passes the library through to
            // the linker unexamined, which is what makes a text-file `-l` input
            // — the mechanism glibc's own `libc.so`/`libm.so` scripts use —
            // usable here at all. Verified against this crate as a library with
            // a downstream binary and against `mold` (#70).
            println!("cargo:rustc-link-lib=static:-bundle={STATIC_GROUP_LIB}");
        }
        "linux" => {
            // #72: the dynamic path is an explicit opt-in, and this is the one
            // place that knows which path was taken early enough to say so.
            //
            // `static` is what a consumer gets by declaring nothing, so the
            // only way to reach this arm is to have asked for `dynamic` (or to
            // depend on this crate directly rather than on the wrapper, which
            // is the one case with no target-gated edge to supply `static`) —
            // both are callers that want the threaded `mkl_rt` dispatcher and
            // now need to know what their own binaries owe it. Emitted as a
            // warning rather than an error because the *requirement* is real
            // but satisfiable outside this build (an rpath from the consumer's
            // own build script, or LD_LIBRARY_PATH where the binary runs), and
            // because there is nothing here that can look at a dependent's link
            // line to check whether either was done.
            //
            // The dynamic half of the issue is *not* silent any more: before
            // #72 the default path led here with no diagnostic and failed at
            // load time with `libmkl_rt.so.3: cannot open shared object file`,
            // naming the loader rather than this crate.
            if feature_enabled("DYNAMIC") {
                println!(
                    "cargo:warning=nuvai-mkl: linking the dynamic oneMKL runtime (`mkl_rt`) is \
                     the explicit `dynamic` opt-in (#72) — the threaded path, whose binaries \
                     must find `libmkl_rt.so.3` and the OpenMP runtime in the acquisition cache \
                     at load time. No directive this crate can emit puts a runtime search path \
                     on a *dependent's* binary (ADR-0006), so a binary that links this build \
                     from another package needs either a `build.rs` calling \
                     `nuvai_mkl_src::emit_binary_link_args()` (with `nuvai-mkl-src` in *both* \
                     `[dependencies]` and `[build-dependencies]`), or `LD_LIBRARY_PATH` set to \
                     the cache's `lib` directories wherever it runs. Without `dynamic` the \
                     default is the static link, which needs neither."
                );
            }
            println!("cargo:rustc-link-lib=dylib=mkl_rt");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=m");
            if let Some(omp) = &info.omp_lib_dir {
                println!("cargo:rustc-link-search=native={}", omp.display());
                println!("cargo:rustc-link-lib=dylib=iomp5");
                // Only emitted with the runtime it references on the link line:
                // the object's whole purpose is to leave `omp_*` undefined so
                // `libiomp5` is kept (see `force_runtime.c`), and emitted
                // without `-liomp5` — which is what a system oneAPI install
                // publishes no `omp_lib_dir` for — it would be an undefined
                // symbol nothing on the line could resolve.
                force_obj = force_runtime_object();
            }
            // The runtime rpath is deliberately *not* emitted here:
            // `cargo:rustc-link-arg` only applies to the emitting package's
            // own targets, and this crate owns none. The crate that owns a
            // binary emits it — `nuvai-mkl/build.rs` for the workspace's
            // binaries, `nuvai_mkl_src::emit_binary_link_args` for a
            // downstream consumer's (#70).
            //
            // `force_runtime.c` is compiled here rather than by whoever links
            // it because its *path* has to reach their build script, and
            // `cargo::metadata` — published below, from this crate's `links =
            // "mkl"` key — is the only channel that does.
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
    // Read back by the `nuvai-mkl-src`-dependent build scripts, which branch on
    // it to skip the dynamic-only linker tricks (rpath, `force_runtime.c`) that
    // a static archive link neither needs nor has any use for. The static group
    // link itself is *not* among them: it propagates from this build script as
    // a `-l` input, so a dependent is linked by the same directive this crate's
    // own build scripts emit (#70).
    println!(
        "cargo::metadata=STATIC_LINK={}",
        if info.static_link { "1" } else { "0" }
    );
    if let Some(obj) = &force_obj {
        println!("cargo::metadata=FORCE_OBJ={}", obj.display());
    }
}

/// The `-l` name of the generated static group's linker script: `-lnuvai_mkl_static_group`
/// resolves to `libnuvai_mkl_static_group.a`.
const STATIC_GROUP_LIB: &str = "nuvai_mkl_static_group";

/// The linker script that makes Intel's three static archives resolve as one
/// group, as a `-l` input Cargo propagates to dependents (#70, ADR-0005).
///
/// `GROUP` is not a convenience: the three archives resolve symbols
/// *circularly*, so whichever is listed last in a linear order still has
/// unresolved references into one listed earlier. `GROUP` tells the linker to
/// re-scan its members until nothing new resolves — the same semantics as
/// `--start-group`/`--end-group`, and the recipe Intel's own link-line advisor
/// prescribes for MKL. Absolute paths, because the script is read from
/// `OUT_DIR` rather than from the archives' directory.
///
/// Verified against both `ld` and `mold` (#70): each parses a `-l` input by its
/// contents, so a linker script reached through `-l` works exactly as a bare
/// `-Wl,--start-group,…` argument did — while propagating to dependents, which
/// an argument cannot.
fn static_group_script(info: &MklInfo) -> String {
    let lib = info.lib_dir.display();
    format!("GROUP ( {lib}/libmkl_intel_lp64.a {lib}/libmkl_sequential.a {lib}/libmkl_core.a )\n")
}

/// Write the group script into `OUT_DIR` and return its path.
fn write_static_group_script(info: &MklInfo, dir: &Path) -> PathBuf {
    let path = dir.join(format!("lib{STATIC_GROUP_LIB}.a"));
    if let Err(e) = fs::write(&path, static_group_script(info)) {
        panic!("nuvai-mkl-src: cannot write {}: {e}", path.display());
    }
    path
}

/// Compile `build/force_runtime.c`, for the dynamic `x86_64-unknown-linux-gnu`
/// link only, and return the object path.
///
/// The object is what keeps `libm`/`libiomp5` in the executable's DT_NEEDED
/// under mold: `-l` flags cannot, because mold records `-l` inputs by resolved
/// file and discards a repeat mention before consulting `--no-as-needed`
/// (#44). It is compiled *here* rather than by each linker because only this
/// crate's `links = "mkl"` metadata can hand a path to a dependent's build
/// script, and because compiling it once means every consumer — the wrapper's
/// own test binaries and a downstream crate's alike — uses the same object
/// instead of a second compilation of the same translation unit (#70).
fn force_runtime_object() -> Option<PathBuf> {
    let source = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("build/force_runtime.c");
    println!("cargo:rerun-if-changed={}", source.display());
    // `compile_intermediates` rather than `compile`: the resulting archive
    // would only be pulled in if something referenced a symbol it *defines*,
    // and this object exists for the symbols it leaves undefined — so the
    // object is passed to the linker as a regular input instead.
    cc::Build::new()
        .file(&source)
        .compile_intermediates()
        .into_iter()
        .next()
}
