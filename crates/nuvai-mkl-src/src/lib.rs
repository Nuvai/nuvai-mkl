//! # nuvai-mkl-src
//!
//! Acquires and links a numerical backend into your crate.
//!
//! On x86_64 Linux/Windows this crate locates Intel oneMKL **2026.1.0** — the system oneAPI
//! install via `MKLROOT`, or a conda-forge download (Linux, Windows) into a
//! shared cache — and emits the linker directives.
//!
//! On `aarch64-apple-darwin` (Apple Silicon) Intel ships no oneMKL, so the
//! crate instead emits the **Accelerate** framework link directive (default) or
//! **OpenBLAS** (`openblas` feature). On `aarch64-unknown-linux-gnu` it emits
//! **OpenBLAS** (the only backend there — no Accelerate). Selection is explicit
//! and queryable via [`backend`]; see [`Backend`].
//!
//! Any crate that transitively depends on this one links against the selected
//! backend (`mkl_rt` on Intel, `-framework Accelerate`/`-lopenblas` on Apple
//! Silicon, `-lopenblas` on Linux aarch64).
//!
//! ## Consuming this backend from your own crate
//!
//! Cargo propagates `cargo:rustc-link-lib`/`cargo:rustc-link-search` from a
//! build script to every link that depends on the emitting package, but
//! **not** `cargo:rustc-link-arg` — that one applies to the emitting package's
//! own targets. Everything this crate emits is therefore of the propagating
//! kind, which is what lets a plain dependency-only consumer link *and* run.
//! The exceptions are the arguments only the owner of a binary can supply — an
//! executable's runtime rpath, and the object that keeps the OpenMP runtime in
//! its DT_NEEDED against mold (#44) — and those are emitted from your own
//! `build.rs` by [`emit_binary_link_args`]:
//!
//! ```toml
//! [build-dependencies]
//! nuvai-mkl-src = { git = "…", rev = "…" }   # the revision nuvai-mkl resolves to
//! ```
//!
//! ```no_run
//! #![allow(clippy::needless_doctest_main)] // a build.rs really does have a `main`
//! fn main() {
//!     nuvai_mkl_src::emit_binary_link_args();
//! }
//! ```
//!
//! With the `static` feature it emits nothing — a statically linked binary has
//! no shared object to point an rpath at — so a static consumer needs no build
//! script and no build-dependency at all (#70).
//!
//! The acquisition above happens in the **build script**, not in this library:
//! it needs `ureq`/`zip`/`zstd`/`tar`/`sha2`, which are build-dependencies so
//! that they stay out of the runtime graph of every downstream build (#24). A
//! dependent build script reads the resolved paths back through
//! [`MklInfo::from_build_metadata`] rather than acquiring a second time, so the
//! expensive path — a conda-forge download — runs once per build graph.
//!
//! ## Platform support
//!
//! | Target | Backend |
//! |---|---|
//! | `x86_64-unknown-linux-gnu` | Intel oneMKL — download (conda-forge) or system |
//! | `x86_64-pc-windows-msvc` | Intel oneMKL — conda-forge `mkl` + `mkl-include` + `mkl-devel` + `llvm-openmp` + `tbb` (links `mkl_rt` → `mkl_rt.3.dll`; runtime DLLs `libiomp5md.dll`/`tbb12.dll` on `PATH`), or system oneAPI (`MKLROOT`; runtime DLL dir on `PATH`) |
//! | `x86_64-apple-darwin` | unsupported (Intel ended macOS oneMKL after 2023.2.0) |
//! | `aarch64-apple-darwin` (Apple Silicon) | Accelerate (default) or OpenBLAS |
//! | `aarch64-unknown-linux-gnu` | OpenBLAS (BLAS/LAPACK only; FFT/VML/VSL/sparse unsupported) |

// `acquire.rs` is deliberately absent: it needs the acquisition crates, and
// this is the target those must not reach (#24). The acquisition itself runs in
// `build.rs`, which includes both files; what the library exposes here is the
// version, the `MklInfo` shape, and the reader that rebuilds one from the
// metadata that build script published.
//
// `link_args.rs` is the one other module the build script does *not* include:
// it is the entry point a *dependent's* build script calls (#70), so it has to
// be compiled into this crate's library target — `nuvai-mkl/build.rs` is its
// first caller, and a downstream consumer's `build.rs` is the second.
include!("mkl_info.rs");
include!("backend.rs");
include!("link_args.rs");
