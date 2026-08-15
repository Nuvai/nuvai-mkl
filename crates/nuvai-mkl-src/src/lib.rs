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
//! **OpenBLAS** (`openblas` feature). Selection is explicit and queryable via
//! [`backend`]; see [`Backend`].
//!
//! Any crate that transitively depends on this one links against the selected
//! backend (`mkl_rt` on Intel, `-framework Accelerate`/`-lopenblas` on Apple
//! Silicon).
//!
//! ## Platform support
//!
//! | Target | Backend |
//! |---|---|
//! | `x86_64-unknown-linux-gnu` | Intel oneMKL — download (conda-forge) or system |
//! | `x86_64-pc-windows-msvc` | Intel oneMKL — conda-forge `mkl` + `mkl-include` + `mkl-devel` + `llvm-openmp` + `tbb` (links `mkl_rt` → `mkl_rt.3.dll`; runtime DLLs `libiomp5md.dll`/`tbb12.dll` on `PATH`), or system oneAPI (`MKLROOT`; runtime DLL dir on `PATH`) |
//! | `x86_64-apple-darwin` | unsupported (Intel ended macOS oneMKL after 2023.2.0) |
//! | `aarch64-apple-darwin` (Apple Silicon) | Accelerate (default) or OpenBLAS |
//! | `aarch64-unknown-linux-gnu` | not yet wired (plan: OpenBLAS) |

include!("acquire.rs");
include!("backend.rs");
