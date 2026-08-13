//! # nuvai-mkl-src
//!
//! Acquires Intel oneMKL **2026.1.0** and links it into your crate.
//!
//! This crate is the modern replacement for the abandoned `intel-mkl-src`
//! (frozen at MKL 2020.1): its build script locates MKL 2026.1.0 on the
//! system via `MKLROOT`/oneAPI, or downloads and extracts it from conda-forge
//! (Linux, Windows) into a shared cache, then emits the linker directives.
//!
//! Any crate that transitively depends on this one links against `mkl_rt`
//! (the single runtime-dispatch library covering the whole MKL surface).
//!
//! ## Platform support
//!
//! | Target | Status |
//! |---|---|
//! | `x86_64-unknown-linux-gnu` | ✅ download (conda-forge) or system |
//! | `x86_64-pc-windows-msvc` | ✅ download (conda-forge) or system |
//! | `x86_64-apple-darwin` | system only (NuGet wiring pending) |
//! | `aarch64-*` (Apple Silicon, Linux ARM) | ❌ Intel ships no MKL — use the `accelerate`/`openblas` fallback |

include!("acquire.rs");
