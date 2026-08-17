//! # nuvai-mkl
//!
//! Safe, idiomatic Rust wrapper over Intel oneMKL **2026.1.0**.
//!
//! Modules mirror the oneMKL domains:
//!
//! | Domain | Module | Notes |
//! |---|---|---|
//! | BLAS | [`blas`] | dense level-1/3 ops |
//! | LAPACK | [`lapack`] | dense solves & factorizations |
//! | FFT | [`fft`] | 1D DFT via DFTI |
//! | VML | [`vml`] | vector math functions |
//! | VSL | [`vsl`] | random-number generation |
//! | Sparse (PARDISO) | [`pardiso`] | parallel direct solver |
//! | Sparse (DSS) | [`dss`] | lightweight direct solver |
//!
//! ## Backends
//!
//! On x86_64 Linux/Windows every domain runs on Intel oneMKL 2026.1.0. On
//! `aarch64-apple-darwin` (Apple Silicon) and `aarch64-unknown-linux-gnu`
//! (ARM64 Linux) Intel ships no oneMKL, so the backend is selected by `cfg`,
//! never silently (ADR-0003):
//!
//! | Target | BLAS/LAPACK | FFT/VML/VSL/PARDISO/DSS |
//! |---|---|---|
//! | `aarch64-apple-darwin` | Accelerate vecLib (or OpenBLAS via the `openblas` feature) | Accelerate vDSP/vForce/Sparse + `rand` |
//! | `aarch64-unknown-linux-gnu` | OpenBLAS (system `libopenblas-dev`) | [`ErrorKind::Unsupported`] |
//!
//! [`ErrorKind::Unsupported`]: error::ErrorKind::Unsupported

#![allow(clippy::too_many_arguments)]
// Every `unsafe` block below carries a `// SAFETY:` rationale, and the wrapper
// defines no `unsafe fn`, so these lints are clean today and deny regressions:
// a new `unsafe` block without a written justification, or an un-scoped `unsafe`
// operation creeping into a future `unsafe fn`, fails the build instead of
// passing review unnoticed.
#![deny(unsafe_op_in_unsafe_fn)]
// `undocumented_unsafe_blocks` is a Clippy restriction lint (there is no
// rustc-level equivalent yet), so it is namespaced accordingly.
#![deny(clippy::undocumented_unsafe_blocks)]

pub use nuvai_mkl_sys::MKL_VERSION;

/// The oneMKL version this crate was built against.
pub const VERSION: &str = MKL_VERSION;

pub mod blas;
pub mod dss;
pub mod error;
pub mod fft;
pub mod lapack;
pub mod layout;
pub mod pardiso;
pub mod vml;
pub mod vsl;

/// Convenience re-exports.
pub mod prelude {
    pub use crate::error::{Error, ErrorKind, Result};
    pub use crate::layout::{Layout, Transpose};
}
