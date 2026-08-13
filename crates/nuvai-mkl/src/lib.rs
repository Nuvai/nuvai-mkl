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

#![allow(clippy::too_many_arguments)]

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
