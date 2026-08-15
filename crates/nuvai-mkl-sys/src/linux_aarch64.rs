// Hand-written FFI surface for `aarch64-unknown-linux-gnu`.
//
// Intel ships no oneMKL build for `aarch64-unknown-linux-gnu`, so
// `nuvai-mkl-sys` cannot generate bindings from the oneMKL headers on this
// target. Instead this module declares the small, bounded set of C symbols the
// OpenBLAS backend calls — netlib CBLAS and the Fortran LAPACK `_` entry
// points — directly against OpenBLAS. The Intel x86_64 bindgen output is
// untouched, and no second bindgen pass is run over the OpenBLAS headers
// (ADR-0003, decision 3).
//
// OpenBLAS covers only BLAS/LAPACK on this target. There is no vDSP/vForce/
// Sparse/SparseSolve equivalent, so this surface deliberately omits those
// symbols; the safe wrapper returns `ErrorKind::Unsupported` for FFT/VML/VSL/
// PARDISO/DSS rather than linking anything absent here (never degrade
// silently).
//
// The CBLAS symbols are re-exported under the *same names* as the Intel
// bindgen output because both MKL and OpenBLAS implement the netlib CBLAS
// ABI; the safe wrapper's `blas` module therefore needs no per-backend
// dispatch (see ADR-0003, decision 4). LAPACK is exposed through the Fortran
// `_` entry points (`SUBROUTINE`s returning `void`, status via `*info`), the
// same ABI as the Apple Silicon Accelerate surface.
//
// Type layouts: `c_int` = i32, CBLAS enums = `u32` (non-Windows), matching the
// non-Windows path used by `nuvai-mkl::blas` and `nuvai-mkl::lapack`.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(improper_ctypes)]
#![allow(clippy::all)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_imports)]

use std::os::raw::c_int;

// ---------------------------------------------------------------------------
// Complex types (layout-identical to the MKL `MKL_Complex*` structs)
// ---------------------------------------------------------------------------

/// Single-precision complex, interleaved (`real` then `imag`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MKL_Complex8 {
    pub real: f32,
    pub imag: f32,
}

/// Double-precision complex, interleaved (`real` then `imag`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MKL_Complex16 {
    pub real: f64,
    pub imag: f64,
}

// ---------------------------------------------------------------------------
// CBLAS (netlib CBLAS ABI — identical constants and calling convention to MKL)
// ---------------------------------------------------------------------------

pub const CblasRowMajor: u32 = 101;
pub const CblasColMajor: u32 = 102;
pub const CblasNoTrans: u32 = 111;
pub const CblasTrans: u32 = 112;
pub const CblasConjTrans: u32 = 113;

// ---------------------------------------------------------------------------
// LAPACK (OpenBLAS exposes only the Fortran `_` entry points; no LAPACKE)
// ---------------------------------------------------------------------------
//
// On LP64 targets `__CLPK_integer` is `int` (4 bytes). The `?gesv`/`?getrf`
// Fortran routines are `SUBROUTINE`s, so they return `void`; the status is
// written through the trailing `*info` argument.

unsafe extern "C" {
    // --- CBLAS (level 1 and 3; netlib ABI, symbol-compatible with MKL) ---
    pub fn cblas_sgemm(
        order: u32,
        transa: u32,
        transb: u32,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
    pub fn cblas_dgemm(
        order: u32,
        transa: u32,
        transb: u32,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f64,
        a: *const f64,
        lda: c_int,
        b: *const f64,
        ldb: c_int,
        beta: f64,
        c: *mut f64,
        ldc: c_int,
    );
    pub fn cblas_saxpy(
        n: c_int,
        alpha: f32,
        x: *const f32,
        incx: c_int,
        y: *mut f32,
        incy: c_int,
    );
    pub fn cblas_daxpy(
        n: c_int,
        alpha: f64,
        x: *const f64,
        incx: c_int,
        y: *mut f64,
        incy: c_int,
    );
    pub fn cblas_sdot(n: c_int, x: *const f32, incx: c_int, y: *const f32, incy: c_int) -> f32;
    pub fn cblas_ddot(n: c_int, x: *const f64, incx: c_int, y: *const f64, incy: c_int) -> f64;
    pub fn cblas_sscal(n: c_int, alpha: f32, x: *mut f32, incx: c_int);
    pub fn cblas_dscal(n: c_int, alpha: f64, x: *mut f64, incx: c_int);

    // --- LAPACK Fortran `_` entry points (void `SUBROUTINE`s; `info` out-arg) ---
    pub fn sgesv_(
        n: *const c_int,
        nrhs: *const c_int,
        a: *mut f32,
        lda: *const c_int,
        ipiv: *mut c_int,
        b: *mut f32,
        ldb: *const c_int,
        info: *mut c_int,
    );
    pub fn dgesv_(
        n: *const c_int,
        nrhs: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        ipiv: *mut c_int,
        b: *mut f64,
        ldb: *const c_int,
        info: *mut c_int,
    );
    pub fn sgetrf_(
        m: *const c_int,
        n: *const c_int,
        a: *mut f32,
        lda: *const c_int,
        ipiv: *mut c_int,
        info: *mut c_int,
    );
    pub fn dgetrf_(
        m: *const c_int,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        ipiv: *mut c_int,
        info: *mut c_int,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Link-level smoke test: exercises a real OpenBLAS symbol so the P2 gate
    /// ("sys links clean on aarch64-linux") proves the `-lopenblas` link, not
    /// merely that the externs type-check.
    #[test]
    fn openblas_cblas_links() {
        let x = [1.0f32, 2.0, 3.0];
        let y = [4.0f32, 5.0, 6.0];
        // SAFETY: `x`/`y` have exactly 3 elements and `incx`/`incy` are 1, so
        // `cblas_sdot(3, …)` reads exactly the three elements of each array.
        let dot = unsafe { cblas_sdot(3, x.as_ptr(), 1, y.as_ptr(), 1) };
        assert!((dot - 32.0).abs() < 1e-6, "cblas_sdot = {dot}");
    }
}
