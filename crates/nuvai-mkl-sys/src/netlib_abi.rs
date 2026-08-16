// Shared netlib CBLAS + Fortran-LAPACK FFI declarations.
//
// This fragment is `include!`d by both hand-written aarch64 surfaces —
// `aarch64.rs` (Apple Silicon, Accelerate) and `linux_aarch64.rs` (OpenBLAS) —
// because both backends implement the netlib CBLAS ABI (identical symbol names
// and calling convention to MKL) and expose LAPACK through the Fortran `_`
// entry points. A single declaration set keeps the two surfaces from drifting:
// `cblas_*` and `?gesv`/`?getrf` have exactly one source of truth.
//
// Type layouts: `c_int` = i32, CBLAS enums = `u32` (non-Windows), matching the
// non-Windows path used by `nuvai-mkl::blas` and `nuvai-mkl::lapack`.

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
// LAPACK (both backends expose only the Fortran `_` entry points; no LAPACKE)
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
