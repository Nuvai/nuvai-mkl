//! BLAS (Basic Linear Algebra Subprograms) via the CBLAS interface.
//!
//! Dense linear-algebra primitives. Every routine takes slices and leading
//! dimensions; the caller is responsible for sizing buffers to the BLAS
//! convention (documented per function). MKL performs no bounds checking, so
//! an under-sized slice is undefined behaviour exactly as in C.

use crate::error::Result;
use crate::layout::{Layout, Transpose};

#[inline]
fn cblas_layout(layout: Layout) -> u32 {
    match layout {
        Layout::RowMajor => nuvai_mkl_sys::CblasRowMajor,
        Layout::ColMajor => nuvai_mkl_sys::CblasColMajor,
    }
}

#[inline]
fn cblas_trans(trans: Transpose) -> u32 {
    match trans {
        Transpose::NoTrans => nuvai_mkl_sys::CblasNoTrans,
        Transpose::Trans => nuvai_mkl_sys::CblasTrans,
        Transpose::ConjTrans => nuvai_mkl_sys::CblasConjTrans,
    }
}

/// `C := alpha * op(A) * op(B) + beta * C`, single precision.
///
/// `A` is `m × k` (or `k × m` when transposed), `B` is `k × n` (or `n × k`),
/// `C` is `m × n`. `lda`/`ldb`/`ldc` are the leading dimensions.
#[allow(clippy::too_many_arguments)]
pub fn sgemm(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    b: &[f32],
    ldb: i32,
    beta: f32,
    c: &mut [f32],
    ldc: i32,
) -> Result<()> {
    unsafe {
        nuvai_mkl_sys::cblas_sgemm(
            cblas_layout(layout),
            cblas_trans(transa),
            cblas_trans(transb),
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr(),
            ldc,
        );
    }
    Ok(())
}

/// `C := alpha * op(A) * op(B) + beta * C`, double precision.
#[allow(clippy::too_many_arguments)]
pub fn dgemm(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &[f64],
    ldb: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
) -> Result<()> {
    unsafe {
        nuvai_mkl_sys::cblas_dgemm(
            cblas_layout(layout),
            cblas_trans(transa),
            cblas_trans(transb),
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr(),
            ldc,
        );
    }
    Ok(())
}

/// `y := alpha * x + y`, single precision.
pub fn saxpy(n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) -> Result<()> {
    unsafe {
        nuvai_mkl_sys::cblas_saxpy(n, alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy);
    }
    Ok(())
}

/// `y := alpha * x + y`, double precision.
pub fn daxpy(n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32) -> Result<()> {
    unsafe {
        nuvai_mkl_sys::cblas_daxpy(n, alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy);
    }
    Ok(())
}

/// `dot := xᵀ · y`, single precision.
pub fn sdot(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    unsafe { nuvai_mkl_sys::cblas_sdot(n, x.as_ptr(), incx, y.as_ptr(), incy) }
}

/// `dot := xᵀ · y`, double precision.
pub fn ddot(n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
    unsafe { nuvai_mkl_sys::cblas_ddot(n, x.as_ptr(), incx, y.as_ptr(), incy) }
}

/// `x := alpha * x`, single precision.
pub fn sscal(n: i32, alpha: f32, x: &mut [f32], incx: i32) -> Result<()> {
    unsafe {
        nuvai_mkl_sys::cblas_sscal(n, alpha, x.as_mut_ptr(), incx);
    }
    Ok(())
}

/// `x := alpha * x`, double precision.
pub fn dscal(n: i32, alpha: f64, x: &mut [f64], incx: i32) -> Result<()> {
    unsafe {
        nuvai_mkl_sys::cblas_dscal(n, alpha, x.as_mut_ptr(), incx);
    }
    Ok(())
}
