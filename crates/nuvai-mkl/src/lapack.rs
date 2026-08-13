//! LAPACK dense solvers and factorizations via the LAPACKE C interface.

use crate::error::{Error, Result};
use crate::layout::Layout;

#[inline]
fn lapacke_layout(layout: Layout) -> i32 {
    match layout {
        Layout::RowMajor => nuvai_mkl_sys::LAPACK_ROW_MAJOR as i32,
        Layout::ColMajor => nuvai_mkl_sys::LAPACK_COL_MAJOR as i32,
    }
}

/// Solve `A * X = B` for a general (non-symmetric) single-precision matrix.
///
/// `a` is `n × n` in `layout` order (`lda ≥ n`), `b` is `n × nrhs` (`ldb ≥ n`),
/// and `ipiv` must have length `n`. On success `a` is overwritten by its LU
/// factorization and `b` by the solution `X`.
pub fn sgesv(
    layout: Layout,
    n: i32,
    nrhs: i32,
    a: &mut [f32],
    lda: i32,
    ipiv: &mut [i32],
    b: &mut [f32],
    ldb: i32,
) -> Result<()> {
    let info = unsafe {
        nuvai_mkl_sys::LAPACKE_sgesv(
            lapacke_layout(layout),
            n,
            nrhs,
            a.as_mut_ptr(),
            lda,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            ldb,
        )
    };
    if info != 0 {
        return Err(Error::mkl(info, "LAPACKE_sgesv"));
    }
    Ok(())
}

/// Solve `A * X = B` for a general double-precision matrix.
pub fn dgesv(
    layout: Layout,
    n: i32,
    nrhs: i32,
    a: &mut [f64],
    lda: i32,
    ipiv: &mut [i32],
    b: &mut [f64],
    ldb: i32,
) -> Result<()> {
    let info = unsafe {
        nuvai_mkl_sys::LAPACKE_dgesv(
            lapacke_layout(layout),
            n,
            nrhs,
            a.as_mut_ptr(),
            lda,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            ldb,
        )
    };
    if info != 0 {
        return Err(Error::mkl(info, "LAPACKE_dgesv"));
    }
    Ok(())
}

/// LU factorization of a general single-precision `m × n` matrix `a`
/// (no pivoting applied yet — returns the factorization and pivot vector).
pub fn sgetrf(
    layout: Layout,
    m: i32,
    n: i32,
    a: &mut [f32],
    lda: i32,
    ipiv: &mut [i32],
) -> Result<()> {
    let info = unsafe {
        nuvai_mkl_sys::LAPACKE_sgetrf(
            lapacke_layout(layout),
            m,
            n,
            a.as_mut_ptr(),
            lda,
            ipiv.as_mut_ptr(),
        )
    };
    if info != 0 {
        return Err(Error::mkl(info, "LAPACKE_sgetrf"));
    }
    Ok(())
}

/// LU factorization of a general double-precision `m × n` matrix `a`.
pub fn dgetrf(
    layout: Layout,
    m: i32,
    n: i32,
    a: &mut [f64],
    lda: i32,
    ipiv: &mut [i32],
) -> Result<()> {
    let info = unsafe {
        nuvai_mkl_sys::LAPACKE_dgetrf(
            lapacke_layout(layout),
            m,
            n,
            a.as_mut_ptr(),
            lda,
            ipiv.as_mut_ptr(),
        )
    };
    if info != 0 {
        return Err(Error::mkl(info, "LAPACKE_dgetrf"));
    }
    Ok(())
}
