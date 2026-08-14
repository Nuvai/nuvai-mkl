//! LAPACK dense solvers and factorizations.
//!
//! On Intel targets this uses the LAPACKE C interface. On Apple Silicon
//! (`aarch64-apple-darwin`) Accelerate exposes only the Fortran `_` entry
//! points (`sgesv_`, `dgesv_`, `sgetrf_`, `dgetrf_`) — no LAPACKE — so the
//! same public functions dispatch to those and translate `Layout::RowMajor`
//! by transposing into column-major buffers, exactly as LAPACKE does
//! internally (see ADR-0003, decision 5).

use crate::error::{Error, Result};
use crate::layout::Layout;

/// Translate a [`Layout`] into the LAPACKE `matrix_layout` constant
/// (`LAPACK_ROW_MAJOR` = 101, `LAPACK_COL_MAJOR` = 102). Only defined where
/// LAPACKE exists (Intel oneMKL); the aarch64 backend uses the Fortran `_`
/// entry points directly.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[inline]
fn lapacke_layout(layout: Layout) -> i32 {
    match layout {
        Layout::RowMajor => nuvai_mkl_sys::LAPACK_ROW_MAJOR as i32,
        Layout::ColMajor => nuvai_mkl_sys::LAPACK_COL_MAJOR as i32,
    }
}

/// Validate a `?gesv` call's buffers before any pointer reaches C/Fortran.
///
/// The underlying routines write `lda·n` elements of `A` and `ldb·nrhs`
/// elements of `B` and `n` pivots, so an undersized slice is a heap
/// out-of-bounds write reachable through the safe API. Reject it here.
/// Row-major input is transposed through scratch first, so its minimum
/// sizes are the trailing-element bounds `(n-1)·lda + n` and
/// `(n-1)·ldb + nrhs` rather than the full leading-dimension products.
fn check_solve_dims(
    layout: Layout,
    n: i32,
    nrhs: i32,
    a_len: usize,
    lda: i32,
    ipiv_len: usize,
    b_len: usize,
    ldb: i32,
) -> Result<()> {
    if n <= 0 || nrhs <= 0 {
        return Err(Error::invalid("lapack: n and nrhs must be positive"));
    }
    if ipiv_len < n as usize {
        return Err(Error::invalid("lapack: ipiv too short"));
    }
    let (a_min, b_min) = match layout {
        Layout::ColMajor => {
            if lda < n {
                return Err(Error::invalid("lapack: lda < n"));
            }
            if ldb < n {
                return Err(Error::invalid("lapack: ldb < n"));
            }
            (lda as usize * n as usize, ldb as usize * nrhs as usize)
        }
        Layout::RowMajor => {
            if lda < n {
                return Err(Error::invalid("lapack: lda < n (row-major)"));
            }
            if ldb < nrhs {
                return Err(Error::invalid("lapack: ldb < nrhs (row-major)"));
            }
            (
                (n - 1) as usize * lda as usize + n as usize,
                (n - 1) as usize * ldb as usize + nrhs as usize,
            )
        }
    };
    if a_len < a_min {
        return Err(Error::invalid("lapack: a too short"));
    }
    if b_len < b_min {
        return Err(Error::invalid("lapack: b too short"));
    }
    Ok(())
}

/// Validate a `?getrf` call's buffers (writes `m × n` with leading
/// dimension `lda`, plus `min(m,n)` pivots).
fn check_factor_dims(
    layout: Layout,
    m: i32,
    n: i32,
    a_len: usize,
    lda: i32,
    ipiv_len: usize,
) -> Result<()> {
    if m <= 0 || n <= 0 {
        return Err(Error::invalid("lapack: m and n must be positive"));
    }
    if lda < m {
        return Err(Error::invalid("lapack: lda < m"));
    }
    if ipiv_len < m.min(n) as usize {
        return Err(Error::invalid("lapack: ipiv too short"));
    }
    let a_min = match layout {
        Layout::ColMajor => lda as usize * n as usize,
        Layout::RowMajor => (m - 1) as usize * lda as usize + n as usize,
    };
    if a_len < a_min {
        return Err(Error::invalid("lapack: a too short"));
    }
    Ok(())
}

/// Solve `A * X = B` for a general (non-symmetric) single-precision matrix.
///
/// `a` is `n × n` in `layout` order (`lda ≥ n`) and `ipiv` must have length
/// `n`. `b` is `n × nrhs`; its leading dimension is layout-dependent, mirroring
/// LAPACKE: `ldb ≥ n` for `ColMajor`, `ldb ≥ nrhs` for `RowMajor`. On success
/// `a` is overwritten by its LU factorization and `b` by the solution `X`.
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
    check_solve_dims(layout, n, nrhs, a.len(), lda, ipiv.len(), b.len(), ldb)?;
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        // SAFETY: `a`, `ipiv` and `b` cover at least the `lda·n`, `n` and
        // `ldb·nrhs` elements (layout-adjusted) that `LAPACKE_sgesv` reads and
        // writes, as enforced by `check_solve_dims` above. `n`, `lda` and `ldb`
        // are non-negative by the same check.
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
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        aarch64::sgesv(layout, n, nrhs, a, lda, ipiv, b, ldb)
    }
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
    check_solve_dims(layout, n, nrhs, a.len(), lda, ipiv.len(), b.len(), ldb)?;
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        // SAFETY: buffers cover the `lda·n` / `n` / `ldb·nrhs` elements that
        // `LAPACKE_dgesv` touches, per `check_solve_dims` above.
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
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        aarch64::dgesv(layout, n, nrhs, a, lda, ipiv, b, ldb)
    }
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
    check_factor_dims(layout, m, n, a.len(), lda, ipiv.len())?;
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        // SAFETY: `a` covers `lda·n` elements and `ipiv` covers `min(m,n)`,
        // per `check_factor_dims` above; `m`, `n`, `lda` are non-negative.
        let info = unsafe {
            nuvai_mkl_sys::LAPACKE_sgetrf(lapacke_layout(layout), m, n, a.as_mut_ptr(), lda, ipiv.as_mut_ptr())
        };
        if info != 0 {
            return Err(Error::mkl(info, "LAPACKE_sgetrf"));
        }
        Ok(())
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        aarch64::sgetrf(layout, m, n, a, lda, ipiv)
    }
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
    check_factor_dims(layout, m, n, a.len(), lda, ipiv.len())?;
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        // SAFETY: `a` covers `lda·n` elements and `ipiv` covers `min(m,n)`,
        // per `check_factor_dims` above.
        let info = unsafe {
            nuvai_mkl_sys::LAPACKE_dgetrf(lapacke_layout(layout), m, n, a.as_mut_ptr(), lda, ipiv.as_mut_ptr())
        };
        if info != 0 {
            return Err(Error::mkl(info, "LAPACKE_dgetrf"));
        }
        Ok(())
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        aarch64::dgetrf(layout, m, n, a, lda, ipiv)
    }
}

/// Accelerate (`aarch64-apple-darwin`) LAPACK backend.
///
/// Calls the Fortran `_` entry points directly. Column-major input is passed
/// through unchanged; row-major input is transposed to column-major, solved,
/// and the result transposed back — mirroring the transpose LAPACKE performs
/// internally on the Intel path, so the two backends agree on row-major
/// semantics.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod aarch64 {
    use super::*;

    /// Copy a row-major `rows × cols` matrix (`lda ≥ cols`) into a
    /// column-major `rows × cols` buffer (leading dimension `rows`).
    fn row_to_col_f32(src: &[f32], rows: i32, cols: i32, lda: i32, dst: &mut [f32]) {
        for i in 0..rows {
            let row_off = (i * lda) as usize;
            for j in 0..cols {
                dst[(j * rows + i) as usize] = src[row_off + j as usize];
            }
        }
    }

    fn row_to_col_f64(src: &[f64], rows: i32, cols: i32, lda: i32, dst: &mut [f64]) {
        for i in 0..rows {
            let row_off = (i * lda) as usize;
            for j in 0..cols {
                dst[(j * rows + i) as usize] = src[row_off + j as usize];
            }
        }
    }

    /// Copy a column-major `rows × cols` matrix (leading dimension `rows`)
    /// into a row-major `rows × cols` buffer (`ldb ≥ cols`).
    fn col_to_row_f32(src: &[f32], rows: i32, cols: i32, dst: &mut [f32], ldb: i32) {
        for i in 0..rows {
            for j in 0..cols {
                dst[(i * ldb + j) as usize] = src[(j * rows + i) as usize];
            }
        }
    }

    fn col_to_row_f64(src: &[f64], rows: i32, cols: i32, dst: &mut [f64], ldb: i32) {
        for i in 0..rows {
            for j in 0..cols {
                dst[(i * ldb + j) as usize] = src[(j * rows + i) as usize];
            }
        }
    }

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
        check_solve_dims(layout, n, nrhs, a.len(), lda, ipiv.len(), b.len(), ldb)?;
        let info = match layout {
            Layout::ColMajor => {
                let mut info = 0i32;
                // SAFETY: `a`/`ipiv`/`b` cover the `lda·n`/`n`/`ldb·nrhs`
                // elements `sgesv_` touches (validated above); `n`, `nrhs`,
                // `lda`, `ldb` and `info` are valid `const`/mutable refs.
                unsafe {
                    nuvai_mkl_sys::sgesv_(
                        &n,
                        &nrhs,
                        a.as_mut_ptr(),
                        &lda,
                        ipiv.as_mut_ptr(),
                        b.as_mut_ptr(),
                        &ldb,
                        &mut info,
                    );
                }
                info
            }
            Layout::RowMajor => {
                // `a` is n×n row-major (lda ≥ n); `b` is n×nrhs row-major (ldb ≥ nrhs).
                let mut a_cm = vec![0.0f32; (n * n) as usize];
                let mut b_cm = vec![0.0f32; (n * nrhs) as usize];
                row_to_col_f32(a, n, n, lda, &mut a_cm);
                row_to_col_f32(b, n, nrhs, ldb, &mut b_cm);
                let lda_cm = n;
                let ldb_cm = n;
                let mut info = 0i32;
                // SAFETY: `a_cm`/`b_cm` are freshly allocated to exactly
                // `n·n` and `n·nrhs`; `ipiv` covers `n` pivots and `info` is
                // a valid out-arg (validated above).
                unsafe {
                    nuvai_mkl_sys::sgesv_(
                        &n,
                        &nrhs,
                        a_cm.as_mut_ptr(),
                        &lda_cm,
                        ipiv.as_mut_ptr(),
                        b_cm.as_mut_ptr(),
                        &ldb_cm,
                        &mut info,
                    );
                }
                if info == 0 {
                    // Copy the factored A and the solution X back to row-major
                    // (LAPACKE parity: transpose in, transpose out).
                    col_to_row_f32(&a_cm, n, n, a, lda);
                    col_to_row_f32(&b_cm, n, nrhs, b, ldb);
                }
                info
            }
        };
        if info != 0 {
            return Err(Error::mkl(info, "sgesv_"));
        }
        Ok(())
    }

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
        check_solve_dims(layout, n, nrhs, a.len(), lda, ipiv.len(), b.len(), ldb)?;
        let info = match layout {
            Layout::ColMajor => {
                let mut info = 0i32;
                // SAFETY: buffers cover the `lda·n`/`n`/`ldb·nrhs` elements
                // `dgesv_` touches (validated above); `info` is a valid out-arg.
                unsafe {
                    nuvai_mkl_sys::dgesv_(
                        &n,
                        &nrhs,
                        a.as_mut_ptr(),
                        &lda,
                        ipiv.as_mut_ptr(),
                        b.as_mut_ptr(),
                        &ldb,
                        &mut info,
                    );
                }
                info
            }
            Layout::RowMajor => {
                let mut a_cm = vec![0.0f64; (n * n) as usize];
                let mut b_cm = vec![0.0f64; (n * nrhs) as usize];
                row_to_col_f64(a, n, n, lda, &mut a_cm);
                row_to_col_f64(b, n, nrhs, ldb, &mut b_cm);
                let lda_cm = n;
                let ldb_cm = n;
                let mut info = 0i32;
                // SAFETY: `a_cm`/`b_cm` are freshly allocated to exactly
                // `n·n`/`n·nrhs`; `ipiv` covers `n` pivots; `info` is valid.
                unsafe {
                    nuvai_mkl_sys::dgesv_(
                        &n,
                        &nrhs,
                        a_cm.as_mut_ptr(),
                        &lda_cm,
                        ipiv.as_mut_ptr(),
                        b_cm.as_mut_ptr(),
                        &ldb_cm,
                        &mut info,
                    );
                }
                if info == 0 {
                    col_to_row_f64(&a_cm, n, n, a, lda);
                    col_to_row_f64(&b_cm, n, nrhs, b, ldb);
                }
                info
            }
        };
        if info != 0 {
            return Err(Error::mkl(info, "dgesv_"));
        }
        Ok(())
    }

    pub fn sgetrf(layout: Layout, m: i32, n: i32, a: &mut [f32], lda: i32, ipiv: &mut [i32]) -> Result<()> {
        check_factor_dims(layout, m, n, a.len(), lda, ipiv.len())?;
        let info = match layout {
            Layout::ColMajor => {
                let mut info = 0i32;
                // SAFETY: `a` covers `lda·n` and `ipiv` covers `min(m,n)`
                // (validated above); `info` is a valid out-arg.
                unsafe {
                    nuvai_mkl_sys::sgetrf_(&m, &n, a.as_mut_ptr(), &lda, ipiv.as_mut_ptr(), &mut info);
                }
                info
            }
            Layout::RowMajor => {
                // `a` is m×n row-major (lda ≥ n); transpose to column-major (lda = m).
                let mut a_cm = vec![0.0f32; (m * n) as usize];
                row_to_col_f32(a, m, n, lda, &mut a_cm);
                let lda_cm = m;
                let mut info = 0i32;
                // SAFETY: `a_cm` is freshly allocated to `m·n`; `ipiv` covers
                // `min(m,n)`; `info` is a valid out-arg.
                unsafe {
                    nuvai_mkl_sys::sgetrf_(&m, &n, a_cm.as_mut_ptr(), &lda_cm, ipiv.as_mut_ptr(), &mut info);
                }
                if info == 0 {
                    // Copy the factored matrix back to row-major (LAPACKE parity).
                    col_to_row_f32(&a_cm, m, n, a, lda);
                }
                info
            }
        };
        if info != 0 {
            return Err(Error::mkl(info, "sgetrf_"));
        }
        Ok(())
    }

    pub fn dgetrf(layout: Layout, m: i32, n: i32, a: &mut [f64], lda: i32, ipiv: &mut [i32]) -> Result<()> {
        check_factor_dims(layout, m, n, a.len(), lda, ipiv.len())?;
        let info = match layout {
            Layout::ColMajor => {
                let mut info = 0i32;
                // SAFETY: `a` covers `lda·n` and `ipiv` covers `min(m,n)`
                // (validated above); `info` is a valid out-arg.
                unsafe {
                    nuvai_mkl_sys::dgetrf_(&m, &n, a.as_mut_ptr(), &lda, ipiv.as_mut_ptr(), &mut info);
                }
                info
            }
            Layout::RowMajor => {
                let mut a_cm = vec![0.0f64; (m * n) as usize];
                row_to_col_f64(a, m, n, lda, &mut a_cm);
                let lda_cm = m;
                let mut info = 0i32;
                // SAFETY: `a_cm` is freshly allocated to `m·n`; `ipiv` covers
                // `min(m,n)`; `info` is a valid out-arg.
                unsafe {
                    nuvai_mkl_sys::dgetrf_(&m, &n, a_cm.as_mut_ptr(), &lda_cm, ipiv.as_mut_ptr(), &mut info);
                }
                if info == 0 {
                    col_to_row_f64(&a_cm, m, n, a, lda);
                }
                info
            }
        };
        if info != 0 {
            return Err(Error::mkl(info, "dgetrf_"));
        }
        Ok(())
    }
}
