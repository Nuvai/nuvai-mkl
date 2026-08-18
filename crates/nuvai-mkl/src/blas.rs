//! BLAS (Basic Linear Algebra Subprograms) via the CBLAS interface.
//!
//! Dense linear-algebra primitives. Every routine takes slices and leading
//! dimensions and validates them against the BLAS sizing convention before any
//! pointer reaches CBLAS, mirroring `lapack::check_*_dims`. An undersized slice
//! or leading dimension is rejected as `ErrorKind::InvalidArgument` rather than
//! forwarded to MKL — which performs no bounds checking and would otherwise
//! read/write out of bounds. Level-1 strides must be positive (a zero or
//! negative stride would index below the slice).

use crate::error::{Error, Result};
use crate::layout::{Layout, Transpose};

/// Integer type the FFI CBLAS enums use on this target. bindgen maps the C
/// `CBLAS_LAYOUT`/`CBLAS_TRANSPOSE` enums to `u32` on Unix (all values fit an
/// `unsigned int`) and to `i32` on Windows (MSVC C enums default to `int`); the
/// aarch64 hand-written FFI surface uses `u32`. Casting through this alias keeps
/// the safe wrapper type-correct on every target (see PR #14 / task #7).
#[cfg(target_os = "windows")]
type CblasEnum = i32;
#[cfg(not(target_os = "windows"))]
type CblasEnum = u32;

#[inline]
fn cblas_layout(layout: Layout) -> CblasEnum {
    match layout {
        Layout::RowMajor => nuvai_mkl_sys::CblasRowMajor as CblasEnum,
        Layout::ColMajor => nuvai_mkl_sys::CblasColMajor as CblasEnum,
    }
}

#[inline]
fn cblas_trans(trans: Transpose) -> CblasEnum {
    match trans {
        Transpose::NoTrans => nuvai_mkl_sys::CblasNoTrans as CblasEnum,
        Transpose::Trans => nuvai_mkl_sys::CblasTrans as CblasEnum,
        Transpose::ConjTrans => nuvai_mkl_sys::CblasConjTrans as CblasEnum,
    }
}

/// Validate a `?gemm` call's buffers before any pointer reaches CBLAS.
///
/// `op(A)` is `m × k` (`k × m` when transposed), `op(B)` is `k × n` (`n × k`
/// when transposed) and `C` is `m × n`. Each operand's leading dimension must
/// cover its stored columns (row-major) or rows (column-major), and its slice
/// must reach the trailing element CBLAS touches. Rejects undersized buffers —
/// otherwise a safe call would be a heap out-of-bounds read/write inside MKL.
///
/// The leading dimensions are checked even when `alpha == 0`: BLAS enforces
/// `lda`/`ldb`/`ldc` before its "quick return", so an invalid `ld` is rejected
/// regardless of whether the operand values are actually read.
fn check_gemm_dims(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: i32,
    n: i32,
    k: i32,
    a_len: usize,
    lda: i32,
    b_len: usize,
    ldb: i32,
    c_len: usize,
    ldc: i32,
) -> Result<()> {
    if m < 0 || n < 0 || k < 0 {
        return Err(Error::invalid("blas: m, n and k must be non-negative"));
    }
    let (a_rows, a_cols) = match transa {
        Transpose::NoTrans => (m, k),
        Transpose::Trans | Transpose::ConjTrans => (k, m),
    };
    let (b_rows, b_cols) = match transb {
        Transpose::NoTrans => (k, n),
        Transpose::Trans | Transpose::ConjTrans => (n, k),
    };
    check_matrix(layout, a_rows, a_cols, lda, a_len, "a", "lda")?;
    check_matrix(layout, b_rows, b_cols, ldb, b_len, "b", "ldb")?;
    check_matrix(layout, m, n, ldc, c_len, "c", "ldc")?;
    Ok(())
}

/// Validate one `rows × cols` operand with leading dimension `ld`.
fn check_matrix(
    layout: Layout,
    rows: i32,
    cols: i32,
    ld: i32,
    len: usize,
    name: &str,
    ld_name: &str,
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let min_ld = match layout {
        Layout::RowMajor => cols,
        Layout::ColMajor => rows,
    };
    if ld < min_ld {
        return Err(Error::invalid(format!("blas: {ld_name} < {min_ld}")));
    }
    let min_len = match layout {
        Layout::RowMajor => (rows - 1) as usize * ld as usize + cols as usize,
        Layout::ColMajor => (cols - 1) as usize * ld as usize + rows as usize,
    };
    if len < min_len {
        return Err(Error::invalid(format!("blas: {name} too short")));
    }
    Ok(())
}

/// Validate a level-1 vector argument: `n` elements at stride `inc`.
///
/// `inc` must be positive. A zero increment is invalid per the BLAS spec, and a
/// negative one is rejected outright: the wrapper passes the slice's *first*
/// element as the CBLAS base, so a negative stride would make CBLAS index below
/// the slice — a heap out-of-bounds access that a `&[T]` cannot represent.
fn check_vector(n: i32, len: usize, inc: i32, name: &str) -> Result<()> {
    if n < 0 {
        return Err(Error::invalid(format!("blas: {name} count is negative")));
    }
    if n == 0 {
        return Ok(());
    }
    if inc <= 0 {
        return Err(Error::invalid(format!(
            "blas: {name} increment must be positive"
        )));
    }
    let required = 1 + (n - 1) as usize * inc as usize;
    if len < required {
        return Err(Error::invalid(format!("blas: {name} too short")));
    }
    Ok(())
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
    check_gemm_dims(
        layout,
        transa,
        transb,
        m,
        n,
        k,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;
    // SAFETY: `a`, `b` and `c` cover the transpose-adjusted leading-dimension
    // region `cblas_sgemm` reads/writes, as enforced by `check_gemm_dims` above.
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
    check_gemm_dims(
        layout,
        transa,
        transb,
        m,
        n,
        k,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;
    // SAFETY: `a`, `b` and `c` cover the transpose-adjusted leading-dimension
    // region `cblas_dgemm` reads/writes, as enforced by `check_gemm_dims` above.
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
    check_vector(n, x.len(), incx, "x")?;
    check_vector(n, y.len(), incy, "y")?;
    // SAFETY: `x` and `y` hold `n` elements at strides `incx`/`incy`, as
    // enforced by `check_vector` above; `cblas_saxpy` reads `x`, reads/writes
    // `y`.
    unsafe {
        nuvai_mkl_sys::cblas_saxpy(n, alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy);
    }
    Ok(())
}

/// `y := alpha * x + y`, double precision.
pub fn daxpy(n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32) -> Result<()> {
    check_vector(n, x.len(), incx, "x")?;
    check_vector(n, y.len(), incy, "y")?;
    // SAFETY: `x` and `y` hold `n` elements at strides `incx`/`incy`, as
    // enforced by `check_vector` above; `cblas_daxpy` reads `x`, reads/writes
    // `y`.
    unsafe {
        nuvai_mkl_sys::cblas_daxpy(n, alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy);
    }
    Ok(())
}

/// `dot := xᵀ · y`, single precision.
pub fn sdot(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> Result<f32> {
    check_vector(n, x.len(), incx, "x")?;
    check_vector(n, y.len(), incy, "y")?;
    // SAFETY: `x` and `y` hold `n` elements at strides `incx`/`incy`, as
    // enforced by `check_vector` above; `cblas_sdot` only reads them.
    Ok(unsafe { nuvai_mkl_sys::cblas_sdot(n, x.as_ptr(), incx, y.as_ptr(), incy) })
}

/// `dot := xᵀ · y`, double precision.
pub fn ddot(n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> Result<f64> {
    check_vector(n, x.len(), incx, "x")?;
    check_vector(n, y.len(), incy, "y")?;
    // SAFETY: `x` and `y` hold `n` elements at strides `incx`/`incy`, as
    // enforced by `check_vector` above; `cblas_ddot` only reads them.
    Ok(unsafe { nuvai_mkl_sys::cblas_ddot(n, x.as_ptr(), incx, y.as_ptr(), incy) })
}

/// `x := alpha * x`, single precision.
pub fn sscal(n: i32, alpha: f32, x: &mut [f32], incx: i32) -> Result<()> {
    check_vector(n, x.len(), incx, "x")?;
    // SAFETY: `x` holds `n` elements at stride `incx`, as enforced by
    // `check_vector` above; `cblas_sscal` writes them.
    unsafe {
        nuvai_mkl_sys::cblas_sscal(n, alpha, x.as_mut_ptr(), incx);
    }
    Ok(())
}

/// `x := alpha * x`, double precision.
pub fn dscal(n: i32, alpha: f64, x: &mut [f64], incx: i32) -> Result<()> {
    check_vector(n, x.len(), incx, "x")?;
    // SAFETY: `x` holds `n` elements at stride `incx`, as enforced by
    // `check_vector` above; `cblas_dscal` writes them.
    unsafe {
        nuvai_mkl_sys::cblas_dscal(n, alpha, x.as_mut_ptr(), incx);
    }
    Ok(())
}
