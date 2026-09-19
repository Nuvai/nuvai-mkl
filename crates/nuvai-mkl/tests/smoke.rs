//! End-to-end smoke tests exercising all six oneMKL domains against the real
//! MKL 2026.1.0 acquired by `nuvai-mkl-src`.

// The FFT round-trip tests (and their complex types) do not run on
// aarch64-unknown-linux-gnu, where FFT returns Unsupported.
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
use nuvai_mkl::fft::{MKL_Complex8, MKL_Complex16};
use nuvai_mkl::layout::{Layout, Transpose};
use nuvai_mkl::{blas, dss, fft, lapack, pardiso, vml, vsl};

fn assert_close(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= eps, "got {a}, expected {e} (eps {eps})");
    }
}

fn assert_close64(actual: &[f64], expected: &[f64], eps: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= eps, "got {a}, expected {e} (eps {eps})");
    }
}

/// A strictly diagonally dominant `m × n` matrix, row-major packed (`lda == n`):
/// every entry is `1` except the leading `min(m, n)` diagonal entries, which are
/// `6`. Dominance is what keeps `?getrf` from interchanging rows, so the
/// factorization can be verified by reconstructing `A` as `L·U` without undoing
/// a permutation — which keeps the check independent of the two backends' pivot
/// conventions (LAPACKE reports 0-based pivot indices, the aarch64 Fortran shim
/// 1-based).
fn diagonal_dominant(m: usize, n: usize) -> Vec<f64> {
    let mut a = vec![1.0f64; m * n];
    for i in 0..m.min(n) {
        a[i * n + i] = 6.0;
    }
    a
}

/// Reconstruct `L·U` from a packed `?getrf` factorization of an `m × n` matrix,
/// reading element `(i, j)` of the packed buffer through `at`.
///
/// `L` is the `m × min(m,n)` unit lower triangular factor stored below the
/// diagonal, `U` the `min(m,n) × n` upper triangular factor stored on and above
/// it — the packing `?getrf` produces in either layout, so `at` is the only
/// layout-dependent part (row-major: `a[i·lda + j]`, column-major:
/// `a[j·lda + i]`). Takes `f64` so the `f32` tests can reuse it.
fn lu_product<F: Fn(usize, usize) -> f64>(m: usize, n: usize, at: F) -> Vec<f64> {
    let k = m.min(n);
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for t in 0..k {
                let l = if i > t {
                    at(i, t)
                } else if i == t {
                    1.0
                } else {
                    0.0
                };
                let u = if t <= j { at(t, j) } else { 0.0 };
                acc += l * u;
            }
            out[i * n + j] = acc;
        }
    }
    out
}

#[test]
fn blas_sgemm_2x2() {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]] (row-major), C = A*B.
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let mut c = [0.0f32; 4];
    blas::sgemm(
        Layout::RowMajor,
        Transpose::NoTrans,
        Transpose::NoTrans,
        2,
        2,
        2,
        1.0,
        &a,
        2,
        &b,
        2,
        0.0,
        &mut c,
        2,
    )
    .unwrap();
    assert_close(&c, &[19.0, 22.0, 43.0, 50.0], 1e-5);
}

#[test]
fn blas_axpy_dot() {
    let x = [1.0f32, 2.0, 3.0];
    let mut y = [10.0f32, 20.0, 30.0];
    blas::saxpy(3, 2.0, &x, 1, &mut y, 1).unwrap();
    assert_close(&y, &[12.0, 24.0, 36.0], 1e-5);

    let dot = blas::sdot(3, &x, 1, &[4.0, 5.0, 6.0], 1).unwrap();
    assert!((dot - 32.0).abs() <= 1e-5, "dot = {dot}");
}

#[test]
fn blas_rejects_undersized_slices() {
    // #20: an undersized operand must error, not drive MKL into a heap OOB
    // read/write reachable from safe code. The huge `m` with a 4-element `a`
    // is the exact UB shape from the issue.
    let a = [1.0f32; 4];
    let b = [1.0f32; 4];
    let mut c = [0.0f32; 4];
    assert!(
        blas::sgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            100_000,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        )
        .is_err()
    );

    // Leading dimension below the stored width (row-major `lda < k`).
    let a = [1.0f32; 4];
    let b = [1.0f32; 4];
    let mut c = [0.0f32; 4];
    assert!(
        blas::sgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            1,
            &b,
            2,
            0.0,
            &mut c,
            2,
        )
        .is_err()
    );

    // Level-1: undersized, zero/negative stride, and negative count are all
    // invalid. A negative stride is rejected up front — the wrapper passes the
    // slice's first element as the CBLAS base, so a negative stride would walk
    // *below* the slice (heap OOB from safe code).
    let x = [1.0f32; 2];
    let mut y = [0.0f32; 2];
    assert!(blas::saxpy(3, 1.0, &x, 1, &mut y, 1).is_err());
    assert!(blas::saxpy(3, 1.0, &x, 0, &mut y, 1).is_err());
    assert!(blas::saxpy(3, 1.0, &x, -1, &mut y, 1).is_err());
    assert!(blas::saxpy(-1, 1.0, &x, 1, &mut y, 1).is_err());

    // `sdot`/`ddot` now return `Result`.
    assert!(blas::sdot(3, &x, 1, &[1.0f32; 2], 1).is_err());
    assert!(blas::ddot(3, &[1.0f64; 2], 1, &[1.0f64; 2], 1).is_err());

    // `sscal`/`dscal` undersized or negative stride.
    let mut xs = [1.0f32; 2];
    assert!(blas::sscal(3, 2.0, &mut xs, 1).is_err());
    assert!(blas::sscal(3, 2.0, &mut xs, -1).is_err());
    let mut xd = [1.0f64; 2];
    assert!(blas::dscal(3, 2.0, &mut xd, 1).is_err());
}

#[test]
fn lapack_sgesv_2x2() {
    // Solve A x = b, A = [[4,1],[1,3]] (col-major), b = [5,4] -> x = [1,1].
    let mut a = [4.0f32, 1.0, 1.0, 3.0];
    let mut b = [5.0f32, 4.0];
    let mut ipiv = [0i32; 2];
    lapack::sgesv(Layout::ColMajor, 2, 1, &mut a, 2, &mut ipiv, &mut b, 2).unwrap();
    assert_close(&b, &[1.0, 1.0], 1e-5);
}

#[test]
fn lapack_dgesv_2x2() {
    let mut a = [4.0f64, 1.0, 1.0, 3.0];
    let mut b = [5.0f64, 4.0];
    let mut ipiv = [0i32; 2];
    lapack::dgesv(Layout::ColMajor, 2, 1, &mut a, 2, &mut ipiv, &mut b, 2).unwrap();
    assert_close64(&b, &[1.0, 1.0], 1e-12);
}

#[test]
fn lapack_sgesv_rowmajor_2x2() {
    // Non-symmetric A (row-major) so the row-major transpose shim is actually
    // exercised: A = [[2,0],[1,3]], b = [4,10] -> x = [2, 8/3]. ldb = nrhs = 1
    // is the packed row-major right-hand side (LAPACKE row-major contract).
    let mut a = [2.0f32, 0.0, 1.0, 3.0];
    let mut b = [4.0f32, 10.0];
    let mut ipiv = [0i32; 2];
    lapack::sgesv(Layout::RowMajor, 2, 1, &mut a, 2, &mut ipiv, &mut b, 1).unwrap();
    assert_close(&b, &[2.0, 8.0 / 3.0], 1e-5);
}

#[test]
fn lapack_dgesv_rowmajor_2x2() {
    let mut a = [2.0f64, 0.0, 1.0, 3.0];
    let mut b = [4.0f64, 10.0];
    let mut ipiv = [0i32; 2];
    lapack::dgesv(Layout::RowMajor, 2, 1, &mut a, 2, &mut ipiv, &mut b, 1).unwrap();
    assert_close64(&b, &[2.0, 8.0 / 3.0], 1e-12);
}

#[test]
fn lapack_sgetrf_rowmajor_tall_accepted() {
    // #22: row-major storage strides *rows* by `lda`, so consecutive rows stay
    // disjoint only when `lda >= n` — the bound is the column count, not the
    // row count. `check_factor_dims` enforced `lda >= m` for both layouts, so a
    // tightly packed tall matrix (`lda == n < m`) was rejected outright even
    // though its length bound `(m-1)·lda + n` is exactly `m·n`. Pre-fix these
    // calls returned `InvalidArgument`.
    //
    // The input is strictly diagonally dominant, so the reconstruction is
    // `A == L·U` with no row interchanges to undo. That makes "accepted" mean
    // "factored the caller's matrix", not merely "did not error".

    // `m = 4, n = 2, lda = 2`: the tightest legal row-major stride here.
    let m = 4i32;
    let n = 2i32;
    let lda = 2i32;
    let expected = diagonal_dominant(m as usize, n as usize);
    let mut a: Vec<f32> = expected.iter().map(|&v| v as f32).collect();
    let mut ipiv = vec![0i32; m.min(n) as usize];
    lapack::sgetrf(Layout::RowMajor, m, n, &mut a, lda, &mut ipiv).unwrap();
    let lu = lu_product(m as usize, n as usize, |i, j| {
        a[i * lda as usize + j] as f64
    });
    assert_close64(&lu, &expected, 1e-4);

    // The issue's exact case: a tightly packed 10×3 row-major matrix.
    let m = 10i32;
    let n = 3i32;
    let lda = 3i32;
    let expected = diagonal_dominant(m as usize, n as usize);
    let mut a: Vec<f32> = expected.iter().map(|&v| v as f32).collect();
    let mut ipiv = vec![0i32; m.min(n) as usize];
    lapack::sgetrf(Layout::RowMajor, m, n, &mut a, lda, &mut ipiv).unwrap();
    let lu = lu_product(m as usize, n as usize, |i, j| {
        a[i * lda as usize + j] as f64
    });
    assert_close64(&lu, &expected, 1e-4);
}

#[test]
fn lapack_dgetrf_rowmajor_tall_accepted() {
    // Double-precision analogue of `lapack_sgetrf_rowmajor_tall_accepted`.
    let m = 4i32;
    let n = 2i32;
    let lda = 2i32;
    let expected = diagonal_dominant(m as usize, n as usize);
    let mut a = expected.clone();
    let mut ipiv = vec![0i32; m.min(n) as usize];
    lapack::dgetrf(Layout::RowMajor, m, n, &mut a, lda, &mut ipiv).unwrap();
    let lu = lu_product(m as usize, n as usize, |i, j| a[i * lda as usize + j]);
    assert_close64(&lu, &expected, 1e-12);

    // The issue's exact case: a tightly packed 10×3 row-major matrix.
    let m = 10i32;
    let n = 3i32;
    let lda = 3i32;
    let expected = diagonal_dominant(m as usize, n as usize);
    let mut a = expected.clone();
    let mut ipiv = vec![0i32; m.min(n) as usize];
    lapack::dgetrf(Layout::RowMajor, m, n, &mut a, lda, &mut ipiv).unwrap();
    let lu = lu_product(m as usize, n as usize, |i, j| a[i * lda as usize + j]);
    assert_close64(&lu, &expected, 1e-12);
}

#[test]
fn lapack_sgetrf_rowmajor_rejects_overlapping_rows() {
    // #22, the other direction: when `m <= lda < n` the old `lda >= m` bound was
    // satisfied, yet rows of an `m × n` row-major matrix overlap — row `i` starts
    // at `i·lda` and runs to `i·lda + n - 1`, past the start of row `i + 1`
    // whenever `lda < n`. The trailing-element length bound `(m-1)·lda + n` is
    // still met, so nothing downstream catches it: MKL factors a matrix the
    // caller never described (on aarch64 the transpose shim reads across logical
    // row boundaries) and returns a silently wrong answer instead of faulting.
    // Pre-fix these calls returned `Ok`.
    //
    // `InvalidArgument` specifically, not merely `is_err()`: it proves the
    // safe-Rust guard rejected the call before any pointer reached the backend.
    // A corrupting call that happened to return a non-zero MKL status would
    // satisfy `is_err()` while the wrong-answer bug still occurred.
    use nuvai_mkl::error::ErrorKind;

    for lda in [2i32, 3i32] {
        // Sized to exactly the trailing-element bound `(m-1)·lda + n`, so the
        // rejection is unambiguously about `lda` and not about `a.len()`.
        let len = (2 - 1) as usize * lda as usize + 4;
        let mut a = vec![1.0f32; len];
        let mut ipiv = vec![0i32; 2];
        let err = lapack::sgetrf(Layout::RowMajor, 2, 4, &mut a, lda, &mut ipiv).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidArgument, "lda = {lda}");
    }
}

#[test]
fn lapack_dgetrf_rowmajor_rejects_overlapping_rows() {
    // Double-precision analogue of `lapack_sgetrf_rowmajor_rejects_overlapping_rows`.
    use nuvai_mkl::error::ErrorKind;

    for lda in [2i32, 3i32] {
        let len = (2 - 1) as usize * lda as usize + 4;
        let mut a = vec![1.0f64; len];
        let mut ipiv = vec![0i32; 2];
        let err = lapack::dgetrf(Layout::RowMajor, 2, 4, &mut a, lda, &mut ipiv).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidArgument, "lda = {lda}");
    }
}

#[test]
fn lapack_sgetrf_rowmajor_lda_boundary() {
    // The row-major contract is `lda >= n` for an `m × n` matrix (LAPACKE:
    // `lda >= max(1, n)`), so `lda == n` is the tightest accepted stride and
    // `lda == n - 1` the widest rejected one. Pre-fix the boundary sat at `m`,
    // which rejected `lda == n` and admitted `lda == n - 1` whenever `n < m` —
    // exactly the two cases below. `lda > n` (padded rows) must also stay
    // accepted, so the `a` length bound may not be tightened to `lda·n`.
    use nuvai_mkl::error::ErrorKind;

    // Accepted: the tight `lda == n == 2` and the padded `lda == 3 > n`. Each
    // buffer is the exact trailing-element bound `(m-1)·lda + n`.
    for lda in [2i32, 3i32] {
        let expected = diagonal_dominant(4, 2);
        let mut a = vec![0.0f32; 3 * lda as usize + 2];
        for i in 0..4usize {
            for j in 0..2usize {
                a[i * lda as usize + j] = expected[i * 2 + j] as f32;
            }
        }
        let mut ipiv = vec![0i32; 2];
        lapack::sgetrf(Layout::RowMajor, 4, 2, &mut a, lda, &mut ipiv).unwrap();
        let lu = lu_product(4, 2, |i, j| a[i * lda as usize + j] as f64);
        assert_close64(&lu, &expected, 1e-4);
    }

    // Rejected: `lda == n - 1 == 3` for a 2×4 matrix (`m <= lda < n`).
    let mut a = vec![1.0f32; 7];
    let mut ipiv = vec![0i32; 2];
    let err = lapack::sgetrf(Layout::RowMajor, 2, 4, &mut a, 3, &mut ipiv).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
}

#[test]
fn lapack_dgetrf_rowmajor_lda_boundary() {
    // Double-precision analogue of `lapack_sgetrf_rowmajor_lda_boundary`.
    use nuvai_mkl::error::ErrorKind;

    for lda in [2i32, 3i32] {
        let expected = diagonal_dominant(4, 2);
        let mut a = vec![0.0f64; 3 * lda as usize + 2];
        for i in 0..4usize {
            for j in 0..2usize {
                a[i * lda as usize + j] = expected[i * 2 + j];
            }
        }
        let mut ipiv = vec![0i32; 2];
        lapack::dgetrf(Layout::RowMajor, 4, 2, &mut a, lda, &mut ipiv).unwrap();
        let lu = lu_product(4, 2, |i, j| a[i * lda as usize + j]);
        assert_close64(&lu, &expected, 1e-12);
    }

    let mut a = vec![1.0f64; 7];
    let mut ipiv = vec![0i32; 2];
    let err = lapack::dgetrf(Layout::RowMajor, 2, 4, &mut a, 3, &mut ipiv).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
}

#[test]
fn lapack_sgetrf_colmajor_lda_bound_control() {
    // Column-major storage strides *columns* by `lda`, so its bound stays
    // `lda >= m` and must not move with the row-major fix. `m = 3, n = 2,
    // lda = 3` is the tightest legal stride here and keeps factoring correctly;
    // `lda = 2 < m` stays rejected.
    use nuvai_mkl::error::ErrorKind;

    let expected = diagonal_dominant(3, 2);
    let lda = 3usize;
    let mut a = vec![0.0f32; lda * 2];
    for i in 0..3usize {
        for j in 0..2usize {
            a[j * lda + i] = expected[i * 2 + j] as f32;
        }
    }
    let mut ipiv = vec![0i32; 2];
    lapack::sgetrf(Layout::ColMajor, 3, 2, &mut a, 3, &mut ipiv).unwrap();
    let lu = lu_product(3, 2, |i, j| a[j * lda + i] as f64);
    assert_close64(&lu, &expected, 1e-4);

    let mut a = vec![1.0f32; 6];
    let mut ipiv = vec![0i32; 2];
    let err = lapack::sgetrf(Layout::ColMajor, 3, 2, &mut a, 2, &mut ipiv).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
}

#[test]
fn lapack_dgetrf_colmajor_lda_bound_control() {
    // Double-precision analogue of `lapack_sgetrf_colmajor_lda_bound_control`.
    use nuvai_mkl::error::ErrorKind;

    let expected = diagonal_dominant(3, 2);
    let lda = 3usize;
    let mut a = vec![0.0f64; lda * 2];
    for i in 0..3usize {
        for j in 0..2usize {
            a[j * lda + i] = expected[i * 2 + j];
        }
    }
    let mut ipiv = vec![0i32; 2];
    lapack::dgetrf(Layout::ColMajor, 3, 2, &mut a, 3, &mut ipiv).unwrap();
    let lu = lu_product(3, 2, |i, j| a[j * lda + i]);
    assert_close64(&lu, &expected, 1e-12);

    let mut a = vec![1.0f64; 6];
    let mut ipiv = vec![0i32; 2];
    let err = lapack::dgetrf(Layout::ColMajor, 3, 2, &mut a, 2, &mut ipiv).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
}

/// #37: LAPACK defines a zero dimension as a legal *quick return* — `?gesv`
/// with `n == 0` or `nrhs == 0`, `?getrf` with `m == 0` or `n == 0` — that
/// reads and writes nothing. The wrapper rejected those outright, which also
/// made `lapack` disagree with `blas`, whose `check_matrix`/`check_vector` have
/// always accepted a zero dimension as a no-op.
///
/// These tests assert the *no-op*, not merely `Ok`: every buffer is pre-filled
/// with a value the backend would never produce and compared afterwards, so a
/// call that quietly reached C/Fortran (and, say, transposed through the
/// row-major scratch buffers, or wrote a pivot) fails here instead of passing
/// as "accepted". They are backend-independent: the wrapper returns before it
/// dispatches, so the Accelerate/OpenBLAS and LAPACKE arms must agree.
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn lapack_sgesv_zero_dims_are_noops() {
    use nuvai_mkl::error::ErrorKind;

    let mut a = [7.0f32; 4];
    let mut ipiv = [5i32; 4];
    let mut b = [9.0f32; 4];

    // `n == 0`: a zero-order system. `lda`/`ldb` are still validated (LAPACK
    // checks them ahead of its quick return), so both are 1 here.
    for layout in [Layout::ColMajor, Layout::RowMajor] {
        lapack::sgesv(layout, 0, 1, &mut a, 1, &mut ipiv, &mut b, 1).unwrap();
        assert_eq!(
            (a, ipiv, b),
            ([7.0f32; 4], [5i32; 4], [9.0f32; 4]),
            "{layout:?}"
        );
    }

    // `nrhs == 0`: the quick return precedes the factorization as well, so `a`
    // and `ipiv` are left untouched too (the old code would have factored `a`
    // into an LU the caller never asked for, had it accepted the call). `ldb`
    // is still checked against `n` here, which is the point of validating the
    // leading dimensions *before* the no-op return: `ldb = 1` is rejected even
    // though nothing would be read.
    for layout in [Layout::ColMajor, Layout::RowMajor] {
        lapack::sgesv(layout, 2, 0, &mut a, 2, &mut ipiv, &mut b, 2).unwrap();
        assert_eq!(
            (a, ipiv, b),
            ([7.0f32; 4], [5i32; 4], [9.0f32; 4]),
            "{layout:?}"
        );
    }
    let err = lapack::sgesv(Layout::ColMajor, 2, 0, &mut a, 2, &mut ipiv, &mut b, 1).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);

    // The fully degenerate 0×0 problem with zero leading dimensions. This is
    // the one case where the wrapper is deliberately more permissive than the
    // Fortran routines, whose bound is `LDA >= max(1, n)`: no call is made, so
    // there is nothing for that bound to protect, and the buffer lengths are
    // unconstrained for the same reason (nothing is read).
    lapack::sgesv(Layout::ColMajor, 0, 0, &mut [], 0, &mut [], &mut [], 0).unwrap();

    // Negative dimensions stay rejected — that is LAPACK's own argument check,
    // reported by the wrapper before any pointer moves. The kind is asserted
    // rather than `is_err()` so a call that merely happened to fail inside MKL
    // cannot be mistaken for the guard firing.
    for (n, nrhs) in [(-1i32, 1i32), (2, -1)] {
        let err =
            lapack::sgesv(Layout::ColMajor, n, nrhs, &mut a, 2, &mut ipiv, &mut b, 2).unwrap_err();
        assert_eq!(
            err.kind(),
            ErrorKind::InvalidArgument,
            "n = {n}, nrhs = {nrhs}"
        );
    }
}

/// Double-precision analogue of `lapack_sgesv_zero_dims_are_noops`.
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn lapack_dgesv_zero_dims_are_noops() {
    use nuvai_mkl::error::ErrorKind;

    let mut a = [7.0f64; 4];
    let mut ipiv = [5i32; 4];
    let mut b = [9.0f64; 4];

    for layout in [Layout::ColMajor, Layout::RowMajor] {
        lapack::dgesv(layout, 0, 1, &mut a, 1, &mut ipiv, &mut b, 1).unwrap();
        assert_eq!(
            (a, ipiv, b),
            ([7.0f64; 4], [5i32; 4], [9.0f64; 4]),
            "{layout:?}"
        );

        // `ldb >= n` still applies when only `nrhs` is zero — see the
        // single-precision test for why that check survives the no-op.
        lapack::dgesv(layout, 2, 0, &mut a, 2, &mut ipiv, &mut b, 2).unwrap();
        assert_eq!(
            (a, ipiv, b),
            ([7.0f64; 4], [5i32; 4], [9.0f64; 4]),
            "{layout:?}"
        );
    }
    let err = lapack::dgesv(Layout::ColMajor, 2, 0, &mut a, 2, &mut ipiv, &mut b, 1).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);

    lapack::dgesv(Layout::ColMajor, 0, 0, &mut [], 0, &mut [], &mut [], 0).unwrap();

    for (n, nrhs) in [(-1i32, 1i32), (2, -1)] {
        let err =
            lapack::dgesv(Layout::ColMajor, n, nrhs, &mut a, 2, &mut ipiv, &mut b, 2).unwrap_err();
        assert_eq!(
            err.kind(),
            ErrorKind::InvalidArgument,
            "n = {n}, nrhs = {nrhs}"
        );
    }
}

/// #37 for `?getrf`: `m == 0` or `n == 0` is the quick return, so the matrix
/// and the pivot vector are left exactly as they were.
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn lapack_sgetrf_zero_dims_are_noops() {
    use nuvai_mkl::error::ErrorKind;

    // `lda = 3` clears the layout-dependent bound in all four combinations
    // below (`lda >= m` column-major, `lda >= n` row-major), so the only thing
    // that can reject these calls is the zero dimension itself.
    let mut a = [7.0f32; 6];
    let mut ipiv = [5i32; 3];

    for (m, n) in [(0i32, 3i32), (3, 0), (0, 0)] {
        for layout in [Layout::ColMajor, Layout::RowMajor] {
            lapack::sgetrf(layout, m, n, &mut a, 3, &mut ipiv).unwrap();
            assert_eq!(
                (a, ipiv),
                ([7.0f32; 6], [5i32; 3]),
                "m = {m}, n = {n}, {layout:?}"
            );
        }
    }

    for (m, n) in [(-1i32, 3i32), (3, -1)] {
        let err = lapack::sgetrf(Layout::ColMajor, m, n, &mut a, 3, &mut ipiv).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidArgument, "m = {m}, n = {n}");
    }
}

/// Double-precision analogue of `lapack_sgetrf_zero_dims_are_noops`.
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn lapack_dgetrf_zero_dims_are_noops() {
    use nuvai_mkl::error::ErrorKind;

    let mut a = [7.0f64; 6];
    let mut ipiv = [5i32; 3];

    for (m, n) in [(0i32, 3i32), (3, 0), (0, 0)] {
        for layout in [Layout::ColMajor, Layout::RowMajor] {
            lapack::dgetrf(layout, m, n, &mut a, 3, &mut ipiv).unwrap();
            assert_eq!(
                (a, ipiv),
                ([7.0f64; 6], [5i32; 3]),
                "m = {m}, n = {n}, {layout:?}"
            );
        }
    }

    for (m, n) in [(-1i32, 3i32), (3, -1)] {
        let err = lapack::dgetrf(Layout::ColMajor, m, n, &mut a, 3, &mut ipiv).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidArgument, "m = {m}, n = {n}");
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn fft_roundtrip_c32() {
    let plan = fft::FftPlan::new_c32(4).unwrap();
    // Impulse δ = [1,0,0,0] -> forward = [1,1,1,1].
    let input = [
        MKL_Complex8 {
            real: 1.0,
            imag: 0.0,
        },
        MKL_Complex8 {
            real: 0.0,
            imag: 0.0,
        },
        MKL_Complex8 {
            real: 0.0,
            imag: 0.0,
        },
        MKL_Complex8 {
            real: 0.0,
            imag: 0.0,
        },
    ];
    let mut freq = [MKL_Complex8 {
        real: 0.0,
        imag: 0.0,
    }; 4];
    plan.forward_c32(&input, &mut freq).unwrap();
    for f in &freq {
        assert!((f.real - 1.0).abs() <= 1e-5, "real = {}", f.real);
        assert!(f.imag.abs() <= 1e-5, "imag = {}", f.imag);
    }

    // Backward of [1,1,1,1] recovers the impulse (default 1/n scaling).
    let mut out = [MKL_Complex8 {
        real: 0.0,
        imag: 0.0,
    }; 4];
    plan.backward_c32(&freq, &mut out).unwrap();
    assert!(
        (out[0].real - 1.0).abs() <= 1e-5,
        "out[0] = {}",
        out[0].real
    );
    for o in &out[1..] {
        assert!(o.real.abs() <= 1e-5 && o.imag.abs() <= 1e-5);
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn fft_roundtrip_c64() {
    let plan = fft::FftPlan::new_c64(4).unwrap();
    // Same impulse test as the c32 variant, in double precision.
    let input = [
        MKL_Complex16 {
            real: 1.0,
            imag: 0.0,
        },
        MKL_Complex16 {
            real: 0.0,
            imag: 0.0,
        },
        MKL_Complex16 {
            real: 0.0,
            imag: 0.0,
        },
        MKL_Complex16 {
            real: 0.0,
            imag: 0.0,
        },
    ];
    let mut freq = [MKL_Complex16 {
        real: 0.0,
        imag: 0.0,
    }; 4];
    plan.forward_c64(&input, &mut freq).unwrap();
    for f in &freq {
        assert!((f.real - 1.0).abs() <= 1e-9, "real = {}", f.real);
        assert!(f.imag.abs() <= 1e-9, "imag = {}", f.imag);
    }

    let mut out = [MKL_Complex16 {
        real: 0.0,
        imag: 0.0,
    }; 4];
    plan.backward_c64(&freq, &mut out).unwrap();
    assert!(
        (out[0].real - 1.0).abs() <= 1e-9,
        "out[0] = {}",
        out[0].real
    );
    for o in &out[1..] {
        assert!(o.real.abs() <= 1e-9 && o.imag.abs() <= 1e-9);
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn fft_roundtrip_c32_len2() {
    // 2 is the smallest split-complex fallback length (the interleaved family
    // needs n >= 2, i.e. length >= 8). The impulse δ = [1,0] DFTs to all-ones
    // and its inverse recovers it.
    let plan = fft::FftPlan::new_c32(2).unwrap();
    let input = [
        MKL_Complex8 {
            real: 1.0,
            imag: 0.0,
        },
        MKL_Complex8 {
            real: 0.0,
            imag: 0.0,
        },
    ];
    let mut freq = [MKL_Complex8 {
        real: 0.0,
        imag: 0.0,
    }; 2];
    plan.forward_c32(&input, &mut freq).unwrap();
    for f in &freq {
        assert!((f.real - 1.0).abs() <= 1e-5, "real = {}", f.real);
        assert!(f.imag.abs() <= 1e-5, "imag = {}", f.imag);
    }

    let mut out = [MKL_Complex8 {
        real: 0.0,
        imag: 0.0,
    }; 2];
    plan.backward_c32(&freq, &mut out).unwrap();
    assert!(
        (out[0].real - 1.0).abs() <= 1e-5,
        "out[0] = {}",
        out[0].real
    );
    for o in &out[1..] {
        assert!(o.real.abs() <= 1e-5 && o.imag.abs() <= 1e-5);
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn fft_roundtrip_c64_len2() {
    // Double-precision analogue of `fft_roundtrip_c32_len2`.
    let plan = fft::FftPlan::new_c64(2).unwrap();
    let input = [
        MKL_Complex16 {
            real: 1.0,
            imag: 0.0,
        },
        MKL_Complex16 {
            real: 0.0,
            imag: 0.0,
        },
    ];
    let mut freq = [MKL_Complex16 {
        real: 0.0,
        imag: 0.0,
    }; 2];
    plan.forward_c64(&input, &mut freq).unwrap();
    for f in &freq {
        assert!((f.real - 1.0).abs() <= 1e-9, "real = {}", f.real);
        assert!(f.imag.abs() <= 1e-9, "imag = {}", f.imag);
    }

    let mut out = [MKL_Complex16 {
        real: 0.0,
        imag: 0.0,
    }; 2];
    plan.backward_c64(&freq, &mut out).unwrap();
    assert!(
        (out[0].real - 1.0).abs() <= 1e-9,
        "out[0] = {}",
        out[0].real
    );
    for o in &out[1..] {
        assert!(o.real.abs() <= 1e-9 && o.imag.abs() <= 1e-9);
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn fft_roundtrip_c32_non_pow2() {
    // 24 = 3·2^3 is not a power of two, so this exercises the non-power-of-two
    // path (DFTI on Intel; vDSP's interleaved `f·2^n` family on aarch64). As for
    // any length, the DFT of the impulse is all-ones and its inverse recovers it.
    let plan = fft::FftPlan::new_c32(24).unwrap();
    let mut input = vec![
        MKL_Complex8 {
            real: 0.0,
            imag: 0.0
        };
        24
    ];
    input[0].real = 1.0;
    let mut freq = vec![
        MKL_Complex8 {
            real: 0.0,
            imag: 0.0
        };
        24
    ];
    plan.forward_c32(&input, &mut freq).unwrap();
    for f in &freq {
        assert!((f.real - 1.0).abs() <= 1e-5, "real = {}", f.real);
        assert!(f.imag.abs() <= 1e-5, "imag = {}", f.imag);
    }

    let mut out = vec![
        MKL_Complex8 {
            real: 0.0,
            imag: 0.0
        };
        24
    ];
    plan.backward_c32(&freq, &mut out).unwrap();
    assert!(
        (out[0].real - 1.0).abs() <= 1e-5,
        "out[0] = {}",
        out[0].real
    );
    for o in &out[1..] {
        assert!(o.real.abs() <= 1e-5 && o.imag.abs() <= 1e-5);
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn fft_roundtrip_c32_len8() {
    // 8 = 2·2^2 is the smallest length the interleaved-complex family plans
    // (macOS 12.0+ on aarch64; DFTI on Intel), so this exercises that path
    // directly rather than the split-complex fallback used for lengths 2 and 4.
    let plan = fft::FftPlan::new_c32(8).unwrap();
    let mut input = vec![
        MKL_Complex8 {
            real: 0.0,
            imag: 0.0
        };
        8
    ];
    input[0].real = 1.0;
    let mut freq = vec![
        MKL_Complex8 {
            real: 0.0,
            imag: 0.0
        };
        8
    ];
    plan.forward_c32(&input, &mut freq).unwrap();
    for f in &freq {
        assert!((f.real - 1.0).abs() <= 1e-5, "real = {}", f.real);
        assert!(f.imag.abs() <= 1e-5, "imag = {}", f.imag);
    }

    let mut out = vec![
        MKL_Complex8 {
            real: 0.0,
            imag: 0.0
        };
        8
    ];
    plan.backward_c32(&freq, &mut out).unwrap();
    assert!(
        (out[0].real - 1.0).abs() <= 1e-5,
        "out[0] = {}",
        out[0].real
    );
    for o in &out[1..] {
        assert!(o.real.abs() <= 1e-5 && o.imag.abs() <= 1e-5);
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn fft_roundtrip_c64_len8() {
    // Double-precision analogue of `fft_roundtrip_c32_len8`.
    let plan = fft::FftPlan::new_c64(8).unwrap();
    let mut input = vec![
        MKL_Complex16 {
            real: 0.0,
            imag: 0.0
        };
        8
    ];
    input[0].real = 1.0;
    let mut freq = vec![
        MKL_Complex16 {
            real: 0.0,
            imag: 0.0
        };
        8
    ];
    plan.forward_c64(&input, &mut freq).unwrap();
    for f in &freq {
        assert!((f.real - 1.0).abs() <= 1e-9, "real = {}", f.real);
        assert!(f.imag.abs() <= 1e-9, "imag = {}", f.imag);
    }

    let mut out = vec![
        MKL_Complex16 {
            real: 0.0,
            imag: 0.0
        };
        8
    ];
    plan.backward_c64(&freq, &mut out).unwrap();
    assert!(
        (out[0].real - 1.0).abs() <= 1e-9,
        "out[0] = {}",
        out[0].real
    );
    for o in &out[1..] {
        assert!(o.real.abs() <= 1e-9 && o.imag.abs() <= 1e-9);
    }
}

/// Exercise every VML function (all 11) in single precision against known
/// values. The f32/f64 variants run through both backends (MKL VML on Intel,
/// Accelerate vForce on aarch64).
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn vml_full_surface_f32() {
    let e = std::f32::consts::E;
    let (pi2, pi4) = (std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_4);

    let mut dst = [0.0f32; 3];
    vml::exp(&[0.0, 1.0, 2.0], &mut dst).unwrap();
    assert_close(&dst, &[1.0, e, e * e], 1e-4);

    vml::ln(&[1.0, e, e * e], &mut dst).unwrap();
    assert_close(&dst, &[0.0, 1.0, 2.0], 1e-4);

    vml::log10(&[1.0, 10.0, 100.0], &mut dst).unwrap();
    assert_close(&dst, &[0.0, 1.0, 2.0], 1e-4);

    let mut dst = [0.0f32; 4];
    vml::sqrt(&[0.0, 1.0, 4.0, 9.0], &mut dst).unwrap();
    assert_close(&dst, &[0.0, 1.0, 2.0, 3.0], 1e-5);

    vml::cbrt(&[0.0, 1.0, 8.0, 27.0], &mut dst).unwrap();
    assert_close(&dst, &[0.0, 1.0, 2.0, 3.0], 1e-5);

    let mut dst = [0.0f32; 2];
    vml::sin(&[0.0, pi2], &mut dst).unwrap();
    assert_close(&dst, &[0.0, 1.0], 1e-5);

    vml::cos(&[0.0, pi2], &mut dst).unwrap();
    assert_close(&dst, &[1.0, 0.0], 1e-5);

    vml::tan(&[0.0, pi4], &mut dst).unwrap();
    assert_close(&dst, &[0.0, 1.0], 1e-5);

    vml::asin(&[0.0, 1.0], &mut dst).unwrap();
    assert_close(&dst, &[0.0, pi2], 1e-5);

    vml::acos(&[1.0, 0.0], &mut dst).unwrap();
    assert_close(&dst, &[0.0, pi2], 1e-5);

    vml::atan(&[0.0, 1.0], &mut dst).unwrap();
    assert_close(&dst, &[0.0, pi4], 1e-5);
}

/// Every VML function in double precision (vForce `D` variants on aarch64).
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn vml_full_surface_f64() {
    let e = std::f64::consts::E;
    let (pi2, pi4) = (std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_4);

    let mut dst = [0.0f64; 3];
    vml::dexp(&[0.0, 1.0, 2.0], &mut dst).unwrap();
    assert_close64(&dst, &[1.0, e, e * e], 1e-12);

    vml::dln(&[1.0, e, e * e], &mut dst).unwrap();
    assert_close64(&dst, &[0.0, 1.0, 2.0], 1e-12);

    vml::dlog10(&[1.0, 10.0, 100.0], &mut dst).unwrap();
    assert_close64(&dst, &[0.0, 1.0, 2.0], 1e-12);

    let mut dst = [0.0f64; 4];
    vml::dsqrt(&[0.0, 1.0, 4.0, 9.0], &mut dst).unwrap();
    assert_close64(&dst, &[0.0, 1.0, 2.0, 3.0], 1e-12);

    vml::dcbrt(&[0.0, 1.0, 8.0, 27.0], &mut dst).unwrap();
    assert_close64(&dst, &[0.0, 1.0, 2.0, 3.0], 1e-12);

    let mut dst = [0.0f64; 2];
    vml::dsin(&[0.0, pi2], &mut dst).unwrap();
    assert_close64(&dst, &[0.0, 1.0], 1e-12);

    vml::dcos(&[0.0, pi2], &mut dst).unwrap();
    assert_close64(&dst, &[1.0, 0.0], 1e-12);

    vml::dtan(&[0.0, pi4], &mut dst).unwrap();
    assert_close64(&dst, &[0.0, 1.0], 1e-12);

    vml::dasin(&[0.0, 1.0], &mut dst).unwrap();
    assert_close64(&dst, &[0.0, pi2], 1e-12);

    vml::dacos(&[1.0, 0.0], &mut dst).unwrap();
    assert_close64(&dst, &[0.0, pi2], 1e-12);

    vml::datan(&[0.0, 1.0], &mut dst).unwrap();
    assert_close64(&dst, &[0.0, pi4], 1e-12);
}

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn vsl_uniform_gaussian() {
    let stream = vsl::Stream::new(12345).unwrap();

    let mut u = vec![0.0f32; 100_000];
    stream.uniform(0.0, 1.0, &mut u).unwrap();
    let mean = u.iter().sum::<f32>() / u.len() as f32;
    assert!(u.iter().all(|&v| (0.0..1.0).contains(&v)));
    assert!((mean - 0.5).abs() <= 0.01, "uniform mean = {mean}");

    let mut g = vec![0.0f32; 100_000];
    stream.gaussian(0.0, 1.0, &mut g).unwrap();
    let gmean = g.iter().sum::<f32>() / g.len() as f32;
    assert!((gmean - 0.0).abs() <= 0.01, "gaussian mean = {gmean}");

    let mut u64 = vec![0.0f64; 100_000];
    stream.uniform64(0.0, 1.0, &mut u64).unwrap();
    let mean64 = u64.iter().sum::<f64>() / u64.len() as f64;
    assert!(u64.iter().all(|&v| (0.0..1.0).contains(&v)));
    assert!((mean64 - 0.5).abs() <= 0.01, "uniform64 mean = {mean64}");

    let mut g64 = vec![0.0f64; 100_000];
    stream.gaussian64(0.0, 1.0, &mut g64).unwrap();
    let gmean64 = g64.iter().sum::<f64>() / g64.len() as f64;
    assert!((gmean64 - 0.0).abs() <= 0.01, "gaussian64 mean = {gmean64}");
}

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn pardiso_solve_3x3() {
    // A = [[2,1,0],[1,3,1],[0,1,2]] (nonsymmetric, full CSR, 1-based).
    let ia = [1i32, 3, 6, 8];
    let ja = [1i32, 2, 1, 2, 3, 2, 3];
    let a = [2.0f64, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
    let b = [4.0f64, 10.0, 8.0];
    let mut solver = pardiso::Pardiso::new(pardiso::mtype::NONSYMMETRIC);
    let x = solver.solve(&ia, &ja, &a, &b).unwrap();
    assert_close64(&x, &[1.0, 2.0, 3.0], 1e-9);
}

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn pardiso_rejects_empty_ia() {
    // #26: `n` was derived as `ia.len() - 1`, evaluated in `usize`, so an empty
    // `ia` underflowed the subtraction *before* the `as i32` cast — a panic
    // under `overflow-checks` (every debug build, so `cargo test`) and a wrap to
    // `usize::MAX` in release. The trailing `n <= 0` guard rejected the wrapped
    // value only by coincidence (`usize::MAX as i32 == -1`), so the expression is
    // now total rather than relying on that coupling.
    //
    // `InvalidArgument` specifically, not merely `is_err()`: it proves the
    // safe-Rust guard fired before any pointer reached the backend. The Intel arm
    // passes `n` into the Fortran ABI with no length argument, so a corrupting
    // call could not be told from a rejected one by `is_err()` alone.
    use nuvai_mkl::error::ErrorKind;
    let mut solver = pardiso::Pardiso::new(pardiso::mtype::NONSYMMETRIC);

    // Empty `ia` describes no rows; `ja`/`a` are empty too, so the
    // `ja.len() != a.len()` check passes and the subtraction is reached.
    let err = solver.solve(&[], &[], &[], &[]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);

    // Same guard with a populated `ja`/`a` and right-hand side.
    let err = solver
        .solve(&[], &[1i32], &[1.0f64], &[1.0f64])
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);

    // `ia = [1]` is well-formed CSR but describes a zero-order system; it stays
    // rejected, through the length guard rather than the empty-`ia` one.
    let err = solver.solve(&[1i32], &[], &[], &[]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);

    // The well-formed path still solves — the new guard rejects nothing valid.
    let ia = [1i32, 3, 6, 8];
    let ja = [1i32, 2, 1, 2, 3, 2, 3];
    let a = [2.0f64, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
    let x = solver.solve(&ia, &ja, &a, &[4.0f64, 10.0, 8.0]).unwrap();
    assert_close64(&x, &[1.0, 2.0, 3.0], 1e-9);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn pardiso_reuses_factorization_on_aarch64() {
    // Same nonsymmetric matrix as `pardiso_solve_3x3`, solved twice with two
    // different right-hand sides. The second `solve` must reuse the cached QR
    // factorization (`b` is not part of the cache key) and still return the
    // correct solution.
    let ia = [1i32, 3, 6, 8];
    let ja = [1i32, 2, 1, 2, 3, 2, 3];
    let a = [2.0f64, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];

    let mut solver = pardiso::Pardiso::new(pardiso::mtype::NONSYMMETRIC);

    let x1 = solver.solve(&ia, &ja, &a, &[4.0f64, 10.0, 8.0]).unwrap();
    assert_close64(&x1, &[1.0, 2.0, 3.0], 1e-9);

    // A·[3,1,2]ᵀ = [7,8,5]ᵀ — a distinct RHS that must not invalidate the
    // cached factorization.
    let x2 = solver.solve(&ia, &ja, &a, &[7.0f64, 8.0, 5.0]).unwrap();
    assert_close64(&x2, &[3.0, 1.0, 2.0], 1e-9);
}

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn dss_solve_2x2() {
    // A = [[4,1],[1,3]] symmetric positive-definite, upper triangle, 0-based.
    let row_index = [0i32, 2, 3];
    let columns = [0i32, 1, 1];
    let values = [4.0f64, 1.0, 3.0];
    let dss = dss::Dss::factor_symmetric(&row_index, &columns, &values).unwrap();
    let x = dss.solve(&[5.0f64, 4.0]).unwrap();
    assert_close64(&x, &[1.0, 1.0], 1e-9);
}

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn dss_solve_rejects_rhs_length_mismatch() {
    // #21: the backend writes exactly `n` values into the solution buffer
    // regardless of `rhs.len()`. Pre-fix the Intel arm sized that buffer to
    // `rhs.len()`, so an undersized RHS drove a heap out-of-bounds *write*
    // reachable from safe code, and an oversized one returned a `Vec` padded
    // past the `n` values actually solved for.
    //
    // Same SPD matrix as `dss_solve_2x2`: A = [[4,1],[1,3]], so n == 2.
    let row_index = [0i32, 2, 3];
    let columns = [0i32, 1, 1];
    let values = [4.0f64, 1.0, 3.0];
    let dss = dss::Dss::factor_symmetric(&row_index, &columns, &values).unwrap();

    // `InvalidArgument` specifically, not merely `is_err()`: it proves the
    // safe-Rust guard rejected the call before any pointer reached the backend.
    // A corrupting call that happened to return a non-zero MKL status would
    // satisfy `is_err()` while the UB still occurred.
    use nuvai_mkl::error::ErrorKind;

    // Undersized: pre-fix this sized `sol` to 1 while DSS wrote 2.
    let err = dss.solve(&[5.0f64]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);

    // Oversized: only the first 2 values are ever solved for.
    let err = dss.solve(&[5.0f64, 4.0, 9.0]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);

    // Empty.
    let err = dss.solve(&[]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);

    // The valid-length path still solves correctly.
    let x = dss.solve(&[5.0f64, 4.0]).unwrap();
    assert_close64(&x, &[1.0, 1.0], 1e-9);
}

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn vsl_uniform_rejects_empty_range() {
    let stream = vsl::Stream::new(7).unwrap();
    // Empty (a == b), inverted (a > b), and NaN ranges must error, not panic:
    // Intel VSL reports BADARGS, but the aarch64 `rand` backend would panic on
    // an empty range, so the guard normalizes both to an error.
    let mut out = [0.0f32; 8];
    assert!(stream.uniform(1.0, 1.0, &mut out).is_err());
    assert!(stream.uniform(2.0, 1.0, &mut out).is_err());
    assert!(stream.uniform(f32::NAN, 1.0, &mut out).is_err());
    assert!(stream.uniform(0.0, f32::NAN, &mut out).is_err());

    let mut out64 = [0.0f64; 8];
    assert!(stream.uniform64(1.0, 1.0, &mut out64).is_err());
    assert!(stream.uniform64(2.0, 1.0, &mut out64).is_err());
    assert!(stream.uniform64(f64::NAN, 1.0, &mut out64).is_err());
    assert!(stream.uniform64(0.0, f64::NAN, &mut out64).is_err());
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn vsl_uniform_rejects_nonfinite_span_on_aarch64() {
    // `a < b` holds for `MIN..MAX`, but the span `b - a` overflows to infinity,
    // which `rand`'s `Uniform::new` rejects. The aarch64 backend must surface
    // that as an error (matching Intel VSL) rather than panicking on the old
    // `.expect("a < b was validated above")`.
    let stream = vsl::Stream::new(11).unwrap();
    let mut out = [0.0f32; 4];
    assert!(stream.uniform(f32::MIN, f32::MAX, &mut out).is_err());

    let mut out64 = [0.0f64; 4];
    assert!(stream.uniform64(f64::MIN, f64::MAX, &mut out64).is_err());
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn pardiso_rejects_symmetric_mtype_on_aarch64() {
    // The Accelerate QR backend factors a full matrix; symmetric `mtype`s store
    // only one triangle and must be rejected rather than silently mis-solved.
    let ia = [1i32, 3, 5];
    let ja = [1i32, 2, 1, 2];
    let a = [2.0f64, 1.0, 1.0, 3.0];
    let b = [4.0f64, 10.0];
    let mut solver = pardiso::Pardiso::new(pardiso::mtype::SPD);
    assert!(solver.solve(&ia, &ja, &a, &b).is_err());
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn pardiso_detects_singular_on_aarch64() {
    // A = [[1,1],[1,1]] is singular; QR still "succeeds" and returns a
    // least-squares solution, so the residual check must turn it into an error.
    let ia = [1i32, 3, 5];
    let ja = [1i32, 2, 1, 2];
    let a = [1.0f64, 1.0, 1.0, 1.0];
    let b = [1.0f64, 0.0];
    let mut solver = pardiso::Pardiso::new(pardiso::mtype::NONSYMMETRIC);
    assert!(solver.solve(&ia, &ja, &a, &b).is_err());
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn pardiso_releases_failed_factorization_zero_matrix_on_aarch64() {
    use nuvai_mkl::error::ErrorKind;

    // Regression test for the non-OK destroy path in `Pardiso::solve` (#30,
    // which proposed *skipping* the release — doing so would leak).
    //
    // The all-zero 2x2 reaches the branch as state 3 — see the comment at the
    // release in `Pardiso::solve`: `_SparseFactorQR_Double` itself fails
    // (`status == -2`, `symbolicFactorization.status == 0`, non-NULL
    // `numericFactorization`), so there is an allocated numeric factor the
    // wrapper must release. This is *not* caught earlier by `check_residual`,
    // which is what handles the singular-but-non-zero matrix in
    // `pardiso_detects_singular_on_aarch64`.
    //
    // The kind assertion is load-bearing for the same reason as in
    // `dss_releases_failed_factorization_on_aarch64`: `InvalidArgument` (the
    // residual check) or `Unsupported` would mean this branch never ran.
    let ia = [1i32, 3, 5];
    let ja = [1i32, 2, 1, 2];
    let a = [0.0f64, 0.0, 0.0, 0.0];
    let b = [1.0f64, 1.0];
    let mut solver = pardiso::Pardiso::new(pardiso::mtype::NONSYMMETRIC);
    let err = solver
        .solve(&ia, &ja, &a, &b)
        .expect_err("QR of the all-zero matrix must fail");
    assert_eq!(err.kind(), ErrorKind::Mkl, "{err}");
}

// A structurally empty row (`ia = [1,2,2]`, `ja = [1]`) reaches the same
// branch as a *different* state — state 1, `status == -2`,
// `symbolicFactorization.status == -3`, NULL numeric — and there is
// deliberately no test for it. Measured on the `macos-14` CI runner,
// `_SparseFactorQR_Double` does not return that state there: it aborts the
// process with SIGTRAP, so a test asserting on it can never pass in CI. A
// probe that called the factorization and leaked the result without ever
// calling `_SparseDestroyOpaqueNumeric` aborted identically, which places the
// abort inside the factorization call and not in this wrapper's release.
// macOS 26 returns the state-1 object normally.
//
// Nothing is lost by leaving it uncovered: state 1 keeps nothing valid, so
// there is no memory to leak — the leak #30 was about is state 3, which the
// test above does pin. Gating a state-1 test on the macOS version instead
// would only look like coverage, since every CI runner is macOS 14.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn dss_rejects_lower_triangle_on_aarch64() {
    // Same SPD matrix as `dss_solve_2x2` but stored as the *lower* triangle,
    // which the Accelerate Cholesky backend does not accept.
    //
    // This test does *not* reach Accelerate: `Dss::factor_symmetric` rejects
    // lower-triangle storage in its own CSR validation (`csr_to_csc`, the
    // `upper_only` check) before `_SparseFactorSymmetric_Double` is called. It
    // is a test of the wrapper's validation. The non-OK *destroy* path (#30) is
    // covered separately by `dss_releases_failed_factorization_on_aarch64` —
    // asserting only `is_err()` here cannot tell the two apart.
    let row_index = [0i32, 1, 3];
    let columns = [0i32, 0, 1];
    let values = [4.0f64, 1.0, 3.0];
    assert!(dss::Dss::factor_symmetric(&row_index, &columns, &values).is_err());
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn dss_releases_failed_factorization_on_aarch64() {
    use nuvai_mkl::error::ErrorKind;

    // Regression test for the non-OK destroy path in `Dss::factor_symmetric`
    // (#30, which proposed *skipping* the release — doing so would leak).
    //
    // A = [[1, 2], [2, 1]] is symmetric indefinite (eigenvalues 3 and -1).
    // Stored as the *upper* triangle so it clears the wrapper's `upper_only`
    // validation and genuinely reaches Accelerate, where
    // `SparseFactorizationCholesky` cannot factor it.
    //
    // The assertion on the *kind* is the whole point: `Mkl` means the
    // factorization call ran and returned `status != SparseStatusOK`, so the
    // release below it executed against a real failed factor. An
    // `InvalidArgument` or `Unsupported` here would mean the matrix was
    // rejected before Accelerate was ever called and the destroy never ran —
    // which is exactly the false coverage that
    // `dss_rejects_lower_triangle_on_aarch64` was mistakenly believed to give.
    //
    // Measured on macOS 26, this matrix returns state 3 (see the comment at the
    // release): `status == -1`, `symbolicFactorization.status == 0`, non-NULL
    // `numericFactorization`.
    let row_index = [0i32, 2, 3];
    let columns = [0i32, 1, 1];
    let values = [1.0f64, 2.0, 1.0];
    let err = dss::Dss::factor_symmetric(&row_index, &columns, &values)
        .err()
        .expect("Cholesky of an indefinite matrix must fail");
    assert_eq!(err.kind(), ErrorKind::Mkl, "{err}");
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn fft_rejects_unsupported_length_on_aarch64() {
    use nuvai_mkl::error::ErrorKind;

    // 7 is prime and not a product of {2,3,5}, so vDSP cannot plan it. The
    // wrapper must surface an error rather than fail on a null setup.
    assert!(fft::FftPlan::new_c32(7).is_err());
    assert!(fft::FftPlan::new_c64(7).is_err());

    // The *kind* is the point of #31: a length vDSP cannot plan is
    // `Unsupported` ("give up"), which is a different answer from the
    // `ResourceExhausted` a failed setup now returns ("this works, but not right
    // now"). Pinning it here keeps a blanket reclassification of all three
    // `create` arms from passing unnoticed.
    //
    // The `ResourceExhausted` arm is deliberately not covered: forcing vDSP to
    // return a null setup needs a length large enough to exhaust the allocator,
    // and measured on macOS 26 the planner grinds on such a length for tens of
    // seconds rather than failing fast (a probe at 2^40 was still allocating
    // after 45s of CPU and had to be killed), so a test for it would hang CI.
    for err in [
        fft::FftPlan::new_c32(7).err().unwrap(),
        fft::FftPlan::new_c64(7).err().unwrap(),
    ] {
        assert_eq!(err.kind(), ErrorKind::Unsupported, "{err}");
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn pardiso_rejects_bad_csr_indices_on_aarch64() {
    // row_index[0] must equal the 1-based index base; anything else would
    // silently mis-index entries, so it must be rejected up front.
    let ia = [0i32, 2, 3];
    let ja = [1i32, 2, 1];
    let a = [2.0f64, 1.0, 3.0];
    let b = [4.0f64, 10.0];
    let mut solver = pardiso::Pardiso::new(pardiso::mtype::NONSYMMETRIC);
    assert!(solver.solve(&ia, &ja, &a, &b).is_err());
}

/// On `aarch64-unknown-linux-gnu` OpenBLAS covers only BLAS/LAPACK, so every
/// other domain must return `ErrorKind::Unsupported` — never a silent no-op or
/// a panic (ADR-0003, decision 2).
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
mod linux_aarch64_unsupported {
    use super::*;
    use nuvai_mkl::error::ErrorKind;

    #[test]
    fn fft_plan_unsupported() {
        // `.err().unwrap()` rather than `.unwrap_err()`: the Ok type
        // (`FftPlan`) does not implement `Debug`.
        assert_eq!(
            fft::FftPlan::new_c32(4).err().unwrap().kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(
            fft::FftPlan::new_c64(4).err().unwrap().kind(),
            ErrorKind::Unsupported
        );
    }

    #[test]
    fn vml_unsupported() {
        let mut dst = [0.0f32; 2];
        assert_eq!(
            vml::exp(&[0.0, 1.0], &mut dst).err().unwrap().kind(),
            ErrorKind::Unsupported
        );
    }

    #[test]
    fn vsl_stream_unsupported() {
        assert_eq!(
            vsl::Stream::new(1).err().unwrap().kind(),
            ErrorKind::Unsupported
        );
    }

    #[test]
    fn pardiso_solve_unsupported() {
        let ia = [1i32, 3, 5];
        let ja = [1i32, 2, 1, 2];
        let a = [2.0f64, 1.0, 1.0, 3.0];
        let b = [4.0f64, 10.0];
        let mut solver = pardiso::Pardiso::new(pardiso::mtype::NONSYMMETRIC);
        assert_eq!(
            solver.solve(&ia, &ja, &a, &b).err().unwrap().kind(),
            ErrorKind::Unsupported
        );
    }

    #[test]
    fn dss_factor_unsupported() {
        let row_index = [0i32, 2, 3];
        let columns = [0i32, 1, 1];
        let values = [4.0f64, 1.0, 3.0];
        assert_eq!(
            dss::Dss::factor_symmetric(&row_index, &columns, &values)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::Unsupported
        );
    }
}
