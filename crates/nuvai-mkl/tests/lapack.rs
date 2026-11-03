//! LAPACK (`nuvai_mkl::lapack`) smoke tests against the real MKL 2026.1.0 acquired by `nuvai-mkl-src`.

use nuvai_mkl::lapack;
use nuvai_mkl::layout::Layout;
mod common;
use common::assert_close;
use common::assert_close64;

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
