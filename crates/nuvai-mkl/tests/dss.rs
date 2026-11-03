//! DSS (`nuvai_mkl::dss`) smoke tests against the real MKL 2026.1.0 acquired by `nuvai-mkl-src`.

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
use nuvai_mkl::dss;
mod common;
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
use common::assert_close64;

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

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn dss_rejects_structurally_empty_row_on_aarch64() {
    use nuvai_mkl::error::ErrorKind;

    // The same guard, through the shared CSR→CSC transposition that DSS's
    // Cholesky path also calls (#58). A single empty row is a zero row of the
    // full symmetric matrix, so it is singular here too — and the guard runs
    // before either backend's factorization.
    //
    // `n = 2`, upper triangle, 0-based: row 0 carries the (0,1) entry and row 1
    // is empty.
    let row_index = [0i32, 1, 1];
    let columns = [1i32];
    let values = [1.0f64];
    let err = dss::Dss::factor_symmetric(&row_index, &columns, &values)
        .err()
        .expect("a structurally empty row describes a singular matrix");
    assert_eq!(err.kind(), ErrorKind::InvalidArgument, "{err}");

    // A complete upper triangle of the same shape still factors, so the guard
    // rejects only the degenerate case.
    let row_index = [0i32, 2, 3];
    let columns = [0i32, 1, 1];
    let values = [2.0f64, 1.0, 3.0];
    let solver = dss::Dss::factor_symmetric(&row_index, &columns, &values).unwrap();
    let x = solver.solve(&[4.0f64, 7.0]).unwrap();
    assert_close64(&x, &[1.0, 2.0], 1e-9);
}

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
fn dss_rejects_csr_row_count_mismatch_on_aarch64() {
    // The same widened comparison, with the 0-based base DSS passes — so this
    // covers the `i64::from(base)` term at its other value. `row_index[n]` must
    // equal `nnz`, and here it does not (2 against 1).
    use nuvai_mkl::error::ErrorKind;
    let row_index = [0i32, 1, 2];
    let columns = [0i32];
    let values = [1.0f64];
    let err = dss::Dss::factor_symmetric(&row_index, &columns, &values)
        .err()
        .unwrap();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
}

/// On `aarch64-unknown-linux-gnu` OpenBLAS covers only BLAS/LAPACK, so every
/// other domain must return `ErrorKind::Unsupported` — never a silent no-op or
/// a panic (ADR-0003, decision 2).
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
mod linux_aarch64 {
    use nuvai_mkl::dss;
    use nuvai_mkl::error::ErrorKind;

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
