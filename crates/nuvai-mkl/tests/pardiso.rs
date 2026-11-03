//! PARDISO (`nuvai_mkl::pardiso`) smoke tests against the real MKL 2026.1.0 acquired by `nuvai-mkl-src`.

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
use nuvai_mkl::pardiso;
mod common;
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
use common::assert_close64;

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

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn pardiso_rejects_structurally_empty_row_on_aarch64() {
    use nuvai_mkl::error::ErrorKind;

    // Regression test for #58: a CSR matrix with a structurally empty row used
    // to be passed straight to `_SparseFactorQR_Double`, and on macOS 14 that
    // call does not return — it aborts the process with SIGTRAP, which no
    // caller can catch. `n = 2`; row 0 has one entry, row 1 has none.
    //
    // This test previously could not exist, and the block it replaces said so
    // rather than asserting anything: the abort happens inside Accelerate, and
    // every CI runner is macOS 14, while this was authored on macOS 26 where
    // the same input returns a state-1 error object normally — so the case was
    // observable only as a comment explaining why it went untested. Rejecting
    // the input before the factorization is what makes it testable *and*
    // portable: the assertion below now holds on both macOS versions, because
    // the call never reaches Accelerate.
    //
    // `InvalidArgument` specifically, not `is_err()`: it pins that the
    // wrapper's own guard fired, rather than an error that arrived from
    // Accelerate. A sanitizer or a macOS-14 CI job would be needed to observe
    // the abort itself; the guard is what prevents it.
    let ia = [1i32, 2, 2];
    let ja = [1i32];
    let a = [1.0f64];
    let b = [1.0f64, 1.0];
    let mut solver = pardiso::Pardiso::new(pardiso::mtype::NONSYMMETRIC);
    let err = solver
        .solve(&ia, &ja, &a, &b)
        .expect_err("a structurally empty row describes a singular matrix");
    assert_eq!(err.kind(), ErrorKind::InvalidArgument, "{err}");

    // The guard is positional-agnostic: the same single entry, moved to row 1,
    // leaves row 0 empty and is rejected identically.
    let ia = [1i32, 1, 2];
    let err = solver
        .solve(&ia, &ja, &a, &b)
        .expect_err("leading empty row");
    assert_eq!(err.kind(), ErrorKind::InvalidArgument, "{err}");

    // And the guard rejects nothing valid: a full 2x2 still solves.
    let ia = [1i32, 3, 5];
    let ja = [1i32, 2, 1, 2];
    let a = [2.0f64, 1.0, 1.0, 3.0];
    let x = solver.solve(&ia, &ja, &a, &[4.0f64, 7.0]).unwrap();
    assert_close64(&x, &[1.0, 2.0], 1e-9);
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

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn pardiso_rejects_csr_row_count_mismatch_on_aarch64() {
    // `row_index[n]` must equal `nnz + base`. That comparison is now widened to
    // `i64` (it read `nnz as i32 + base`, which truncates above `i32::MAX`), so
    // this pins that widening it changed no verdict for values that fit: the
    // mismatch is still rejected, and the well-formed CSR below still solves.
    //
    // The case the widening actually fixes — `nnz` above `i32::MAX` — is not
    // reachable from a test: it needs a `&[i32]` of more than 2^31 elements,
    // 8 GiB no test can allocate. Same reason `conv::len_to_c_int` keeps its
    // rule in one unit-tested function instead of at each call site.
    use nuvai_mkl::error::ErrorKind;
    let mut solver = pardiso::Pardiso::new(pardiso::mtype::NONSYMMETRIC);

    // `ia = [1, 1, 2]` with `nnz = 2` implies one stored entry, not two. The
    // first two entries agree with the 1-based base, so the mismatch check is
    // what rejects this.
    let err = solver
        .solve(&[1i32, 1, 2], &[1i32, 1], &[1.0f64, 1.0], &[1.0f64, 1.0])
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);

    // The well-formed path still solves — the widened check rejects nothing valid.
    let ia = [1i32, 3, 6, 8];
    let ja = [1i32, 2, 1, 2, 3, 2, 3];
    let a = [2.0f64, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
    let x = solver.solve(&ia, &ja, &a, &[4.0f64, 10.0, 8.0]).unwrap();
    assert_close64(&x, &[1.0, 2.0, 3.0], 1e-9);
}

/// On `aarch64-unknown-linux-gnu` OpenBLAS covers only BLAS/LAPACK, so every
/// other domain must return `ErrorKind::Unsupported` — never a silent no-op or
/// a panic (ADR-0003, decision 2).
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
mod linux_aarch64 {
    use nuvai_mkl::error::ErrorKind;
    use nuvai_mkl::pardiso;

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
}
