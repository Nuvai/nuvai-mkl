//! BLAS (`nuvai_mkl::blas`) smoke tests against the real MKL 2026.1.0 acquired by `nuvai-mkl-src`.

use nuvai_mkl::blas;
use nuvai_mkl::layout::{Layout, Transpose};
mod common;
use common::assert_close;

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

/// Half-precision GEMM (#47), against oneMKL's `cblas_hgemm`.
///
/// Intel targets only — both aarch64 backends return `Unsupported`, asserted
/// separately below. The operands and the product are all integers below 2048,
/// which IEEE-754 binary16 represents exactly, so the result is asserted
/// bit-for-bit rather than within a tolerance. The bit patterns are the ones
/// `struct.pack('<e', v)` produces in Python for these decimals.
#[cfg(not(target_arch = "aarch64"))]
#[test]
fn blas_hgemm_2x2() {
    // A = [[1,2],[3,4]] and B = [[5,6],[7,8]], row-major — the same operands as
    // `blas_sgemm_2x2`, converted to binary16.
    let a = [0x3C00u16, 0x4000, 0x4200, 0x4400]; //  1.0  2.0  3.0  4.0
    let b = [0x4500u16, 0x4600, 0x4700, 0x4800]; //  5.0  6.0  7.0  8.0
    let mut c = [0u16; 4];
    blas::hgemm(
        Layout::RowMajor,
        Transpose::NoTrans,
        Transpose::NoTrans,
        2,
        2,
        2,
        0x3C00, // alpha = 1.0
        &a,
        2,
        &b,
        2,
        0x0000, // beta = 0.0
        &mut c,
        2,
    )
    .unwrap();
    // A·B = [[19,22],[43,50]].
    assert_eq!(c, [0x4CC0u16, 0x4D80, 0x5160, 0x5240]);
}

/// The fp16 path validates exactly as `sgemm` does (#47): an undersized operand
/// must error rather than reach CBLAS, which performs no bounds checking.
#[cfg(not(target_arch = "aarch64"))]
#[test]
fn blas_hgemm_rejects_undersized_slices() {
    let a = [0x3C00u16; 4];
    let b = [0x3C00u16; 4];
    let mut c = [0u16; 4];
    assert!(
        blas::hgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            // `m` far past what the 4-element `a` covers — the same shape as
            // `blas_rejects_undersized_slices`.
            100_000,
            2,
            2,
            0x3C00,
            &a,
            2,
            &b,
            2,
            0x0000,
            &mut c,
            2,
        )
        .is_err()
    );
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

/// aarch64 has no fp16 GEMM, on either OS. Accelerate's CBLAS (macOS) carries no
/// half-precision entry point, and OpenBLAS (Linux, and macOS under the
/// `openblas` feature) exposes no `hgemm`/`cblas_hgemm` symbol. #47 requires the
/// fp16 path to fail loudly instead of falling back to an fp32 one, and before
/// validating its arguments (ADR-0003, decision 2). `blas::hgemm` gates on
/// `target_arch = "aarch64"` alone, so this is one contract for both OSes.
#[cfg(target_arch = "aarch64")]
#[test]
fn blas_hgemm_unsupported_on_aarch64() {
    use nuvai_mkl::error::ErrorKind;
    let a = [0x3C00u16; 4];
    let b = [0x3C00u16; 4];
    let mut c = [0u16; 4];
    assert_eq!(
        blas::hgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            0x3C00,
            &a,
            2,
            &b,
            2,
            0x0000,
            &mut c,
            2,
        )
        .err()
        .unwrap()
        .kind(),
        ErrorKind::Unsupported
    );
}

/// Unsupported comes before argument validation (#47), so an undersized operand
/// still yields `Unsupported` rather than `InvalidArgument`. Mirrors
/// `blas_hgemm_rejects_undersized_slices`, which runs on the other targets.
#[cfg(target_arch = "aarch64")]
#[test]
fn blas_hgemm_unsupported_before_validation() {
    use nuvai_mkl::error::ErrorKind;
    let a = [0x3C00u16; 4];
    let b = [0x3C00u16; 4];
    let mut c = [0u16; 4];
    let err = blas::hgemm(
        Layout::RowMajor,
        Transpose::NoTrans,
        Transpose::NoTrans,
        100_000,
        2,
        2,
        0x3C00,
        &a,
        2,
        &b,
        2,
        0x0000,
        &mut c,
        2,
    )
    .err()
    .unwrap();
    assert_eq!(err.kind(), ErrorKind::Unsupported, "{err}");
    assert!(err.to_string().contains("hgemm"), "{err}");
}
