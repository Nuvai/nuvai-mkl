//! The three calls both fixtures make, so the pair differs in exactly one
//! thing: the link plumbing (#70).
//!
//! Reached through `#[path = "../../checks.rs"] mod checks;` from each
//! fixture's `src/main.rs` — the file is not a crate of its own, and the
//! fixtures are separate workspaces, so neither can depend on the other.

use nuvai_mkl::layout::{Layout, Transpose};
use nuvai_mkl::{blas, vml};

/// Call one function from each of the domains in the issue's repro — a VML
/// vector op, an sgemm and an hgemm — and assert the results.
///
/// The numbers are the issue's, and the fp16 operands are the bit patterns
/// `smoke.rs` uses: every value is an integer below 2048, which binary16
/// represents exactly, so the product is asserted bit-for-bit.
pub fn run() {
    // VML.
    let mut exp = [0.0f32; 3];
    vml::exp(&[0.0, 1.0, 2.0], &mut exp).expect("vml::exp");
    assert!(
        (exp[0] - 1.0).abs() < 1e-6 && (exp[1] - std::f32::consts::E).abs() < 1e-6,
        "vml::exp gave {exp:?}"
    );

    // BLAS: C := A·B for A = [[1,2],[3,4]], B = [[5,6],[7,8]], row-major.
    let (a, b) = ([1.0f32, 2.0, 3.0, 4.0], [5.0f32, 6.0, 7.0, 8.0]);
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
    .expect("blas::sgemm");
    assert_eq!(c, [19.0, 22.0, 43.0, 50.0], "blas::sgemm gave {c:?}");

    // Half-precision GEMM. It is a oneMKL extension with no aarch64 equivalent,
    // so it is asserted on the Intel targets only — these fixtures are x86_64
    // Linux, and the `cfg` is what keeps the check honest if one is ever run
    // elsewhere.
    #[cfg(not(target_arch = "aarch64"))]
    {
        let (a, b) = (
            [0x3C00u16, 0x4000, 0x4200, 0x4400], //  1.0  2.0  3.0  4.0
            [0x4500u16, 0x4600, 0x4700, 0x4800], //  5.0  6.0  7.0  8.0
        );
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
        .expect("blas::hgemm");
        assert_eq!(
            c,
            [0x4CC0u16, 0x4D80, 0x5160, 0x5240],
            "blas::hgemm gave {c:?}"
        );
    }
}
