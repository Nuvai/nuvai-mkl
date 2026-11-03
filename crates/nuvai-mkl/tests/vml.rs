//! VML (`nuvai_mkl::vml`) smoke tests against the real MKL 2026.1.0 acquired by `nuvai-mkl-src`.

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
use nuvai_mkl::vml;
mod common;
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
use common::assert_close;
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
use common::assert_close64;

/// Exercise every unary VML function (all 13) in single precision against known
/// values. The f32/f64 variants run through both backends (MKL VML on Intel,
/// Accelerate vForce — and vDSP for `sqr` — on aarch64).
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

    // #49. `tanh` is the activation function; `sqr` is the one unary operation
    // the Apple Silicon backend takes from vDSP rather than vForce, which has no
    // squaring function.
    let mut dst = [0.0f32; 3];
    vml::tanh(&[0.0, 1.0, -1.0], &mut dst).unwrap();
    assert_close(&dst, &[0.0, 0.761_594_2, -0.761_594_2], 1e-5);

    vml::sqr(&[0.0, 1.5, -3.0], &mut dst).unwrap();
    assert_close(&dst, &[0.0, 2.25, 9.0], 1e-5);
}

/// Every unary VML function in double precision (vForce `D` variants on
/// aarch64, `vDSP_vsqD` for `sqr`).
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

    let mut dst = [0.0f64; 3];
    vml::dtanh(&[0.0, 1.0, -1.0], &mut dst).unwrap();
    assert_close64(
        &dst,
        &[0.0, 0.761_594_155_955_765, -0.761_594_155_955_765],
        1e-12,
    );

    vml::dsqr(&[0.0, 1.5, -3.0], &mut dst).unwrap();
    assert_close64(&dst, &[0.0, 2.25, 9.0], 1e-12);
}

/// Every binary VML operation in single precision (#48).
///
/// `sub` and `div` are the cases that would expose a wrong argument order on
/// Apple Silicon: vDSP declares `vDSP_vsub`/`vDSP_vdiv` as `(B, A)`, so dropping
/// the compensating swap turns `a - b` into `b - a` and `a / b` into `b / a`
/// while leaving `add`/`mul`/`fmax`/`fmin` correct — which is why the operands
/// here are asymmetric and the two differ in value.
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn vml_binary_ops_f32() {
    let a = [1.0f32, 4.0, -2.0];
    let b = [2.0f32, 2.0, 3.0];
    let mut r = [0.0f32; 3];

    vml::add(&a, &b, &mut r).unwrap();
    assert_close(&r, &[3.0, 6.0, 1.0], 1e-6);

    vml::sub(&a, &b, &mut r).unwrap();
    assert_close(&r, &[-1.0, 2.0, -5.0], 1e-6);

    vml::mul(&a, &b, &mut r).unwrap();
    assert_close(&r, &[2.0, 8.0, -6.0], 1e-6);

    vml::div(&a, &b, &mut r).unwrap();
    assert_close(&r, &[0.5, 2.0, -2.0 / 3.0], 1e-6);

    vml::fmax(&a, &b, &mut r).unwrap();
    assert_close(&r, &[2.0, 4.0, 3.0], 1e-6);

    vml::fmin(&a, &b, &mut r).unwrap();
    assert_close(&r, &[1.0, 2.0, -2.0], 1e-6);
}

/// Every binary VML operation in double precision.
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn vml_binary_ops_f64() {
    let a = [1.0f64, 4.0, -2.0];
    let b = [2.0f64, 2.0, 3.0];
    let mut r = [0.0f64; 3];

    vml::dadd(&a, &b, &mut r).unwrap();
    assert_close64(&r, &[3.0, 6.0, 1.0], 1e-12);

    vml::dsub(&a, &b, &mut r).unwrap();
    assert_close64(&r, &[-1.0, 2.0, -5.0], 1e-12);

    vml::dmul(&a, &b, &mut r).unwrap();
    assert_close64(&r, &[2.0, 8.0, -6.0], 1e-12);

    vml::ddiv(&a, &b, &mut r).unwrap();
    assert_close64(&r, &[0.5, 2.0, -2.0 / 3.0], 1e-12);

    vml::dfmax(&a, &b, &mut r).unwrap();
    assert_close64(&r, &[2.0, 4.0, 3.0], 1e-12);

    vml::dfmin(&a, &b, &mut r).unwrap();
    assert_close64(&r, &[1.0, 2.0, -2.0], 1e-12);
}

/// The binary path validates all three buffers, on the same rule the unary
/// path's `src`/`dst` check applies (#48): unequal lengths are rejected before
/// any pointer reaches the backend, whichever buffer is the odd one out.
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
#[test]
fn vml_binary_rejects_length_mismatch() {
    use nuvai_mkl::error::ErrorKind;

    let a = [1.0f32, 2.0, 3.0];
    let mut r = [0.0f32; 3];

    // `b` short of `a`.
    assert_eq!(
        vml::add(&a, &[1.0, 2.0], &mut r).err().unwrap().kind(),
        ErrorKind::InvalidArgument
    );
    // `r` short of both.
    assert_eq!(
        vml::add(&a, &[1.0, 2.0, 3.0], &mut r[..2])
            .err()
            .unwrap()
            .kind(),
        ErrorKind::InvalidArgument
    );
}

/// On `aarch64-unknown-linux-gnu` OpenBLAS covers only BLAS/LAPACK, so every
/// other domain must return `ErrorKind::Unsupported` — never a silent no-op or
/// a panic (ADR-0003, decision 2).
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
mod linux_aarch64 {
    use nuvai_mkl::error::ErrorKind;
    use nuvai_mkl::vml;

    #[test]
    fn vml_unsupported() {
        let mut dst = [0.0f32; 2];
        assert_eq!(
            vml::exp(&[0.0, 1.0], &mut dst).err().unwrap().kind(),
            ErrorKind::Unsupported
        );

        // The unary additions (#49) and every binary operation (#48) take the
        // same path — Unsupported before any argument is inspected.
        assert_eq!(
            vml::sqr(&[0.0, 1.0], &mut dst).err().unwrap().kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(
            vml::tanh(&[0.0, 1.0], &mut dst).err().unwrap().kind(),
            ErrorKind::Unsupported
        );

        let mut r = [0.0f32; 2];
        assert_eq!(
            vml::add(&[0.0, 1.0], &[1.0, 2.0], &mut r)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::Unsupported
        );
        // Asserted with mismatched operand lengths on purpose: `Unsupported`
        // comes before validation, so this must not be an `InvalidArgument`.
        assert_eq!(
            vml::sub(&[0.0, 1.0], &[1.0], &mut r).err().unwrap().kind(),
            ErrorKind::Unsupported
        );
    }

    #[test]
    fn vml_unsupported_names_domain() {
        // Mismatched lengths are invalid on every backend that validates; the
        // domain name in the message is asserted so a copy-pasted domain string
        // in `src/vml.rs` cannot pass.
        let mut r = [0.0f32; 2];
        let err = vml::add(&[0.0, 1.0], &[1.0], &mut r).err().unwrap();
        assert_eq!(err.kind(), ErrorKind::Unsupported, "{err}");
        assert!(err.to_string().contains("VML"), "{err}");
    }
}
