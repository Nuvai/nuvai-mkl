//! End-to-end smoke tests exercising all six oneMKL domains against the real
//! MKL 2026.1.0 acquired by `nuvai-mkl-src`.

use nuvai_mkl::fft::{MKL_Complex16, MKL_Complex8};
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

    let dot = blas::sdot(3, &x, 1, &[4.0, 5.0, 6.0], 1);
    assert!((dot - 32.0).abs() <= 1e-5, "dot = {dot}");
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
fn fft_roundtrip_c32() {
    let plan = fft::FftPlan::new_c32(4).unwrap();
    // Impulse δ = [1,0,0,0] -> forward = [1,1,1,1].
    let input = [
        MKL_Complex8 { real: 1.0, imag: 0.0 },
        MKL_Complex8 { real: 0.0, imag: 0.0 },
        MKL_Complex8 { real: 0.0, imag: 0.0 },
        MKL_Complex8 { real: 0.0, imag: 0.0 },
    ];
    let mut freq = [MKL_Complex8 { real: 0.0, imag: 0.0 }; 4];
    plan.forward_c32(&input, &mut freq).unwrap();
    for f in &freq {
        assert!((f.real - 1.0).abs() <= 1e-5, "real = {}", f.real);
        assert!(f.imag.abs() <= 1e-5, "imag = {}", f.imag);
    }

    // Backward of [1,1,1,1] recovers the impulse (default 1/n scaling).
    let mut out = [MKL_Complex8 { real: 0.0, imag: 0.0 }; 4];
    plan.backward_c32(&freq, &mut out).unwrap();
    assert!((out[0].real - 1.0).abs() <= 1e-5, "out[0] = {}", out[0].real);
    for o in &out[1..] {
        assert!(o.real.abs() <= 1e-5 && o.imag.abs() <= 1e-5);
    }
}

#[test]
fn fft_roundtrip_c64() {
    let plan = fft::FftPlan::new_c64(4).unwrap();
    // Same impulse test as the c32 variant, in double precision.
    let input = [
        MKL_Complex16 { real: 1.0, imag: 0.0 },
        MKL_Complex16 { real: 0.0, imag: 0.0 },
        MKL_Complex16 { real: 0.0, imag: 0.0 },
        MKL_Complex16 { real: 0.0, imag: 0.0 },
    ];
    let mut freq = [MKL_Complex16 { real: 0.0, imag: 0.0 }; 4];
    plan.forward_c64(&input, &mut freq).unwrap();
    for f in &freq {
        assert!((f.real - 1.0).abs() <= 1e-9, "real = {}", f.real);
        assert!(f.imag.abs() <= 1e-9, "imag = {}", f.imag);
    }

    let mut out = [MKL_Complex16 { real: 0.0, imag: 0.0 }; 4];
    plan.backward_c64(&freq, &mut out).unwrap();
    assert!((out[0].real - 1.0).abs() <= 1e-9, "out[0] = {}", out[0].real);
    for o in &out[1..] {
        assert!(o.real.abs() <= 1e-9 && o.imag.abs() <= 1e-9);
    }
}

/// Exercise every VML function (all 11) in single precision against known
/// values. The f32/f64 variants run through both backends (MKL VML on Intel,
/// Accelerate vForce on aarch64).
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
}

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
