//! End-to-end smoke tests exercising all six oneMKL domains against the real
//! MKL 2026.1.0 acquired by `nuvai-mkl-src`.

use nuvai_mkl::fft::MKL_Complex8;
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
fn vml_exp_sqrt() {
    let src = [0.0f32, 1.0, 2.0];
    let mut dst = [0.0f32; 3];
    vml::exp(&src, &mut dst).unwrap();
    assert_close(&dst, &[1.0, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E], 1e-4);

    let src = [0.0f32, 1.0, 4.0, 9.0];
    let mut dst = [0.0f32; 4];
    vml::sqrt(&src, &mut dst).unwrap();
    assert_close(&dst, &[0.0, 1.0, 2.0, 3.0], 1e-5);
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
