//! VSL (`nuvai_mkl::vsl`) smoke tests against the real MKL 2026.1.0 acquired by `nuvai-mkl-src`.

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
use nuvai_mkl::vsl;

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

/// On `aarch64-unknown-linux-gnu` OpenBLAS covers only BLAS/LAPACK, so every
/// other domain must return `ErrorKind::Unsupported` — never a silent no-op or
/// a panic (ADR-0003, decision 2).
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
mod linux_aarch64 {
    use nuvai_mkl::error::ErrorKind;
    use nuvai_mkl::vsl;

    #[test]
    fn vsl_stream_unsupported() {
        assert_eq!(
            vsl::Stream::new(1).err().unwrap().kind(),
            ErrorKind::Unsupported
        );
    }

    #[test]
    fn vsl_unsupported_names_domain() {
        let err = vsl::Stream::new(0).err().unwrap();
        assert_eq!(err.kind(), ErrorKind::Unsupported, "{err}");
        assert!(err.to_string().contains("VSL"), "{err}");
    }
}
