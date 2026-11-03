//! FFT (`nuvai_mkl::fft`) smoke tests against the real MKL 2026.1.0 acquired by `nuvai-mkl-src`.
//!
//! On `aarch64-unknown-linux-gnu` FFT returns `Unsupported`, so only the
//! `linux_aarch64` module compiles there.

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
use nuvai_mkl::fft;
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
use nuvai_mkl::fft::{MKL_Complex8, MKL_Complex16};

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

/// On `aarch64-unknown-linux-gnu` OpenBLAS covers only BLAS/LAPACK, so every
/// other domain must return `ErrorKind::Unsupported` — never a silent no-op or
/// a panic (ADR-0003, decision 2).
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
mod linux_aarch64 {
    use nuvai_mkl::error::ErrorKind;
    use nuvai_mkl::fft;

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
}
