#![cfg(all(target_os = "macos", target_arch = "aarch64"))]
//! DIAGNOSTIC ONLY — throwaway instrumentation for issue #53.
//!
//! Not a real test: it prints evidence about the macOS 14 vDSP DFT runtime and
//! then panics on purpose, because libtest only reveals captured stdout for
//! *failing* tests (CI runs `cargo test` without `--nocapture`). Delete this
//! file; it is not part of the suite.

use std::f64::consts::PI;

/// Print *and flush*, so output survives a hard abort mid-run (libtest captures
/// stdout, and a SIGABRT discards whatever is still buffered).
macro_rules! out {
    ($($arg:tt)*) => {{
        println!($($arg)*);
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }};
}

use nuvai_mkl::fft::{FftPlan, MKL_Complex16, MKL_Complex8};
use nuvai_mkl_sys::{
    vDSP_DFT_DestroySetup, vDSP_DFT_DestroySetupD, vDSP_DFT_Execute, vDSP_DFT_ExecuteD,
    vDSP_DFT_FORWARD, vDSP_DFT_INVERSE, vDSP_DFT_Interleaved_ComplextoComplex,
    vDSP_DFT_Interleaved_CreateSetup, vDSP_DFT_Interleaved_CreateSetupD,
    vDSP_DFT_Interleaved_DestroySetup, vDSP_DFT_Interleaved_DestroySetupD,
    vDSP_DFT_Interleaved_Execute, vDSP_DFT_Interleaved_ExecuteD, vDSP_DFT_zop_CreateSetup,
    vDSP_DFT_zop_CreateSetupD, DSPComplex, DSPDoubleComplex,
};

fn env_info() -> String {
    let mut s = String::new();
    for cmd in [
        &["sw_vers"][..],
        &["uname", "-a"][..],
        &["/usr/bin/arch"][..],
        &["sh", "-c", "sysctl -n machdep.cpu.brand_string"][..],
    ] {
        match std::process::Command::new(cmd[0]).args(&cmd[1..]).output() {
            Ok(o) => {
                s.push_str(&format!("$ {}\n{}", cmd.join(" "), String::from_utf8_lossy(&o.stdout)));
                if !o.stderr.is_empty() {
                    s.push_str(&format!("[stderr] {}", String::from_utf8_lossy(&o.stderr)));
                }
            }
            Err(e) => s.push_str(&format!("$ {} -> {e}\n", cmd.join(" "))),
        }
    }
    s
}

/// Deterministic, non-degenerate complex input. Deliberately *not* a unit
/// impulse: a delta at index 0 is invariant to twiddle-table errors (only j=0
/// contributes, e^0 = 1), so it cannot distinguish "wrong algorithm" from
/// "correct algorithm, wrong twiddles".
fn probe_input(n: usize) -> (Vec<f64>, Vec<f64>) {
    let re = (0..n).map(|j| ((j * 7 + 3) % 5) as f64 - 2.0).collect();
    let im = (0..n).map(|j| ((j * 11 + 1) % 7) as f64 - 3.0).collect();
    (re, im)
}

fn dft_analytic(re: &[f64], im: &[f64]) -> Vec<(f64, f64)> {
    let n = re.len();
    (0..n)
        .map(|k| {
            let mut sr = 0.0f64;
            let mut si = 0.0f64;
            for j in 0..n {
                let th = -2.0 * PI * (j * k) as f64 / n as f64;
                let (c, s) = (th.cos(), th.sin());
                sr += re[j] * c - im[j] * s;
                si += re[j] * s + im[j] * c;
            }
            (sr, si)
        })
        .collect()
}

fn max_err(got: &[(f64, f64)], want: &[(f64, f64)]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g.0 - w.0).abs().max((g.1 - w.1).abs()))
        .fold(0.0f64, f64::max)
}

fn fmt_err(e: Option<f64>) -> String {
    match e {
        Some(v) if v < 1e-9 => format!("{v:.3e} OK"),
        Some(v) if v < 1e-4 => format!("{v:.3e} ok?"),
        Some(v) => format!("{v:.3e} **WRONG**"),
        None => "setup NULL".to_string(),
    }
}

/// Interleaved family, single precision, forward, complex-to-complex.
fn ilv32_err(n: usize) -> Option<f64> {
    let (re, im) = probe_input(n);
    unsafe {
        let setup = vDSP_DFT_Interleaved_CreateSetup(
            std::ptr::null_mut(),
            n as _,
            vDSP_DFT_FORWARD,
            vDSP_DFT_Interleaved_ComplextoComplex,
        );
        if setup.is_null() {
            return None;
        }
        let input: Vec<DSPComplex> = (0..n)
            .map(|j| DSPComplex { real: re[j] as f32, imag: im[j] as f32 })
            .collect();
        let mut out = vec![DSPComplex { real: 0.0, imag: 0.0 }; n];
        vDSP_DFT_Interleaved_Execute(setup, input.as_ptr(), out.as_mut_ptr());
        vDSP_DFT_Interleaved_DestroySetup(setup);
        let got: Vec<(f64, f64)> = out.iter().map(|c| (c.real as f64, c.imag as f64)).collect();
        Some(max_err(&got, &dft_analytic(&re, &im)))
    }
}

fn ilv64_err(n: usize) -> Option<f64> {
    let (re, im) = probe_input(n);
    unsafe {
        let setup = vDSP_DFT_Interleaved_CreateSetupD(
            std::ptr::null_mut(),
            n as _,
            vDSP_DFT_FORWARD,
            vDSP_DFT_Interleaved_ComplextoComplex,
        );
        if setup.is_null() {
            return None;
        }
        let input: Vec<DSPDoubleComplex> = (0..n)
            .map(|j| DSPDoubleComplex { real: re[j], imag: im[j] })
            .collect();
        let mut out = vec![DSPDoubleComplex { real: 0.0, imag: 0.0 }; n];
        vDSP_DFT_Interleaved_ExecuteD(setup, input.as_ptr(), out.as_mut_ptr());
        vDSP_DFT_Interleaved_DestroySetupD(setup);
        let got: Vec<(f64, f64)> = out.iter().map(|c| (c.real, c.imag)).collect();
        Some(max_err(&got, &dft_analytic(&re, &im)))
    }
}

fn zop32_err(n: usize) -> Option<f64> {
    let (re, im) = probe_input(n);
    unsafe {
        let setup = vDSP_DFT_zop_CreateSetup(std::ptr::null_mut(), n as _, vDSP_DFT_FORWARD);
        if setup.is_null() {
            return None;
        }
        let ir: Vec<f32> = re.iter().map(|&v| v as f32).collect();
        let ii: Vec<f32> = im.iter().map(|&v| v as f32).collect();
        let mut or = vec![0.0f32; n];
        let mut oi = vec![0.0f32; n];
        vDSP_DFT_Execute(setup, ir.as_ptr(), ii.as_ptr(), or.as_mut_ptr(), oi.as_mut_ptr());
        vDSP_DFT_DestroySetup(setup);
        let got: Vec<(f64, f64)> = (0..n).map(|k| (or[k] as f64, oi[k] as f64)).collect();
        Some(max_err(&got, &dft_analytic(&re, &im)))
    }
}

fn zop64_err(n: usize) -> Option<f64> {
    let (re, im) = probe_input(n);
    unsafe {
        let setup = vDSP_DFT_zop_CreateSetupD(std::ptr::null_mut(), n as _, vDSP_DFT_FORWARD);
        if setup.is_null() {
            return None;
        }
        let mut or = vec![0.0f64; n];
        let mut oi = vec![0.0f64; n];
        vDSP_DFT_ExecuteD(setup, re.as_ptr(), im.as_ptr(), or.as_mut_ptr(), oi.as_mut_ptr());
        vDSP_DFT_DestroySetupD(setup);
        let got: Vec<(f64, f64)> = (0..n).map(|k| (or[k], oi[k])).collect();
        Some(max_err(&got, &dft_analytic(&re, &im)))
    }
}

fn crate32_err(n: usize) -> Option<f64> {
    let (re, im) = probe_input(n);
    let plan = FftPlan::new_c32(n).ok()?;
    let input: Vec<MKL_Complex8> =
        (0..n).map(|j| MKL_Complex8 { real: re[j] as f32, imag: im[j] as f32 }).collect();
    let mut out = vec![MKL_Complex8 { real: 0.0, imag: 0.0 }; n];
    plan.forward_c32(&input, &mut out).ok()?;
    let got: Vec<(f64, f64)> = out.iter().map(|c| (c.real as f64, c.imag as f64)).collect();
    Some(max_err(&got, &dft_analytic(&re, &im)))
}

fn crate64_err(n: usize) -> Option<f64> {
    let (re, im) = probe_input(n);
    let plan = FftPlan::new_c64(n).ok()?;
    let input: Vec<MKL_Complex16> = (0..n).map(|j| MKL_Complex16 { real: re[j], imag: im[j] }).collect();
    let mut out = vec![MKL_Complex16 { real: 0.0, imag: 0.0 }; n];
    plan.forward_c64(&input, &mut out).ok()?;
    let got: Vec<(f64, f64)> = out.iter().map(|c| (c.real, c.imag)).collect();
    Some(max_err(&got, &dft_analytic(&re, &im)))
}

/// Feed every basis vector e_j through the interleaved f32 family at length `n`
/// and dump the resulting linear map M[k][j], alongside the analytic
/// e^{-2*pi*i*j*k/n}. This characterises *what* the runtime actually computed.
fn dump_matrix(n: usize) {
    out!("--- basis-vector map, interleaved f32, n={n} (rows k = output bin, cols j = input index) ---");
    unsafe {
        let setup = vDSP_DFT_Interleaved_CreateSetup(
            std::ptr::null_mut(),
            n as _,
            vDSP_DFT_FORWARD,
            vDSP_DFT_Interleaved_ComplextoComplex,
        );
        if setup.is_null() {
            out!("  setup NULL at n={n}");
            return;
        }
        for j in 0..n {
            let mut input = vec![DSPComplex { real: 0.0, imag: 0.0 }; n];
            input[j].real = 1.0;
            let mut out = vec![DSPComplex { real: 0.0, imag: 0.0 }; n];
            vDSP_DFT_Interleaved_Execute(setup, input.as_ptr(), out.as_mut_ptr());
            let got: Vec<(f64, f64)> =
                out.iter().map(|c| (c.real as f64, c.imag as f64)).collect();
            let want_re: Vec<f64> = vec![1.0; n];
            let want_im: Vec<f64> = (0..n)
                .map(|k| (-2.0 * PI * (j * k) as f64 / n as f64).sin())
                .collect();
            let want = dft_analytic(&{
                let mut v = vec![0.0; n];
                v[j] = 1.0;
                v
            }, &vec![0.0; n]);
            let _ = (want_re, want_im);
            let e = max_err(&got, &want);
            let flagged: Vec<String> = got
                .iter()
                .zip(&want)
                .enumerate()
                .filter(|(_, (g, w))| (g.0 - w.0).abs() > 1e-4 || (g.1 - w.1).abs() > 1e-4)
                .map(|(k, (g, _))| format!("k{k}:({:.4},{:.4})", g.0, g.1))
                .collect();
            out!("  j={j}: max_err={e:.3e} {}", if flagged.is_empty() {
                "all bins OK".to_string()
            } else {
                format!("MISMATCH {}", flagged.join(" "))
            });
        }
        vDSP_DFT_Interleaved_DestroySetup(setup);
    }
}

/// Dump the full output array for the unit impulse (the failing test's input).
fn dump_delta(n: usize) {
    let (mut re, im) = (vec![0.0f64; n], vec![0.0f64; n]);
    re[0] = 1.0;
    out!("--- unit-impulse (delta at index 0) full output, n={n} ---");
    out!("  expected: every bin = (1.0, 0.0)");
    unsafe {
        let setup = vDSP_DFT_Interleaved_CreateSetup(
            std::ptr::null_mut(),
            n as _,
            vDSP_DFT_FORWARD,
            vDSP_DFT_Interleaved_ComplextoComplex,
        );
        if setup.is_null() {
            out!("  interleaved f32 setup NULL");
        } else {
            let input: Vec<DSPComplex> =
                (0..n).map(|j| DSPComplex { real: re[j] as f32, imag: im[j] as f32 }).collect();
            let mut out = vec![DSPComplex { real: 0.0, imag: 0.0 }; n];
            vDSP_DFT_Interleaved_Execute(setup, input.as_ptr(), out.as_mut_ptr());
            vDSP_DFT_Interleaved_DestroySetup(setup);
            let s: Vec<String> =
                out.iter().map(|c| format!("({}, {})", c.real, c.imag)).collect();
            out!("  interleaved f32 raw: {}", s.join(" "));
        }

        let setup = vDSP_DFT_Interleaved_CreateSetupD(
            std::ptr::null_mut(),
            n as _,
            vDSP_DFT_FORWARD,
            vDSP_DFT_Interleaved_ComplextoComplex,
        );
        if setup.is_null() {
            out!("  interleaved f64 setup NULL");
        } else {
            let input: Vec<DSPDoubleComplex> =
                (0..n).map(|j| DSPDoubleComplex { real: re[j], imag: im[j] }).collect();
            let mut out = vec![DSPDoubleComplex { real: 0.0, imag: 0.0 }; n];
            vDSP_DFT_Interleaved_ExecuteD(setup, input.as_ptr(), out.as_mut_ptr());
            vDSP_DFT_Interleaved_DestroySetupD(setup);
            let s: Vec<String> =
                out.iter().map(|c| format!("({}, {})", c.real, c.imag)).collect();
            out!("  interleaved f64 raw: {}", s.join(" "));
        }

        let setup = vDSP_DFT_zop_CreateSetup(std::ptr::null_mut(), n as _, vDSP_DFT_FORWARD);
        if setup.is_null() {
            out!("  split f32 setup NULL");
        } else {
            let ir: Vec<f32> = re.iter().map(|&v| v as f32).collect();
            let ii: Vec<f32> = im.iter().map(|&v| v as f32).collect();
            let (mut or, mut oi) = (vec![0.0f32; n], vec![0.0f32; n]);
            vDSP_DFT_Execute(setup, ir.as_ptr(), ii.as_ptr(), or.as_mut_ptr(), oi.as_mut_ptr());
            vDSP_DFT_DestroySetup(setup);
            let s: Vec<String> = (0..n).map(|k| format!("({}, {})", or[k], oi[k])).collect();
            out!("  split f32 raw:       {}", s.join(" "));
        }
    }
    // What the crate's own public API returns for the same input.
    if let Ok(plan) = FftPlan::new_c32(n) {
        let input: Vec<MKL_Complex8> = (0..n)
            .map(|j| MKL_Complex8 { real: re[j] as f32, imag: im[j] as f32 })
            .collect();
        let mut out = vec![MKL_Complex8 { real: 0.0, imag: 0.0 }; n];
        plan.forward_c32(&input, &mut out).unwrap();
        let s: Vec<String> = out.iter().map(|c| format!("({}, {})", c.real, c.imag)).collect();
        out!("  crate forward_c32:   {}", s.join(" "));
    }
}

/// Is the interleaved result sensitive to buffer alignment at n=8?
fn dump_alignment(n: usize) {
    out!("--- alignment sensitivity, interleaved f32, n={n} ---");
    let (re, im) = probe_input(n);
    for off in [0usize, 1, 2, 3] {
        let mut input = vec![DSPComplex { real: 0.0, imag: 0.0 }; n + 4];
        let mut out = vec![DSPComplex { real: 0.0, imag: 0.0 }; n + 4];
        let ip = unsafe { input.as_mut_ptr().add(off) };
        let op = unsafe { out.as_mut_ptr().add(off) };
        for j in 0..n {
            unsafe {
                (*ip.add(j)).real = re[j] as f32;
                (*ip.add(j)).imag = im[j] as f32;
            }
        }
        unsafe {
            let setup = vDSP_DFT_Interleaved_CreateSetup(
                std::ptr::null_mut(),
                n as _,
                vDSP_DFT_FORWARD,
                vDSP_DFT_Interleaved_ComplextoComplex,
            );
            if setup.is_null() {
                out!("  offset {off}: setup NULL");
                return;
            }
            vDSP_DFT_Interleaved_Execute(setup, ip, op);
            vDSP_DFT_Interleaved_DestroySetup(setup);
        }
        let got: Vec<(f64, f64)> = (0..n)
            .map(|k| unsafe {
                ((*op.add(k)).real as f64, (*op.add(k)).imag as f64)
            })
            .collect();
        out!(
            "  input ptr {:p} (align {}), err = {}",
            ip,
            (ip as usize) % 16,
            fmt_err(Some(max_err(&got, &dft_analytic(&re, &im))))
        );
    }
}

/// Which direction was actually used? Forward vs inverse on the same probe.
fn dump_directions(n: usize) {
    out!("--- direction check, interleaved f32, n={n} ---");
    let (re, im) = probe_input(n);
    let analytic = dft_analytic(&re, &im);
    for (name, dir) in [("FORWARD", vDSP_DFT_FORWARD), ("INVERSE", vDSP_DFT_INVERSE)] {
        unsafe {
            let setup = vDSP_DFT_Interleaved_CreateSetup(
                std::ptr::null_mut(),
                n as _,
                dir,
                vDSP_DFT_Interleaved_ComplextoComplex,
            );
            if setup.is_null() {
                out!("  {name}: setup NULL");
                continue;
            }
            let input: Vec<DSPComplex> =
                (0..n).map(|j| DSPComplex { real: re[j] as f32, imag: im[j] as f32 }).collect();
            let mut out = vec![DSPComplex { real: 0.0, imag: 0.0 }; n];
            vDSP_DFT_Interleaved_Execute(setup, input.as_ptr(), out.as_mut_ptr());
            vDSP_DFT_Interleaved_DestroySetup(setup);
            let got: Vec<(f64, f64)> =
                out.iter().map(|c| (c.real as f64, c.imag as f64)).collect();
            out!("  {name}: err vs analytic = {:.3e}", max_err(&got, &analytic));
        }
    }
    let _ = (re, im);
}

#[test]
fn diag53() {
    out!("===== DIAG-53 FFT/vDSP evidence =====");
    out!("os/arch: {} / {}", std::env::consts::OS, std::env::consts::ARCH);
    out!("{}", env_info());

    out!("--- setup acceptance + correctness vs analytic DFT (forward, complextocomplex) ---");
    out!(
        "{:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "len", "ilv32_set", "ilv64_set", "zop32_set", "zop64_set", "crate_ok", "crate32_err"
    );
    for n in [
        2usize, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 24, 25, 32, 36, 40, 48, 60, 64, 100, 120,
        128, 256,
    ] {
        let ilv32 = unsafe {
            let s = vDSP_DFT_Interleaved_CreateSetup(
                std::ptr::null_mut(),
                n as _,
                vDSP_DFT_FORWARD,
                vDSP_DFT_Interleaved_ComplextoComplex,
            );
            if !s.is_null() {
                vDSP_DFT_Interleaved_DestroySetup(s);
            }
            !s.is_null()
        };
        let ilv64 = unsafe {
            let s = vDSP_DFT_Interleaved_CreateSetupD(
                std::ptr::null_mut(),
                n as _,
                vDSP_DFT_FORWARD,
                vDSP_DFT_Interleaved_ComplextoComplex,
            );
            if !s.is_null() {
                vDSP_DFT_Interleaved_DestroySetupD(s);
            }
            !s.is_null()
        };
        let zop32 = unsafe {
            let s = vDSP_DFT_zop_CreateSetup(std::ptr::null_mut(), n as _, vDSP_DFT_FORWARD);
            if !s.is_null() {
                vDSP_DFT_DestroySetup(s);
            }
            !s.is_null()
        };
        let zop64 = unsafe {
            let s = vDSP_DFT_zop_CreateSetupD(std::ptr::null_mut(), n as _, vDSP_DFT_FORWARD);
            if !s.is_null() {
                vDSP_DFT_DestroySetupD(s);
            }
            !s.is_null()
        };
        out!(
            "{:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}",
            n,
            ilv32,
            ilv64,
            zop32,
            zop64,
            FftPlan::new_c32(n).is_ok(),
            fmt_err(crate32_err(n))
        );
    }

    out!("--- correctness errors (small = correct) ---");
    out!(
        "{:>5} {:>20} {:>20} {:>20} {:>20} {:>20}",
        "len", "ilv32", "ilv64", "zop32", "zop64", "crate64_err"
    );
    for n in [8usize, 12, 16, 20, 24, 32, 36, 40, 48, 60, 64, 100, 120, 128, 256] {
        out!(
            "{:>5} {:>20} {:>20} {:>20} {:>20} {:>20}",
            n,
            fmt_err(ilv32_err(n)),
            fmt_err(ilv64_err(n)),
            fmt_err(zop32_err(n)),
            fmt_err(zop64_err(n)),
            fmt_err(crate64_err(n))
        );
    }

    dump_delta(8);
    dump_delta(24);
    dump_matrix(8);
    dump_matrix(24);
    dump_alignment(8);
    dump_directions(8);
    dump_directions(24);

    out!("===== DIAG-53 END =====");
    panic!("DIAG-53: intentional panic to surface the captured stdout above");
}
