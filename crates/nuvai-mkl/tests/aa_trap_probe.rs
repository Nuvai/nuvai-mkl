//! TEMPORARY diagnostic — delete before merge.
//!
//! Why this exists: `smoke.rs`'s PARDISO non-OK tests abort the test binary
//! with SIGTRAP on the `macos-14` CI runner (both darwin jobs red), while the
//! same suite is green on macOS 26. libtest reports only the tests that
//! finished, and reports them as their threads complete — so a printed list
//! cannot say which input was responsible. A signal kills the process, so
//! attribution has to survive a dead process.
//!
//! Each case therefore runs in a *child process* of this test binary: the
//! parent observes the child's wait status (signal included) and reports every
//! case, whatever happens to any one of them.
//!
//! Rust's stdout is line-buffered even when it is a pipe, so a line printed by
//! the child *before* the aborting call still reaches the parent. That is what
//! lets the `factor_*` modes below separate a trap inside the factorization
//! call from one inside the destroy.
//!
//! Round 1 result (macos-14 runner): `zero2x2` completed with no signal and
//! reported `Err kind=Mkl`, while `emptyrow` was killed by signal 5. So the
//! all-zero matrix is not the problem — its state-3 destroy ran fine, matching
//! the DSS test that also passes on that runner.
//!
//! The parent deliberately panics with the report so libtest prints it without
//! needing `--nocapture` in CI.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::os::unix::process::ExitStatusExt;
use std::os::raw::c_long;
use std::process::Command;

const CASES: [&str; 4] = [
    // Control: known to pass on macOS 14 (round 1).
    "zero2x2_via_api",
    // Control: known to trap on macOS 14 (round 1) — confirms the harness.
    "emptyrow_via_api",
    // The empty row again, but driving the sys API so the factor call and the
    // destroy can be separated.
    "emptyrow_factor_only",
    "emptyrow_factor_destroy",
];

/// CSC for the structurally empty second row: n=2, 1-based CSR
/// `ia=[1,2,2], ja=[1], a=[1.0]` — row 0 holds (0,0)=1.0, row 1 is empty.
/// Mirrors `csr_to_csc(2, .., base=1, upper_only=false)`.
fn empty_row_matrix() -> (
    Vec<i64>,
    Vec<i32>,
    Vec<f64>,
    nuvai_mkl_sys::SparseSymbolicFactorOptions,
    nuvai_mkl_sys::SparseNumericFactorOptions,
) {
    let column_starts = vec![0i64, 1, 1];
    let row_indices = vec![0i32];
    let values = vec![1.0f64];
    let sfoptions = nuvai_mkl_sys::SparseSymbolicFactorOptions {
        control: nuvai_mkl_sys::SparseDefaultControl,
        orderMethod: nuvai_mkl_sys::SparseOrderDefault,
        order: std::ptr::null_mut(),
        ignoreRowsAndColumns: std::ptr::null_mut(),
        malloc: nuvai_mkl_sys::malloc,
        free: nuvai_mkl_sys::free,
        reportError: None,
    };
    let nfoptions = nuvai_mkl_sys::SparseNumericFactorOptions {
        control: nuvai_mkl_sys::SparseDefaultControl,
        scalingMethod: nuvai_mkl_sys::SparseScalingDefault,
        scaling: std::ptr::null_mut(),
        pivotTolerance: 0.0,
        zeroTolerance: 0.0,
    };
    (column_starts, row_indices, values, sfoptions, nfoptions)
}

fn run_via_api(case: &str) {
    let (ia, ja, a): (&[i32], &[i32], &[f64]) = match case {
        "zero2x2_via_api" => (&[1, 3, 5], &[1, 2, 1, 2], &[0.0, 0.0, 0.0, 0.0]),
        "emptyrow_via_api" => (&[1, 2, 2], &[1], &[1.0]),
        other => panic!("not an api case: {other}"),
    };
    let b = [1.0f64, 1.0];
    let mut solver = nuvai_mkl::pardiso::Pardiso::new(nuvai_mkl::pardiso::mtype::NONSYMMETRIC);
    match solver.solve(ia, ja, a, &b) {
        Ok(_) => println!("CHILD {case}: Ok"),
        Err(e) => println!("CHILD {case}: Err kind={:?} msg={e}", e.kind()),
    }
}

/// Drives `_SparseFactorQR_Double` directly, so the destroy can be included or
/// skipped. `Pardiso::solve` always destroys before returning, which is exactly
/// the ambiguity being resolved.
fn run_factor_mode(case: &str) {
    let (column_starts, row_indices, values, sfoptions, nfoptions) = empty_row_matrix();
    let mut matrix = nuvai_mkl_sys::SparseMatrix_Double {
        structure: nuvai_mkl_sys::SparseMatrixStructure {
            rowCount: 2,
            columnCount: 2,
            columnStarts: column_starts.as_ptr() as *mut c_long,
            rowIndices: row_indices.as_ptr() as *mut i32,
            attributes: nuvai_mkl_sys::SparseAttributes_t::ordinary(),
            blockSize: 1,
        },
        data: values.as_ptr() as *mut f64,
    };
    let factor = unsafe {
        nuvai_mkl_sys::_SparseFactorQR_Double(
            nuvai_mkl_sys::SparseFactorizationQR,
            &matrix,
            &sfoptions,
            &nfoptions,
        )
    };
    // Line-buffered stdout: this line survives an abort in the destroy below.
    println!(
        "CHILD {case}: REACHED post-factor status={} symbolic.status={} numeric_is_null={}",
        factor.status,
        factor.symbolicFactorization.status,
        factor.numericFactorization.is_null()
    );
    if case == "emptyrow_factor_only" {
        // Deliberately leaked: the point is to never call the destroy.
        std::mem::forget(factor);
        println!("CHILD {case}: REACHED end (leaked on purpose)");
        return;
    }
    let mut factor = factor;
    unsafe { nuvai_mkl_sys::_SparseDestroyOpaqueNumeric_Double(&mut factor) };
    println!("CHILD {case}: REACHED post-destroy");
}

/// Re-entry point: only does work when the parent sets `PROBE_CASE`.
#[test]
fn case_runner() {
    let Ok(case) = std::env::var("PROBE_CASE") else {
        return;
    };
    if case.ends_with("_via_api") {
        run_via_api(&case);
    } else {
        run_factor_mode(&case);
    }
}

#[test]
fn trap_attribution() {
    if std::env::var("PROBE_CASE").is_ok() {
        return;
    }
    let exe = std::env::current_exe().expect("current_exe");
    let mut report = String::from("trap attribution:");
    for case in CASES {
        let out = Command::new(&exe)
            .args(["--exact", "--nocapture", "case_runner"])
            .env("PROBE_CASE", case)
            .output()
            .expect("spawn child");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let child_lines: Vec<&str> = stdout
            .lines()
            .filter(|l| l.starts_with("CHILD "))
            .collect();
        report.push_str(&format!(
            "\n  {case}: exit={:?} signal={:?}",
            out.status.code(),
            out.status.signal(),
        ));
        if child_lines.is_empty() {
            report.push_str("\n      <no CHILD line reached>");
        }
        for line in child_lines {
            report.push_str("\n      ");
            report.push_str(line);
        }
    }
    panic!("{report}");
}
