//! TEMPORARY diagnostic — delete before merge.
//!
//! Why this exists: `smoke.rs`'s PARDISO non-OK tests abort the test binary
//! with SIGTRAP on the `macos-14` CI runner (both darwin jobs red), while the
//! same suite is green on macOS 26. libtest reports only the tests that
//! finished, and it reports them as their threads complete — so with the
//! aborting test still in flight alongside ~10 others, the printed list
//! cannot say which input was responsible. Splitting the cases into separate
//! `#[test]`s does not help for the same reason: both were still running when
//! the abort landed.
//!
//! A signal kills the process, so attribution has to survive a dead process.
//! Each case therefore runs in a *child process* of this test binary: the
//! parent observes the child's wait status (signal included) and can report
//! every case, whatever happens to any one of them.
//!
//! The parent deliberately panics with the report so that libtest prints it
//! without needing `--nocapture` in CI.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::os::unix::process::ExitStatusExt;
use std::process::Command;

/// The two inputs from the split PARDISO tests, in the order they were tried.
const CASES: [&str; 2] = ["zero2x2", "emptyrow"];

fn case_inputs(case: &str) -> (&'static [i32], &'static [i32], &'static [f64]) {
    match case {
        // all-zero 2x2, same CSR structure as `pardiso_detects_singular_on_aarch64`
        "zero2x2" => (&[1, 3, 5], &[1, 2, 1, 2], &[0.0, 0.0, 0.0, 0.0]),
        // structurally empty second row
        "emptyrow" => (&[1, 2, 2], &[1], &[1.0]),
        other => panic!("unknown case {other}"),
    }
}

fn run_case(case: &str) {
    let (ia, ja, a) = case_inputs(case);
    let b = [1.0f64, 1.0];
    let mut solver = nuvai_mkl::pardiso::Pardiso::new(nuvai_mkl::pardiso::mtype::NONSYMMETRIC);
    match solver.solve(ia, ja, a, &b) {
        Ok(_) => println!("CHILD {case}: Ok"),
        Err(e) => println!("CHILD {case}: Err kind={:?} msg={e}", e.kind()),
    }
}

/// Re-entry point: only does work when the parent sets `PROBE_CASE`.
#[test]
fn case_runner() {
    let Ok(case) = std::env::var("PROBE_CASE") else {
        return;
    };
    run_case(&case);
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
        let child_line = stdout
            .lines()
            .find(|l| l.starts_with("CHILD "))
            .unwrap_or("<no CHILD line>");
        report.push_str(&format!(
            "\n  {case}: exit={:?} signal={:?} :: {child_line}",
            out.status.code(),
            out.status.signal(),
        ));
    }
    panic!("{report}");
}
