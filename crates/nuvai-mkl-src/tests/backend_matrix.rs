//! The backend `nuvai-mkl-src` actually selects, asserted per CI target (#11).
//!
//! `src/backend.rs`'s unit tests pin the *rules* — the four `aarch64-apple-darwin`
//! feature pairs, and the target triples that are rejected — but nothing asserted
//! the outcome a caller depends on: that on each target CI builds, `backend()`
//! returns the backend that target is supposed to have. `backend()`'s last arm is
//! a catch-all returning `IntelMkl`, so a target nobody considered does not fail;
//! it quietly claims Intel MKL and emits x86_64 link directives for a toolchain
//! that cannot use them. This file closes that hole by making the *absence* of an
//! expectation a compile error, so "explicit, never silent" (ADR-0003) holds for
//! the target dimension as well as the feature dimension.
//!
//! Being in `tests/`, this file is also the "builds + links" half of #11: an
//! integration test is a separate binary linked from outside the crate, so it
//! cannot run at all unless the link directives `nuvai-mkl-src`'s build script
//! emitted for the current job resolved. The assertions stay at the selection
//! level deliberately — *calling* into a backend is `nuvai-mkl/tests/smoke.rs`'s
//! job, and `nuvai-mkl` is the crate whose build script supplies the
//! `--no-as-needed` / `force_runtime.c` treatment the x86_64 Linux link needs
//! (#44). This crate's test binaries reference no MKL symbol and need none of it.
//!
//! The binary is linked exactly as `src/backend.rs`'s lib-test harness already is
//! — same package, same build script, same directives — and that harness is green
//! on all five CI jobs today, which is why adding this file cannot newly break a
//! job that passes.

use nuvai_mkl_src::{Backend, backend, backend_for_target, backend_tag};

// ---------------------------------------------------------------------------
// The matrix: one arm per (target, feature selection) pair CI builds. A pair
// with no arm is a compile error (below), never a pass. The two Apple Silicon
// jobs share a target triple and differ only in features, so the `not(...)`
// guards are what make them assert *different* backends instead of both
// accepting whichever one the build happened to pick.
// ---------------------------------------------------------------------------

/// `aarch64-darwin` — default features (`accelerate`).
#[cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    feature = "accelerate",
    not(feature = "openblas")
))]
const EXPECTED: Backend = Backend::Accelerate;

/// `aarch64-darwin-openblas` — `--no-default-features --features openblas`.
#[cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    feature = "openblas",
    not(feature = "accelerate")
))]
const EXPECTED: Backend = Backend::OpenBlas;

/// `aarch64-linux` — OpenBLAS is the only backend there; both features are inert.
#[cfg(all(target_os = "linux", target_arch = "aarch64", target_env = "gnu"))]
const EXPECTED: Backend = Backend::OpenBlas;

/// `x86_64-linux` — Intel oneMKL; the backend features are inert.
#[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
const EXPECTED: Backend = Backend::IntelMkl;

/// `x86_64-windows` — Intel oneMKL; the backend features are inert.
#[cfg(all(target_os = "windows", target_arch = "x86_64", target_env = "msvc"))]
const EXPECTED: Backend = Backend::IntelMkl;

/// The tag [`backend_tag`] must return for [`EXPECTED`].
///
/// Restated here as its own literal table rather than derived from the function
/// under test, so that a change to the mapping is *caught* instead of mirrored.
/// This is the string the build script publishes as `cargo:metadata=BACKEND=…`,
/// so it is a contract, not a debug aid.
const EXPECTED_TAG: &str = match EXPECTED {
    Backend::IntelMkl => "mkl",
    Backend::Accelerate => "accelerate",
    Backend::OpenBlas => "openblas",
};

// The negation of the five arms above.
//
// A target the matrix does not cover fails to compile here rather than passing
// on `backend()`'s `IntelMkl` catch-all. The targets this actually catches are
// the ones the crate does *not* already `compile_error!` on —
// `x86_64-unknown-linux-musl`, `x86_64-pc-windows-gnu`,
// `riscv64gc-unknown-linux-gnu`, `i686-*` — i.e. exactly the ones that would
// otherwise test green while linking nothing usable. `target_env` is part of the
// arms for that reason: without it, `x86_64-unknown-linux-musl` would silently
// be blessed with `IntelMkl`. Adding a target to CI means adding it here and
// above.
//
// A plain `//` comment, not `///`: this annotates a macro invocation rather than
// an item, so a doc comment here is reported as unused on exactly the targets
// that reach it.
#[cfg(not(any(
    all(
        target_os = "macos",
        target_arch = "aarch64",
        feature = "accelerate",
        not(feature = "openblas")
    ),
    all(
        target_os = "macos",
        target_arch = "aarch64",
        feature = "openblas",
        not(feature = "accelerate")
    ),
    all(target_os = "linux", target_arch = "aarch64", target_env = "gnu"),
    all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"),
    all(target_os = "windows", target_arch = "x86_64", target_env = "msvc"),
)))]
compile_error!(
    "tests/backend_matrix.rs: this target + feature selection is not in the CI matrix, so no \
     backend expectation is recorded for it — `backend()` would otherwise pass here on its \
     `IntelMkl` catch-all. Add the target to `.github/workflows/ci.yml` and to the two `cfg` \
     lists in this file."
);

#[test]
fn backend_matches_the_matrix_expectation() {
    let actual = backend();

    // EXPECTED differs per job, so a job that selected the wrong backend fails
    // here rather than reporting whatever it happened to pick. On the two macOS
    // jobs this is what proves the `--no-default-features --features openblas`
    // job really links OpenBLAS and the default job really links Accelerate —
    // the assertion is non-vacuous precisely because the two arms disagree.
    assert_eq!(
        actual, EXPECTED,
        "backend() reported {actual:?}, expected {EXPECTED:?} on this target"
    );

    // A correct enum with a wrong tag is still a silent mis-link for anything
    // reading the published metadata, so the tag is asserted too.
    assert_eq!(
        backend_tag(actual),
        EXPECTED_TAG,
        "backend_tag({actual:?}) disagrees with the matrix"
    );

    // The catch-all must never fire on ARM: every aarch64 arm above is a fallback
    // backend, and `IntelMkl` there would mean the `cfg` arms stopped matching and
    // x86_64 MKL directives were emitted for an ARM toolchain.
    if cfg!(target_arch = "aarch64") {
        assert_ne!(
            actual,
            Backend::IntelMkl,
            "aarch64 must never select Intel MKL"
        );
    }
}

/// `backend_for_target` — the selector `nuvai-mkl-src`'s build script uses,
/// because a build script is host-compiled and its own `cfg(...)` would describe
/// the *host* — must agree with `backend()` for the triple this job is building.
///
/// A disagreement is exactly the silent mis-link both selectors exist to avoid:
/// the build script would emit one backend's linker directives while the library
/// reports the other. `src/backend.rs` covers this for `aarch64-apple-darwin`
/// alone; this covers every job, which is where the x86_64 and ARM64 Linux arms
/// get their first cross-check.
#[test]
fn backend_for_target_agrees_for_this_target() {
    // `std::env::consts` has OS and ARCH but no ENV, and `CARGO_CFG_TARGET_ENV`
    // is exported to build scripts only — not to this compilation — so the libc
    // component has to come from `cfg`.
    let target_env = if cfg!(target_env = "gnu") {
        Some("gnu")
    } else if cfg!(target_env = "msvc") {
        Some("msvc")
    } else if cfg!(target_env = "musl") {
        Some("musl")
    } else {
        // macOS leaves `target_env` unset, which is what `backend_for_target`'s
        // macOS arms expect.
        None
    };

    assert_eq!(
        backend_for_target(std::env::consts::OS, std::env::consts::ARCH, target_env),
        Ok(backend()),
        "the build-script selector must mirror backend() on this target"
    );
}
