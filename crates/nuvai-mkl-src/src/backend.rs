// Backend selection for `nuvai-mkl-src`.
//
// This crate links Intel oneMKL on x86_64 Linux/Windows and falls back on the
// aarch64 targets where Intel ships no MKL: Accelerate or OpenBLAS on
// `aarch64-apple-darwin`, and OpenBLAS on `aarch64-unknown-linux-gnu` (where
// it is the *only* backend — OpenBLAS covers BLAS/LAPACK, with no vDSP/vForce/
// Sparse equivalent). The selection is *explicit* — never silent:
//
// * `cfg(target_arch = "aarch64")` mandates the non-MKL path (Intel ships no
//   oneMKL for any aarch64 target), and
// * on `aarch64-apple-darwin` a Cargo feature (`accelerate`, the default, vs
//   `openblas`) chooses between the available fallback backends; on
//   `aarch64-unknown-linux-gnu` both features are inert (OpenBLAS is the only
//   backend).
//
// `nuvai-mkl-src` keeps `links = "mkl"` on every platform and remains the sole
// emitter of linker directives, so no second `links` provider is needed.

// On `aarch64-apple-darwin` the fallback is mandatory but its *choice* is a
// Cargo feature, so disabling defaults (`--no-default-features`) with no
// explicit replacement must fail loudly rather than silently fall back to
// Accelerate. Selection is explicit, never silent (ADR-0003).
#[cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    not(feature = "accelerate"),
    not(feature = "openblas")
))]
compile_error!(
    "nuvai-mkl-src: aarch64-apple-darwin requires exactly one backend feature — \
     enable `accelerate` (default) or `openblas`"
);

// `x86_64-apple-darwin` is unsupported: Intel ended oneMKL for macOS after the
// 2023.2.0 release (well short of the 2026.1.0 this crate links), so no
// supported acquisition path exists. Reject at compile time rather than
// half-linking against a stale manual oneAPI install.
#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
compile_error!(
    "nuvai-mkl-src: x86_64-apple-darwin is not supported — Intel ships no 2026.x \
     oneMKL for macOS (last macOS build: 2023.2.0). Use aarch64-apple-darwin \
     (Apple Silicon, Accelerate/OpenBLAS) or an x86_64 Linux/Windows target."
);

// The aarch64 fallback exists on exactly two targets: `aarch64-apple-darwin`
// (Accelerate or OpenBLAS) and `aarch64-unknown-linux-gnu` (OpenBLAS — the
// only backend there). Every other aarch64 target — musl Linux (no glibc
// OpenBLAS), Android, Windows, FreeBSD — has no oneMKL and no supported
// fallback. Reject at compile time rather than silently returning `IntelMkl`
// and emitting x86_64 MKL directives an aarch64 toolchain cannot link (the
// pre-fallback behaviour of returning `IntelMkl` from the catch-all arm
// produced exactly that). Selection is explicit, never silent (ADR-0003).
#[cfg(all(
    target_arch = "aarch64",
    not(target_os = "macos"),
    not(all(target_os = "linux", target_env = "gnu"))
))]
compile_error!(
    "nuvai-mkl-src: unsupported aarch64 target — Intel ships no oneMKL for aarch64, \
     and the fallback backends exist only on aarch64-apple-darwin (Accelerate/OpenBLAS) \
     and aarch64-unknown-linux-gnu (OpenBLAS, glibc). For Linux use the glibc target \
     (aarch64-unknown-linux-gnu); musl/Android/Windows/FreeBSD aarch64 have no backend."
);

/// The link backend selected for the current build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Intel oneMKL (`mkl_rt`) — the x86_64 Linux/Windows path.
    IntelMkl,
    /// Apple's Accelerate framework (`-framework Accelerate`) — the default
    /// `aarch64-apple-darwin` fallback.
    Accelerate,
    /// OpenBLAS (`-lopenblas`) — opt-in alternative on aarch64.
    OpenBlas,
}

/// Select the link backend for the current target and Cargo features.
///
/// On `aarch64-apple-darwin` the `openblas` feature selects OpenBLAS; the
/// `accelerate` feature (default) selects Accelerate. Exactly one of the two
/// must be enabled — a `compile_error!` above rejects the no-feature case. On
/// `aarch64-unknown-linux-gnu` OpenBLAS is the only backend (no vDSP/vForce/
/// Sparse exists on Linux) and both features are inert. On every other target
/// Intel MKL is used and the feature flags are ignored.
///
/// This is the *library-time* selector: its `#[cfg(...)]` reflects the target
/// the crate is compiled for. Build scripts (which compile for the *host*)
/// must use [`backend_for_target`] instead, passing the target triple from the
/// `CARGO_CFG_TARGET_*` env vars, so cross-compilation picks the target's
/// backend rather than the host's.
pub fn backend() -> Backend {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if cfg!(feature = "openblas") {
            Backend::OpenBlas
        } else {
            Backend::Accelerate
        }
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64", target_env = "gnu"))]
    {
        // OpenBLAS is the only backend on aarch64-unknown-linux-gnu; the
        // `accelerate`/`openblas` features are inert on this target.
        Backend::OpenBlas
    }
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "aarch64", target_env = "gnu")
    )))]
    {
        Backend::IntelMkl
    }
}

/// Select the link backend for an explicit target triple.
///
/// Build scripts compile for and run on the *host*, so their `#[cfg(...)]`
/// cannot describe the crate they are building for — but Cargo still sets
/// `CARGO_CFG_TARGET_OS` / `CARGO_CFG_TARGET_ARCH` / `CARGO_CFG_TARGET_ENV` to
/// the real *target* triple. Pass those here to select the target's backend.
///
/// Mirrors [`backend`]'s `#[cfg]` selection and rejects the same unsupported
/// targets with an `Err` (rather than a compile error, since the build script
/// is host-compiled and cannot `compile_error!` for the target).
pub fn backend_for_target(
    target_os: &str,
    target_arch: &str,
    target_env: Option<&str>,
) -> Result<Backend, &'static str> {
    match (target_os, target_arch, target_env) {
        ("macos", "aarch64", _) => Ok(if cfg!(feature = "openblas") {
            Backend::OpenBlas
        } else {
            Backend::Accelerate
        }),
        ("linux", "aarch64", _) if target_env == Some("gnu") => Ok(Backend::OpenBlas),
        ("linux", "aarch64", _) => Err(
            "nuvai-mkl-src: aarch64 Linux with a non-gnu libc (musl/android) is \
             unsupported — the OpenBLAS fallback requires a glibc target \
             (aarch64-unknown-linux-gnu)",
        ),
        (_, "aarch64", _) => Err(
            "nuvai-mkl-src: unsupported aarch64 target — Intel ships no oneMKL for \
             aarch64; the fallback backends exist only on aarch64-apple-darwin and \
             aarch64-unknown-linux-gnu",
        ),
        ("macos", _, _) => Err(
            "nuvai-mkl-src: x86_64-apple-darwin is unsupported — Intel's last macOS \
             oneMKL was 2023.2.0",
        ),
        _ => Ok(Backend::IntelMkl),
    }
}

/// Human-readable backend tag used for `cargo:metadata=BACKEND=…`.
pub const fn backend_tag(backend: Backend) -> &'static str {
    match backend {
        Backend::IntelMkl => "mkl",
        Backend::Accelerate => "accelerate",
        Backend::OpenBlas => "openblas",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_is_explicit_and_tagged() {
        // Selection must be deterministic and map to a non-empty tag.
        let b = backend();
        assert!(matches!(
            b,
            Backend::IntelMkl | Backend::Accelerate | Backend::OpenBlas
        ));
        assert!(!backend_tag(b).is_empty());
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64", target_env = "gnu"))]
    #[test]
    fn linux_aarch64_selects_openblas() {
        // OpenBLAS is the only backend on aarch64-unknown-linux-gnu.
        assert_eq!(backend(), Backend::OpenBlas);
        assert_eq!(backend_tag(backend()), "openblas");
    }

    #[test]
    fn backend_for_target_is_exhaustive() {
        use super::{Backend, backend_for_target};

        // The target-triple selector used by build scripts must agree with the
        // cfg-based `backend()` for the supported targets, and reject the
        // unsupported ones explicitly (never silently selecting IntelMkl).
        assert_eq!(backend_for_target("linux", "aarch64", Some("gnu")), Ok(Backend::OpenBlas));
        assert_eq!(backend_for_target("linux", "x86_64", Some("gnu")), Ok(Backend::IntelMkl));
        assert_eq!(backend_for_target("windows", "x86_64", Some("msvc")), Ok(Backend::IntelMkl));
        assert!(backend_for_target("linux", "aarch64", Some("musl")).is_err());
        assert!(backend_for_target("linux", "aarch64", Some("android")).is_err());
        assert!(backend_for_target("windows", "aarch64", Some("msvc")).is_err());
        assert!(backend_for_target("macos", "x86_64", None).is_err());
    }
}
