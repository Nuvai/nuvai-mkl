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
pub fn backend() -> Backend {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if cfg!(feature = "openblas") {
            Backend::OpenBlas
        } else {
            Backend::Accelerate
        }
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        // OpenBLAS is the only backend on aarch64-unknown-linux-gnu; the
        // `accelerate`/`openblas` features are inert on this target.
        Backend::OpenBlas
    }
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "aarch64")
    )))]
    {
        Backend::IntelMkl
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

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    #[test]
    fn linux_aarch64_selects_openblas() {
        // OpenBLAS is the only backend on aarch64-unknown-linux-gnu.
        assert_eq!(backend(), Backend::OpenBlas);
        assert_eq!(backend_tag(backend()), "openblas");
    }
}
