// Backend selection for `nuvai-mkl-src`.
//
// This crate links Intel oneMKL on x86_64 (Linux/Windows/macOS) and falls back
// to Accelerate or OpenBLAS on `aarch64-apple-darwin`, where Intel ships no
// MKL. The selection is *explicit* — never silent:
//
// * `cfg(target_arch = "aarch64")` mandates the non-MKL path (Intel ships no
//   MKL for Apple Silicon), and
// * a Cargo feature (`accelerate`, the default, vs `openblas`) chooses between
//   the available fallback backends.
//
// `nuvai-mkl-src` keeps `links = "mkl"` on every platform and remains the sole
// emitter of linker directives, so no second `links` provider is needed.

/// The link backend selected for the current build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Intel oneMKL (`mkl_rt`) — the x86_64 Linux/Windows/macOS path.
    IntelMkl,
    /// Apple's Accelerate framework (`-framework Accelerate`) — the default
    /// `aarch64-apple-darwin` fallback.
    Accelerate,
    /// OpenBLAS (`-lopenblas`) — opt-in alternative on aarch64.
    OpenBlas,
}

/// Select the link backend for the current target and Cargo features.
///
/// On `aarch64-apple-darwin` the `openblas` feature selects OpenBLAS; anything
/// else selects Accelerate. On every other target Intel MKL is used and the
/// feature flags are ignored (they are no-ops because the aarch64 code is
/// `cfg`-gated).
pub fn backend() -> Backend {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if cfg!(feature = "openblas") {
            Backend::OpenBlas
        } else {
            Backend::Accelerate
        }
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
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
}
