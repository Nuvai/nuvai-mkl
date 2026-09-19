//! Fallible integer conversions at the C/Fortran boundary.
//!
//! Every wrapped MKL entry point takes a vector or matrix *count* as a 32-bit C
//! integer — `MKL_INT` is `int` under the `MKL_LP64` layer, LAPACK's
//! `__CLPK_integer` and the CBLAS/VSL/VML counts likewise — while a Rust slice
//! length is a `usize`. Casting between the two with `as` truncates silently
//! above `i32::MAX`, and the failure is invisible: MKL would process only a
//! prefix of the caller's buffer and return success, so the caller gets a
//! partly-filled result with no error to notice. [`len_to_c_int`] is the single
//! guard for that conversion.

use std::os::raw::c_int;

use crate::error::{Error, Result};

/// Convert a slice length to the `c_int` an MKL routine takes, rejecting a
/// length that would truncate instead of casting it.
///
/// `name` identifies the caller in the error message: a routine name where one
/// exists (`vsRngUniform`), otherwise the domain (`PARDISO`, `DSS`).
// On `aarch64-unknown-linux-gnu` the only caller (`vml::check`) is itself
// unreachable — every `vml` function returns `Unsupported` on that target
// before it looks at its arguments — so this is dead there. Gating on the arch
// alone would not do: on `aarch64-apple-darwin` and on Intel the `vml` fast
// paths do call it. Same pattern as the `allow` on `Error::resource_exhausted`
// in `error.rs`.
#[cfg_attr(all(target_os = "linux", target_arch = "aarch64"), allow(dead_code))]
pub(crate) fn len_to_c_int(len: usize, name: &str) -> Result<c_int> {
    c_int::try_from(len).map_err(|_| Error::invalid(format!("{name}: length exceeds i32::MAX")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

    /// The boundary itself. This is the only place the guard can be reached: a
    /// slice longer than `i32::MAX` would be 8 GiB of `f32` (16 GiB of `f64`),
    /// which no test can allocate — the reason the rule lives in one shared
    /// function rather than as four inline `try_from` calls.
    #[test]
    fn accepts_up_to_i32_max_and_rejects_beyond() {
        assert_eq!(len_to_c_int(0, "x").unwrap(), 0);
        assert_eq!(len_to_c_int(1, "x").unwrap(), 1);
        assert_eq!(len_to_c_int(i32::MAX as usize, "x").unwrap(), i32::MAX);

        let err = len_to_c_int(i32::MAX as usize + 1, "vsRngUniform").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidArgument);
        assert_eq!(
            err.to_string(),
            "invalid argument: vsRngUniform: length exceeds i32::MAX"
        );
    }

    /// `usize::MAX` is the value an unchecked length computation would wrap to,
    /// so it must be rejected rather than truncated to `-1`.
    #[test]
    fn rejects_usize_max() {
        assert!(len_to_c_int(usize::MAX, "DSS").is_err());
    }
}
