//! VML — Vector Math Library: element-wise transcendental and algebraic
//! functions computed over a whole vector.
//!
//! On Intel targets these call the oneMKL VML functions (`(n, src, dst)`
//! argument order). On Apple Silicon (`aarch64-apple-darwin`) they call the
//! Accelerate vForce functions, which use the reversed `(dst, src, n)` order
//! and different symbol names; the [`vml_unary!`] macro carries both symbols
//! and each cfg branch emits the matching call.

use std::os::raw::c_int;

use crate::error::{Error, Result};

/// Validate `src`/`dst` lengths and return the vector length as `c_int`.
#[inline]
fn check(src: usize, dst: usize, name: &str) -> Result<c_int> {
    if src != dst {
        return Err(Error::invalid(format!("{name}: src/dst length mismatch")));
    }
    // The C routine takes the length as `int`; a vector longer than `i32::MAX`
    // would truncate and silently process only a prefix. Reject it up front.
    c_int::try_from(src).map_err(|_| Error::invalid(format!("{name}: length exceeds i32::MAX")))
}

/// Generate one unary VML function.
///
/// `$mkl` is the oneMKL VML symbol (used on Intel, `(n, src, dst)` order) and
/// `$vforce` is the Accelerate vForce symbol (used on aarch64, `(dst, src, n)`
/// order).
macro_rules! vml_unary {
    ($(#[$doc:meta])* $name:ident, $mkl:ident, $vforce:ident, $ty:ty) => {
        $(#[$doc])*
        pub fn $name(src: &[$ty], dst: &mut [$ty]) -> Result<()> {
            let n = check(src.len(), dst.len(), stringify!($name))?;
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                // SAFETY: `src`/`dst` have equal length `n` (checked above);
                // vForce reads `n` elements from `src` and writes `n` to `dst`.
                unsafe { nuvai_mkl_sys::$vforce(dst.as_mut_ptr(), src.as_ptr(), &n) };
                Ok(())
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                // SAFETY: `src`/`dst` have equal length `n` (checked above);
                // VML reads `n` elements from `src` and writes `n` to `dst`.
                unsafe { nuvai_mkl_sys::$mkl(n, src.as_ptr(), dst.as_mut_ptr()) };
                Ok(())
            }
        }
    };
}

vml_unary! {
    /// `r[i] = exp(a[i])` (single precision).
    exp, vsExp, vvexpf, f32
}
vml_unary! {
    /// `r[i] = exp(a[i])` (double precision).
    dexp, vdExp, vvexp, f64
}
vml_unary! {
    /// `r[i] = ln(a[i])` (single precision).
    ln, vsLn, vvlogf, f32
}
vml_unary! {
    /// `r[i] = ln(a[i])` (double precision).
    dln, vdLn, vvlog, f64
}
vml_unary! {
    /// `r[i] = sqrt(a[i])` (single precision).
    sqrt, vsSqrt, vvsqrtf, f32
}
vml_unary! {
    /// `r[i] = sqrt(a[i])` (double precision).
    dsqrt, vdSqrt, vvsqrt, f64
}
vml_unary! {
    /// `r[i] = sin(a[i])` (single precision).
    sin, vsSin, vvsinf, f32
}
vml_unary! {
    /// `r[i] = sin(a[i])` (double precision).
    dsin, vdSin, vvsin, f64
}
vml_unary! {
    /// `r[i] = cos(a[i])` (single precision).
    cos, vsCos, vvcosf, f32
}
vml_unary! {
    /// `r[i] = cos(a[i])` (double precision).
    dcos, vdCos, vvcos, f64
}
vml_unary! {
    /// `r[i] = tan(a[i])` (single precision).
    tan, vsTan, vvtanf, f32
}
vml_unary! {
    /// `r[i] = tan(a[i])` (double precision).
    dtan, vdTan, vvtan, f64
}
vml_unary! {
    /// `r[i] = log10(a[i])` (single precision).
    log10, vsLog10, vvlog10f, f32
}
vml_unary! {
    /// `r[i] = log10(a[i])` (double precision).
    dlog10, vdLog10, vvlog10, f64
}
vml_unary! {
    /// `r[i] = cbrt(a[i])` (single precision).
    cbrt, vsCbrt, vvcbrtf, f32
}
vml_unary! {
    /// `r[i] = cbrt(a[i])` (double precision).
    dcbrt, vdCbrt, vvcbrt, f64
}
vml_unary! {
    /// `r[i] = asin(a[i])` (single precision).
    asin, vsAsin, vvasinf, f32
}
vml_unary! {
    /// `r[i] = asin(a[i])` (double precision).
    dasin, vdAsin, vvasin, f64
}
vml_unary! {
    /// `r[i] = acos(a[i])` (single precision).
    acos, vsAcos, vvacosf, f32
}
vml_unary! {
    /// `r[i] = acos(a[i])` (double precision).
    dacos, vdAcos, vvacos, f64
}
vml_unary! {
    /// `r[i] = atan(a[i])` (single precision).
    atan, vsAtan, vvatanf, f32
}
vml_unary! {
    /// `r[i] = atan(a[i])` (double precision).
    datan, vdAtan, vvatan, f64
}
