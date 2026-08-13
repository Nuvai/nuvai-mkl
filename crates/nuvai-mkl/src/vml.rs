//! VML — Vector Math Library: element-wise transcendental and algebraic
//! functions computed in parallel over a whole vector.

use std::os::raw::c_int;

use crate::error::{Error, Result};

/// Validate `src`/`dst` lengths and return the vector length as `c_int`.
#[inline]
fn check(src: usize, dst: usize, name: &str) -> Result<c_int> {
    if src != dst {
        return Err(Error::invalid(format!("{name}: src/dst length mismatch")));
    }
    Ok(src as c_int)
}

macro_rules! vml_unary {
    ($(#[$doc:meta])* $name:ident, $ffi:ident, $ty:ty) => {
        $(#[$doc])*
        pub fn $name(src: &[$ty], dst: &mut [$ty]) -> Result<()> {
            let n = check(src.len(), dst.len(), stringify!($name))?;
            unsafe { nuvai_mkl_sys::$ffi(n, src.as_ptr(), dst.as_mut_ptr()) };
            Ok(())
        }
    };
}

vml_unary! {
    /// `r[i] = exp(a[i])` (single precision).
    exp, vsExp, f32
}
vml_unary! {
    /// `r[i] = exp(a[i])` (double precision).
    dexp, vdExp, f64
}
vml_unary! {
    /// `r[i] = ln(a[i])` (single precision).
    ln, vsLn, f32
}
vml_unary! {
    /// `r[i] = ln(a[i])` (double precision).
    dln, vdLn, f64
}
vml_unary! {
    /// `r[i] = sqrt(a[i])` (single precision).
    sqrt, vsSqrt, f32
}
vml_unary! {
    /// `r[i] = sqrt(a[i])` (double precision).
    dsqrt, vdSqrt, f64
}
vml_unary! {
    /// `r[i] = sin(a[i])` (single precision).
    sin, vsSin, f32
}
vml_unary! {
    /// `r[i] = sin(a[i])` (double precision).
    dsin, vdSin, f64
}
vml_unary! {
    /// `r[i] = cos(a[i])` (single precision).
    cos, vsCos, f32
}
vml_unary! {
    /// `r[i] = cos(a[i])` (double precision).
    dcos, vdCos, f64
}
vml_unary! {
    /// `r[i] = tan(a[i])` (single precision).
    tan, vsTan, f32
}
vml_unary! {
    /// `r[i] = tan(a[i])` (double precision).
    dtan, vdTan, f64
}
vml_unary! {
    /// `r[i] = log10(a[i])` (single precision).
    log10, vsLog10, f32
}
vml_unary! {
    /// `r[i] = log10(a[i])` (double precision).
    dlog10, vdLog10, f64
}
vml_unary! {
    /// `r[i] = cbrt(a[i])` (single precision).
    cbrt, vsCbrt, f32
}
vml_unary! {
    /// `r[i] = cbrt(a[i])` (double precision).
    dcbrt, vdCbrt, f64
}
vml_unary! {
    /// `r[i] = asin(a[i])` (single precision).
    asin, vsAsin, f32
}
vml_unary! {
    /// `r[i] = asin(a[i])` (double precision).
    dasin, vdAsin, f64
}
vml_unary! {
    /// `r[i] = acos(a[i])` (single precision).
    acos, vsAcos, f32
}
vml_unary! {
    /// `r[i] = acos(a[i])` (double precision).
    dacos, vdAcos, f64
}
vml_unary! {
    /// `r[i] = atan(a[i])` (single precision).
    atan, vsAtan, f32
}
vml_unary! {
    /// `r[i] = atan(a[i])` (double precision).
    datan, vdAtan, f64
}
