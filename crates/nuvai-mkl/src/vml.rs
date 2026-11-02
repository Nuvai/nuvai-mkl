//! VML — Vector Math Library: element-wise unary and binary functions
//! computed over a whole vector.
//!
//! On Intel targets these call the oneMKL VML functions — unary
//! `(n, src, dst)`, binary `(n, a, b, r)`. On Apple Silicon
//! (`aarch64-apple-darwin`) they call Accelerate, and *which* Accelerate
//! framework provides a given function depends on what that framework carries:
//! the unary transcendental functions (`exp`, `ln`, `sin`, …, `tanh`) map to
//! **vForce**, whose `(dst, src, n)` argument order is the reverse of MKL's,
//! while `sqr` and every binary operation map to **vDSP**, whose shape is
//! `(A, IA, B, IB, C, IC, N)`. vForce has no squaring function and carries only
//! one of the six binary operations (`vvdiv`), so mapping those to vDSP is what
//! keeps the surface complete instead of partially unsupported; the symbol
//! survey behind that choice is in `nuvai-mkl-sys/src/aarch64.rs`.
//! [`vml_unary!`] and [`vml_binary!`] carry both backend symbols and each cfg
//! branch emits the matching call, so the backend is never chosen silently
//! (ADR-0003). On `aarch64-unknown-linux-gnu` there is no VML/vForce/vDSP
//! backend (OpenBLAS covers only BLAS/LAPACK), so every function returns
//! [`ErrorKind::Unsupported`].

use std::os::raw::c_int;

use crate::conv::len_to_c_int;
use crate::error::{Error, Result};

/// Validate `src`/`dst` lengths and return the vector length as `c_int`.
// On `aarch64-unknown-linux-gnu` this is unreachable: every generated function
// below returns `Unsupported` before it looks at its arguments, so nothing
// calls `check` there. On the other targets the fast paths do.
#[cfg_attr(all(target_os = "linux", target_arch = "aarch64"), allow(dead_code))]
#[inline]
fn check(src: usize, dst: usize, name: &str) -> Result<c_int> {
    if src != dst {
        return Err(Error::invalid(format!("{name}: src/dst length mismatch")));
    }
    // The C routine takes the length as `int`; a vector longer than `i32::MAX`
    // would truncate and silently process only a prefix. `len_to_c_int` rejects
    // it up front, sharing that rule with the VSL/PARDISO/DSS call sites.
    len_to_c_int(src, name)
}

/// Validate `a`/`b`/`r` lengths and return the vector length as `c_int`.
///
/// The binary counterpart of [`check`], under the same rule: all three buffers
/// must be the same length, and that length goes through `len_to_c_int` so an
/// over-`i32::MAX` vector is rejected rather than truncated. Every binary
/// routine is documented as computing `r[i]` from `a[i]` and `b[i]`, so a
/// length that differed between them would leave the tail of `r` unwritten on
/// one backend and read out of bounds on another.
// Same reachability as `check` above.
#[cfg_attr(all(target_os = "linux", target_arch = "aarch64"), allow(dead_code))]
#[inline]
fn check_binary(a: usize, b: usize, r: usize, name: &str) -> Result<c_int> {
    if a != b || a != r {
        return Err(Error::invalid(format!("{name}: a/b/r length mismatch")));
    }
    len_to_c_int(a, name)
}

/// Call an Accelerate routine whose shape depends on which framework carries it.
///
/// vForce's unary functions take `(dst, src, n)` and vDSP's take
/// `(A, IA, C, IC, N)`; this expands to the call the `$shape` selects, so
/// [`vml_unary!`] can keep one body for both. vDSP's stride is a `vDSP_Stride`
/// (signed) and its length a `vDSP_Length` (unsigned) — the length is the
/// `c_int` `check` returned, which is non-negative by construction, so the cast
/// is exact.
// Only invoked from `vml_unary!`'s macOS arm, so the definition is `cfg`-gated
// rather than left unused on the other targets.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
macro_rules! apple_unary {
    (vforce $sym:ident, $src:ident, $dst:ident, $n:ident) => {
        nuvai_mkl_sys::$sym($dst.as_mut_ptr(), $src.as_ptr(), &$n)
    };
    (vdsp $sym:ident, $src:ident, $dst:ident, $n:ident) => {
        nuvai_mkl_sys::$sym(
            $src.as_ptr(),
            1,
            $dst.as_mut_ptr(),
            1,
            $n as nuvai_mkl_sys::vDSP_Length,
        )
    };
}

/// Call an Accelerate vDSP routine carrying both inputs.
///
/// `direct` is for the routines whose two input pointers are in name order;
/// `swapped` is for `vDSP_vsub`/`vDSP_vdiv`, which the Accelerate SDK declares
/// as `(B, A)` — the callee computes `(second pointer) op (first pointer)`, so
/// the pointers are passed reversed here to compute the `a op b` the safe
/// routine promises. The sys declarations reproduce that C order verbatim; this
/// is the one place the wrapper compensates for it.
// Same `cfg`-gating rationale as `apple_unary!`.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
macro_rules! apple_binary {
    (direct $sym:ident, $a:ident, $b:ident, $r:ident, $n:ident) => {
        nuvai_mkl_sys::$sym(
            $a.as_ptr(),
            1,
            $b.as_ptr(),
            1,
            $r.as_mut_ptr(),
            1,
            $n as nuvai_mkl_sys::vDSP_Length,
        )
    };
    (swapped $sym:ident, $a:ident, $b:ident, $r:ident, $n:ident) => {
        nuvai_mkl_sys::$sym(
            $b.as_ptr(),
            1,
            $a.as_ptr(),
            1,
            $r.as_mut_ptr(),
            1,
            $n as nuvai_mkl_sys::vDSP_Length,
        )
    };
}

/// Generate one unary VML function.
///
/// `$mkl` is the oneMKL VML symbol (used on Intel, `(n, src, dst)` order).
/// The Accelerate symbol follows either a `vforce` tag — the default, for the
/// functions vForce carries, called as `(dst, src, n)` — or a `vdsp` tag, for
/// the ones it does not, called as `(src, stride, dst, stride, n)`.
///
/// The Apple call is emitted by `apple_unary!` rather than written out here, so
/// that this body exists once for both shapes. It cannot instead be passed *in*
/// as an expression from the arm above: macro hygiene gives each expansion its
/// own syntax context, so an expression written there cannot name the
/// `src`/`dst`/`n` this arm introduces — it fails to resolve. Forwarding the
/// *identifiers* into a nested macro keeps them in this expansion's context,
/// which is why the tag is forwarded and the call is not.
macro_rules! vml_unary {
    ($(#[$doc:meta])* $name:ident, $mkl:ident, $vforce:ident, $ty:ty) => {
        vml_unary!(@emit $(#[$doc])* $name, $mkl, $ty, vforce $vforce);
    };
    ($(#[$doc:meta])* $name:ident, $mkl:ident, vdsp $vdsp:ident, $ty:ty) => {
        vml_unary!(@emit $(#[$doc])* $name, $mkl, $ty, vdsp $vdsp);
    };
    (@emit $(#[$doc:meta])* $name:ident, $mkl:ident, $ty:ty, $shape:ident $apple:ident) => {
        $(#[$doc])*
        pub fn $name(src: &[$ty], dst: &mut [$ty]) -> Result<()> {
            #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
            {
                // No VML/vForce backend on aarch64-unknown-linux-gnu
                // (OpenBLAS covers only BLAS/LAPACK). Return Unsupported before
                // validating the arguments so feature-detection callers get the
                // documented error regardless of their inputs.
                let _ = (src, dst);
                Err(Error::unsupported_linux_aarch64("VML"))
            }
            #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
            {
                let n = check(src.len(), dst.len(), stringify!($name))?;
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                {
                    // SAFETY: `src`/`dst` have equal length `n` (checked above);
                    // the Accelerate routine reads `n` elements from `src` and
                    // writes `n` to `dst`, at unit stride.
                    unsafe { apple_unary!($shape $apple, src, dst, n) };
                    Ok(())
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    // SAFETY: `src`/`dst` have equal length `n` (checked above);
                    // VML reads `n` elements from `src` and writes `n` to `dst`.
                    unsafe { nuvai_mkl_sys::$mkl(n, src.as_ptr(), dst.as_mut_ptr()) };
                    Ok(())
                }
            }
        }
    };
}

/// Generate one binary VML function.
///
/// `$mkl` is the oneMKL VML symbol (used on Intel, `(n, a, b, r)` order) and
/// `$vdsp` the Accelerate vDSP one (used on aarch64, `(A, IA, B, IB, C, IC, N)`
/// order) — vForce carries only one of the six, so these all go through vDSP.
/// Prefixing `$vdsp` with the `swapped` tag marks the routines the SDK declares
/// as `(B, A)`: see `apple_binary!`. The call is emitted by that macro for the
/// same hygiene reason given on [`vml_unary!`].
macro_rules! vml_binary {
    ($(#[$doc:meta])* $name:ident, $mkl:ident, $vdsp:ident, $ty:ty) => {
        vml_binary!(@emit $(#[$doc])* $name, $mkl, $ty, direct $vdsp);
    };
    ($(#[$doc:meta])* $name:ident, $mkl:ident, swapped $vdsp:ident, $ty:ty) => {
        vml_binary!(@emit $(#[$doc])* $name, $mkl, $ty, swapped $vdsp);
    };
    (@emit $(#[$doc:meta])* $name:ident, $mkl:ident, $ty:ty, $shape:ident $vdsp:ident) => {
        $(#[$doc])*
        pub fn $name(a: &[$ty], b: &[$ty], r: &mut [$ty]) -> Result<()> {
            #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
            {
                // As for the unary path: `Unsupported` before validation, so
                // feature detection does not depend on the arguments.
                let _ = (a, b, r);
                Err(Error::unsupported_linux_aarch64("VML"))
            }
            #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
            {
                let n = check_binary(a.len(), b.len(), r.len(), stringify!($name))?;
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                {
                    // SAFETY: `a`, `b` and `r` have equal length `n` (checked
                    // above); the routine reads `n` elements from each input and
                    // writes `n` to `r`, at unit stride.
                    unsafe { apple_binary!($shape $vdsp, a, b, r, n) };
                    Ok(())
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    // SAFETY: `a`, `b` and `r` have equal length `n` (checked
                    // above); VML reads `n` elements from each input and writes
                    // `n` to `r`.
                    unsafe { nuvai_mkl_sys::$mkl(n, a.as_ptr(), b.as_ptr(), r.as_mut_ptr()) };
                    Ok(())
                }
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
vml_unary! {
    /// `r[i] = tanh(a[i])` (single precision).
    tanh, vsTanh, vvtanhf, f32
}
vml_unary! {
    /// `r[i] = tanh(a[i])` (double precision).
    dtanh, vdTanh, vvtanh, f64
}
vml_unary! {
    /// `r[i] = a[i]²` (single precision).
    ///
    /// vForce has no squaring function, so this is the one unary operation the
    /// Apple Silicon backend takes from vDSP (`vDSP_vsq`) rather than vForce.
    sqr, vsSqr, vdsp vDSP_vsq, f32
}
vml_unary! {
    /// `r[i] = a[i]²` (double precision).
    ///
    /// As for [`sqr`], via vDSP's `vDSP_vsqD`.
    dsqr, vdSqr, vdsp vDSP_vsqD, f64
}

// ---------------------------------------------------------------------------
// Binary operations
//
// `r[i] = a[i] <op> b[i]` over two equal-length operands. All six go through
// vDSP on Apple Silicon: vForce carries only `vvdiv` of the six, and mixing a
// single vForce call into an otherwise vDSP-shaped family would buy nothing.
// ---------------------------------------------------------------------------

vml_binary! {
    /// `r[i] = a[i] + b[i]` (single precision).
    add, vsAdd, vDSP_vadd, f32
}
vml_binary! {
    /// `r[i] = a[i] + b[i]` (double precision).
    dadd, vdAdd, vDSP_vaddD, f64
}
vml_binary! {
    /// `r[i] = a[i] - b[i]` (single precision).
    sub, vsSub, swapped vDSP_vsub, f32
}
vml_binary! {
    /// `r[i] = a[i] - b[i]` (double precision).
    dsub, vdSub, swapped vDSP_vsubD, f64
}
vml_binary! {
    /// `r[i] = a[i] * b[i]` (single precision).
    mul, vsMul, vDSP_vmul, f32
}
vml_binary! {
    /// `r[i] = a[i] * b[i]` (double precision).
    dmul, vdMul, vDSP_vmulD, f64
}
vml_binary! {
    /// `r[i] = a[i] / b[i]` (single precision).
    div, vsDiv, swapped vDSP_vdiv, f32
}
vml_binary! {
    /// `r[i] = a[i] / b[i]` (double precision).
    ddiv, vdDiv, swapped vDSP_vdivD, f64
}
vml_binary! {
    /// `r[i] = max(a[i], b[i])` (single precision).
    ///
    /// The larger of the two *values*, not of their magnitudes. `NaN` handling
    /// is **not unified across backends** (ADR-0003 records the same kind of
    /// difference for the VSL generators, decision 7): oneMKL's `vsFmax`
    /// returns the non-`NaN` operand when exactly one of the pair is `NaN`,
    /// while Accelerate's `vDSP_vmax` propagates it — measured directly against
    /// the framework, `vDSP_vmax(NaN, 5)` is `NaN`. A caller for which the
    /// difference matters should apply its own `NaN` policy rather than rely on
    /// this function's.
    fmax, vsFmax, vDSP_vmax, f32
}
vml_binary! {
    /// `r[i] = max(a[i], b[i])` (double precision).
    ///
    /// `NaN` handling differs per backend as for [`fmax`].
    dfmax, vdFmax, vDSP_vmaxD, f64
}
vml_binary! {
    /// `r[i] = min(a[i], b[i])` (single precision).
    ///
    /// The smaller of the two *values*, not of their magnitudes. `NaN` handling
    /// differs per backend as for [`fmax`]: oneMKL's `vsFmin` returns the
    /// non-`NaN` operand, Accelerate's `vDSP_vmin` propagates `NaN`.
    fmin, vsFmin, vDSP_vmin, f32
}
vml_binary! {
    /// `r[i] = min(a[i], b[i])` (double precision).
    ///
    /// `NaN` handling differs per backend as for [`fmax`].
    dfmin, vdFmin, vDSP_vminD, f64
}
