// Hand-written FFI surface for Apple Silicon (`aarch64-apple-darwin`).
//
// Intel ships no oneMKL build for `aarch64-apple-darwin`, so `nuvai-mkl-sys`
// cannot generate bindings from the oneMKL headers on this target. Instead
// this module declares the small, bounded set of C symbols the fallback
// backends call — Accelerate's CBLAS, the Fortran LAPACK `_` entry points,
// the vDSP DFT routines, the vDSP vector-arithmetic routines, the vForce
// vector-math routines, and the Sparse/SparseSolve direct solvers — directly
// against the Accelerate framework. The Intel x86_64 bindgen output is
// untouched.
//
// The CBLAS symbols are re-exported under the *same names* as the Intel
// bindgen output because both MKL and Accelerate implement the netlib CBLAS
// ABI; the safe wrapper's `blas` module therefore needs no per-backend
// dispatch (see ADR-0003, decision 4).
//
// Type layouts below were verified against the macOS SDK headers by a C probe
// (`sizeof`/`offsetof`), not assumed from documentation.

// The declarations below deliberately keep the C SDK's own identifiers —
// `CblasRowMajor`, `vDSP_DFT_Setup`, `SparseMatrixStructure.rowCount` — so that
// this surface can be read against the Accelerate headers it mirrors, and the
// three C-naming lints are suppressed for the module rather than renaming it
// away from the ABI it describes.
//
// These three are the *only* suppressions the module needs (#36). They were
// measured, not assumed: with the crate-root `allow`s lifted, `cargo check`
// reports nothing here except these naming lints, and `cargo clippy` adds
// exactly one finding — which is fixed rather than suppressed (see the
// `AmbiguousIfCopy` guard below, whose `?Sized` bound was redundant). In
// particular `dead_code` is unnecessary: every declaration is reachable through
// the `pub use aarch64::*` in `lib.rs`, so none of them is dead, and
// `improper_ctypes`/`clippy::all` — the two that were covering hand-written,
// hand-reviewed code — are gone.
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::{c_long, c_ulong, c_void};

// The netlib CBLAS + Fortran-LAPACK declarations are shared verbatim with the
// OpenBLAS surface (`linux_aarch64.rs`) — both Accelerate and OpenBLAS
// implement the netlib CBLAS ABI and expose LAPACK through the `_` entry
// points, so one declaration set serves both (see `netlib_abi.rs` for the full
// ABI notes).
include!("netlib_abi.rs");

// ---------------------------------------------------------------------------
// vDSP DFT
// ---------------------------------------------------------------------------

/// `unsigned long` — 8 bytes on LP64.
pub type vDSP_Length = c_ulong;
/// `long` — the stride type of the `vDSP_v*` vector routines (8 bytes on LP64,
/// and *signed*, unlike `vDSP_Length`).
pub type vDSP_Stride = c_long;
/// DFT direction (`vDSP_ENUM(int, vDSP_DFT_Direction)`).
pub type vDSP_DFT_Direction = c_int;

pub const vDSP_DFT_FORWARD: vDSP_DFT_Direction = 1;
pub const vDSP_DFT_INVERSE: vDSP_DFT_Direction = -1;

/// Opaque DFT setup (`struct vDSP_DFT_SetupStruct *`).
pub type vDSP_DFT_Setup = *mut c_void;
/// Opaque double-precision DFT setup (`struct vDSP_DFT_SetupStructD *`).
pub type vDSP_DFT_SetupD = *mut c_void;

/// Opaque interleaved-complex DFT setup (`struct vDSP_DFT_Interleaved_SetupStruct *`).
pub type vDSP_DFT_Interleaved_Setup = *mut c_void;
/// Opaque double-precision interleaved-complex DFT setup.
pub type vDSP_DFT_Interleaved_SetupD = *mut c_void;

/// Real-to-complex flag for the interleaved DFT (`vDSP_ENUM(bool,
/// vDSP_DFT_RealtoComplex)`). The C enum's underlying type is `_Bool` (1 byte),
/// so it is declared `u8` — not `c_int` (4 bytes) — to match the callee's ABI
/// register width on ARM64. Only `0`/`1` is ever passed.
pub type vDSP_DFT_RealtoComplex = u8;
pub const vDSP_DFT_Interleaved_ComplextoComplex: vDSP_DFT_RealtoComplex = 0;
pub const vDSP_DFT_Interleaved_RealtoComplex: vDSP_DFT_RealtoComplex = 1;

/// Interleaved single-precision complex (layout-identical to `MKL_Complex8`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DSPComplex {
    pub real: f32,
    pub imag: f32,
}

/// Interleaved double-precision complex (layout-identical to `MKL_Complex16`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DSPDoubleComplex {
    pub real: f64,
    pub imag: f64,
}

// ---------------------------------------------------------------------------
// vForce (note argument order: `(y, x, n)` — unlike MKL's `(n, src, dst)`)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Sparse / SparseSolve (direct solvers)
// ---------------------------------------------------------------------------

/// Status code returned by `_Sparse*` routines (`sparse_status`).
pub const SPARSE_SUCCESS: sparse_status = 0;
pub const SPARSE_ILLEGAL_PARAMETER: sparse_status = -1000;
pub const SPARSE_CANNOT_SET_PROPERTY: sparse_status = -1001;
pub const SPARSE_SYSTEM_ERROR: sparse_status = -1002;

/// Status field of a factorization object (`SparseStatus`).
pub const SparseStatusOK: SparseStatus_t = 0;
pub const SparseFactorizationFailed: SparseStatus_t = -1;
pub const SparseMatrixIsSingular: SparseStatus_t = -2;
pub const SparseInternalError: SparseStatus_t = -3;
pub const SparseParameterError: SparseStatus_t = -4;

pub type sparse_status = c_int;
pub type SparseStatus_t = c_int;
pub type SparseControl_t = u32;
pub type SparseOrder_t = u8;
pub type SparseScaling_t = u8;
pub type SparseFactorization_t = u8;
pub type SparseKind_t = u32;
pub type SparseTriangle_t = u8;
/// `void *(*)(size_t)` allocator callback used by the sparse options.
pub type SparseAllocator_t = unsafe extern "C" fn(size: usize) -> *mut c_void;

pub const SparseDefaultControl: SparseControl_t = 0;

pub const SparseOrderDefault: SparseOrder_t = 0;
pub const SparseOrderUser: SparseOrder_t = 1;
pub const SparseOrderAMD: SparseOrder_t = 2;
pub const SparseOrderMetis: SparseOrder_t = 3;
pub const SparseOrderCOLAMD: SparseOrder_t = 4;

pub const SparseScalingDefault: SparseScaling_t = 0;
pub const SparseScalingUser: SparseScaling_t = 1;
pub const SparseScalingEquilibriationInf: SparseScaling_t = 2;

pub const SparseFactorizationCholesky: SparseFactorization_t = 0;
pub const SparseFactorizationLDLT: SparseFactorization_t = 1;
pub const SparseFactorizationLDLTUnpivoted: SparseFactorization_t = 2;
pub const SparseFactorizationLDLTSBK: SparseFactorization_t = 3;
pub const SparseFactorizationLDLTTPP: SparseFactorization_t = 4;
pub const SparseFactorizationQR: SparseFactorization_t = 40;
pub const SparseFactorizationCholeskyAtA: SparseFactorization_t = 41;
pub const SparseFactorizationLU: SparseFactorization_t = 80;
pub const SparseFactorizationLUUnpivoted: SparseFactorization_t = 81;
pub const SparseFactorizationLUSPP: SparseFactorization_t = 82;
pub const SparseFactorizationLUTPP: SparseFactorization_t = 83;

pub const SparseOrdinary: SparseKind_t = 0;
pub const SparseTriangular: SparseKind_t = 1;
pub const SparseUnitTriangular: SparseKind_t = 2;
pub const SparseSymmetric: SparseKind_t = 3;
pub const SparseHermitian: SparseKind_t = 7;

pub const SparseUpperTriangle: SparseTriangle_t = 0;
pub const SparseLowerTriangle: SparseTriangle_t = 1;

/// Bit-field struct of matrix attributes (C `SparseAttributes_t`).
///
/// The C bit-field layout packs into a 4-byte storage unit (`_reserved` is an
/// `unsigned int`, so clang chooses a 4-byte unit). Bit positions:
/// `transpose` = bit 0, `triangle` = bit 1, `kind` = bits 2..3,
/// `_reserved` = bits 4..14, `_allocatedBySparse` = bit 15.
///
/// `kind` is **2 bits** here, so representable kinds are `SparseOrdinary`(0)
/// through `SparseSymmetric`(3) only. This is not an oversight: the SDK
/// declares a *second*, separate attributes struct for complex matrices —
/// `SparseAttributesComplex_t`, which widens `kind` to 3 bits and adds a
/// `conjugate_transpose` bit so that `SparseHermitian`(7) becomes
/// representable. The two types are not interchangeable, and the 2-bit field
/// in the real-valued struct can never hold `SparseHermitian`.
///
/// This crate wraps the real (non-complex) sparse path only — the owner of
/// this type is `SparseMatrixStructure`, never `SparseMatrixStructureComplex`
/// — so `SparseAttributes_t` is the correct mirror and the 2-bit `kind` mask
/// is correct. A `& 0x7` mask would read bit 4 (unused/reserved here, and
/// `conjugate_transpose` in the *complex* struct) as part of `kind`, so it
/// must not be widened. See `sparse_attributes_layout_matches_sdk_probe`
/// below, which pins the layout verified against the SDK headers.
///
/// (Verified by a C `sizeof`/bit probe against the macOS SDK: `sizeof == 4`,
/// `kind = SparseSymmetric` encodes to `0x0c`, `_allocatedBySparse` to
/// `0x8000`.)
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseAttributes_t(pub u32);

impl SparseAttributes_t {
    /// Ordinary (non-symmetric) matrix, upper triangle, not transposed.
    pub const fn ordinary() -> Self {
        Self(0)
    }

    /// Real symmetric matrix (`kind = SparseSymmetric`, `triangle` is
    /// interpreted as upper/lower; `transpose` = false).
    pub const fn symmetric() -> Self {
        Self(SparseSymmetric << 2)
    }

    /// The `transpose` bit.
    pub const fn transpose(&self) -> bool {
        (self.0 & 0x1) != 0
    }

    /// The `triangle` bit (`SparseUpperTriangle` = 0, `SparseLowerTriangle` = 1).
    pub const fn triangle(&self) -> SparseTriangle_t {
        ((self.0 >> 1) & 0x1) as SparseTriangle_t
    }

    /// The `kind` field: `SparseOrdinary`(0), `SparseTriangular`(1),
    /// `SparseUnitTriangular`(2) or `SparseSymmetric`(3).
    ///
    /// The mask is `0x3` because this struct's `kind` is 2 bits wide — see the
    /// type-level docs for why `SparseHermitian`(7) is *not* reachable here.
    pub const fn kind(&self) -> SparseKind_t {
        (self.0 >> 2) & 0x3
    }

    /// Whether `kind == SparseSymmetric`.
    pub const fn is_symmetric(&self) -> bool {
        self.kind() == SparseSymmetric
    }
}

/// CSC sparse-matrix structure. `columnStarts` has `columnCount + 1` entries;
/// `rowIndices` has `nnz` entries. (Verified: `sizeof == 32`.)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SparseMatrixStructure {
    pub rowCount: i32,
    pub columnCount: i32,
    pub columnStarts: *mut c_long,
    pub rowIndices: *mut i32,
    pub attributes: SparseAttributes_t,
    pub blockSize: u8,
}

/// Double-precision sparse matrix: a CSC structure plus the value array.
/// (Verified: `sizeof == 40`.)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SparseMatrix_Double {
    pub structure: SparseMatrixStructure,
    pub data: *mut f64,
}

/// Double-precision dense matrix with explicit column stride.
/// (Verified: `sizeof == 24`.)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseMatrix_Double {
    pub rowCount: i32,
    pub columnCount: i32,
    pub columnStride: i32,
    pub attributes: SparseAttributes_t,
    pub data: *mut f64,
}

/// Double-precision dense vector. (Verified: `sizeof == 16`.)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseVector_Double {
    pub count: i32,
    pub data: *mut f64,
}

/// Semi-opaque symbolic factorization. Layout is public and reproduced here
/// because `SparseOpaqueFactorization_Double` embeds one by value.
/// (Verified: `sizeof == 64`.)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SparseOpaqueSymbolicFactorization {
    pub status: SparseStatus_t,
    pub rowCount: i32,
    pub columnCount: i32,
    pub attributes: SparseAttributes_t,
    pub blockSize: u8,
    pub type_: SparseFactorization_t,
    pub factorization: *mut c_void,
    pub workspaceSize_Float: usize,
    pub workspaceSize_Double: usize,
    pub factorSize_Float: usize,
    pub factorSize_Double: usize,
}

/// Semi-opaque numeric factorization. Returned by value from
/// `_SparseFactorQR_Double`/`_SparseFactorSymmetric_Double`, so the exact
/// layout is required for the struct-return ABI. (Verified: `sizeof == 104`.)
///
/// Deliberately **not `Copy`** — see the `AmbiguousIfCopy` guard immediately
/// below, which fails the build if `Copy` is ever re-added. This type owns
/// heap allocations through `numericFactorization` (and, via
/// `symbolicFactorization`, its `factorization` pointer), both released with
/// `_SparseDestroyOpaqueNumeric_Double`. An implicit `Copy` would duplicate
/// the only owning handle, so dropping both copies frees the same allocation
/// twice. Returning it by value from an `extern "C"` fn does not require
/// `Copy`; move semantics are sufficient and are what the callers rely on.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct SparseOpaqueFactorization_Double {
    pub status: SparseStatus_t,
    pub attributes: SparseAttributes_t,
    pub symbolicFactorization: SparseOpaqueSymbolicFactorization,
    pub userFactorStorage: bool,
    pub numericFactorization: *mut c_void,
    pub solveWorkspaceRequiredStatic: usize,
    pub solveWorkspaceRequiredPerRHS: usize,
}

// Compile-time guard: `SparseOpaqueFactorization_Double` must never be `Copy`
// again (see its docs — an implicit copy double-frees the numeric
// factorization). This is `static_assertions::assert_not_impl_any!`'s trick
// without the dependency: if the type were `Copy`, both `AmbiguousIfCopy`
// impls would apply, so the `_` in the `AmbiguousIfCopy<_>` projection becomes
// uninferable and the build fails here with "type annotations needed".
//
// Placed at module scope rather than inside `mod tests` so it is checked by
// every `cargo build`/`check`, not only by `cargo test`.
const _: fn() = || {
    trait AmbiguousIfCopy<A> {
        fn some_item() {}
    }
    impl<T: ?Sized> AmbiguousIfCopy<()> for T {}
    // No `?Sized` on this one: `Copy` requires `Sized` (through `Clone`), so the
    // relaxation is ignored — which is what `cargo clippy` reported here
    // ("`?Sized` bound is ignored because of a `Sized` requirement") once
    // `#![allow(clippy::all)]` stopped covering this module (#36). Dropping it
    // does not narrow the impl: a type that is `Copy` was never unsized to
    // begin with, so the guard still applies to exactly the types that would
    // make the projection ambiguous.
    impl<T: Copy> AmbiguousIfCopy<u8> for T {}
    let _ = <SparseOpaqueFactorization_Double as AmbiguousIfCopy<_>>::some_item;
};

/// Options for the symbolic stage of a sparse factorization.
/// (Verified: `sizeof == 48`.)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SparseSymbolicFactorOptions {
    pub control: SparseControl_t,
    pub orderMethod: SparseOrder_t,
    pub order: *mut c_int,
    pub ignoreRowsAndColumns: *mut c_int,
    pub malloc: SparseAllocator_t,
    pub free: unsafe extern "C" fn(*mut c_void),
    pub reportError: Option<unsafe extern "C" fn(*const i8)>,
}

/// Options for the numeric stage of a sparse factorization.
/// (Verified: `sizeof == 32`.)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SparseNumericFactorOptions {
    pub control: SparseControl_t,
    pub scalingMethod: SparseScaling_t,
    pub scaling: *mut c_void,
    pub pivotTolerance: f64,
    pub zeroTolerance: f64,
}

unsafe extern "C" {
    // --- vDSP DFT (complex-to-complex, split real/imag arrays) ---
    pub fn vDSP_DFT_zop_CreateSetup(
        previous: vDSP_DFT_Setup,
        length: vDSP_Length,
        direction: vDSP_DFT_Direction,
    ) -> vDSP_DFT_Setup;
    pub fn vDSP_DFT_zop_CreateSetupD(
        previous: vDSP_DFT_SetupD,
        length: vDSP_Length,
        direction: vDSP_DFT_Direction,
    ) -> vDSP_DFT_SetupD;
    pub fn vDSP_DFT_Execute(
        setup: vDSP_DFT_Setup,
        ir: *const f32,
        ii: *const f32,
        or: *mut f32,
        oi: *mut f32,
    );
    pub fn vDSP_DFT_ExecuteD(
        setup: vDSP_DFT_SetupD,
        ir: *const f64,
        ii: *const f64,
        or: *mut f64,
        oi: *mut f64,
    );
    pub fn vDSP_DFT_DestroySetup(setup: vDSP_DFT_Setup);
    pub fn vDSP_DFT_DestroySetupD(setup: vDSP_DFT_SetupD);

    // --- vDSP DFT (interleaved complex, macOS 12.0+) ---
    // Unlike the split family above, these transform an interleaved
    // `DSP*Complex` buffer directly (no split/reinterleave copy). Availability
    // is `API_AVAILABLE(macos(12.0), …)` — later than the split family's 10.7.
    pub fn vDSP_DFT_Interleaved_CreateSetup(
        previous: vDSP_DFT_Interleaved_Setup,
        length: vDSP_Length,
        direction: vDSP_DFT_Direction,
        real_to_complex: vDSP_DFT_RealtoComplex,
    ) -> vDSP_DFT_Interleaved_Setup;
    pub fn vDSP_DFT_Interleaved_CreateSetupD(
        previous: vDSP_DFT_Interleaved_SetupD,
        length: vDSP_Length,
        direction: vDSP_DFT_Direction,
        real_to_complex: vDSP_DFT_RealtoComplex,
    ) -> vDSP_DFT_Interleaved_SetupD;
    pub fn vDSP_DFT_Interleaved_Execute(
        setup: vDSP_DFT_Interleaved_Setup,
        ir: *const DSPComplex,
        or: *mut DSPComplex,
    );
    pub fn vDSP_DFT_Interleaved_ExecuteD(
        setup: vDSP_DFT_Interleaved_SetupD,
        ir: *const DSPDoubleComplex,
        or: *mut DSPDoubleComplex,
    );
    pub fn vDSP_DFT_Interleaved_DestroySetup(setup: vDSP_DFT_Interleaved_Setup);
    pub fn vDSP_DFT_Interleaved_DestroySetupD(setup: vDSP_DFT_Interleaved_SetupD);

    // --- vForce (vector math; `(y, x, n)` argument order) ---
    pub fn vvexpf(y: *mut f32, x: *const f32, n: *const c_int);
    pub fn vvexp(y: *mut f64, x: *const f64, n: *const c_int);
    pub fn vvlogf(y: *mut f32, x: *const f32, n: *const c_int);
    pub fn vvlog(y: *mut f64, x: *const f64, n: *const c_int);
    pub fn vvlog10f(y: *mut f32, x: *const f32, n: *const c_int);
    pub fn vvlog10(y: *mut f64, x: *const f64, n: *const c_int);
    pub fn vvsqrtf(y: *mut f32, x: *const f32, n: *const c_int);
    pub fn vvsqrt(y: *mut f64, x: *const f64, n: *const c_int);
    pub fn vvcbrtf(y: *mut f32, x: *const f32, n: *const c_int);
    pub fn vvcbrt(y: *mut f64, x: *const f64, n: *const c_int);
    pub fn vvsinf(y: *mut f32, x: *const f32, n: *const c_int);
    pub fn vvsin(y: *mut f64, x: *const f64, n: *const c_int);
    pub fn vvcosf(y: *mut f32, x: *const f32, n: *const c_int);
    pub fn vvcos(y: *mut f64, x: *const f64, n: *const c_int);
    pub fn vvtanf(y: *mut f32, x: *const f32, n: *const c_int);
    pub fn vvtan(y: *mut f64, x: *const f64, n: *const c_int);
    pub fn vvasinf(y: *mut f32, x: *const f32, n: *const c_int);
    pub fn vvasin(y: *mut f64, x: *const f64, n: *const c_int);
    pub fn vvacosf(y: *mut f32, x: *const f32, n: *const c_int);
    pub fn vvacos(y: *mut f64, x: *const f64, n: *const c_int);
    pub fn vvatanf(y: *mut f32, x: *const f32, n: *const c_int);
    pub fn vvatan(y: *mut f64, x: *const f64, n: *const c_int);
    pub fn vvtanhf(y: *mut f32, x: *const f32, n: *const c_int);
    pub fn vvtanh(y: *mut f64, x: *const f64, n: *const c_int);

    // --- vDSP vector arithmetic (`(A, IA, B, IB, C, IC, N)` order) ---
    //
    // Element-wise binary and squaring routines for the VML surface (#48, #49).
    // vForce carries only *one* of the six binary operations VML exposes
    // (`vvdivf`/`vvdiv`) and has no squaring function at all — its 84 `vv*`
    // symbols are unary transcendentals plus `vvpow`/`vvfmod`/`vvremainder`/
    // `vvatan2`/`vvcopysign`/`vvnextafter` — so the Apple Silicon backend for
    // these maps to vDSP, which covers all seven uniformly, rather than mixing
    // a single vForce call into a vDSP-shaped family.
    //
    // `vDSP_vsub` and `vDSP_vdiv` are declared with their two input pointers
    // *swapped* relative to their names, and the SDK header says so explicitly
    // ("Caution: A and B are swapped!"). Both compute `A op B` where `A` is the
    // **second** input argument, so a caller wanting `a - b` passes
    // `(b, 1, a, 1, r, 1, n)`. The safe wrapper compensates; these declarations
    // reproduce the C ABI verbatim.
    pub fn vDSP_vadd(
        a: *const f32,
        ia: vDSP_Stride,
        b: *const f32,
        ib: vDSP_Stride,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    pub fn vDSP_vaddD(
        a: *const f64,
        ia: vDSP_Stride,
        b: *const f64,
        ib: vDSP_Stride,
        c: *mut f64,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    pub fn vDSP_vsub(
        b: *const f32,
        ib: vDSP_Stride,
        a: *const f32,
        ia: vDSP_Stride,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    pub fn vDSP_vsubD(
        b: *const f64,
        ib: vDSP_Stride,
        a: *const f64,
        ia: vDSP_Stride,
        c: *mut f64,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    pub fn vDSP_vmul(
        a: *const f32,
        ia: vDSP_Stride,
        b: *const f32,
        ib: vDSP_Stride,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    pub fn vDSP_vmulD(
        a: *const f64,
        ia: vDSP_Stride,
        b: *const f64,
        ib: vDSP_Stride,
        c: *mut f64,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    pub fn vDSP_vdiv(
        b: *const f32,
        ib: vDSP_Stride,
        a: *const f32,
        ia: vDSP_Stride,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    pub fn vDSP_vdivD(
        b: *const f64,
        ib: vDSP_Stride,
        a: *const f64,
        ia: vDSP_Stride,
        c: *mut f64,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    pub fn vDSP_vmax(
        a: *const f32,
        ia: vDSP_Stride,
        b: *const f32,
        ib: vDSP_Stride,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    pub fn vDSP_vmaxD(
        a: *const f64,
        ia: vDSP_Stride,
        b: *const f64,
        ib: vDSP_Stride,
        c: *mut f64,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    pub fn vDSP_vmin(
        a: *const f32,
        ia: vDSP_Stride,
        b: *const f32,
        ib: vDSP_Stride,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    pub fn vDSP_vminD(
        a: *const f64,
        ia: vDSP_Stride,
        b: *const f64,
        ib: vDSP_Stride,
        c: *mut f64,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    /// `C[i] = A[i] * A[i]` — the squaring routine VML's `vsSqr`/`vdSqr` maps
    /// to (#49). vForce has no squaring function.
    pub fn vDSP_vsq(a: *const f32, ia: vDSP_Stride, c: *mut f32, ic: vDSP_Stride, n: vDSP_Length);
    pub fn vDSP_vsqD(a: *const f64, ia: vDSP_Stride, c: *mut f64, ic: vDSP_Stride, n: vDSP_Length);

    // --- Sparse direct solvers ---
    pub fn _SparseFactorSymmetric_Double(
        factorType: SparseFactorization_t,
        matrix: *const SparseMatrix_Double,
        sfoptions: *const SparseSymbolicFactorOptions,
        nfoptions: *const SparseNumericFactorOptions,
    ) -> SparseOpaqueFactorization_Double;
    pub fn _SparseFactorQR_Double(
        factorType: SparseFactorization_t,
        matrix: *const SparseMatrix_Double,
        sfoptions: *const SparseSymbolicFactorOptions,
        nfoptions: *const SparseNumericFactorOptions,
    ) -> SparseOpaqueFactorization_Double;
    pub fn _SparseSolveOpaque_Double(
        factored: *const SparseOpaqueFactorization_Double,
        rhs: *const DenseMatrix_Double,
        soln: *const DenseMatrix_Double,
        workspace: *mut c_void,
    );
    pub fn _SparseDestroyOpaqueNumeric_Double(toFree: *mut SparseOpaqueFactorization_Double);

    // --- libc allocator (backs the default sparse `SparseSymbolicFactorOptions`)
    pub fn malloc(size: usize) -> *mut c_void;
    pub fn free(ptr: *mut c_void);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the `SparseAttributes_t` bit layout to what a `sizeof`/bit probe
    /// against the macOS SDK headers actually produces (the crate's stated
    /// convention for these hand-written layouts, rather than trusting the
    /// docs).
    ///
    /// The `0x30`/`0x8000` assertions are the point of this test: they fail if
    /// `kind()` is ever widened from `0x3` to `0x7`. Bit 4 is the low bit of
    /// the *3-bit* `kind` in the **separate** `SparseAttributesComplex_t`, and
    /// bit 5 is that struct's `conjugate_transpose`; neither exists in this
    /// struct. Widening the mask would decode a complex-struct bit pattern as
    /// this struct's `kind`.
    #[test]
    fn sparse_attributes_layout_matches_sdk_probe() {
        // Probe: sizeof(SparseAttributes_t) == 4 (clang picks a 4-byte unit
        // because `_reserved` is declared `unsigned int`).
        assert_eq!(core::mem::size_of::<SparseAttributes_t>(), 4);

        // `transpose` = bit 0.
        assert!(SparseAttributes_t(0x1).transpose());
        assert!(!SparseAttributes_t(0x0).transpose());

        // `triangle` = bit 1 (upper = 0, lower = 1).
        assert_eq!(SparseAttributes_t(0x2).triangle(), SparseLowerTriangle);
        assert_eq!(SparseAttributes_t(0x0).triangle(), SparseUpperTriangle);

        // `kind` = bits 2..3: the probe encodes SparseSymmetric(3) as 0x0c.
        assert_eq!(SparseAttributes_t::symmetric().0, 0x0c);
        assert!(SparseAttributes_t::symmetric().is_symmetric());
        assert_eq!(SparseAttributes_t::ordinary().kind(), SparseOrdinary);
        for kind in [
            SparseOrdinary,
            SparseTriangular,
            SparseUnitTriangular,
            SparseSymmetric,
        ] {
            assert_eq!(SparseAttributes_t(kind << 2).kind(), kind);
        }

        // Bits 4..5 are reserved in this struct, so they must not alias into
        // `kind` (this is what a `& 0x7` mask would get wrong).
        assert_eq!(SparseAttributes_t(0x30).kind(), SparseOrdinary);

        // `_allocatedBySparse` = bit 15: the probe encodes it as 0x8000, which
        // is also outside `kind`.
        assert_eq!(SparseAttributes_t(0x8000).kind(), SparseOrdinary);
    }

    /// `SparseOpaqueFactorization_Double` owns heap through raw pointers and is
    /// freed by `_SparseDestroyOpaqueNumeric_Double`, so implicit duplication
    /// must be impossible. The module-scope `AmbiguousIfCopy` const above
    /// proves the negative at compile time (and is checked by `cargo build`);
    /// this asserts the positive half — that a *move* still works, so the
    /// guard has not simply made the type unusable.
    #[test]
    fn factorization_handle_moves_but_does_not_copy() {
        fn consume(f: SparseOpaqueFactorization_Double) -> SparseStatus_t {
            f.status
        }
        let f = SparseOpaqueFactorization_Double {
            status: SparseStatusOK,
            attributes: SparseAttributes_t::ordinary(),
            symbolicFactorization: SparseOpaqueSymbolicFactorization {
                status: SparseStatusOK,
                rowCount: 1,
                columnCount: 1,
                attributes: SparseAttributes_t::ordinary(),
                blockSize: 1,
                type_: SparseFactorizationQR,
                factorization: core::ptr::null_mut(),
                workspaceSize_Float: 0,
                workspaceSize_Double: 0,
                factorSize_Float: 0,
                factorSize_Double: 0,
            },
            userFactorStorage: false,
            numericFactorization: core::ptr::null_mut(),
            solveWorkspaceRequiredStatic: 0,
            solveWorkspaceRequiredPerRHS: 0,
        };
        // Moved into `consume` — compiles only because it is not `Copy`
        // *and* move semantics are intact.
        assert_eq!(consume(f), SparseStatusOK);
    }

    /// Link-level smoke test: exercises a real Accelerate symbol so the P2
    /// gate ("sys links clean on aarch64") proves the framework link, not
    /// merely that the externs type-check.
    #[test]
    fn accelerate_cblas_links() {
        let x = [1.0f32, 2.0, 3.0];
        let y = [4.0f32, 5.0, 6.0];
        // SAFETY: `x`/`y` have exactly 3 elements and `incx`/`incy` are 1, so
        // `cblas_sdot(3, …)` reads exactly the three elements of each array.
        let dot = unsafe { cblas_sdot(3, x.as_ptr(), 1, y.as_ptr(), 1) };
        assert!((dot - 32.0).abs() < 1e-6, "cblas_sdot = {dot}");
    }
}
