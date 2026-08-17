// Hand-written FFI surface for Apple Silicon (`aarch64-apple-darwin`).
//
// Intel ships no oneMKL build for `aarch64-apple-darwin`, so `nuvai-mkl-sys`
// cannot generate bindings from the oneMKL headers on this target. Instead
// this module declares the small, bounded set of C symbols the fallback
// backends call — Accelerate's CBLAS, the Fortran LAPACK `_` entry points,
// the vDSP DFT routines, the vForce vector-math routines, and the
// Sparse/SparseSolve direct solvers — directly against the Accelerate
// framework. The Intel x86_64 bindgen output is untouched.
//
// The CBLAS symbols are re-exported under the *same names* as the Intel
// bindgen output because both MKL and Accelerate implement the netlib CBLAS
// ABI; the safe wrapper's `blas` module therefore needs no per-backend
// dispatch (see ADR-0003, decision 4).
//
// Type layouts below were verified against the macOS SDK headers by a C probe
// (`sizeof`/`offsetof`), not assumed from documentation.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(improper_ctypes)]
#![allow(clippy::all)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_imports)]

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

    /// The `kind` field (`SparseOrdinary`..`SparseHermitian`).
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
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SparseOpaqueFactorization_Double {
    pub status: SparseStatus_t,
    pub attributes: SparseAttributes_t,
    pub symbolicFactorization: SparseOpaqueSymbolicFactorization,
    pub userFactorStorage: bool,
    pub numericFactorization: *mut c_void,
    pub solveWorkspaceRequiredStatic: usize,
    pub solveWorkspaceRequiredPerRHS: usize,
}

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
