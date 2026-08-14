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

use std::os::raw::{c_int, c_long, c_ulong, c_void};

// ---------------------------------------------------------------------------
// Complex types (layout-identical to the MKL `MKL_Complex*` structs)
// ---------------------------------------------------------------------------

/// Single-precision complex, interleaved (`real` then `imag`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MKL_Complex8 {
    pub real: f32,
    pub imag: f32,
}

/// Double-precision complex, interleaved (`real` then `imag`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MKL_Complex16 {
    pub real: f64,
    pub imag: f64,
}

// ---------------------------------------------------------------------------
// CBLAS (netlib CBLAS ABI — identical constants and calling convention to MKL)
// ---------------------------------------------------------------------------

pub const CblasRowMajor: u32 = 101;
pub const CblasColMajor: u32 = 102;
pub const CblasNoTrans: u32 = 111;
pub const CblasTrans: u32 = 112;
pub const CblasConjTrans: u32 = 113;

// ---------------------------------------------------------------------------
// LAPACK (Accelerate exposes only the Fortran `_` entry points; no LAPACKE)
// ---------------------------------------------------------------------------
//
// On LP64 targets `__CLPK_integer` is `int` (4 bytes). The `?gesv`/`?getrf`
// Fortran routines are `SUBROUTINE`s, so they return `void`; the status is
// written through the trailing `*info` argument.

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
    // --- CBLAS (level 1 and 3; netlib ABI, symbol-compatible with MKL) ---
    pub fn cblas_sgemm(
        order: u32,
        transa: u32,
        transb: u32,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
    pub fn cblas_dgemm(
        order: u32,
        transa: u32,
        transb: u32,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f64,
        a: *const f64,
        lda: c_int,
        b: *const f64,
        ldb: c_int,
        beta: f64,
        c: *mut f64,
        ldc: c_int,
    );
    pub fn cblas_saxpy(
        n: c_int,
        alpha: f32,
        x: *const f32,
        incx: c_int,
        y: *mut f32,
        incy: c_int,
    );
    pub fn cblas_daxpy(
        n: c_int,
        alpha: f64,
        x: *const f64,
        incx: c_int,
        y: *mut f64,
        incy: c_int,
    );
    pub fn cblas_sdot(n: c_int, x: *const f32, incx: c_int, y: *const f32, incy: c_int) -> f32;
    pub fn cblas_ddot(n: c_int, x: *const f64, incx: c_int, y: *const f64, incy: c_int) -> f64;
    pub fn cblas_sscal(n: c_int, alpha: f32, x: *mut f32, incx: c_int);
    pub fn cblas_dscal(n: c_int, alpha: f64, x: *mut f64, incx: c_int);

    // --- LAPACK Fortran `_` entry points (void `SUBROUTINE`s; `info` out-arg) ---
    pub fn sgesv_(
        n: *const c_int,
        nrhs: *const c_int,
        a: *mut f32,
        lda: *const c_int,
        ipiv: *mut c_int,
        b: *mut f32,
        ldb: *const c_int,
        info: *mut c_int,
    );
    pub fn dgesv_(
        n: *const c_int,
        nrhs: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        ipiv: *mut c_int,
        b: *mut f64,
        ldb: *const c_int,
        info: *mut c_int,
    );
    pub fn sgetrf_(
        m: *const c_int,
        n: *const c_int,
        a: *mut f32,
        lda: *const c_int,
        ipiv: *mut c_int,
        info: *mut c_int,
    );
    pub fn dgetrf_(
        m: *const c_int,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        ipiv: *mut c_int,
        info: *mut c_int,
    );

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
    pub fn _SparseDestroyOpaqueNumeric_Double(
        toFree: *mut SparseOpaqueFactorization_Double,
    );

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
