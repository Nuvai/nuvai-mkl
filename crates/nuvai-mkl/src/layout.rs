//! Matrix storage-order and transpose conventions shared across BLAS/LAPACK.

/// Storage order for dense matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Row-major (C-style) storage.
    RowMajor,
    /// Column-major (Fortran-style) storage — MKL's native order.
    ColMajor,
}

/// Matrix transpose flag for BLAS/LAPACK routines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transpose {
    /// No transpose: `op(A) = A`.
    NoTrans,
    /// Transpose: `op(A) = Aᵀ`.
    Trans,
    /// Conjugate transpose: `op(A) = Aᴴ` (only meaningful for complex types).
    ConjTrans,
}
