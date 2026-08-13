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

impl Transpose {
    /// The single-character code used by the C BLAS interface.
    #[allow(dead_code)] // reserved for the Fortran-style `*_` entry points
    pub(crate) const fn as_char(self) -> u8 {
        match self {
            Transpose::NoTrans => b'N',
            Transpose::Trans => b'T',
            Transpose::ConjTrans => b'C',
        }
    }
}
