//! Error type shared across all `nuvai-mkl` modules.

use std::fmt;

/// Result alias for the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// The category of an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// The underlying MKL routine returned a non-zero error code.
    Mkl,
    /// A safe-Rust argument was invalid (bad dimensions, layout, length).
    InvalidArgument,
    /// The requested operation is unavailable on this backend / platform.
    Unsupported,
}

/// Error type for all `nuvai-mkl` operations.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    code: i32,
    message: String,
}

impl Error {
    /// Wrap an error code returned by an MKL routine.
    pub(crate) fn mkl(code: i32, message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Mkl,
            code,
            message: message.into(),
        }
    }

    /// An invalid argument detected on the safe-Rust side.
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidArgument,
            code: 0,
            message: message.into(),
        }
    }

    /// An operation not available on this backend.
    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Unsupported,
            code: 0,
            message: message.into(),
        }
    }

    /// The error category.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The raw MKL error code (zero for non-MKL errors).
    pub fn code(&self) -> i32 {
        self.code
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ErrorKind::Mkl => write!(f, "MKL error (code {}): {}", self.code, self.message),
            ErrorKind::InvalidArgument => write!(f, "invalid argument: {}", self.message),
            ErrorKind::Unsupported => write!(f, "unsupported: {}", self.message),
        }
    }
}

impl std::error::Error for Error {}
