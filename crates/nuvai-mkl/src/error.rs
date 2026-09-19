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
    /// The operation is supported here, but a backend resource could not be
    /// allocated to carry it out (for example a vDSP DFT setup handle).
    ///
    /// Distinct from [`ErrorKind::Unsupported`] on purpose: `Unsupported` tells
    /// the caller "this backend cannot do this, try another or give up", while
    /// this one means "this backend can do this, but not right now". Retrying,
    /// shrinking the problem, or freeing memory are all sensible responses to
    /// this kind and pointless for `Unsupported`.
    ResourceExhausted,
}

/// Which library's error-code space a raw code came from (#33).
///
/// The wrapped libraries do not share a namespace, and they do not even agree
/// on a sign convention: DFTI counts *up* from zero, while VSL, PARDISO and
/// Accelerate Sparse count *down*. Code `3` therefore means three unrelated
/// things depending on who produced it (`DFTI_INCONSISTENT_CONFIGURATION`,
/// `VSL_ERROR_BADARGS` and "reordering problem" respectively), so a decode
/// table keyed on the number alone would confidently report the wrong failure.
/// Every decode below is keyed on the space, and a code with no entry in its
/// space stays undecoded rather than borrowing another space's meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodeSpace {
    /// DFTI status codes (`mkl_dfti.h`), returned by the `fft` module's
    /// `Dfti*` calls.
    Dfti,
    /// VSL status codes (`mkl_vsl_defines.h`), returned by the `vsl` module.
    Vsl,
    /// The `error` out-parameter of PARDISO's `pardiso` routine.
    Pardiso,
    /// DSS status codes (`mkl_dss.h`), returned by the `dss` module's Intel arm.
    Dss,
    /// Accelerate `SparseStatus_t`, returned by `_Sparse*` on Apple Silicon.
    Sparse,
}

/// Error type for all `nuvai-mkl` operations.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    code: i32,
    /// Which code space [`Error::code`] belongs to, when it is one of the
    /// wrapped libraries' (`None` for non-MKL errors and for codes this crate
    /// has no table for). Kept out of the public API: callers that need the
    /// meaning get it through [`Error::description`].
    space: Option<CodeSpace>,
    message: String,
}

impl Error {
    /// Wrap an error code returned by an MKL routine, with no decode table.
    ///
    /// Used for codes this crate has no table for. That is currently the LAPACK
    /// `info` out-parameter (#33 scoped the tables to the PARDISO/DFTI/DSS/VSL
    /// status codes): `info` is positional rather than tabular — negative is the
    /// negated index of the offending argument, positive means `U(i,i)` was
    /// exactly zero — so decoding it needs the routine's argument list, not a
    /// code-to-text map. Prefer the typed constructors ([`Error::dfti`],
    /// [`Error::vsl`], [`Error::pardiso`], [`Error::dss`], [`Error::sparse`])
    /// wherever the code space is known, so the code is decoded in `Display`.
    pub(crate) fn mkl(code: i32, message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Mkl,
            code,
            space: None,
            message: message.into(),
        }
    }

    /// An MKL error whose code is decodable in `space`.
    fn coded(space: CodeSpace, code: i32, message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Mkl,
            code,
            space: Some(space),
            message: message.into(),
        }
    }

    /// A DFTI status code from the `fft` module.
    // Its callers are all in `fft`'s `not(target_arch = "aarch64")` arm, so it
    // is dead on the aarch64 targets, where `fft` never reaches DFTI. Each of
    // the four tabular constructors below carries the same arch-scoped `allow`
    // for the same reason; `sparse` is the mirror image (Apple Silicon only).
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(crate) fn dfti(code: i32, routine: impl Into<String>) -> Self {
        Self::coded(CodeSpace::Dfti, code, routine)
    }

    /// A VSL status code from the `vsl` module.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(crate) fn vsl(code: i32, routine: impl Into<String>) -> Self {
        Self::coded(CodeSpace::Vsl, code, routine)
    }

    /// A PARDISO `error` code from the `pardiso` module's Intel arm.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(crate) fn pardiso(code: i32, routine: impl Into<String>) -> Self {
        Self::coded(CodeSpace::Pardiso, code, routine)
    }

    /// A DSS status code from the `dss` module's Intel arm.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(crate) fn dss(code: i32, routine: impl Into<String>) -> Self {
        Self::coded(CodeSpace::Dss, code, routine)
    }

    /// An Accelerate `SparseStatus_t` from the sparse backends on Apple
    /// Silicon (`dss`, and `pardiso`'s QR arm).
    // Dead on Intel *and* on aarch64-unknown-linux-gnu: both callers sit behind
    // `all(target_os = "macos", target_arch = "aarch64")`.
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        allow(dead_code)
    )]
    pub(crate) fn sparse(code: i32, routine: impl Into<String>) -> Self {
        Self::coded(CodeSpace::Sparse, code, routine)
    }

    /// An invalid argument detected on the safe-Rust side.
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidArgument,
            code: 0,
            space: None,
            message: message.into(),
        }
    }

    /// An operation not available on this backend.
    // Every caller sits behind `target_arch = "aarch64"`: the Accelerate arms of
    // `fft`/`pardiso` on macOS, and `unsupported_linux_aarch64` on
    // aarch64-unknown-linux-gnu. On Intel every domain has a real MKL backend,
    // so nothing constructs this and it is dead there — not on the aarch64
    // targets, so the `allow` stays scoped to the arch that needs it.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Unsupported,
            code: 0,
            space: None,
            message: message.into(),
        }
    }

    /// A backend resource could not be allocated, though the operation itself
    /// is supported. See [`ErrorKind::ResourceExhausted`].
    // Its only callers are the vDSP setup arms of `fft`, which sit behind
    // `all(target_os = "macos", target_arch = "aarch64")` — so it is dead on
    // Intel *and* on aarch64-unknown-linux-gnu, where no vDSP backend exists.
    // Gating on `target_arch` alone is not enough: that would leave the
    // linux-aarch64 CI job (`cargo check`/`test`/`clippy`) warning that this is
    // never used. Unlike `unsupported` above, which `unsupported_linux_aarch64`
    // keeps reachable there.
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        allow(dead_code)
    )]
    pub(crate) fn resource_exhausted(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::ResourceExhausted,
            code: 0,
            space: None,
            message: message.into(),
        }
    }

    /// An operation unavailable on `aarch64-unknown-linux-gnu`, where OpenBLAS
    /// covers only BLAS/LAPACK (no FFT/VML/VSL/sparse backend exists).
    #[cfg(all(target_os = "linux", target_arch = "aarch64", target_env = "gnu"))]
    pub(crate) fn unsupported_linux_aarch64(domain: &str) -> Self {
        Self::unsupported(format!(
            "{domain} is not supported on aarch64-unknown-linux-gnu \
             (OpenBLAS covers only BLAS/LAPACK)"
        ))
    }

    /// The error category.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The raw MKL error code (zero for non-MKL errors).
    pub fn code(&self) -> i32 {
        self.code
    }

    /// What the raw code means, when this crate knows (#33).
    ///
    /// The same information `Display` appends to an [`ErrorKind::Mkl`] message,
    /// for callers that want to branch on it rather than parse the text. `None`
    /// means the code belongs to no space this crate decodes — a non-MKL error,
    /// a code with no documented meaning here, or one produced by a library
    /// entry point this crate does not wrap.
    pub fn description(&self) -> Option<&'static str> {
        self.space.and_then(|space| describe(space, self.code))
    }
}

/// Decode a raw code from `space` into a short description (#33).
///
/// Tables are transcribed from the oneMKL 2026.1.0 headers `nuvai-mkl-sys`
/// binds — `mkl_dfti.h`, `mkl_vsl_defines.h`, `mkl_dss.h`, `mkl_pardiso.h` —
/// with the C constant name recorded beside each entry, so an entry can be
/// re-checked against the header without a build. PARDISO's `error` table is
/// the exception: the header only declares the values (and only for its
/// extension-entry-point "error classes"), so the meanings come from the
/// oneMKL Developer Reference's `pardiso` page.
///
/// Success (`0`) is absent everywhere on purpose: every call site in this crate
/// tests the returned code *before* constructing an error, so an `Error` can
/// never carry it. An unlisted code returns `None` and renders without a
/// description, exactly as this crate behaved before the table existed — the
/// code is never guessed at.
fn describe(space: CodeSpace, code: i32) -> Option<&'static str> {
    match space {
        // `mkl_dfti.h` > "Error classes". DFTI is the one space that counts
        // *up*: a positive status is a failure class, not a severity. 10 is
        // unassigned, and 1001+ are `CDFT_*` cluster-FFT codes from
        // `mkl_cdft_types.h` — this crate wraps no cluster FFT, so they are
        // absent rather than decoded.
        CodeSpace::Dfti => match code {
            // DFTI_MEMORY_ERROR
            1 => Some("memory allocation failed"),
            // DFTI_INVALID_CONFIGURATION
            2 => Some("invalid descriptor configuration"),
            // DFTI_INCONSISTENT_CONFIGURATION
            3 => Some("inconsistent configuration or transform parameters"),
            // DFTI_MULTITHREADED_ERROR
            4 => Some("OpenMP runtime error"),
            // DFTI_BAD_DESCRIPTOR
            5 => Some("descriptor is unusable (not committed, or already freed)"),
            // DFTI_UNIMPLEMENTED
            6 => Some("the requested configuration is not implemented"),
            // DFTI_MKL_INTERNAL_ERROR
            7 => Some("internal library error"),
            // DFTI_NUMBER_OF_THREADS_ERROR
            8 => Some("thread count at compute differs from the count committed"),
            // DFTI_1D_LENGTH_EXCEEDS_INT32 and DFTI_1D_MEMORY_EXCEEDS_INT32 are
            // two names for this one value in the header, so they cannot be
            // told apart here and the description names both.
            9 => Some("1D transform length or data size exceeds INT32_MAX"),
            // DFTI_NO_WORKSPACE
            11 => Some("no workspace is available for the transform"),
            _ => None,
        },

        // `mkl_vsl_defines.h`. VSL numbers failures *down*, in two families
        // that never collide: the generic `VSL_ERROR_*` codes in `[-6, -1]` and
        // the generator-specific `VSL_RNG_ERROR_*` codes at `-1000` and below.
        //
        // The generator codes listed are those the five wrapped routines
        // (`vslNewStream`, `vsRngUniform`, `vdRngUniform`, `vsRngGaussian`,
        // `vdRngGaussian`) document as returnable. The rest of the header's
        // `VSL_RNG_ERROR_*` set belongs to entry points this crate does not
        // wrap (stream file I/O, leapfrog/skip-ahead, the extended stream
        // constructors), so decoding them would describe failures that cannot
        // reach a caller here.
        CodeSpace::Vsl => match code {
            // VSL_ERROR_FEATURE_NOT_IMPLEMENTED
            -1 => Some("the requested feature is not implemented"),
            // VSL_ERROR_UNKNOWN
            -2 => Some("unknown error"),
            // VSL_ERROR_BADARGS
            -3 => Some("bad arguments"),
            // VSL_ERROR_MEM_FAILURE
            -4 => Some("memory allocation failed"),
            // VSL_ERROR_NULL_PTR
            -5 => Some("a NULL pointer was passed"),
            // VSL_ERROR_CPU_NOT_SUPPORTED
            -6 => Some("the CPU does not support the requested feature"),
            // VSL_RNG_ERROR_INVALID_BRNG_INDEX
            -1000 => Some("invalid basic generator index"),
            // VSL_RNG_ERROR_BAD_STREAM
            -1006 => Some("invalid or corrupted stream"),
            // VSL_RNG_ERROR_QRNG_PERIOD_ELAPSED
            -1012 => Some("the quasi-random generator's period has elapsed"),
            // VSL_RNG_ERROR_BAD_UPDATE
            -1120 => Some("the stream cannot be updated in its current state"),
            // VSL_RNG_ERROR_NO_NUMBERS
            -1121 => Some("the generator cannot produce the requested numbers"),
            // VSL_RNG_ERROR_NONDETERM_NOT_SUPPORTED
            -1130 => Some("the non-deterministic generator is not supported"),
            // VSL_RNG_ERROR_NONDETERM_NRETRIES_EXCEEDED
            -1131 => Some("the non-deterministic generator's retry limit was exceeded"),
            // VSL_RNG_ERROR_ARS5_NOT_SUPPORTED
            -1140 => Some("the ARS5 generator is not supported"),
            _ => None,
        },

        // The `error` out-parameter of `pardiso`. `mkl_pardiso.h` defines an
        // "Error classes" block of its own — PARDISO_UNIMPLEMENTED (-101),
        // PARDISO_NULL_HANDLE (-102), PARDISO_MEMORY_ERROR (-103) — but those
        // belong to the oneMKL *extension* entry points (`pardiso_getdiag`,
        // `pardiso_export`, …), which this crate does not call, so they are
        // deliberately absent from this space. Positive values are not
        // documented as errors by Intel; the perturbed-pivot count a caller
        // might mistake for one is `iparm[13]`, a different argument.
        CodeSpace::Pardiso => match code {
            -1 => Some("input inconsistent"),
            -2 => Some("not enough memory"),
            -3 => Some("reordering problem"),
            -4 => Some("zero pivot; numerical factorization or iterative refinement failed"),
            -5 => Some("unclassified internal error"),
            -6 => Some("reordering failed (matrix types 11 and 13 only)"),
            -7 => Some("the diagonal matrix is singular"),
            -8 => Some("32-bit integer overflow"),
            -9 => Some("not enough memory for out-of-core factorization"),
            -10 => Some("could not open an out-of-core temporary file"),
            -11 => Some("read/write error on the out-of-core data file"),
            // Also stated in `mkl_pardiso.h` above `pardiso_64`'s declaration
            // ("If called on IA-32, error = -12 is returned").
            -12 => Some("the 64-bit interface was called from a 32-bit library"),
            _ => None,
        },

        // `mkl_dss.h` > "Return status values" — one contiguous block, mirrored
        // whole. (`MKL_DSS_DEFAULTS`, `MKL_DSS_SYMMETRIC`, `MKL_DSS_MSG_LVL_*`
        // and the other `MKL_DSS_*` macros in that header are *option* values,
        // not statuses, and the statistics codes belong to `dss_statistics_`,
        // which this crate does not wrap.)
        CodeSpace::Dss => match code {
            // MKL_DSS_ZERO_PIVOT
            -1 => Some("zero pivot encountered"),
            // MKL_DSS_OUT_OF_MEMORY
            -2 => Some("out of memory"),
            // MKL_DSS_FAILURE
            -3 => Some("general failure"),
            // MKL_DSS_ROW_ERR
            -4 => Some("row index error"),
            // MKL_DSS_COL_ERR
            -5 => Some("column index error"),
            // MKL_DSS_TOO_FEW_VALUES
            -6 => Some("too few values supplied"),
            // MKL_DSS_TOO_MANY_VALUES
            -7 => Some("too many values supplied"),
            // MKL_DSS_NOT_SQUARE
            -8 => Some("matrix is not square"),
            // MKL_DSS_STATE_ERR
            -9 => Some("routine called out of sequence"),
            // MKL_DSS_INVALID_OPTION
            -10 => Some("invalid option"),
            // MKL_DSS_OPTION_CONFLICT
            -11 => Some("conflicting options"),
            // MKL_DSS_MSG_LVL_ERR
            -12 => Some("invalid message level"),
            // MKL_DSS_TERM_LVL_ERR
            -13 => Some("invalid termination level"),
            // MKL_DSS_STRUCTURE_ERR
            -14 => Some("structure definition error"),
            // MKL_DSS_REORDER_ERR
            -15 => Some("reordering error"),
            // MKL_DSS_VALUES_ERR
            -16 => Some("values error"),
            // MKL_DSS_STATISTICS_INVALID_MATRIX
            -17 => Some("statistics: invalid matrix"),
            // MKL_DSS_STATISTICS_INVALID_STATE
            -18 => Some("statistics: invalid state"),
            // MKL_DSS_STATISTICS_INVALID_STRING
            -19 => Some("statistics: invalid string"),
            // MKL_DSS_REORDER1_ERR
            -20 => Some("reordering (stage 1) error"),
            // MKL_DSS_PREORDER_ERR
            -21 => Some("preordering error"),
            // MKL_DSS_DIAG_ERR
            -22 => Some("diagonal error"),
            // MKL_DSS_I32BIT_ERR
            -23 => Some("32-bit integer overflow"),
            // MKL_DSS_OOC_MEM_ERR
            -24 => Some("out-of-core memory error"),
            // MKL_DSS_OOC_OC_ERR
            -25 => Some("out-of-core open/close error"),
            // MKL_DSS_OOC_RW_ERR
            -26 => Some("out-of-core read/write error"),
            _ => None,
        },

        // Accelerate's `SparseStatus_t`, as declared by
        // `nuvai-mkl-sys`'s hand-written Apple Silicon surface (`SparseStatusOK`
        // and friends). Also counts *down*.
        CodeSpace::Sparse => match code {
            // SparseFactorizationFailed
            -1 => Some("factorization failed"),
            // SparseMatrixIsSingular
            -2 => Some("matrix is singular"),
            // SparseInternalError
            -3 => Some("internal Accelerate sparse error"),
            // SparseParameterError
            -4 => Some("invalid sparse parameter"),
            _ => None,
        },
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ErrorKind::Mkl => {
                write!(f, "MKL error (code {}): {}", self.code, self.message)?;
                // Append the decoded meaning when there is one (#33). The raw
                // code stays in the message either way: it is what a bug report
                // or a vendor's own documentation is keyed on.
                match self.description() {
                    Some(description) => write!(f, " — {description}"),
                    None => Ok(()),
                }
            }
            ErrorKind::InvalidArgument => write!(f, "invalid argument: {}", self.message),
            ErrorKind::Unsupported => write!(f, "unsupported: {}", self.message),
            ErrorKind::ResourceExhausted => {
                write!(f, "resource exhausted: {}", self.message)
            }
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each space decodes its own codes, and the same number means different
    /// things in different spaces — the reason the table is keyed by space at
    /// all. `-3` is the sharpest case: VSL's BADARGS, PARDISO's reordering
    /// problem, DSS's general failure.
    #[test]
    fn decodes_each_space_independently() {
        let vsl = Error::vsl(-3, "vsRngUniform");
        assert_eq!(vsl.description(), Some("bad arguments"));
        assert_eq!(
            vsl.to_string(),
            "MKL error (code -3): vsRngUniform — bad arguments"
        );

        let pardiso = Error::pardiso(-3, "pardiso phase 11");
        assert_eq!(pardiso.description(), Some("reordering problem"));

        let dss = Error::dss(-3, "dss_create");
        assert_eq!(dss.description(), Some("general failure"));

        // DFTI is the one space that counts up, so `3` — not `-3` — is its
        // third failure class.
        let dfti = Error::dfti(3, "DftiCommitDescriptor");
        assert_eq!(
            dfti.description(),
            Some("inconsistent configuration or transform parameters")
        );
        assert_eq!(Error::dfti(-3, "DftiCommitDescriptor").description(), None);
    }

    /// The code space is stored per error, so a space's table cannot be
    /// consulted for another's number: `-1` decodes in every space here, and
    /// each answer must match its own space (`Error::mkl` has no space at all).
    #[test]
    fn does_not_borrow_another_spaces_table() {
        assert_eq!(
            Error::dfti(1, "x").description(),
            Some("memory allocation failed")
        );
        assert_eq!(
            Error::sparse(-1, "x").description(),
            Some("factorization failed")
        );
        assert_eq!(
            Error::pardiso(-1, "x").description(),
            Some("input inconsistent")
        );
        assert_eq!(Error::mkl(1, "x").description(), None);
        assert_eq!(Error::mkl(-1, "x").description(), None);
    }

    /// A code with no entry anywhere renders exactly as it did before the table
    /// existed, and the raw code is never dropped from the message.
    #[test]
    fn unknown_codes_render_as_before() {
        let err = Error::dss(-99, "dss_solve_real");
        assert_eq!(err.description(), None);
        assert_eq!(err.to_string(), "MKL error (code -99): dss_solve_real");

        // A VSL code that exists in the header but belongs to an entry point
        // this crate does not wrap (`VSL_RNG_ERROR_FILE_OPEN`, -1101).
        let err = Error::vsl(-1101, "vslNewStream");
        assert_eq!(err.description(), None);
    }

    /// DFTI's code 9 carries two header names for one value
    /// (`DFTI_1D_LENGTH_EXCEEDS_INT32`, `DFTI_1D_MEMORY_EXCEEDS_INT32`), so the
    /// description has to name both rather than pick one.
    #[test]
    fn dfti_9_describes_both_meanings() {
        let desc = Error::dfti(9, "DftiCreateDescriptor")
            .description()
            .unwrap();
        assert!(desc.contains("length"));
        assert!(desc.contains("data size"));
    }

    /// DFTI statuses are positive and VSL/PARDISO/DSS are negative, so a sign
    /// mix-up in a table would be caught here rather than shipped.
    #[test]
    fn sign_conventions_are_per_space() {
        // Positive codes decode only in DFTI.
        for code in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11] {
            assert!(
                Error::dfti(code, "x").description().is_some(),
                "DFTI code {code}"
            );
            assert!(Error::vsl(code, "x").description().is_none(), "VSL {code}");
            assert!(Error::dss(code, "x").description().is_none(), "DSS {code}");
            assert!(
                Error::pardiso(code, "x").description().is_none(),
                "PARDISO {code}"
            );
        }
        // PARDISO's error table is dense from -1 to -12.
        for code in -12..=-1 {
            assert!(
                Error::pardiso(code, "x").description().is_some(),
                "PARDISO {code}"
            );
        }
        assert!(Error::pardiso(0, "x").description().is_none());
        assert!(Error::pardiso(-13, "x").description().is_none());
        // DSS's is dense from -1 to -26.
        for code in -26..=-1 {
            assert!(Error::dss(code, "x").description().is_some(), "DSS {code}");
        }
        assert!(Error::dss(-27, "x").description().is_none());
    }

    /// The decoded Accelerate `SparseStatus_t` values must equal the constants
    /// `nuvai-mkl-sys` declares for them on Apple Silicon, so the table cannot
    /// drift from the FFI surface it describes.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn sparse_table_matches_the_sys_constants() {
        for (status, expected) in [
            (
                nuvai_mkl_sys::SparseFactorizationFailed,
                "factorization failed",
            ),
            (nuvai_mkl_sys::SparseMatrixIsSingular, "matrix is singular"),
            (
                nuvai_mkl_sys::SparseInternalError,
                "internal Accelerate sparse error",
            ),
            (
                nuvai_mkl_sys::SparseParameterError,
                "invalid sparse parameter",
            ),
        ] {
            assert_eq!(Error::sparse(status, "x").description(), Some(expected));
        }
        assert_eq!(
            Error::sparse(nuvai_mkl_sys::SparseStatusOK, "x").description(),
            None
        );
    }

    /// Non-MKL kinds keep their exact previous formatting, and their `code` is
    /// zero, so nothing can decode through them.
    #[test]
    fn non_mkl_kinds_are_unchanged() {
        for err in [
            Error::invalid("blas: m, n and k must be non-negative"),
            Error::unsupported("PARDISO mtype 2 is not supported"),
            Error::resource_exhausted("vDSP setup could not be created"),
        ] {
            assert_eq!(err.description(), None);
            assert_eq!(err.code(), 0);
        }
        assert_eq!(Error::invalid("x").to_string(), "invalid argument: x");
    }
}
