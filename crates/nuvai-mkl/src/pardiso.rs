//! PARDISO — parallel sparse direct solver (double precision).
//!
//! On Intel targets matrices are supplied in CSR (3-array) form with 1-based
//! indexing (PARDISO's default) and [`Pardiso::solve`] runs the analysis →
//! factorization → solve phases. On Apple Silicon (`aarch64-apple-darwin`) the
//! same CSR input is transposed to CSC and solved with the Accelerate
//! Sparse/SparseSolve backend (`_SparseFactorQR_Double` +
//! `_SparseSolveOpaque_Double`, ADR-0003 decision 7). On
//! `aarch64-unknown-linux-gnu` there is no sparse backend (OpenBLAS covers only
//! BLAS/LAPACK), so every [`Pardiso::solve`] returns
//! [`ErrorKind::Unsupported`].

// `c_void`/`ptr` are used by the Intel path and the Accelerate (macOS-aarch64)
// helpers but not by the linux-aarch64 Unsupported path, so gate them to every
// target except the inert linux-aarch64 arm.
use std::marker::PhantomData;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::os::raw::c_long;
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64"),
    not(target_arch = "aarch64")
))]
use std::os::raw::c_void;
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64"),
    not(target_arch = "aarch64")
))]
use std::ptr;

// Used by the Intel arm and the Accelerate arm, but not by the inert
// linux-aarch64 one, so it carries the same gate as `c_void`/`ptr` above.
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64"),
    not(target_arch = "aarch64")
))]
use crate::conv::len_to_c_int;
use crate::error::{Error, Result};

/// PARDISO matrix types (`mtype`).
#[allow(non_upper_case_globals)]
pub mod mtype {
    /// Real, symmetric positive definite.
    pub const SPD: i32 = 2;
    /// Real, symmetric indefinite.
    pub const SYMMETRIC_INDEFINITE: i32 = -2;
    /// Real, structurally nonsymmetric.
    pub const NONSYMMETRIC: i32 = 11;
}

/// A PARDISO solver handle (double precision).
///
/// On Intel targets this holds the PARDISO `pt`/`iparm` state. On Apple
/// Silicon the Accelerate backend caches the QR factorization of the most
/// recently solved matrix and reuses it when the matrix is unchanged, so only
/// `mtype` and that cache are kept — the 768-byte PARDISO state is cfg'd out.
/// The handle is `!Send + !Sync` on every backend (Intel's raw `pt` pointers,
/// the Apple Silicon factor's raw pointers, and an explicit raw-pointer
/// `PhantomData` on the inert linux-aarch64 handle), so share it behind a lock
/// for cross-thread use. On `aarch64-unknown-linux-gnu` there is no PARDISO
/// backend and the handle is inert (every `solve` returns
/// [`ErrorKind::Unsupported`]).
pub struct Pardiso {
    /// PARDISO internal state (Intel targets only).
    #[cfg(not(target_arch = "aarch64"))]
    pt: [*mut c_void; 64],
    /// The matrix type — read by every backend that can solve; stored but
    /// never read on linux-aarch64 (where no `solve` ever runs).
    #[cfg_attr(all(target_os = "linux", target_arch = "aarch64"), allow(dead_code))]
    mtype: i32,
    /// PARDISO control array (Intel targets only).
    #[cfg(not(target_arch = "aarch64"))]
    iparm: [i32; 64],
    /// System size from the most recent `solve` (Intel targets only).
    #[cfg(not(target_arch = "aarch64"))]
    n: i32,
    /// True once `pardisoinit` has run, i.e. once a phase-`-1` release is both
    /// legal and necessary (Intel targets only).
    ///
    /// Set in [`Pardiso::new`], immediately after `pardisoinit` — *not* once the
    /// analysis phase has succeeded, which is what this flag used to mean
    /// (#23). Phase 11 can fail (error `-2`, out of memory, among others) after
    /// partially allocating, and a handle whose flag was still false skipped
    /// the release in [`Drop`] and leaked that memory. `pardisoinit` is what
    /// makes a later phase `-1` legal, so it is the condition that gates the
    /// release; being set at construction it cannot be missed by a failing
    /// phase. It is true for every Intel handle today, and stays a field rather
    /// than dropping the gate because the invariant it records — "PARDISO state
    /// that only phase `-1` frees may exist" — is what `Drop` is keyed on.
    #[cfg(not(target_arch = "aarch64"))]
    release_pending: bool,
    /// Cached QR factorization plus the CSR matrix that produced it (Apple
    /// Silicon only). Reused across `solve` calls when the matrix is unchanged.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    cached: Option<CachedFactor>,
    /// Pin auto-trait parity across backends (see `vsl::Stream`'s identical
    /// field). Intel's raw `pt` pointers and the Apple Silicon cached
    /// factorization are `!Send + !Sync`; a raw-pointer `PhantomData` is also
    /// `!Send + !Sync`, so the inert linux-aarch64 handle matches instead of
    /// leaking `Send + Sync` on only one platform.
    _not_send_sync: PhantomData<*const ()>,
}

/// A cached Accelerate QR factorization and the CSR matrix that produced it
/// (Apple Silicon only).
///
/// [`SparseOpaqueFactorization_Double`] owns heap memory through raw pointers
/// and is deliberately **not `Copy`** (returning it by value needs only move
/// semantics), so it is stored here by value and moved into `Drop` — the
/// compiler now rejects any accidental duplication that would alias the
/// allocation and double-free it.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct CachedFactor {
    /// The owned QR factorization (output of `_SparseFactorQR_Double`).
    factor: nuvai_mkl_sys::SparseOpaqueFactorization_Double,
    /// Copies of the caller's CSR input, compared with `==` to detect an
    /// unchanged matrix and skip re-factoring.
    ia: Vec<i32>,
    ja: Vec<i32>,
    a: Vec<f64>,
}

impl Pardiso {
    /// Create a handle for the given matrix type.
    pub fn new(mtype: i32) -> Self {
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut iparm = [0i32; 64];
            iparm[0] = 1; // use default `iparm` values
            let mut pt = [ptr::null_mut::<c_void>(); 64];
            // SAFETY: `pt` and `iparm` are valid, zeroed arrays of the exact
            // size `pardisoinit` expects to initialize; `mtype` is passed by
            // const reference and only read.
            unsafe {
                nuvai_mkl_sys::pardisoinit(
                    pt.as_mut_ptr() as *mut c_void,
                    &mtype,
                    iparm.as_mut_ptr(),
                );
            }
            Self {
                pt,
                mtype,
                iparm,
                n: 0,
                // `pardisoinit` has run, so a phase-`-1` release is legal from
                // here on — see the field's docs for why the flag is set here
                // rather than after a successful analysis (#23).
                release_pending: true,
                _not_send_sync: PhantomData,
            }
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            // The Accelerate backend starts with no cached factorization; it is
            // populated on the first `solve` and reused for unchanged matrices.
            Self {
                mtype,
                cached: None,
                _not_send_sync: PhantomData,
            }
        }
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            // linux-aarch64 has no PARDISO backend (every `solve` returns
            // Unsupported), so the handle carries only `mtype`.
            Self {
                mtype,
                _not_send_sync: PhantomData,
            }
        }
    }

    /// Factor and solve `A x = b` for the matrix in CSR form: `ia` has length
    /// `n + 1`, `ja` (column indices) and `a` (values) have length `nnz`.
    /// Returns the solution `x` (length `n`).
    pub fn solve(&mut self, ia: &[i32], ja: &[i32], a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            // No PARDISO backend on aarch64-unknown-linux-gnu (OpenBLAS covers
            // only BLAS/LAPACK).
            let _ = (ia, ja, a, b);
            Err(Error::unsupported_linux_aarch64("PARDISO"))
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.solve_accelerate(ia, ja, a, b)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            if ja.len() != a.len() {
                return Err(Error::invalid("PARDISO: ja/a length mismatch"));
            }
            // `ia` has length n+1, so an empty `ia` would underflow this
            // subtraction: a panic under `overflow-checks` (every debug build)
            // and a wrap to `usize::MAX` in release. `checked_sub` makes the
            // expression total, and `len_to_c_int` rejects any length the
            // 32-bit `MKL_INT` cannot hold — so `n` no longer relies on the
            // `n <= 0` guard below catching a truncated `usize::MAX as i32 == -1`
            // by coincidence.
            let n = match ia.len().checked_sub(1) {
                Some(n) => len_to_c_int(n, "PARDISO")?,
                None => return Err(Error::invalid("PARDISO: bad ia/b lengths")),
            };
            if n <= 0 || b.len() != n as usize {
                return Err(Error::invalid("PARDISO: bad ia/b lengths"));
            }
            self.n = n;

            let maxfct = 1i32;
            let mnum = 1i32;
            let nrhs = 1i32;
            let msglvl = 0i32;
            let mut x = vec![0.0f64; n as usize];
            let mut error = 0i32;

            // SAFETY: `self.pt`/`self.iparm` are initialized by `pardisoinit`
            // in `new`; `a`/`ia`/`ja` are valid slices describing the CSR
            // matrix (lengths checked above); `b`/`x` are valid `n`-element
            // buffers (`b` read, `x` written) and `error` is a valid out-arg.
            // `pt`/`iparm`/`error` are passed mutably and the scalars by
            // reference, as the Fortran `pardiso` ABI expects.
            //
            // Phase 33's `b` argument is the one cast that needs an invariant
            // written down (#29). `b` is the caller's `&[f64]` and is cast
            // `*const` → `*mut` purely to satisfy the ABI: oneMKL's prototype
            // declares the right-hand side as plain `void *b` (mkl_pardiso.h),
            // with no `const` variant, even though phase 33 only reads it —
            // `a` is declared `const void *a` and is passed as `*const c_void`
            // for exactly that read-only reason. So the cast is sound *only*
            // while "phase 33 reads `b`" holds: the solution is written to `x`,
            // never to `b`.
            //
            // What would break it: any `iparm` setting that makes PARDISO write
            // back into `b` (an overwrite/iterative-refinement output option),
            // or reusing this call shape for a phase with different aliasing. If
            // that ever becomes necessary, `b` must become `&mut [f64]` or be
            // replaced by a scratch copy first — writing through this pointer
            // while the caller still holds the `&[f64]` it came from is UB that
            // no later check can detect.
            unsafe {
                // Phase 11: analysis + reordering.
                let phase = 11i32;
                nuvai_mkl_sys::pardiso(
                    self.pt.as_mut_ptr() as *mut c_void,
                    &maxfct,
                    &mnum,
                    &self.mtype,
                    &phase,
                    &n,
                    a.as_ptr() as *const c_void,
                    ia.as_ptr(),
                    ja.as_ptr(),
                    ptr::null_mut(),
                    &nrhs,
                    self.iparm.as_mut_ptr(),
                    &msglvl,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    &mut error,
                );
                if error != 0 {
                    // No flag to clear on the way out: `release_pending` was
                    // already set in `new`, so `Drop` still runs phase `-1` for
                    // the memory this failed phase may have allocated (#23).
                    return Err(Error::pardiso(error, "pardiso phase 11 (analysis)"));
                }

                // Phase 22: numerical factorization.
                let phase = 22i32;
                nuvai_mkl_sys::pardiso(
                    self.pt.as_mut_ptr() as *mut c_void,
                    &maxfct,
                    &mnum,
                    &self.mtype,
                    &phase,
                    &n,
                    a.as_ptr() as *const c_void,
                    ia.as_ptr(),
                    ja.as_ptr(),
                    ptr::null_mut(),
                    &nrhs,
                    self.iparm.as_mut_ptr(),
                    &msglvl,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    &mut error,
                );
                if error != 0 {
                    return Err(Error::pardiso(error, "pardiso phase 22 (factorization)"));
                }

                // Phase 33: solve (forward/back substitution + refinement).
                let phase = 33i32;
                nuvai_mkl_sys::pardiso(
                    self.pt.as_mut_ptr() as *mut c_void,
                    &maxfct,
                    &mnum,
                    &self.mtype,
                    &phase,
                    &n,
                    a.as_ptr() as *const c_void,
                    ia.as_ptr(),
                    ja.as_ptr(),
                    ptr::null_mut(),
                    &nrhs,
                    self.iparm.as_mut_ptr(),
                    &msglvl,
                    b.as_ptr() as *mut c_void,
                    x.as_mut_ptr() as *mut c_void,
                    &mut error,
                );
                if error != 0 {
                    return Err(Error::pardiso(error, "pardiso phase 33 (solve)"));
                }
            }

            Ok(x)
        }
    }
}

/// Accelerate (`aarch64-apple-darwin`) sparse backend.
///
/// Converts the caller's 1-based CSR input to a 0-based CSC
/// `SparseMatrix_Double`, factors it with QR (`_SparseFactorQR_Double`), and
/// solves with `_SparseSolveOpaque_Double`. The factorization is cached in
/// `self.cached` and reused across `solve` calls when the CSR triple is
/// unchanged; `b` is deliberately absent from the cache key, so a new
/// right-hand side re-solves against the retained factor. Apple's factor
/// routines copy the matrix into the factorization's own storage, so the local
/// CSC buffers can be dropped after factoring.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Pardiso {
    fn solve_accelerate(
        &mut self,
        ia: &[i32],
        ja: &[i32],
        a: &[f64],
        b: &[f64],
    ) -> Result<Vec<f64>> {
        // The Accelerate QR backend factors a *full* (ordinary) matrix. PARDISO's
        // symmetric `mtype`s (SPD / symmetric indefinite) store only one triangle,
        // which QR would read as a full matrix and silently mis-solve. Reject them
        // rather than return a wrong answer (ADR-0003: never degrade silently).
        if self.mtype != mtype::NONSYMMETRIC {
            return Err(Error::unsupported(format!(
                "PARDISO mtype {} is not supported on Apple Silicon; only mtype::NONSYMMETRIC ({}) is available",
                self.mtype,
                mtype::NONSYMMETRIC
            )));
        }
        if ja.len() != a.len() {
            return Err(Error::invalid("PARDISO: ja/a length mismatch"));
        }
        // `ia` has length n+1, so an empty `ia` would underflow this
        // subtraction: a panic under `overflow-checks` (every debug build) and a
        // wrap to `usize::MAX` in release. `checked_sub` makes the expression
        // total, and `len_to_c_int` rejects any length the `i32` row count of
        // `SparseMatrixStructure` cannot hold — so `n` no longer relies on the
        // `n <= 0` guard below catching a truncated `usize::MAX as i32 == -1` by
        // coincidence.
        let n = match ia.len().checked_sub(1) {
            Some(n) => len_to_c_int(n, "PARDISO")?,
            None => return Err(Error::invalid("PARDISO: bad ia/b lengths")),
        };
        if n <= 0 || b.len() != n as usize {
            return Err(Error::invalid("PARDISO: bad ia/b lengths"));
        }

        // Reuse the cached factorization when the matrix is unchanged (exact
        // `==` on the CSR triple). `b` is deliberately absent from the key, so
        // a new right-hand side re-solves against the retained factor. This key
        // comparison is O(nnz) per solve — inherent to safe reuse, since the
        // factor can only be trusted after the matrix is verified — and it
        // short-circuits on `ia` (length n+1) before touching `ja`/`a`.
        let cache_hit = self
            .cached
            .as_ref()
            .is_some_and(|c| c.ia.as_slice() == ia && c.ja.as_slice() == ja && c.a.as_slice() == a);

        if !cache_hit {
            let (col_starts, row_indices, values) = csr_to_csc(n as usize, ia, ja, a, 1, false)?;

            let matrix = nuvai_mkl_sys::SparseMatrix_Double {
                structure: nuvai_mkl_sys::SparseMatrixStructure {
                    rowCount: n,
                    columnCount: n,
                    columnStarts: col_starts.as_ptr() as *mut c_long,
                    rowIndices: row_indices.as_ptr() as *mut i32,
                    attributes: nuvai_mkl_sys::SparseAttributes_t::ordinary(),
                    blockSize: 1,
                },
                data: values.as_ptr() as *mut f64,
            };

            let sfoptions = default_symbolic_options();
            let nfoptions = default_numeric_options();

            // SAFETY: `matrix` borrows the CSC arrays for the duration of the
            // call; `SparseFactorizationQR` and the option structs are valid.
            // The returned `SparseOpaqueFactorization_Double` is owned by value
            // and stored in `self.cached` (released exactly once by `Drop`, or
            // when a later, different matrix replaces it).
            let mut factor = unsafe {
                nuvai_mkl_sys::_SparseFactorQR_Double(
                    nuvai_mkl_sys::SparseFactorizationQR,
                    &matrix,
                    &sfoptions,
                    &nfoptions,
                )
            };
            if factor.status != nuvai_mkl_sys::SparseStatusOK {
                let status = factor.status;
                // A *failed* factorization is still destroyed here — do not
                // "fix" this by skipping the release when `status != OK` (#30).
                // Apple's docs for `SparseOpaqueFactorization_Double`
                // (Sparse/Solve.h) list four states, three of which have
                // `status < 0`:
                //   state 1: `.symbolicFactorization.status < 0` — nothing valid,
                //            so nothing is owned;
                //   state 2: `symbolic >= 0 && status < 0 && numeric == NULL` —
                //            symbolic is valid and "may be used for future calls";
                //   state 3: `symbolic >= 0 && status < 0 && numeric != NULL` —
                //            "factor allocated/initialized correctly, but numeric
                //            factorization failed" (e.g. Cholesky of an
                //            indefinite matrix).
                // The type docs say to free "these objects" with `SparseCleanup`,
                // and `SparseCleanup` (Sparse/SolveImplementationTyped.h) is a
                // static inline that calls `_SparseDestroyOpaqueNumeric`
                // unconditionally — with no status check. States 2 and 3 own
                // memory, so skipping the release would leak it; releasing state
                // 1 frees nothing, but it is what `SparseCleanup` does anyway.
                //
                // State 3 is the one a test pins: the all-zero 2x2 in
                // `pardiso_releases_failed_factorization_zero_matrix_on_aarch64`
                // (`status == -2`, `symbolicFactorization.status == 0`, non-NULL
                // numeric). A structurally empty row reaches state 1 instead
                // (`status == -2`, `symbolicFactorization.status == -3`, NULL
                // numeric), and is deliberately left untested: on the macOS 14 CI
                // runner `_SparseFactorQR_Double` aborts the process on that
                // input rather than returning, so no assertion on it can pass in
                // CI — see the note in `tests/smoke.rs`.
                //
                // SAFETY: `factor` is an owned, initialized factorization in one
                // of those states (the struct is returned by value and fully
                // populated); released exactly once on this error path, since it
                // was never cached.
                unsafe { nuvai_mkl_sys::_SparseDestroyOpaqueNumeric_Double(&mut factor) };
                return Err(Error::sparse(status, "_SparseFactorQR_Double"));
            }

            // Replace any previous cache entry: destroy the old factor exactly
            // once, then store the fresh one together with copies of the CSR
            // input it was factored from.
            if let Some(old) = self.cached.take() {
                let mut old_factor = old.factor;
                // SAFETY: `old_factor` is an owned, initialized factorization
                // that is no longer needed; moved out by value (the handle is
                // not `Copy`) and destroyed exactly once here.
                unsafe { nuvai_mkl_sys::_SparseDestroyOpaqueNumeric_Double(&mut old_factor) };
            }
            self.cached = Some(CachedFactor {
                factor,
                ia: ia.to_vec(),
                ja: ja.to_vec(),
                a: a.to_vec(),
            });
        }

        // `self.cached` is `Some` here: either this was a cache hit, or the
        // branch above just populated it. The `ok_or_else` fallback is
        // unreachable (both paths populate the cache), but returning an error
        // keeps the no-panic guarantee rather than asserting an invariant the
        // compiler cannot prove.
        let cached = self.cached.as_ref().ok_or_else(|| {
            Error::invalid("PARDISO: factorization cache missing after factor/populate")
        })?;
        let x = solve_with_factor(&cached.factor, n, b)?;

        // QR factorization of a singular matrix still succeeds (R is simply
        // rank-deficient), so Accelerate reports no error and the solve returns a
        // meaningless least-squares solution. Detect that with a residual check so
        // singular systems error out like Intel PARDISO's zero-pivot failure.
        check_residual(ia, ja, a, b, &x)?;
        Ok(x)
    }

    /// Release the cached QR factorization and its matrix copies without
    /// dropping the handle.
    ///
    /// The cache retains the factor plus full copies of the caller's CSR input
    /// (`ia`/`ja`/`a`) for the handle's lifetime so repeated solves with an
    /// unchanged matrix skip re-factoring — roughly 2–3× the matrix's `nnz`
    /// storage on top of Accelerate's own factor copy. Call this to release
    /// that memory (e.g. after a one-off solve, or before a long idle period);
    /// the next [`Pardiso::solve`] re-factors from scratch.
    pub fn reset(&mut self) {
        if let Some(cached) = self.cached.take() {
            let mut factor = cached.factor;
            // SAFETY: `factor` is an owned, initialized factorization created by
            // `_SparseFactorQR_Double`; destroyed exactly once here (the same
            // release `Drop` performs).
            unsafe { nuvai_mkl_sys::_SparseDestroyOpaqueNumeric_Double(&mut factor) };
        }
    }
}

/// Build a 0-based CSC (compressed sparse column) representation of the square
/// `n × n` matrix given in CSR form. `base` is the CSR index base (0 for DSS,
/// 1 for PARDISO); `upper_only` requires the stored entries to be the upper
/// triangle of a symmetric matrix (rejects lower-triangle storage). Returns
/// `(column_starts, row_indices, values)` with 0-based indices.
///
/// This is the single validated CSR→CSC transposition shared by the PARDISO
/// (nonsymmetric, 1-based) and DSS (symmetric, 0-based) aarch64 backends, so
/// bounds/monotonicity checks apply to both — as does the rejection of a
/// structurally empty row (a zero row, hence a singular matrix) that would
/// otherwise reach Accelerate's factorization and abort the process on
/// macOS 14 (#58).
///
/// Callers supply the CSR contract rather than this function re-deriving it:
/// `row_index` has `n + 1` entries and `values` is parallel to `columns`. Both
/// are asserted in debug builds, since the alternative is an index panic from
/// inside the transposition loop.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn csr_to_csc(
    n: usize,
    row_index: &[i32],
    columns: &[i32],
    values: &[f64],
    base: i32,
    upper_only: bool,
) -> Result<(Vec<i64>, Vec<i32>, Vec<f64>)> {
    let nnz = columns.len();
    debug_assert_eq!(
        row_index.len(),
        n + 1,
        "CSR row_index must have n + 1 entries"
    );
    debug_assert_eq!(values.len(), nnz, "CSR values must be parallel to columns");

    // A valid `base`-based CSR has row_index[0] == base, row_index[n] == nnz +
    // base, and a non-decreasing row_index. Any other shape would silently drop
    // or mis-index entries.
    if row_index[0] != base {
        return Err(Error::invalid(
            "CSR row_index[0] does not match the index base",
        ));
    }
    // Compared in `i64` instead of narrowing to `row_index[n]`'s `i32`. The
    // narrowing was #25's class: `nnz as i32` truncates above `i32::MAX`, which
    // would let an invalid CSR pass this check and then index `row_indices` out
    // of bounds below, and `nnz + base` overflows `i32` at `nnz == i32::MAX`
    // (a panic under `overflow-checks`). Nothing is guarded instead, because
    // `nnz` is not a length handed to C as 32 bits — the CSC arrays reach
    // Accelerate through an `int64_t` `columnStarts` — so the fix is to drop the
    // narrowing rather than bound it.
    if i64::from(row_index[n]) != nnz as i64 + i64::from(base) {
        return Err(Error::invalid("CSR row_index[n] does not match nnz"));
    }
    for (row, w) in row_index.windows(2).enumerate() {
        if w[1] < w[0] {
            return Err(Error::invalid("CSR row_index must be non-decreasing"));
        }
        // A structurally empty row (`row_index[row] == row_index[row + 1]`) is a
        // zero row whatever the values are, so the matrix is singular and no
        // backend can solve it. The reason it is rejected *here*, rather than
        // left to the solver, is that Accelerate's QR path does not fail
        // cleanly on one: on macOS 14 `_SparseFactorQR_Double` aborts the
        // process with `SIGTRAP` instead of returning a status, and a `SIGTRAP`
        // is not recoverable — a safe API must not let a caller reach it (#58).
        // macOS 26 returns the state-1 error object instead, which is exactly
        // the kind of platform split that makes "the caller will get an error"
        // an unsafe assumption.
        //
        // The kind matches what the residual check reports for a singular
        // matrix, so callers see one answer for "this system has no solution"
        // regardless of how the singularity was detected. Intel PARDISO does
        // not share this function and still reports an empty row as its own
        // zero-pivot error at phase 22 — this guard closes a process abort, not
        // a behavioural difference worth propagating to a path that already
        // handles the input.
        //
        // `upper_only` is unaffected by the reasoning above: an empty row of
        // the stored triangle leaves that row of the full symmetric matrix
        // zero, since the reflected entries live in earlier rows. The check
        // therefore applies to DSS's Cholesky path as well, which shares this
        // transposition.
        if w[1] == w[0] {
            return Err(Error::invalid(format!(
                "CSR row {row} has no entries — the matrix is structurally singular"
            )));
        }
    }

    let mut col_count = vec![0usize; n];
    for &col in columns {
        if col < base || (col - base) as usize >= n {
            return Err(Error::invalid("CSR column index out of range"));
        }
        col_count[(col - base) as usize] += 1;
    }
    let mut col_starts = vec![0i64; n + 1];
    for j in 0..n {
        // `col_count[j] <= nnz`, and a slice of more than `i64::MAX` elements
        // cannot be allocated, so this widening is lossless — the reason the
        // `usize` count is left as `usize` here rather than narrowed as `nnz`
        // used to be above.
        col_starts[j + 1] = col_starts[j] + col_count[j] as i64;
    }
    let mut next = col_starts[..n].to_vec();
    let mut row_indices = vec![0i32; nnz];
    let mut out_values = vec![0.0f64; nnz];
    for i in 0..n {
        let lo = (row_index[i] - base) as usize;
        let hi = (row_index[i + 1] - base) as usize;
        for k in lo..hi {
            let col0 = (columns[k] - base) as usize;
            if upper_only && col0 < i {
                return Err(Error::unsupported(
                    "symmetric lower-triangle storage is not supported on Apple Silicon; store the upper triangle",
                ));
            }
            let pos = next[col0] as usize;
            row_indices[pos] = i as i32;
            out_values[pos] = values[k];
            next[col0] += 1;
        }
    }
    Ok((col_starts, row_indices, out_values))
}

/// Backward-error + growth check for a sparse solve: compute `A·x` in the
/// caller's CSR form and reject the solution when it is not a plausible answer
/// to a nonsingular system. This catches singular matrices, which Accelerate's
/// QR path does not flag on its own.
///
/// A rank-deficient matrix surfaces in one of two ways: the residual
/// `‖A·x − b‖_∞` is not at the roundoff floor of the problem scale (b outside
/// the column space), or the solution magnitude `‖x‖_∞` blows up (division by a
/// ~0 pivot) even though the *backward* error is still tiny. Both are checked,
/// mirroring Intel PARDISO's zero-pivot error.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn check_residual(ia: &[i32], ja: &[i32], a: &[f64], b: &[f64], x: &[f64]) -> Result<()> {
    let n = b.len();
    let mut ax = vec![0.0f64; n];
    let mut a_inf = 0.0f64; // ‖A‖_∞ = max absolute row sum
    for i in 0..n {
        let lo = (ia[i] - 1) as usize;
        let hi = (ia[i + 1] - 1) as usize;
        let mut row_sum = 0.0f64;
        let mut dot = 0.0f64;
        for k in lo..hi {
            let col = (ja[k] - 1) as usize;
            dot += a[k] * x[col];
            row_sum += a[k].abs();
        }
        ax[i] = dot;
        a_inf = a_inf.max(row_sum);
    }
    let mut r_inf = 0.0f64;
    let mut b_inf = 0.0f64;
    let mut x_inf = 0.0f64;
    for i in 0..n {
        r_inf = r_inf.max((ax[i] - b[i]).abs());
        b_inf = b_inf.max(b[i].abs());
        x_inf = x_inf.max(x[i].abs());
    }

    // A valid solve is finite; NaN/inf in the residual or solution is a
    // definite failure, not a "not yet decided" case.
    if !(r_inf.is_finite() && x_inf.is_finite()) {
        return Err(Error::invalid(
            "PARDISO: non-finite residual or solution (singular matrix)",
        ));
    }
    // Backward error: for a stably solved *nonsingular* system the residual is
    // at the roundoff floor of ‖A‖·‖x‖ + ‖b‖. A residual well above that means
    // b was not in the column space (a singular/inconsistent system).
    if r_inf > 1e-8 * (a_inf * x_inf + b_inf) {
        return Err(Error::invalid(
            "PARDISO: singular or ill-conditioned matrix (residual check failed)",
        ));
    }
    // Growth: `‖A‖_∞·‖x‖_∞ / ‖b‖_∞` proxies the condition number. A singular
    // matrix drives it effectively unbounded (the huge-x case above slips the
    // backward-error test because ‖x‖ inflates its tolerance). 1e8 still admits
    // very ill-conditioned but nonsingular systems while catching rank
    // deficiency.
    if b_inf > 0.0 && a_inf * x_inf > 1e8 * b_inf {
        return Err(Error::invalid(
            "PARDISO: singular matrix (solution norm blow-up)",
        ));
    }
    Ok(())
}

/// Factor-and-solve is shared between PARDISO (nonsymmetric, QR) and DSS
/// (symmetric, Cholesky): build the `n × 1` dense right-hand side, run
/// `_SparseSolveOpaque_Double` with the factorization's required workspace, and
/// return the solution.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn solve_with_factor(
    factor: &nuvai_mkl_sys::SparseOpaqueFactorization_Double,
    n: i32,
    b: &[f64],
) -> Result<Vec<f64>> {
    // `DenseMatrix_Double.data` is `double *` (Apple's own struct type), but
    // this RHS is read-only: `_SparseSolveOpaque_Double` takes it as
    // `const DenseMatrix_Double *`. The `*const -> *mut` cast only satisfies
    // the struct field type; the callee never writes through it.
    let rhs = nuvai_mkl_sys::DenseMatrix_Double {
        rowCount: n,
        columnCount: 1,
        columnStride: n,
        attributes: nuvai_mkl_sys::SparseAttributes_t::ordinary(),
        data: b.as_ptr() as *mut f64,
    };
    let mut x = vec![0.0f64; n as usize];
    let soln = nuvai_mkl_sys::DenseMatrix_Double {
        rowCount: n,
        columnCount: 1,
        columnStride: n,
        attributes: nuvai_mkl_sys::SparseAttributes_t::ordinary(),
        data: x.as_mut_ptr(),
    };
    // SparseSolve documents its workspace as solveWorkspaceRequiredStatic +
    // nrhs * solveWorkspaceRequiredPerRHS bytes. It must be 16-byte aligned
    // (Apple's headers note any `malloc()` allocation has this property), so
    // allocate `u128`s — 16-byte alignment by construction — rather than a
    // 1-byte-aligned `Vec<u8>`.
    let ws_size = factor.solveWorkspaceRequiredStatic + factor.solveWorkspaceRequiredPerRHS;
    let ws_elems = ws_size.div_ceil(std::mem::size_of::<u128>());
    let mut workspace = vec![0u128; ws_elems];
    // `_SparseSolveOpaque_Double` is void-returning and takes the factor by
    // `*const`, so it never reports failure: there is no post-solve status to
    // inspect. Singularity is caught elsewhere — factor-time status for the
    // symmetric Cholesky path, or the caller's residual check for the QR path.
    // SAFETY: `factor` is a live, successfully-factored handle; `rhs`/`soln`
    // are correctly-shaped `DenseMatrix_Double`s whose `data` buffers (`b` and
    // `x`) are `n` elements long; `workspace` is at least `ws_size` bytes and
    // 16-byte aligned.
    unsafe {
        nuvai_mkl_sys::_SparseSolveOpaque_Double(
            factor,
            &rhs,
            &soln,
            workspace.as_mut_ptr() as *mut c_void,
        );
    }
    Ok(x)
}

/// Default symbolic-factor options: the same field values the Sparse library's
/// own `SparseSymbolicFactorOptionsDefault()` produces (order method default,
/// libc `malloc`/`free`, no error callback).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn default_symbolic_options() -> nuvai_mkl_sys::SparseSymbolicFactorOptions {
    nuvai_mkl_sys::SparseSymbolicFactorOptions {
        control: nuvai_mkl_sys::SparseDefaultControl,
        orderMethod: nuvai_mkl_sys::SparseOrderDefault,
        order: ptr::null_mut(),
        ignoreRowsAndColumns: ptr::null_mut(),
        malloc: nuvai_mkl_sys::malloc,
        free: nuvai_mkl_sys::free,
        reportError: None,
    }
}

/// Default numeric-factor options, matching
/// `SparseNumericFactorOptionsDefault()`.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn default_numeric_options() -> nuvai_mkl_sys::SparseNumericFactorOptions {
    nuvai_mkl_sys::SparseNumericFactorOptions {
        control: nuvai_mkl_sys::SparseDefaultControl,
        scalingMethod: nuvai_mkl_sys::SparseScalingDefault,
        scaling: ptr::null_mut(),
        pivotTolerance: 0.0,
        zeroTolerance: 0.0,
    }
}

impl Drop for Pardiso {
    fn drop(&mut self) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            // Release the cached QR factorization exactly once. It is moved out
            // by value (`take` into a local); the handle is not `Copy`, so the
            // raw-pointer payload cannot be silently aliased.
            if let Some(cached) = self.cached.take() {
                let mut factor = cached.factor;
                // SAFETY: `factor` is an owned, initialized factorization
                // created by `_SparseFactorQR_Double`; destroyed exactly once.
                unsafe { nuvai_mkl_sys::_SparseDestroyOpaqueNumeric_Double(&mut factor) };
            }
        }
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            // Nothing to release on aarch64-unknown-linux-gnu: no PARDISO
            // backend, so a `Pardiso` can never hold a factorization there.
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // Keyed on "`pardisoinit` ran", not on "the analysis succeeded"
            // (#23). The analysis phase can fail after partially allocating,
            // and a handle that survived `pardisoinit` but not phase 11 still
            // owns that memory — gating the release on a successful analysis
            // leaked it. `self.n` is `0` when no `solve` ever ran, which phase
            // `-1` ignores.
            if self.release_pending {
                // SAFETY: `self.pt`/`self.iparm` are initialized and
                // `self.n`/`self.mtype` are valid (set during `solve`); phase
                // -1 releases PARDISO's internal memory exactly once. The null
                // matrix/permutation pointers are valid "unused" arguments.
                unsafe {
                    let phase = -1i32; // release all internal memory
                    let maxfct = 1i32;
                    let mnum = 1i32;
                    let nrhs = 1i32;
                    let msglvl = 0i32;
                    let mut error = 0i32;
                    nuvai_mkl_sys::pardiso(
                        self.pt.as_mut_ptr() as *mut c_void,
                        &maxfct,
                        &mnum,
                        &self.mtype,
                        &phase,
                        &self.n,
                        ptr::null::<c_void>(),
                        ptr::null::<i32>(),
                        ptr::null::<i32>(),
                        ptr::null_mut::<i32>(),
                        &nrhs,
                        self.iparm.as_mut_ptr(),
                        &msglvl,
                        ptr::null_mut::<c_void>(),
                        ptr::null_mut::<c_void>(),
                        &mut error,
                    );
                }
            }
        }
    }
}

// Intel targets only: the flag under test exists solely on the Intel arm (the
// Accelerate arm releases its cached factorization, and linux-aarch64 has no
// backend at all).
#[cfg(all(test, not(target_arch = "aarch64")))]
mod tests {
    use super::*;

    /// Regression test for #23.
    ///
    /// `Drop` decides whether to release with phase `-1` from this flag, so the
    /// flag must be true for *any* handle whose `pardisoinit` has run — the
    /// invariant is "the handle may own PARDISO memory", not "the analysis
    /// succeeded". Asserting it straight after `new` is the whole test: the
    /// previous code set `analyzed = true` only after phase 11 returned
    /// success, so a phase-11 failure left it false, `Drop` skipped phase `-1`,
    /// and the partially-allocated analysis memory leaked.
    ///
    /// Deliberately not driven through a failing `solve`: forcing a *phase-11*
    /// failure needs an input oneMKL rejects at analysis, and the test would
    /// then be asserting on which phase reported the error rather than on the
    /// invariant. The flag is set before any `pardiso` call can fail, so the
    /// state after `new` is what makes the leak impossible.
    #[test]
    fn a_handle_is_releasable_before_any_solve() {
        let solver = Pardiso::new(mtype::NONSYMMETRIC);
        assert!(
            solver.release_pending,
            "a handle whose `pardisoinit` has run must be released on drop: a phase-11 \
             failure can leave partially-allocated analysis memory that only phase -1 \
             frees (#23)"
        );
    }

    /// The flag must not be cleared on the error paths that run after it is
    /// set — a rejected `solve` still leaves the handle releasable.
    #[test]
    fn a_rejected_solve_still_leaves_the_handle_releasable() {
        let mut solver = Pardiso::new(mtype::NONSYMMETRIC);
        // Fails on argument lengths, before any `pardiso` call.
        assert!(solver.solve(&[], &[], &[], &[]).is_err());
        assert!(solver.release_pending);
    }
}
