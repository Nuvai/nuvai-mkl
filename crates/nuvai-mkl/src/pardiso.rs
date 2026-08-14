//! PARDISO — parallel sparse direct solver (double precision).
//!
//! On Intel targets matrices are supplied in CSR (3-array) form with 1-based
//! indexing (PARDISO's default) and [`Pardiso::solve`] runs the analysis →
//! factorization → solve phases. On Apple Silicon (`aarch64-apple-darwin`) the
//! same CSR input is transposed to CSC and solved with the Accelerate
//! Sparse/SparseSolve backend (`_SparseFactorQR_Double` +
//! `_SparseSolveOpaque_Double`, ADR-0003 decision 7).

use std::os::raw::c_void;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::os::raw::c_long;
use std::ptr;

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
/// Silicon the Accelerate backend performs a self-contained factor+solve per
/// call and keeps only the caller's `mtype`, so the handle carries no inert
/// 768-byte PARDISO state.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub struct Pardiso {
    pt: [*mut c_void; 64],
    mtype: i32,
    iparm: [i32; 64],
    n: i32,
    analyzed: bool,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub struct Pardiso {
    mtype: i32,
}

impl Pardiso {
    /// Create a handle for the given matrix type.
    pub fn new(mtype: i32) -> Self {
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let mut iparm = [0i32; 64];
            iparm[0] = 1; // use default `iparm` values
            let mut pt = [ptr::null_mut::<c_void>(); 64];
            // SAFETY: `pt` and `iparm` are valid, zeroed arrays of the exact
            // size `pardisoinit` expects to initialize; `mtype` is passed by
            // const reference and only read.
            unsafe {
                nuvai_mkl_sys::pardisoinit(pt.as_mut_ptr() as *mut c_void, &mtype, iparm.as_mut_ptr());
            }
            Self {
                pt,
                mtype,
                iparm,
                n: 0,
                analyzed: false,
            }
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            Self { mtype }
        }
    }

    /// Factor and solve `A x = b` for the matrix in CSR form: `ia` has length
    /// `n + 1`, `ja` (column indices) and `a` (values) have length `nnz`.
    /// Returns the solution `x` (length `n`).
    pub fn solve(&mut self, ia: &[i32], ja: &[i32], a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.solve_accelerate(ia, ja, a, b)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            if ja.len() != a.len() {
                return Err(Error::invalid("PARDISO: ja/a length mismatch"));
            }
            let n = (ia.len() - 1) as i32;
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
                    return Err(Error::mkl(error, "pardiso phase 11 (analysis)"));
                }
                self.analyzed = true;

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
                    return Err(Error::mkl(error, "pardiso phase 22 (factorization)"));
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
                    return Err(Error::mkl(error, "pardiso phase 33 (solve)"));
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
/// solves with `_SparseSolveOpaque_Double`. The factorization is local and
/// destroyed before returning; the matrix buffers stay alive for the call.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Pardiso {
    fn solve_accelerate(&mut self, ia: &[i32], ja: &[i32], a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
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
        let n = (ia.len() - 1) as i32;
        if n <= 0 || b.len() != n as usize {
            return Err(Error::invalid("PARDISO: bad ia/b lengths"));
        }

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

        // SAFETY: `matrix` borrows the CSC arrays for the duration of the call;
        // `SparseFactorizationQR` and the option structs are valid. The returned
        // `SparseOpaqueFactorization_Double` is owned by value and freed once
        // below.
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
            // SAFETY: `factor` is an owned, initialized factorization; released
            // exactly once on the error path.
            unsafe { nuvai_mkl_sys::_SparseDestroyOpaqueNumeric_Double(&mut factor) };
            return Err(Error::mkl(status, "_SparseFactorQR_Double"));
        }

        let result = solve_with_factor(&factor, n, b);
        // SAFETY: `factor` is an owned, initialized factorization; released
        // exactly once here after the solve.
        unsafe { nuvai_mkl_sys::_SparseDestroyOpaqueNumeric_Double(&mut factor) };
        let x = result?;

        // QR factorization of a singular matrix still succeeds (R is simply
        // rank-deficient), so Accelerate reports no error and the solve returns a
        // meaningless least-squares solution. Detect that with a residual check so
        // singular systems error out like Intel PARDISO's zero-pivot failure.
        check_residual(ia, ja, a, b, &x)?;
        Ok(x)
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
/// bounds/monotonicity checks apply to both.
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

    // A valid `base`-based CSR has row_index[0] == base, row_index[n] == nnz +
    // base, and a non-decreasing row_index. Any other shape would silently drop
    // or mis-index entries.
    if row_index[0] != base {
        return Err(Error::invalid("CSR row_index[0] does not match the index base"));
    }
    if row_index[n] != nnz as i32 + base {
        return Err(Error::invalid("CSR row_index[n] does not match nnz"));
    }
    for w in row_index.windows(2) {
        if w[1] < w[0] {
            return Err(Error::invalid("CSR row_index must be non-decreasing"));
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
            // The Accelerate backend keeps no persistent handle state: the
            // QR factorization is created and destroyed inside `solve_accelerate`,
            // so there is nothing to release here.
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            if self.analyzed {
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
