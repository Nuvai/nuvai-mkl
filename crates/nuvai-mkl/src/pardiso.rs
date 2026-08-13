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
/// On aarch64 the handle fields are inert: the Accelerate backend performs a
/// self-contained factor+solve per call and keeps no persistent state, so the
/// fields exist only to keep the public type identical across platforms.
#[cfg_attr(all(target_os = "macos", target_arch = "aarch64"), allow(dead_code))]
pub struct Pardiso {
    pt: [*mut c_void; 64],
    mtype: i32,
    iparm: [i32; 64],
    n: i32,
    analyzed: bool,
}

impl Pardiso {
    /// Create a handle for the given matrix type.
    pub fn new(mtype: i32) -> Self {
        let mut iparm = [0i32; 64];
        iparm[0] = 1; // use default `iparm` values
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let mut pt = [ptr::null_mut::<c_void>(); 64];
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let pt = [ptr::null_mut::<c_void>(); 64];
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            unsafe {
                nuvai_mkl_sys::pardisoinit(pt.as_mut_ptr() as *mut c_void, &mtype, iparm.as_mut_ptr());
            }
        }
        Self {
            pt,
            mtype,
            iparm,
            n: 0,
            analyzed: false,
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
        if ja.len() != a.len() {
            return Err(Error::invalid("PARDISO: ja/a length mismatch"));
        }
        let n = (ia.len() - 1) as i32;
        if n <= 0 || b.len() != n as usize {
            return Err(Error::invalid("PARDISO: bad ia/b lengths"));
        }
        self.n = n;

        let (col_starts, row_indices, values) = csr_to_csc(n, ia, ja, a)?;

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
            unsafe { nuvai_mkl_sys::_SparseDestroyOpaqueNumeric_Double(&mut factor) };
            return Err(Error::mkl(status, "_SparseFactorQR_Double"));
        }

        let result = solve_with_factor(&factor, n, b);
        unsafe { nuvai_mkl_sys::_SparseDestroyOpaqueNumeric_Double(&mut factor) };
        result
    }
}

/// Build a 0-based CSC (compressed sparse column) representation of the square
/// `n × n` matrix given in 1-based CSR form (`ia` length `n + 1`, `ja`/`a`
/// length `nnz`). Returns `(column_starts, row_indices, values)`.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn csr_to_csc(n: i32, ia: &[i32], ja: &[i32], a: &[f64]) -> Result<(Vec<i64>, Vec<i32>, Vec<f64>)> {
    let n = n as usize;
    let nnz = ja.len();
    let mut col_count = vec![0usize; n];
    for &col in ja {
        col_count[(col - 1) as usize] += 1;
    }
    let mut col_starts = vec![0i64; n + 1];
    for j in 0..n {
        col_starts[j + 1] = col_starts[j] + col_count[j] as i64;
    }
    let mut next = col_starts[..n].to_vec();
    let mut row_indices = vec![0i32; nnz];
    let mut values = vec![0.0f64; nnz];
    for i in 0..n {
        let lo = (ia[i] - 1) as usize;
        let hi = (ia[i + 1] - 1) as usize;
        for k in lo..hi {
            let col = (ja[k] - 1) as usize;
            let pos = next[col] as usize;
            row_indices[pos] = i as i32;
            values[pos] = a[k];
            next[col] += 1;
        }
    }
    Ok((col_starts, row_indices, values))
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
    // nrhs * solveWorkspaceRequiredPerRHS bytes.
    let ws_size = factor.solveWorkspaceRequiredStatic + factor.solveWorkspaceRequiredPerRHS;
    let mut workspace = vec![0u8; ws_size];
    let mut factor_copy = *factor;
    unsafe {
        nuvai_mkl_sys::_SparseSolveOpaque_Double(
            &mut factor_copy,
            &rhs,
            &soln,
            workspace.as_mut_ptr() as *mut c_void,
        );
    }
    if factor_copy.status != nuvai_mkl_sys::SparseStatusOK {
        return Err(Error::mkl(factor_copy.status, "_SparseSolveOpaque_Double"));
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
        if !self.analyzed {
            return;
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
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
