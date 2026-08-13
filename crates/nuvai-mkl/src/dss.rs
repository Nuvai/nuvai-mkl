//! DSS — Direct Sparse Solver (double precision).
//!
//! On Intel targets this wraps the MKL DSS solver for symmetric systems
//! (matrices in CSR form with 0-based indexing, upper/lower triangle only).
//! On Apple Silicon (`aarch64-apple-darwin`) the same CSR input is transposed
//! to CSC and solved with the Accelerate Sparse/SparseSolve backend
//! (`_SparseFactorSymmetric_Double` + `_SparseSolveOpaque_Double`,
//! ADR-0003 decision 7). The backend is never silently selected: it is chosen
//! by `cfg(target_arch)` exactly as on the other domains.
//!
//! Each `dss_*` routine interprets its own `opt` argument; the flags are *not*
//! interchangeable between routines. In particular the matrix-structure flag
//! (`MKL_DSS_SYMMETRIC`) belongs to [`dss_define_structure_`], the reordering
//! flag (`MKL_DSS_AUTO_ORDER`) to [`dss_reorder_`], the definiteness flag
//! (`MKL_DSS_POSITIVE_DEFINITE`) to [`dss_factor_real_`], and the solve-mode
//! flags to [`dss_solve_real_`]. Only the indexing/precision flags
//! (`MKL_DSS_ZERO_BASED_INDEXING`) are passed to [`dss_create_`].

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use std::os::raw::c_void;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::os::raw::c_long;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use std::ptr;

use crate::error::{Error, Result};

/// Opaque factorized handle. On Intel this is the MKL DSS handle pointer; on
/// aarch64 it is the Accelerate `SparseOpaqueFactorization_Double` (104 bytes)
/// owned by value, since `_SparseFactorSymmetric_Double` returns it by value
/// and `solve`/`Drop` consume it.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
type DssHandle = *mut c_void;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
type DssHandle = nuvai_mkl_sys::SparseOpaqueFactorization_Double;

/// A factorized DSS handle (double precision, real symmetric).
pub struct Dss {
    handle: DssHandle,
}

impl Dss {
    /// Analyze + factor a real symmetric positive-definite matrix given in
    /// CSR (0-based): `row_index` has length `n + 1`; `columns` and `values`
    /// have length `nnz`.
    pub fn factor_symmetric(row_index: &[i32], columns: &[i32], values: &[f64]) -> Result<Self> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let n_rows = (row_index.len() as i32) - 1;
            if n_rows <= 0 || values.len() != columns.len() {
                return Err(Error::invalid("DSS: bad row_index/columns/values lengths"));
            }

            let (col_starts, row_indices, csc_values) = csr_upper_to_csc(n_rows, row_index, columns, values)?;

            let matrix = nuvai_mkl_sys::SparseMatrix_Double {
                structure: nuvai_mkl_sys::SparseMatrixStructure {
                    rowCount: n_rows,
                    columnCount: n_rows,
                    columnStarts: col_starts.as_ptr() as *mut c_long,
                    rowIndices: row_indices.as_ptr() as *mut i32,
                    attributes: nuvai_mkl_sys::SparseAttributes_t::symmetric(),
                    blockSize: 1,
                },
                data: csc_values.as_ptr() as *mut f64,
            };

            let sfoptions = crate::pardiso::default_symbolic_options();
            let nfoptions = crate::pardiso::default_numeric_options();

            let mut factor = unsafe {
                nuvai_mkl_sys::_SparseFactorSymmetric_Double(
                    nuvai_mkl_sys::SparseFactorizationCholesky,
                    &matrix,
                    &sfoptions,
                    &nfoptions,
                )
            };
            if factor.status != nuvai_mkl_sys::SparseStatusOK {
                let status = factor.status;
                unsafe { nuvai_mkl_sys::_SparseDestroyOpaqueNumeric_Double(&mut factor) };
                return Err(Error::mkl(status, "_SparseFactorSymmetric_Double"));
            }
            Ok(Self { handle: factor })
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let n_rows = (row_index.len() as i32) - 1;
            let n_cols = n_rows;
            let n_nonzeros = columns.len() as i32;
            if n_rows <= 0 || values.len() != columns.len() {
                return Err(Error::invalid("DSS: bad row_index/columns/values lengths"));
            }

            let opt_create = nuvai_mkl_sys::MKL_DSS_ZERO_BASED_INDEXING as i32;
            let opt_structure = nuvai_mkl_sys::MKL_DSS_SYMMETRIC as i32;
            let opt_reorder = nuvai_mkl_sys::MKL_DSS_AUTO_ORDER as i32;
            let opt_factor = nuvai_mkl_sys::MKL_DSS_POSITIVE_DEFINITE as i32;

            let mut handle: *mut c_void = ptr::null_mut();

            unsafe {
                let mut status = nuvai_mkl_sys::dss_create_(&mut handle, &opt_create);
                if status != 0 {
                    return Err(Error::mkl(status, "dss_create"));
                }

                status = nuvai_mkl_sys::dss_define_structure_(
                    &mut handle,
                    &opt_structure,
                    row_index.as_ptr(),
                    &n_rows,
                    &n_cols,
                    columns.as_ptr(),
                    &n_nonzeros,
                );
                if status != 0 {
                    nuvai_mkl_sys::dss_delete_(&handle, &opt_create);
                    return Err(Error::mkl(status, "dss_define_structure"));
                }

                let mut perm = vec![0i32; n_rows as usize];
                status = nuvai_mkl_sys::dss_reorder_(&mut handle, &opt_reorder, perm.as_mut_ptr());
                if status != 0 {
                    nuvai_mkl_sys::dss_delete_(&handle, &opt_create);
                    return Err(Error::mkl(status, "dss_reorder"));
                }

                status = nuvai_mkl_sys::dss_factor_real_(
                    &mut handle,
                    &opt_factor,
                    values.as_ptr() as *const c_void,
                );
                if status != 0 {
                    nuvai_mkl_sys::dss_delete_(&handle, &opt_create);
                    return Err(Error::mkl(status, "dss_factor_real"));
                }
            }

            Ok(Self { handle })
        }
    }

    /// Solve for a single right-hand side; returns the solution.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let n = self.handle.symbolicFactorization.rowCount;
            if rhs.len() != n as usize {
                return Err(Error::invalid("DSS: rhs length mismatch"));
            }
            crate::pardiso::solve_with_factor(&self.handle, n, rhs)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let n_rhs = 1i32;
            let opt_solve = 0i32; // normal (non-transpose, non-conjugate) solve
            let mut sol = vec![0.0f64; rhs.len()];
            let mut handle = self.handle; // local copy: DSS takes the handle by address
            let status = unsafe {
                nuvai_mkl_sys::dss_solve_real_(
                    &mut handle,
                    &opt_solve,
                    rhs.as_ptr() as *const c_void,
                    &n_rhs,
                    sol.as_mut_ptr() as *mut c_void,
                )
            };
            if status != 0 {
                return Err(Error::mkl(status, "dss_solve_real"));
            }
            Ok(sol)
        }
    }
}

/// Build a 0-based CSC (compressed sparse column) representation of the square
/// `n × n` matrix given in 0-based upper-triangle CSR form (`row_index` length
/// `n + 1`, `columns`/`values` length `nnz`). Returns
/// `(column_starts, row_indices, values)`.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn csr_upper_to_csc(
    n: i32,
    row_index: &[i32],
    columns: &[i32],
    values: &[f64],
) -> Result<(Vec<i64>, Vec<i32>, Vec<f64>)> {
    let n = n as usize;
    let nnz = columns.len();
    let mut col_count = vec![0usize; n];
    for &col in columns {
        if col < 0 || col as usize >= n {
            return Err(Error::invalid("DSS: column index out of range"));
        }
        col_count[col as usize] += 1;
    }
    let mut col_starts = vec![0i64; n + 1];
    for j in 0..n {
        col_starts[j + 1] = col_starts[j] + col_count[j] as i64;
    }
    let mut next = col_starts[..n].to_vec();
    let mut row_indices = vec![0i32; nnz];
    let mut out_values = vec![0.0f64; nnz];
    for i in 0..n {
        let lo = row_index[i] as usize;
        let hi = row_index[i + 1] as usize;
        if lo > hi || hi > nnz {
            return Err(Error::invalid("DSS: malformed row_index"));
        }
        for k in lo..hi {
            let col = columns[k] as usize;
            let pos = next[col] as usize;
            row_indices[pos] = i as i32;
            out_values[pos] = values[k];
            next[col] += 1;
        }
    }
    Ok((col_starts, row_indices, out_values))
}

impl Drop for Dss {
    fn drop(&mut self) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            unsafe {
                nuvai_mkl_sys::_SparseDestroyOpaqueNumeric_Double(&mut self.handle);
            }
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            // `dss_delete` takes the plain `opt = 0` (it does not accept the
            // zero-based-indexing flag).
            let opt = 0i32;
            unsafe {
                nuvai_mkl_sys::dss_delete_(&self.handle, &opt);
            }
        }
    }
}
