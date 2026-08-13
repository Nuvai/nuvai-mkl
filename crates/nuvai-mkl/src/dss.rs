//! DSS — Direct Sparse Solver (double precision).
//!
//! On Intel targets this wraps the MKL DSS solver for symmetric systems
//! (matrices in CSR form with 0-based indexing, upper/lower triangle only).
//! On Apple Silicon (`aarch64-apple-darwin`) the Accelerate Sparse/SparseSolve
//! backend is wired in a later phase; this module currently reports
//! [`ErrorKind::Unsupported`](crate::error::ErrorKind) rather than degrading
//! silently (ADR-0003, decision 2).
//!
//! Each `dss_*` routine interprets its own `opt` argument; the flags are *not*
//! interchangeable between routines. In particular the matrix-structure flag
//! (`MKL_DSS_SYMMETRIC`) belongs to [`dss_define_structure_`], the reordering
//! flag (`MKL_DSS_AUTO_ORDER`) to [`dss_reorder_`], the definiteness flag
//! (`MKL_DSS_POSITIVE_DEFINITE`) to [`dss_factor_real_`], and the solve-mode
//! flags to [`dss_solve_real_`]. Only the indexing/precision flags
//! (`MKL_DSS_ZERO_BASED_INDEXING`) are passed to [`dss_create_`].

use std::os::raw::c_void;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use std::ptr;

use crate::error::{Error, Result};

/// A factorized DSS handle (double precision, real symmetric).
pub struct Dss {
    handle: *mut c_void,
}

impl Dss {
    /// Analyze + factor a real symmetric positive-definite matrix given in
    /// CSR (0-based): `row_index` has length `n + 1`; `columns` and `values`
    /// have length `nnz`.
    pub fn factor_symmetric(row_index: &[i32], columns: &[i32], values: &[f64]) -> Result<Self> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let _ = (row_index, columns, values);
            return Err(Error::unsupported(
                "DSS on aarch64 requires the Accelerate Sparse backend (not yet wired)",
            ));
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
            let _ = rhs;
            return Err(Error::unsupported(
                "DSS on aarch64 requires the Accelerate Sparse backend (not yet wired)",
            ));
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

impl Drop for Dss {
    fn drop(&mut self) {
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
