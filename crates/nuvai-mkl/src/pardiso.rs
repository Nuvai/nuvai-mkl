//! PARDISO — parallel sparse direct solver (double precision).
//!
//! On Intel targets matrices are supplied in CSR (3-array) form with 1-based
//! indexing (PARDISO's default) and [`Pardiso::solve`] runs the analysis →
//! factorization → solve phases. On Apple Silicon (`aarch64-apple-darwin`) the
//! Accelerate Sparse/SparseSolve backend is wired in a later phase; this module
//! currently reports [`ErrorKind::Unsupported`](crate::error::ErrorKind)
//! rather than degrading silently (ADR-0003, decision 2).

use std::os::raw::c_void;
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
            let _ = (ia, ja, a, b);
            return Err(Error::unsupported(
                "PARDISO on aarch64 requires the Accelerate Sparse backend (not yet wired)",
            ));
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
