//! Discrete Fourier transforms.
//!
//! On Intel targets this uses the DFTI interface ([`FftPlan`] wraps a committed
//! 1D complex-to-complex DFT descriptor). On Apple Silicon
//! (`aarch64-apple-darwin`) Intel ships no oneMKL; the vDSP DFT backend is
//! wired in a later phase and this module currently reports
//! [`ErrorKind::Unsupported`](crate::error::ErrorKind) rather than degrading
//! silently (ADR-0003, decision 2).
//!
//! The forward transform is unnormalized; the backward transform applies the
//! default `1/n` scaling, so `backward(forward(x)) == x`.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use std::ptr;

use crate::error::{Error, Result};

pub use nuvai_mkl_sys::{MKL_Complex16, MKL_Complex8};

/// Opaque plan handle. On Intel this is the committed DFTI descriptor; on
/// aarch64 it is unused (the vDSP backend is not wired yet).
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
type FftHandle = nuvai_mkl_sys::DFTI_DESCRIPTOR_HANDLE;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
type FftHandle = *mut std::os::raw::c_void;

/// A committed 1D complex-to-complex DFT plan.
pub struct FftPlan {
    handle: FftHandle,
    len: usize,
}

impl FftPlan {
    /// Plan a single-precision 1D complex-to-complex DFT of length `len`.
    pub fn new_c32(len: usize) -> Result<Self> {
        Self::create(len, true)
    }

    /// Plan a double-precision 1D complex-to-complex DFT of length `len`.
    pub fn new_c64(len: usize) -> Result<Self> {
        Self::create(len, false)
    }

    fn create(len: usize, single: bool) -> Result<Self> {
        if len == 0 {
            return Err(Error::invalid("FFT length must be positive"));
        }
        let _ = single;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let _ = len;
            return Err(Error::unsupported(
                "FFT on aarch64 requires the vDSP DFT backend (not yet wired)",
            ));
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let len = len as i64;
            let precision = if single {
                nuvai_mkl_sys::DFTI_SINGLE
            } else {
                nuvai_mkl_sys::DFTI_DOUBLE
            };
            let mut handle: nuvai_mkl_sys::DFTI_DESCRIPTOR_HANDLE = ptr::null_mut();
            // Use the public `DftiCreateDescriptor` entry point rather than the
            // internal `*_s_1d`/`*_d_1d` helpers (which the header marks as
            // "INTERNAL INTERFACES … may change in future releases").
            let status = unsafe {
                nuvai_mkl_sys::DftiCreateDescriptor(
                    &mut handle,
                    precision,
                    nuvai_mkl_sys::DFTI_COMPLEX,
                    1,
                    len,
                )
            };
            if status != 0 {
                return Err(Error::mkl(status as i32, "DftiCreateDescriptor"));
            }
            // Explicitly request out-of-place transforms so the caller's distinct
            // input/output buffers are honoured.
            let status = unsafe {
                nuvai_mkl_sys::DftiSetValue(
                    handle,
                    nuvai_mkl_sys::DFTI_PLACEMENT,
                    nuvai_mkl_sys::DFTI_NOT_INPLACE,
                )
            };
            if status != 0 {
                unsafe { nuvai_mkl_sys::DftiFreeDescriptor(&mut handle) };
                return Err(Error::mkl(status as i32, "DftiSetValue(DFTI_PLACEMENT)"));
            }
            // Pin the scaling so `backward(forward(x)) == x` regardless of the MKL
            // version's default backward scale (which is not always `1/n`).
            let status = unsafe {
                nuvai_mkl_sys::DftiSetValue(handle, nuvai_mkl_sys::DFTI_FORWARD_SCALE, 1.0f64)
            };
            if status != 0 {
                unsafe { nuvai_mkl_sys::DftiFreeDescriptor(&mut handle) };
                return Err(Error::mkl(status as i32, "DftiSetValue(DFTI_FORWARD_SCALE)"));
            }
            let backward_scale = 1.0f64 / len as f64;
            let status = unsafe {
                nuvai_mkl_sys::DftiSetValue(handle, nuvai_mkl_sys::DFTI_BACKWARD_SCALE, backward_scale)
            };
            if status != 0 {
                unsafe { nuvai_mkl_sys::DftiFreeDescriptor(&mut handle) };
                return Err(Error::mkl(status as i32, "DftiSetValue(DFTI_BACKWARD_SCALE)"));
            }
            let status = unsafe { nuvai_mkl_sys::DftiCommitDescriptor(handle) };
            if status != 0 {
                unsafe { nuvai_mkl_sys::DftiFreeDescriptor(&mut handle) };
                return Err(Error::mkl(status as i32, "DftiCommitDescriptor"));
            }
            Ok(Self {
                handle,
                len: len as usize,
            })
        }
    }

    fn check(&self, input: usize, output: usize) -> Result<()> {
        if input != self.len || output != self.len {
            return Err(Error::invalid("FFT input/output length mismatch with plan"));
        }
        Ok(())
    }

    /// Forward transform (single precision), `input → output`.
    pub fn forward_c32(&self, input: &[MKL_Complex8], output: &mut [MKL_Complex8]) -> Result<()> {
        self.check(input.len(), output.len())?;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            Err(Error::unsupported(
                "FFT on aarch64 requires the vDSP DFT backend (not yet wired)",
            ))
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let status = unsafe {
                nuvai_mkl_sys::DftiComputeForward(
                    self.handle,
                    input.as_ptr() as *mut std::os::raw::c_void,
                    output.as_mut_ptr() as *mut std::os::raw::c_void,
                )
            };
            if status != 0 {
                return Err(Error::mkl(status as i32, "DftiComputeForward"));
            }
            Ok(())
        }
    }

    /// Backward transform (single precision), `input → output` (scaled by `1/n`).
    pub fn backward_c32(&self, input: &[MKL_Complex8], output: &mut [MKL_Complex8]) -> Result<()> {
        self.check(input.len(), output.len())?;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            Err(Error::unsupported(
                "FFT on aarch64 requires the vDSP DFT backend (not yet wired)",
            ))
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let status = unsafe {
                nuvai_mkl_sys::DftiComputeBackward(
                    self.handle,
                    input.as_ptr() as *mut std::os::raw::c_void,
                    output.as_mut_ptr() as *mut std::os::raw::c_void,
                )
            };
            if status != 0 {
                return Err(Error::mkl(status as i32, "DftiComputeBackward"));
            }
            Ok(())
        }
    }

    /// Forward transform (double precision), `input → output`.
    pub fn forward_c64(&self, input: &[MKL_Complex16], output: &mut [MKL_Complex16]) -> Result<()> {
        self.check(input.len(), output.len())?;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            Err(Error::unsupported(
                "FFT on aarch64 requires the vDSP DFT backend (not yet wired)",
            ))
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let status = unsafe {
                nuvai_mkl_sys::DftiComputeForward(
                    self.handle,
                    input.as_ptr() as *mut std::os::raw::c_void,
                    output.as_mut_ptr() as *mut std::os::raw::c_void,
                )
            };
            if status != 0 {
                return Err(Error::mkl(status as i32, "DftiComputeForward"));
            }
            Ok(())
        }
    }

    /// Backward transform (double precision), `input → output` (scaled by `1/n`).
    pub fn backward_c64(&self, input: &[MKL_Complex16], output: &mut [MKL_Complex16]) -> Result<()> {
        self.check(input.len(), output.len())?;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            Err(Error::unsupported(
                "FFT on aarch64 requires the vDSP DFT backend (not yet wired)",
            ))
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let status = unsafe {
                nuvai_mkl_sys::DftiComputeBackward(
                    self.handle,
                    input.as_ptr() as *mut std::os::raw::c_void,
                    output.as_mut_ptr() as *mut std::os::raw::c_void,
                )
            };
            if status != 0 {
                return Err(Error::mkl(status as i32, "DftiComputeBackward"));
            }
            Ok(())
        }
    }
}

impl Drop for FftPlan {
    fn drop(&mut self) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            // No descriptor on aarch64 (the vDSP backend is not wired yet).
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            unsafe {
                nuvai_mkl_sys::DftiFreeDescriptor(&mut self.handle);
            }
        }
    }
}
