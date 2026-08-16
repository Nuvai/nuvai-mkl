//! Discrete Fourier transforms.
//!
//! On Intel targets this uses the DFTI interface ([`FftPlan`] wraps a committed
//! 1D complex-to-complex DFT descriptor). On Apple Silicon
//! (`aarch64-apple-darwin`) it wraps a pair of vDSP DFT setups (one per
//! direction), since vDSP makes the direction a creation parameter rather than
//! a per-call argument. On `aarch64-unknown-linux-gnu` there is no FFT backend
//! (OpenBLAS covers only BLAS/LAPACK), so every [`FftPlan`] operation returns
//! [`ErrorKind::Unsupported`].
//!
//! The forward transform is unnormalized; the backward transform applies the
//! default `1/n` scaling, so `backward(forward(x)) == x`. On aarch64 the vDSP
//! inverse DFT is itself unnormalized, so the shim applies the `1/n` scale
//! explicitly to preserve that contract.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::cell::RefCell;
#[cfg(not(target_arch = "aarch64"))]
use std::ptr;

use crate::error::{Error, Result};

pub use nuvai_mkl_sys::{MKL_Complex16, MKL_Complex8};

/// Opaque plan handle. On Intel this is the committed DFTI descriptor; on
/// aarch64 it holds the forward + inverse vDSP DFT setups and the precision.
#[cfg(not(target_arch = "aarch64"))]
type FftHandle = nuvai_mkl_sys::DFTI_DESCRIPTOR_HANDLE;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct FftHandle {
    /// Forward DFT setup (`vDSP_DFT_zop_CreateSetup`, direction = FORWARD).
    forward: nuvai_mkl_sys::vDSP_DFT_Setup,
    /// Inverse DFT setup (`vDSP_DFT_zop_CreateSetup`, direction = INVERSE).
    inverse: nuvai_mkl_sys::vDSP_DFT_Setup,
    /// True for the single-precision (f32) variants; selects the `D` vDSP API.
    single: bool,
    /// Reused split-complex scratch (`re`/`im`) for single-precision transforms.
    scratch32: RefCell<Option<(Vec<f32>, Vec<f32>)>>,
    /// Reused split-complex scratch for double-precision transforms.
    scratch64: RefCell<Option<(Vec<f64>, Vec<f64>)>>,
}

/// Uninhabited-in-practice handle on `aarch64-unknown-linux-gnu`: no FFT
/// backend exists there, so every [`FftPlan`] creation returns
/// [`ErrorKind::Unsupported`] and no handle is ever constructed.
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
struct FftHandle;

/// A committed 1D complex-to-complex DFT plan.
// On `aarch64-unknown-linux-gnu` no `FftPlan` can be constructed, so the
// `handle`/`len` fields are never read there.
#[cfg_attr(all(target_os = "linux", target_arch = "aarch64"), allow(dead_code))]
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
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            // No FFT backend on aarch64-unknown-linux-gnu (OpenBLAS covers
            // only BLAS/LAPACK). Return Unsupported before validating `len` so
            // feature-detection callers get the documented error even for
            // `len == 0`.
            let _ = (len, single);
            Err(Error::unsupported_linux_aarch64("FFT"))
        }
        #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
        if len == 0 {
            return Err(Error::invalid("FFT length must be positive"));
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let length = len as nuvai_mkl_sys::vDSP_Length;
            // SAFETY: `CreateSetup`/`CreateSetupD` take a `prev` setup (null
            // here) and the positive `length` validated above, returning a new
            // setup (or null on failure). Each returned handle is owned and
            // released exactly once by `Drop`/the error path below.
            let (forward, inverse) = unsafe {
                let forward = if single {
                    nuvai_mkl_sys::vDSP_DFT_zop_CreateSetup(
                        std::ptr::null_mut(),
                        length,
                        nuvai_mkl_sys::vDSP_DFT_FORWARD,
                    )
                } else {
                    nuvai_mkl_sys::vDSP_DFT_zop_CreateSetupD(
                        std::ptr::null_mut(),
                        length,
                        nuvai_mkl_sys::vDSP_DFT_FORWARD,
                    )
                };
                let inverse = if single {
                    nuvai_mkl_sys::vDSP_DFT_zop_CreateSetup(
                        std::ptr::null_mut(),
                        length,
                        nuvai_mkl_sys::vDSP_DFT_INVERSE,
                    )
                } else {
                    nuvai_mkl_sys::vDSP_DFT_zop_CreateSetupD(
                        std::ptr::null_mut(),
                        length,
                        nuvai_mkl_sys::vDSP_DFT_INVERSE,
                    )
                };
                (forward, inverse)
            };
            if forward.is_null() || inverse.is_null() {
                // Best-effort release of whatever half was allocated, using the
                // precision-matched destroy routine (as Drop does) — a double
                // setup released through the single-precision destroy leaks.
                // SAFETY: `forward`/`inverse` are either null or a valid setup
                // returned by the matching `CreateSetup`/`CreateSetupD` call
                // above, and each non-null setup is destroyed exactly once.
                unsafe {
                    if single {
                        if !forward.is_null() {
                            nuvai_mkl_sys::vDSP_DFT_DestroySetup(forward);
                        }
                        if !inverse.is_null() {
                            nuvai_mkl_sys::vDSP_DFT_DestroySetup(inverse);
                        }
                    } else {
                        if !forward.is_null() {
                            nuvai_mkl_sys::vDSP_DFT_DestroySetupD(forward);
                        }
                        if !inverse.is_null() {
                            nuvai_mkl_sys::vDSP_DFT_DestroySetupD(inverse);
                        }
                    }
                }
                return Err(Error::unsupported(format!(
                    "FFT length {len} is not supported by vDSP (supported: powers of two, and f·2^n for f in {{3, 5, 15}} with n >= 3)"
                )));
            }
            Ok(Self {
                handle: FftHandle {
                    forward,
                    inverse,
                    single,
                    scratch32: RefCell::new(None),
                    scratch64: RefCell::new(None),
                },
                len,
            })
        }
        #[cfg(not(target_arch = "aarch64"))]
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
            // SAFETY: `handle` is a valid out-param; `precision`, `DFTI_COMPLEX`
            // and `1` are constants and `len` is the positive length validated
            // above, so on `status == 0` `handle` holds a valid descriptor.
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
            // SAFETY: `handle` is the valid descriptor returned above; the
            // `DFTI_PLACEMENT`/`DFTI_NOT_INPLACE` config is valid for it.
            let status = unsafe {
                nuvai_mkl_sys::DftiSetValue(
                    handle,
                    nuvai_mkl_sys::DFTI_PLACEMENT,
                    nuvai_mkl_sys::DFTI_NOT_INPLACE,
                )
            };
            if status != 0 {
                // SAFETY: `handle` is a valid descriptor, freed exactly once on
                // this error path and not used afterwards.
                unsafe { nuvai_mkl_sys::DftiFreeDescriptor(&mut handle) };
                return Err(Error::mkl(status as i32, "DftiSetValue(DFTI_PLACEMENT)"));
            }
            // Pin the scaling so `backward(forward(x)) == x` regardless of the MKL
            // version's default backward scale (which is not always `1/n`).
            // SAFETY: `handle` is a valid descriptor; `DFTI_FORWARD_SCALE`/`1.0`
            // is valid config for it.
            let status = unsafe {
                nuvai_mkl_sys::DftiSetValue(handle, nuvai_mkl_sys::DFTI_FORWARD_SCALE, 1.0f64)
            };
            if status != 0 {
                // SAFETY: `handle` is a valid descriptor, freed exactly once here.
                unsafe { nuvai_mkl_sys::DftiFreeDescriptor(&mut handle) };
                return Err(Error::mkl(status as i32, "DftiSetValue(DFTI_FORWARD_SCALE)"));
            }
            let backward_scale = 1.0f64 / len as f64;
            // SAFETY: `handle` is a valid descriptor; `DFTI_BACKWARD_SCALE` with
            // the computed `1/len` value is valid config.
            let status = unsafe {
                nuvai_mkl_sys::DftiSetValue(handle, nuvai_mkl_sys::DFTI_BACKWARD_SCALE, backward_scale)
            };
            if status != 0 {
                // SAFETY: `handle` is a valid descriptor, freed exactly once here.
                unsafe { nuvai_mkl_sys::DftiFreeDescriptor(&mut handle) };
                return Err(Error::mkl(status as i32, "DftiSetValue(DFTI_BACKWARD_SCALE)"));
            }
            // SAFETY: `handle` is a valid, fully-configured descriptor.
            let status = unsafe { nuvai_mkl_sys::DftiCommitDescriptor(handle) };
            if status != 0 {
                // SAFETY: `handle` is a valid descriptor, freed exactly once here.
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
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            Err(Error::unsupported_linux_aarch64("FFT"))
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.transform_c32(input, output, false);
            Ok(())
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // SAFETY: `self.handle` is a committed descriptor; `input`/`output`
            // are exactly `self.len` complex elements (validated by `self.check`)
            // cast to `void*` as the out-of-place DFTI API expects.
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
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            Err(Error::unsupported_linux_aarch64("FFT"))
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.transform_c32(input, output, true);
            Ok(())
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // SAFETY: as `forward_c32` — committed descriptor and length-matched
            // buffers validated by `self.check`.
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
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            Err(Error::unsupported_linux_aarch64("FFT"))
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.transform_c64(input, output, false);
            Ok(())
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // SAFETY: as `forward_c32` — committed descriptor and length-matched
            // buffers validated by `self.check`.
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
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            Err(Error::unsupported_linux_aarch64("FFT"))
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.transform_c64(input, output, true);
            Ok(())
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // SAFETY: as `forward_c32` — committed descriptor and length-matched
            // buffers validated by `self.check`.
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

/// Accelerate (`aarch64-apple-darwin`) FFT backend.
///
/// vDSP operates on *split* complex (separate real/imag arrays), so each call
/// deinterleaves the caller's `MKL_Complex*` buffer into split scratch arrays,
/// runs [`nuvai_mkl_sys::vDSP_DFT_Execute`] in place, and re-interleaves the
/// result. vDSP's inverse DFT is unnormalized, so the `1/n` backward scale is
/// applied explicitly to match the DFTI contract on Intel.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl FftPlan {
    fn transform_c32(&self, input: &[MKL_Complex8], output: &mut [MKL_Complex8], inverse: bool) {
        let n = self.len;
        let mut scratch = self.handle.scratch32.borrow_mut();
        let (re, im) = scratch.get_or_insert_with(|| (vec![0.0f32; n], vec![0.0f32; n]));
        for (i, x) in input.iter().enumerate() {
            re[i] = x.real;
            im[i] = x.imag;
        }
        let setup = if inverse { self.handle.inverse } else { self.handle.forward };
        // SAFETY: `setup` is the non-null forward/inverse setup from `create`;
        // `re`/`im` are length-`n` scratch arrays and `vDSP_DFT_Execute` reads
        // the first `n` elements of each and writes them back in place
        // (split-complex out-of-place transform), which `re`/`im` are sized for.
        unsafe {
            nuvai_mkl_sys::vDSP_DFT_Execute(
                setup,
                re.as_ptr(),
                im.as_ptr(),
                re.as_mut_ptr(),
                im.as_mut_ptr(),
            );
        }
        let scale = if inverse { 1.0 / n as f32 } else { 1.0f32 };
        for (i, y) in output.iter_mut().enumerate() {
            y.real = re[i] * scale;
            y.imag = im[i] * scale;
        }
    }

    fn transform_c64(&self, input: &[MKL_Complex16], output: &mut [MKL_Complex16], inverse: bool) {
        let n = self.len;
        let mut scratch = self.handle.scratch64.borrow_mut();
        let (re, im) = scratch.get_or_insert_with(|| (vec![0.0f64; n], vec![0.0f64; n]));
        for (i, x) in input.iter().enumerate() {
            re[i] = x.real;
            im[i] = x.imag;
        }
        let setup = if inverse { self.handle.inverse } else { self.handle.forward };
        // SAFETY: `setup` is the non-null forward/inverse setup from `create`;
        // `re`/`im` are length-`n` scratch arrays and `vDSP_DFT_ExecuteD` reads
        // and writes exactly the first `n` split-complex elements, which the
        // scratch arrays are sized for.
        unsafe {
            nuvai_mkl_sys::vDSP_DFT_ExecuteD(
                setup,
                re.as_ptr(),
                im.as_ptr(),
                re.as_mut_ptr(),
                im.as_mut_ptr(),
            );
        }
        let scale = if inverse { 1.0 / n as f64 } else { 1.0f64 };
        for (i, y) in output.iter_mut().enumerate() {
            y.real = re[i] * scale;
            y.imag = im[i] * scale;
        }
    }
}

impl Drop for FftPlan {
    fn drop(&mut self) {
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            // No FFT backend on aarch64-unknown-linux-gnu: an `FftPlan` can
            // never be created here, so there is nothing to release.
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            // SAFETY: `self.handle.forward`/`inverse` are valid setups created
            // non-null in `create`, and each is destroyed exactly once here via
            // the precision-matched destroy routine.
            unsafe {
                if self.handle.single {
                    nuvai_mkl_sys::vDSP_DFT_DestroySetup(self.handle.forward);
                    nuvai_mkl_sys::vDSP_DFT_DestroySetup(self.handle.inverse);
                } else {
                    nuvai_mkl_sys::vDSP_DFT_DestroySetupD(self.handle.forward);
                    nuvai_mkl_sys::vDSP_DFT_DestroySetupD(self.handle.inverse);
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // SAFETY: `self.handle` is a valid committed descriptor (or null);
            // freeing it once here releases the MKL resources.
            unsafe {
                nuvai_mkl_sys::DftiFreeDescriptor(&mut self.handle);
            }
        }
    }
}
