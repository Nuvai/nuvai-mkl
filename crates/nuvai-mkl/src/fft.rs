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
    /// Which vDSP DFT family backs this plan.
    backend: FftBackend,
    /// True for the single-precision (f32) variants; selects the `D` vDSP API.
    single: bool,
}

/// The vDSP DFT family backing an aarch64 [`FftHandle`]. Interleaved-complex is
/// preferred — it transforms the caller's `MKL_Complex*` buffer directly (no
/// split/reinterleave copy) — but it can only plan lengths `f·2^n` for
/// `f ∈ {2, 3, 5, 9, 15, 25}` with `n >= 2` (minimum 8) on macOS 12.0+. The
/// split-complex family is the fallback for the two lengths below that (2 and
/// 4), which need a deinterleave → execute → reinterleave round-trip through
/// scratch arrays.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
enum FftBackend {
    /// Interleaved-complex DFT (`vDSP_DFT_Interleaved_*`, macOS 12.0+).
    Interleaved {
        forward: nuvai_mkl_sys::vDSP_DFT_Interleaved_Setup,
        inverse: nuvai_mkl_sys::vDSP_DFT_Interleaved_Setup,
    },
    /// Split-complex DFT (`vDSP_DFT_zop_*`, macOS 10.7+).
    Split {
        forward: nuvai_mkl_sys::vDSP_DFT_Setup,
        inverse: nuvai_mkl_sys::vDSP_DFT_Setup,
        /// Reused split-complex scratch (`re`/`im`) for single-precision transforms.
        scratch32: RefCell<Option<(Vec<f32>, Vec<f32>)>>,
        /// Reused split-complex scratch for double-precision transforms.
        scratch64: RefCell<Option<(Vec<f64>, Vec<f64>)>>,
    },
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
            // Prefer the interleaved family (no split/reinterleave copy); fall
            // back to split-complex for the lengths interleaved cannot plan
            // (2 and 4). vDSP reports an unsupported length as a null setup, so
            // probing the setup is the source of truth for the length predicate
            // rather than re-deriving the `f·2^n` factorization here.
            let backend = Self::create_interleaved(length, single)
                .or_else(|| Self::create_split(length, single));
            let Some(backend) = backend else {
                return Err(Error::unsupported(format!(
                    "FFT length {len} is not supported by vDSP (supported: 2, 4, and f·2^n for f in {{2, 3, 5, 9, 15, 25}} with n >= 2)"
                )));
            };
            Ok(Self {
                handle: FftHandle { backend, single },
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
/// The plan prefers vDSP's interleaved-complex DFT (macOS 12.0+), which
/// transforms the caller's `MKL_Complex*` buffer directly (layout-identical to
/// `DSP*Complex`) with no split/reinterleave copy. For the two lengths that
/// family cannot plan (2 and 4) it falls back to the split-complex DFT, which
/// deinterleaves into `re`/`im` scratch arrays, runs
/// [`nuvai_mkl_sys::vDSP_DFT_Execute`], and re-interleaves. Both families'
/// inverse DFTs are unnormalized, so the `1/n` backward scale is applied
/// explicitly to match the DFTI contract on Intel.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl FftPlan {
    /// Try to plan the interleaved-complex family. Returns `None` when the
    /// length is unsupported (vDSP returns a null setup), signalling `create`
    /// to fall back to the split family.
    fn create_interleaved(
        length: nuvai_mkl_sys::vDSP_Length,
        single: bool,
    ) -> Option<FftBackend> {
        let complextocomplex = nuvai_mkl_sys::vDSP_DFT_Interleaved_ComplextoComplex;
        // SAFETY: `CreateSetup`/`CreateSetupD` take a null `prev` setup and the
        // positive `length`, returning a new setup (or null on failure). Each
        // returned non-null handle is owned and released exactly once below or
        // by `Drop`.
        let (forward, inverse) = unsafe {
            let forward = if single {
                nuvai_mkl_sys::vDSP_DFT_Interleaved_CreateSetup(
                    std::ptr::null_mut(),
                    length,
                    nuvai_mkl_sys::vDSP_DFT_FORWARD,
                    complextocomplex,
                )
            } else {
                nuvai_mkl_sys::vDSP_DFT_Interleaved_CreateSetupD(
                    std::ptr::null_mut(),
                    length,
                    nuvai_mkl_sys::vDSP_DFT_FORWARD,
                    complextocomplex,
                )
            };
            let inverse = if single {
                nuvai_mkl_sys::vDSP_DFT_Interleaved_CreateSetup(
                    std::ptr::null_mut(),
                    length,
                    nuvai_mkl_sys::vDSP_DFT_INVERSE,
                    complextocomplex,
                )
            } else {
                nuvai_mkl_sys::vDSP_DFT_Interleaved_CreateSetupD(
                    std::ptr::null_mut(),
                    length,
                    nuvai_mkl_sys::vDSP_DFT_INVERSE,
                    complextocomplex,
                )
            };
            (forward, inverse)
        };
        if forward.is_null() || inverse.is_null() {
            // Best-effort release of whatever half was allocated, using the
            // precision-matched destroy routine (as Drop does) — a double setup
            // released through the single-precision destroy leaks.
            // SAFETY: each pointer is null or a valid setup from the matching
            // `CreateSetup`/`CreateSetupD` call above, destroyed exactly once.
            unsafe {
                if single {
                    if !forward.is_null() {
                        nuvai_mkl_sys::vDSP_DFT_Interleaved_DestroySetup(forward);
                    }
                    if !inverse.is_null() {
                        nuvai_mkl_sys::vDSP_DFT_Interleaved_DestroySetup(inverse);
                    }
                } else {
                    if !forward.is_null() {
                        nuvai_mkl_sys::vDSP_DFT_Interleaved_DestroySetupD(forward);
                    }
                    if !inverse.is_null() {
                        nuvai_mkl_sys::vDSP_DFT_Interleaved_DestroySetupD(inverse);
                    }
                }
            }
            return None;
        }
        Some(FftBackend::Interleaved { forward, inverse })
    }

    /// Plan the split-complex family (fallback for lengths 2 and 4). Returns
    /// `None` when the length is unsupported.
    fn create_split(length: nuvai_mkl_sys::vDSP_Length, single: bool) -> Option<FftBackend> {
        // SAFETY: `CreateSetup`/`CreateSetupD` take a null `prev` setup and the
        // positive `length`, returning a new setup (or null on failure). Each
        // returned non-null handle is owned and released exactly once below or
        // by `Drop`.
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
            // SAFETY: as `create_interleaved` — null or valid, destroyed once.
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
            return None;
        }
        Some(FftBackend::Split {
            forward,
            inverse,
            scratch32: RefCell::new(None),
            scratch64: RefCell::new(None),
        })
    }

    fn transform_c32(&self, input: &[MKL_Complex8], output: &mut [MKL_Complex8], inverse: bool) {
        let n = self.len;
        match &self.handle.backend {
            FftBackend::Interleaved { forward, inverse: inv } => {
                let setup = if inverse { *inv } else { *forward };
                // `DSPComplex` is layout-identical to `MKL_Complex8` (both
                // `#[repr(C)] { real: f32, imag: f32 }`), so the caller's
                // interleaved buffers cast directly — no deinterleave copy.
                // SAFETY: `setup` is non-null; the pointers describe `n`
                // interleaved complex elements each, and the out-of-place
                // interleaved DFT reads `input` and writes the distinct `output`.
                unsafe {
                    nuvai_mkl_sys::vDSP_DFT_Interleaved_Execute(
                        setup,
                        input.as_ptr() as *const nuvai_mkl_sys::DSPComplex,
                        output.as_mut_ptr() as *mut nuvai_mkl_sys::DSPComplex,
                    );
                }
                // Forward needs no scale; inverse applies the `1/n`
                // normalization to match the DFTI contract on Intel.
                if inverse {
                    let scale = 1.0 / n as f32;
                    for y in output.iter_mut() {
                        y.real *= scale;
                        y.imag *= scale;
                    }
                }
            }
            FftBackend::Split {
                forward,
                inverse: inv,
                scratch32,
                ..
            } => {
                let mut scratch = scratch32.borrow_mut();
                let (re, im) = scratch.get_or_insert_with(|| (vec![0.0f32; n], vec![0.0f32; n]));
                for (i, x) in input.iter().enumerate() {
                    re[i] = x.real;
                    im[i] = x.imag;
                }
                let setup = if inverse { *inv } else { *forward };
                // SAFETY: `setup` is the non-null split setup; `re`/`im` are
                // length-`n` scratch arrays and `vDSP_DFT_Execute` reads and
                // writes exactly the first `n` split-complex elements.
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
        }
    }

    fn transform_c64(&self, input: &[MKL_Complex16], output: &mut [MKL_Complex16], inverse: bool) {
        let n = self.len;
        match &self.handle.backend {
            FftBackend::Interleaved { forward, inverse: inv } => {
                let setup = if inverse { *inv } else { *forward };
                // `DSPDoubleComplex` is layout-identical to `MKL_Complex16`
                // (both `#[repr(C)] { real: f64, imag: f64 }`), so the caller's
                // interleaved buffers cast directly — no deinterleave copy.
                // SAFETY: `setup` is non-null; the pointers describe `n`
                // interleaved complex elements each, and the out-of-place
                // interleaved DFT reads `input` and writes the distinct `output`.
                unsafe {
                    nuvai_mkl_sys::vDSP_DFT_Interleaved_ExecuteD(
                        setup,
                        input.as_ptr() as *const nuvai_mkl_sys::DSPDoubleComplex,
                        output.as_mut_ptr() as *mut nuvai_mkl_sys::DSPDoubleComplex,
                    );
                }
                if inverse {
                    let scale = 1.0 / n as f64;
                    for y in output.iter_mut() {
                        y.real *= scale;
                        y.imag *= scale;
                    }
                }
            }
            FftBackend::Split {
                forward,
                inverse: inv,
                scratch64,
                ..
            } => {
                let mut scratch = scratch64.borrow_mut();
                let (re, im) = scratch.get_or_insert_with(|| (vec![0.0f64; n], vec![0.0f64; n]));
                for (i, x) in input.iter().enumerate() {
                    re[i] = x.real;
                    im[i] = x.imag;
                }
                let setup = if inverse { *inv } else { *forward };
                // SAFETY: `setup` is the non-null split setup; `re`/`im` are
                // length-`n` scratch arrays and `vDSP_DFT_ExecuteD` reads and
                // writes exactly the first `n` split-complex elements.
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
            match &self.handle.backend {
                FftBackend::Interleaved { forward, inverse } => {
                    // SAFETY: `forward`/`inverse` are valid interleaved setups
                    // created non-null in `create_interleaved`, and each is
                    // destroyed exactly once here via the precision-matched
                    // destroy routine.
                    unsafe {
                        if self.handle.single {
                            nuvai_mkl_sys::vDSP_DFT_Interleaved_DestroySetup(*forward);
                            nuvai_mkl_sys::vDSP_DFT_Interleaved_DestroySetup(*inverse);
                        } else {
                            nuvai_mkl_sys::vDSP_DFT_Interleaved_DestroySetupD(*forward);
                            nuvai_mkl_sys::vDSP_DFT_Interleaved_DestroySetupD(*inverse);
                        }
                    }
                }
                FftBackend::Split { forward, inverse, .. } => {
                    // SAFETY: `forward`/`inverse` are valid split setups created
                    // non-null in `create_split`, and each is destroyed exactly
                    // once here via the precision-matched destroy routine.
                    unsafe {
                        if self.handle.single {
                            nuvai_mkl_sys::vDSP_DFT_DestroySetup(*forward);
                            nuvai_mkl_sys::vDSP_DFT_DestroySetup(*inverse);
                        } else {
                            nuvai_mkl_sys::vDSP_DFT_DestroySetupD(*forward);
                            nuvai_mkl_sys::vDSP_DFT_DestroySetupD(*inverse);
                        }
                    }
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
