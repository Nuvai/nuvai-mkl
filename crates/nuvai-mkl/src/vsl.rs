//! VSL — Vector Statistical Library: random-number generation.
//!
//! On Intel targets [`Stream`] wraps a VSL random stream. On Apple Silicon
//! (`aarch64-apple-darwin`) the `rand`/`rand_chacha` backend is wired in a
//! later phase; this module currently reports
//! [`ErrorKind::Unsupported`](crate::error::ErrorKind) rather than degrading
//! silently (ADR-0003, decision 2). Streams hold raw state and are therefore
//! neither `Send` nor `Sync`; share one behind a lock when cross-thread
//! randomness is required.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use std::os::raw::c_int;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use std::ptr;

use crate::error::{Error, Result};

/// Opaque stream state. On Intel this is the VSL stream pointer; on aarch64 it
/// is unused (the `rand` backend is not wired yet).
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
type StreamState = nuvai_mkl_sys::VSLStreamStatePtr;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
type StreamState = *mut std::os::raw::c_void;

/// A random-number stream.
pub struct Stream {
    state: StreamState,
}

impl Stream {
    /// Create a new MT19937 stream seeded with `seed`.
    pub fn new(seed: u32) -> Result<Self> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let _ = seed;
            return Err(Error::unsupported(
                "VSL on aarch64 requires the rand/rand_chacha backend (not yet wired)",
            ));
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let mut state: nuvai_mkl_sys::VSLStreamStatePtr = ptr::null_mut();
            let status = unsafe {
                nuvai_mkl_sys::vslNewStream(
                    &mut state,
                    nuvai_mkl_sys::VSL_BRNG_MT19937 as c_int,
                    seed,
                )
            };
            if status != 0 {
                return Err(Error::mkl(status, "vslNewStream"));
            }
            Ok(Self { state })
        }
    }

    /// Fill `out` with uniforms in `[a, b)` (single precision).
    pub fn uniform(&self, a: f32, b: f32, out: &mut [f32]) -> Result<()> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let _ = (a, b, out);
            return Err(Error::unsupported(
                "VSL on aarch64 requires the rand/rand_chacha backend (not yet wired)",
            ));
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let n = out.len() as c_int;
            let status = unsafe {
                nuvai_mkl_sys::vsRngUniform(
                    nuvai_mkl_sys::VSL_RNG_METHOD_UNIFORM_STD as c_int,
                    self.state,
                    n,
                    out.as_mut_ptr(),
                    a,
                    b,
                )
            };
            if status != 0 {
                return Err(Error::mkl(status, "vsRngUniform"));
            }
            Ok(())
        }
    }

    /// Fill `out` with uniforms in `[a, b)` (double precision).
    pub fn uniform64(&self, a: f64, b: f64, out: &mut [f64]) -> Result<()> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let _ = (a, b, out);
            return Err(Error::unsupported(
                "VSL on aarch64 requires the rand/rand_chacha backend (not yet wired)",
            ));
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let n = out.len() as c_int;
            let status = unsafe {
                nuvai_mkl_sys::vdRngUniform(
                    nuvai_mkl_sys::VSL_RNG_METHOD_UNIFORM_STD as c_int,
                    self.state,
                    n,
                    out.as_mut_ptr(),
                    a,
                    b,
                )
            };
            if status != 0 {
                return Err(Error::mkl(status, "vdRngUniform"));
            }
            Ok(())
        }
    }

    /// Fill `out` with normals `N(mean, sigma²)` (single precision).
    pub fn gaussian(&self, mean: f32, sigma: f32, out: &mut [f32]) -> Result<()> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let _ = (mean, sigma, out);
            return Err(Error::unsupported(
                "VSL on aarch64 requires the rand/rand_chacha backend (not yet wired)",
            ));
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let n = out.len() as c_int;
            let status = unsafe {
                nuvai_mkl_sys::vsRngGaussian(
                    nuvai_mkl_sys::VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2 as c_int,
                    self.state,
                    n,
                    out.as_mut_ptr(),
                    mean,
                    sigma,
                )
            };
            if status != 0 {
                return Err(Error::mkl(status, "vsRngGaussian"));
            }
            Ok(())
        }
    }

    /// Fill `out` with normals `N(mean, sigma²)` (double precision).
    pub fn gaussian64(&self, mean: f64, sigma: f64, out: &mut [f64]) -> Result<()> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let _ = (mean, sigma, out);
            return Err(Error::unsupported(
                "VSL on aarch64 requires the rand/rand_chacha backend (not yet wired)",
            ));
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let n = out.len() as c_int;
            let status = unsafe {
                nuvai_mkl_sys::vdRngGaussian(
                    nuvai_mkl_sys::VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2 as c_int,
                    self.state,
                    n,
                    out.as_mut_ptr(),
                    mean,
                    sigma,
                )
            };
            if status != 0 {
                return Err(Error::mkl(status, "vdRngGaussian"));
            }
            Ok(())
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            // No VSL stream on aarch64 (the rand backend is not wired yet).
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            unsafe {
                nuvai_mkl_sys::vslDeleteStream(&mut self.state);
            }
        }
    }
}
