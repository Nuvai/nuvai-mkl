//! VSL — Vector Statistical Library: random-number generation.
//!
//! On Intel targets [`Stream`] wraps a VSL random stream (MT19937 by default).
//! On Apple Silicon (`aarch64-apple-darwin`) it wraps a ChaCha20 stream from
//! `rand_chacha`, seeded via `rand`'s `SeedableRng` (ADR-0003 decision 6: the
//! sequence is statistically valid but not identical to Intel VSL). Both
//! backends expose the same [`Stream`] API and mutate on `&self` (VSL streams
//! are internally mutable; the aarch64 backend uses [`std::cell::RefCell`]).
//! On `aarch64-unknown-linux-gnu` there is no VSL backend, so every method
//! returns [`ErrorKind::Unsupported`]. Streams are not `Sync`; share one behind
//! a lock when cross-thread randomness is required.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::cell::RefCell;
use std::marker::PhantomData;
#[cfg(not(target_arch = "aarch64"))]
use std::os::raw::c_int;
#[cfg(not(target_arch = "aarch64"))]
use std::ptr;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use rand::RngExt;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use rand::SeedableRng;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use rand_chacha::ChaCha20Rng;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use rand_distr::{Distribution, Normal};

use crate::error::{Error, Result};

/// Opaque stream state. On Intel this is the VSL stream pointer; on aarch64 it
/// is a ChaCha20 RNG behind `RefCell` for interior mutability on `&self`.
#[cfg(not(target_arch = "aarch64"))]
type StreamState = nuvai_mkl_sys::VSLStreamStatePtr;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
type StreamState = RefCell<ChaCha20Rng>;
/// No VSL backend on `aarch64-unknown-linux-gnu`: [`Stream`] is never
/// constructed there (every method returns [`ErrorKind::Unsupported`]).
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
type StreamState = ();

/// A random-number stream.
// On `aarch64-unknown-linux-gnu` no `Stream` can be constructed, so `state` is
// never read there.
#[cfg_attr(all(target_os = "linux", target_arch = "aarch64"), allow(dead_code))]
pub struct Stream {
    state: StreamState,
    /// Pin auto-trait parity across backends. The Intel backend's raw
    /// `VSLStreamStatePtr` is `!Send + !Sync`, but the aarch64
    /// `RefCell<ChaCha20Rng>` would be `Send` (still `!Sync`). A raw-pointer
    /// `PhantomData` is `!Send + !Sync`, so it makes the aarch64 backend match
    /// Intel instead of leaking `Send` on only one platform.
    _not_send_sync: PhantomData<*const ()>,
}

impl Stream {
    /// Create a new stream seeded with `seed`.
    ///
    /// On Intel this is VSL's MT19937 BRNG; on aarch64 it is a ChaCha20 RNG
    /// seeded deterministically from `seed` (both reproduce their sequence for
    /// the same seed, but the two platforms do not produce identical streams).
    pub fn new(seed: u32) -> Result<Self> {
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            // No VSL backend on aarch64-unknown-linux-gnu.
            let _ = seed;
            Err(Error::unsupported_linux_aarch64("VSL"))
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let rng = ChaCha20Rng::seed_from_u64(seed as u64);
            Ok(Self {
                state: RefCell::new(rng),
                _not_send_sync: PhantomData,
            })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut state: nuvai_mkl_sys::VSLStreamStatePtr = ptr::null_mut();
            // SAFETY: `state` is a valid out-param; `VSL_BRNG_MT19937` and
            // `seed` are valid arguments; on `status == 0` `state` holds a
            // valid VSL stream (checked below).
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
            Ok(Self {
                state,
                _not_send_sync: PhantomData,
            })
        }
    }

    /// Fill `out` with uniforms in `[a, b)` (single precision).
    pub fn uniform(&self, a: f32, b: f32, out: &mut [f32]) -> Result<()> {
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            // No VSL backend on aarch64-unknown-linux-gnu. Return Unsupported
            // before validating the range so feature-detection callers get the
            // documented error regardless of their arguments.
            let _ = (a, b, out);
            Err(Error::unsupported_linux_aarch64("VSL"))
        }
        #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
        // `a..b` is empty (or NaN) when `a >= b`: Intel VSL returns
        // `VSL_ERROR_BADARGS`, but the aarch64 `rand` backend would panic.
        // Reject it up front so both backends fail identically with an error.
        // `partial_cmp` is `None` for NaN and `Some(Greater|Equal)` for `a >= b`,
        // so only `Some(Less)` is an acceptable non-empty range.
        if a.partial_cmp(&b) != Some(std::cmp::Ordering::Less) {
            return Err(Error::invalid("uniform: a must be < b (empty range)"));
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let mut rng = self.state.borrow_mut();
            for v in out.iter_mut() {
                *v = rng.random_range(a..b);
            }
            Ok(())
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let n = out.len() as c_int;
            // SAFETY: `self.state` is a valid stream from `new`; `out` is a
            // mutable slice of `n` elements; the method constant and `a`/`b`
            // (validated `a < b` above) are valid arguments.
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
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            // No VSL backend on aarch64-unknown-linux-gnu. Return Unsupported
            // before validating the range so feature-detection callers get the
            // documented error regardless of their arguments.
            let _ = (a, b, out);
            Err(Error::unsupported_linux_aarch64("VSL"))
        }
        #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
        // See `uniform`: reject the empty/NaN range that would panic the
        // aarch64 `rand` backend, matching Intel VSL's error behaviour.
        if a.partial_cmp(&b) != Some(std::cmp::Ordering::Less) {
            return Err(Error::invalid("uniform64: a must be < b (empty range)"));
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let mut rng = self.state.borrow_mut();
            for v in out.iter_mut() {
                *v = rng.random_range(a..b);
            }
            Ok(())
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let n = out.len() as c_int;
            // SAFETY: `self.state` is a valid stream; `out` is a mutable slice
            // of `n` elements; the method constant and `a`/`b` (validated
            // `a < b` above) are valid arguments.
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
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            // No VSL backend on aarch64-unknown-linux-gnu.
            let _ = (mean, sigma, out);
            Err(Error::unsupported_linux_aarch64("VSL"))
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let distr = Normal::new(mean, sigma)
                .map_err(|e| Error::invalid(format!("gaussian: {e}")))?;
            let mut rng = self.state.borrow_mut();
            for v in out.iter_mut() {
                *v = distr.sample(&mut *rng);
            }
            Ok(())
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let n = out.len() as c_int;
            // SAFETY: `self.state` is a valid stream; `out` is a mutable slice
            // of `n` elements; the method constant and `mean`/`sigma` are valid
            // arguments.
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
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            // No VSL backend on aarch64-unknown-linux-gnu.
            let _ = (mean, sigma, out);
            Err(Error::unsupported_linux_aarch64("VSL"))
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let distr = Normal::new(mean, sigma)
                .map_err(|e| Error::invalid(format!("gaussian: {e}")))?;
            let mut rng = self.state.borrow_mut();
            for v in out.iter_mut() {
                *v = distr.sample(&mut *rng);
            }
            Ok(())
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let n = out.len() as c_int;
            // SAFETY: `self.state` is a valid stream; `out` is a mutable slice
            // of `n` elements; the method constant and `mean`/`sigma` are valid
            // arguments.
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
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            // No VSL backend on aarch64-unknown-linux-gnu: a `Stream` can
            // never be constructed there, so there is nothing to release.
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            // `ChaCha20Rng` is owned and needs no teardown.
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // SAFETY: `self.state` is a valid stream created in `new` and is
            // deleted exactly once here.
            unsafe {
                nuvai_mkl_sys::vslDeleteStream(&mut self.state);
            }
        }
    }
}
