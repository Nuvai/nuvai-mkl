// Hand-written FFI surface for `aarch64-unknown-linux-gnu`.
//
// Intel ships no oneMKL build for `aarch64-unknown-linux-gnu`, so
// `nuvai-mkl-sys` cannot generate bindings from the oneMKL headers on this
// target. Instead this module declares the small, bounded set of C symbols the
// OpenBLAS backend calls — netlib CBLAS and the Fortran LAPACK `_` entry
// points — directly against OpenBLAS. The Intel x86_64 bindgen output is
// untouched, and no second bindgen pass is run over the OpenBLAS headers
// (ADR-0003, decision 3).
//
// OpenBLAS covers only BLAS/LAPACK on this target. There is no vDSP/vForce/
// Sparse/SparseSolve equivalent, so this surface deliberately omits those
// symbols; the safe wrapper returns `ErrorKind::Unsupported` for FFT/VML/VSL/
// PARDISO/DSS rather than linking anything absent here (never degrade
// silently).
//
// The CBLAS symbols are re-exported under the *same names* as the Intel
// bindgen output because both MKL and OpenBLAS implement the netlib CBLAS
// ABI; the safe wrapper's `blas` module therefore needs no per-backend
// dispatch (see ADR-0003, decision 4). LAPACK is exposed through the Fortran
// `_` entry points (`SUBROUTINE`s returning `void`, status via `*info`), the
// same ABI as the Apple Silicon Accelerate surface.
//
// Type layouts: `c_int` = i32, CBLAS enums = `u32` (non-Windows), matching the
// non-Windows path used by `nuvai-mkl::blas` and `nuvai-mkl::lapack`.

// One suppression: the shared `netlib_abi.rs` declarations keep the CBLAS
// spelling of their constants (`CblasRowMajor`, not `CBLAS_ROW_MAJOR`). That
// fragment's only other named items are `MKL_Complex8`/`MKL_Complex16` and the
// `cblas_*`/`?gesv_`/`?getrf_` functions, which are Rust-cased already —
// `non_camel_case_types` does not fire on those type names (measured directly:
// compiling this fragment's declarations standalone warns on the five CBLAS
// constants and nothing else), and `non_snake_case` has no camel-cased field to
// fire on here, unlike on the Apple Silicon surface. Everything else the crate
// root used to suppress is gone (#36).
#![allow(non_upper_case_globals)]

// The netlib CBLAS + Fortran-LAPACK declarations are shared verbatim with the
// Apple Silicon surface (`aarch64.rs`) — both Accelerate and OpenBLAS
// implement the netlib CBLAS ABI and expose LAPACK through the `_` entry
// points, so one declaration set serves both (see `netlib_abi.rs` for the full
// ABI notes). This module adds only the target-specific diagnostics/tests.
include!("netlib_abi.rs");

#[cfg(test)]
mod tests {
    use super::*;

    /// Link-level smoke test: exercises a real OpenBLAS symbol so the P2 gate
    /// ("sys links clean on aarch64-linux") proves the `-lopenblas` link, not
    /// merely that the externs type-check.
    #[test]
    fn openblas_cblas_links() {
        let x = [1.0f32, 2.0, 3.0];
        let y = [4.0f32, 5.0, 6.0];
        // SAFETY: `x`/`y` have exactly 3 elements and `incx`/`incy` are 1, so
        // `cblas_sdot(3, …)` reads exactly the three elements of each array.
        let dot = unsafe { cblas_sdot(3, x.as_ptr(), 1, y.as_ptr(), 1) };
        assert!((dot - 32.0).abs() < 1e-6, "cblas_sdot = {dot}");
    }
}
