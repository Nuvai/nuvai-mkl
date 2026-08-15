# ADR-0003: Apple Silicon backend selection for nuvai-mkl

- **Status:** Accepted
- **Date:** 2026-08-13
- **Epic:** #1 — Roadmap: complete the nuvai-mkl wrapper
- **Task:** #3 — Apple Silicon (aarch64-apple-darwin) fallback
- **Deciders:** nuvai-mkl maintainers

## Context

Intel ships no oneMKL build for Apple Silicon (`aarch64-apple-darwin`). The
`nuvai-mkl-src` acquisition path (`MKLROOT` / conda-forge) only resolves Intel
MKL, so the wrapper cannot link on Apple Silicon. Task #3 maps each MKL domain
to an Apple Silicon-native backend behind the existing safe-wrapper API.

## Decision

1. **Single link-provider stays.** `nuvai-mkl-src` keeps `links = "mkl"` and
   remains the sole linker-directive emitter for *both* platforms. On
   `aarch64-apple-darwin` its `build.rs` emits
   `cargo:rustc-link-lib=framework=Accelerate` (default) or `-lopenblas`
   (`openblas` feature) instead of `mkl_rt`; the x86_64 MKL path is
   byte-identical. No second `links` crate, no per-target provider crates.

2. **Backend selection = `cfg` (mandatory reality) + Cargo features (choice),
   never silent.** aarch64 has no MKL, so the non-MKL path is gated by
   `cfg(target_arch = "aarch64")` (mandatory). Where multiple backends exist the
   choice is a Cargo feature: `accelerate` (default on aarch64), `openblas`
   (BLAS/LAPACK), `rustfft` (FFT), `rand` (VSL). A domain with no selected
   backend must fail to compile or return `ErrorKind::Unsupported` — never
   degrade silently.

3. **`nuvai-mkl-sys` aarch64 FFI = hand-written `extern "C"`** (cblas, Fortran
   LAPACK `_`, vDSP DFT, vForce, Sparse/SparseSolve), not a second bindgen pass
   over `Accelerate.framework`. The surface is small and bounded; hand-written
   externs avoid SDK-header bindgen fragility and keep the Intel bindgen output
   untouched.

4. **BLAS drops in via symbol aliasing.** Accelerate's CBLAS uses the same
   symbol names and CBLAS enum values as MKL (both netlib CBLAS ABI). On aarch64
   `nuvai-mkl-sys` re-exports Accelerate's `cblas_sgemm` etc. under the same
   names, so `nuvai-mkl::blas` is unchanged.

5. **LAPACK needs a RowMajor shim.** Accelerate exposes only Fortran `_` entry
   points (`sgesv_`, `dgesv_`, `sgetrf_`, `dgetrf_`), no LAPACKE. On aarch64 the
   existing `lapack::*` functions translate `Layout::RowMajor` by transposing
   into column-major buffers, preserving signatures. (`Transpose::as_char` is
   reserved for this.)

6. **Sparse direct requires CSR→CSC transpose.** Accelerate
   `Sparse/SparseSolve` is column-major CSC; PARDISO/DSS inputs are CSR. The
   aarch64 backend transposes CSR→CSC (`A_csr` row-major ⇔ `Aᵀ_csc`), uses
   `SparseFactor` (Cholesky for symmetric-PD DSS, QR for nonsymmetric PARDISO)
   + `SparseSymmetricSolve`/`SparseSolve`.

7. **VSL RNG is statistically-, not sequence-compatible.** MKL MT19937 and
   `rand_chacha` are not sequence-identical; `Stream::new(seed)` will not
   reproduce MKL's exact draws on aarch64. Documented semantic difference
   (smoke tests assert only statistical properties).

## Consequences

- The x86_64 Intel MKL path is untouched and `main` stays green per
  trunk-based development; all aarch64 work is `cfg(target_arch = "aarch64")`-gated.
- Backend selection is explicit and queryable via `nuvai_mkl_src::backend()`.
- A domain with no selected backend returns `ErrorKind::Unsupported` rather than
  silently degrading.
- Follow-up (not blocking): a bindgen pass for Accelerate may replace the
  hand-written externs if a symbol beyond the current set is ever needed.

## Extension (2026-08-15): aarch64-unknown-linux-gnu (task #9)

Intel ships no oneMKL for ARM64 Linux either, so the same explicit-backend model
extends to `aarch64-unknown-linux-gnu`:

8. **OpenBLAS is the sole backend on aarch64-unknown-linux-gnu.** Unlike Apple
   Silicon there is no OS-provided multi-domain framework: OpenBLAS covers only
   BLAS/LAPACK. The backend is therefore selected unconditionally by
   `cfg(all(target_os = "linux", target_arch = "aarch64"))` — the
   `accelerate`/`openblas` Cargo features are no-ops there — and `nuvai-mkl-src`
   emits `-lopenblas` (adding an `OPENBLAS_ROOT/lib` rpath when `OPENBLAS_ROOT`
   is set, for conda/pip installs).

9. **Unsupported domains return `ErrorKind::Unsupported`.** FFT (no DFTI/vDSP),
   VML (no vForce), VSL (no `rand` backend is wired on Linux), PARDISO and DSS
   (no Sparse/SparseSolve) have no OpenBLAS equivalent; every entry point returns
   `ErrorKind::Unsupported` with an explicit message — never degrade silently
   (decision 2).

10. **The aarch64 LAPACK shim and CBLAS aliasing are shared.** OpenBLAS exposes
    the same netlib Fortran `_` entry points (`sgesv_`/`dgesv_`/`sgetrf_`/
    `dgetrf_`) as Accelerate, so the decision-5 RowMajor transpose shim is reused
    unchanged, and the decision-4 CBLAS symbol aliasing applies verbatim.

### Consequences (extension)

- `nuvai-mkl` on linux-aarch64 covers BLAS/LAPACK numerically against OpenBLAS
  and returns `ErrorKind::Unsupported` for FFT/VML/VSL/PARDISO/DSS.
- The hand-written `nuvai-mkl-sys` surface for this target (`linux_aarch64.rs`)
  mirrors the Apple Silicon one but is limited to CBLAS + Fortran LAPACK `_`.
- A native ARM64 CI job (`ubuntu-24.04-arm` + `libopenblas-dev`) exercises the
  numerical and Unsupported paths on every PR/push.
