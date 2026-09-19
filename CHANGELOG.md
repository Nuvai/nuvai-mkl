# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — though
the crate is pre-1.0, so breaking changes are permitted without a major bump.

## [Unreleased]

### Changed

- **Breaking:** every `blas` routine now validates its operands and returns
  `Result` before any pointer reaches CBLAS (issue #20). `sdot`/`ddot` change
  from `f32`/`f64` to `Result<f32>`/`Result<f64>`; `saxpy`/`daxpy`/`sscal`/
  `dscal`/`sgemm`/`dgemm` change from `()` to `Result<()>`, so calls that
  previously could not fail may now return `ErrorKind::InvalidArgument`.

- Level-1 strides and counts are now validated strictly: `n < 0`, `inc == 0`,
  and `inc < 0` return `ErrorKind::InvalidArgument` instead of the no-op /
  broadcast behavior of reference BLAS, matching the crate's LAPACK validation
  which already rejects non-positive dimensions.

- **Breaking:** `ErrorKind` gains a `ResourceExhausted` variant (issue #31).
  `ErrorKind` is a public enum without `#[non_exhaustive]`, so any downstream
  `match` over it needs a new arm. It is a separate category from `Unsupported`
  because the two call for opposite responses: `Unsupported` means the backend
  cannot do this at all, while `ResourceExhausted` means it can, but a resource
  could not be allocated right now — retrying or shrinking the problem is
  sensible for the latter and futile for the former.

### Fixed

- The MKL runtime's implicit dependencies are now forced into the test binaries
  by an object file, not by linker flags alone (issue #44). `libmkl_core.so.3`
  calls `log`/`exp`/`sin`/… and `libmkl_intel_thread.so.3` calls `omp_*` without
  declaring a `DT_NEEDED` for libm or for the OpenMP runtime, so both have to
  reach the process-global scope through the executable's own `DT_NEEDED` —
  otherwise MKL's shared objects cannot resolve them and the binary aborts at
  load time. The `-Wl,--no-as-needed` flags that were meant to keep them are
  honoured by GNU ld and lld but silently discarded by mold, which records `-l`
  inputs by *resolved file* and drops a repeat mention — however it is spelled
  — before ever consulting the as-needed state. On such a linker the test suite
  died at load time with `undefined symbol: omp_in_parallel` (and, before that,
  could look like it had "mostly passed", because the tests that ran first never
  called into MKL). `crates/nuvai-mkl/build/force_runtime.c` now leaves those
  symbols undefined in a regular object file, which every linker honours; the
  existing flags are kept for GNU ld and lld.

- `blas` level-1 routines reject negative strides. A negative stride was
  previously accepted but walked *below* the slice (the wrapper passes the
  slice's first element as the CBLAS base), making heap out-of-bounds
  reads/writes reachable from safe code.

- `Dss::solve` validates `rhs.len()` against the factored dimension `n` on
  every backend (issue #21). The Intel arm sized the solution buffer to
  `rhs.len()` while `dss_solve_real_` writes `n` elements, so an undersized
  RHS was a heap out-of-bounds write reachable from safe code and an oversized
  one returned a `Vec` padded past the values actually solved for. `n` is now
  captured at factor time (the Intel DSS handle is opaque, so it cannot be
  recovered at solve time) and the check runs ahead of the `cfg` dispatch
  rather than only on the aarch64 arm. A mismatched length now returns
  `ErrorKind::InvalidArgument`.

- `FftPlan::new` no longer reports resource exhaustion as
  `ErrorKind::Unsupported` on Apple Silicon (issue #31). When a length is one
  vDSP *can* plan but the setup handle comes back null, the failure was
  classified `Unsupported` — telling callers to give up on a length that works,
  and contradicting the comment sitting directly above the code. Both arms
  (interleaved and split) now return `ErrorKind::ResourceExhausted`. A genuinely
  unplannable length still returns `Unsupported`.
