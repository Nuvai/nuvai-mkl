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

### Fixed

- `nuvai-mkl-src` now fails loudly when both the `accelerate` and `openblas`
  backend features are enabled on `aarch64-apple-darwin` instead of silently
  resolving to OpenBLAS (issue #35). Enabling both features is as ambiguous as
  enabling neither: the selection now triggers a `compile_error!` at library
  compile time and an `Err` from `backend_for_target` at build-script time (the
  host-compiled path used when cross-compiling, where the library
  `compile_error!` cannot fire), matching ADR-0003 "selection is explicit,
  never silent".

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
