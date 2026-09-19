# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — though
the crate is pre-1.0, so breaking changes are permitted without a major bump.

## [Unreleased]

### Added

- MKL error codes are decoded into descriptions (issue #33). `Error` now records
  which library's code space a status came from, and `Display` appends the
  meaning, so a sparse solve failure reads `MKL error (code -4): pardiso phase
  22 (factorization) — zero pivot; numerical factorization or iterative
  refinement failed` instead of leaving the caller to look the number up. The
  same text is available programmatically through the new `Error::description()`
  (non-breaking; `kind()` and `code()` are unchanged, and `description()` returns
  `None` for any code this crate has no meaning for, which still renders exactly
  as before). The tables are transcribed from the oneMKL 2026.1.0 headers
  `nuvai-mkl-sys` binds, and are keyed **per code space**: the libraries share no
  namespace and do not even agree on a sign — DFTI counts *up* from zero while
  VSL, PARDISO and Accelerate Sparse count *down* — so code `3` means three
  unrelated things depending on who returned it. LAPACK's `info` is deliberately
  not decoded: it is positional (the negated index of the offending argument, or
  "`U(i,i)` was exactly zero") rather than a code-to-text map.

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

- `lapack`'s `?gesv`/`?getrf` accept zero dimensions as the no-ops LAPACK
  defines them to be (issue #37). `n == 0` or `nrhs == 0` (`?gesv`) and `m == 0`
  or `n == 0` (`?getrf`) are LAPACK's documented *quick returns* — they read and
  write nothing — but the wrapper rejected them as `ErrorKind::InvalidArgument`,
  which also made `lapack` disagree with `blas`, whose `check_matrix` and
  `check_vector` have always accepted a zero dimension as a no-op. The wrapper
  now answers `Ok(())` itself rather than calling through, because LAPACK's
  quick return sits *after* its argument checks: a zero-order system handed to
  it with a small `lda` would abort through `XERBLA` rather than return. The
  leading dimensions are still validated (as LAPACK does, ahead of the quick
  return); the buffer *lengths* are not, since nothing is read or written.
  Negative dimensions remain `InvalidArgument`.

- The `len → c_int` conversions in `vsl`, `pardiso` and `dss` reject lengths
  above `i32::MAX` instead of truncating them (issue #25), matching the guard
  `vml` already had. `len_to_c_int` (`src/conv.rs`) is now the single
  implementation of that rule. A truncated length would have the MKL routine
  process only a prefix of the caller's buffer while still returning success, so
  the symptom is a silently partly-computed result rather than an error. The
  guard is unreachable through any allocatable slice (8 GiB of `f32`), which is
  why it lives in one testable function rather than as inline casts.

- `nuvai-mkl-sys` scopes its lint suppressions to the code that needs them
  (issue #36), replacing nine `#![allow(...)]`s at the crate root that also
  covered the hand-written FFI surfaces. The generated bindgen output keeps its
  set inside a new `mod bindings`; each hand-written surface now carries only
  the three C-naming lints its SDK-mirroring identifiers require. In particular
  `clippy::all` and `improper_ctypes` no longer cover hand-reviewed code —
  removing them surfaced a redundant `?Sized` bound in the `AmbiguousIfCopy`
  compile-time guard, which is fixed rather than suppressed. `deref_nullptr`,
  `unused_imports` and `suspicious_runtime_symbol_definitions` are dropped as
  never-firing (the last covers *definitions* of runtime symbols, and this
  workspace defines none).

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

### Documentation

- The `pardiso` phase-33 `SAFETY` comment now states the invariant behind
  casting the caller's `&[f64]` right-hand side from `*const` to `*mut` (issue
  #29). oneMKL's prototype declares `b` as plain `void *` with no `const`
  variant (`mkl_pardiso.h`), so the cast exists only to satisfy the ABI while
  the phase-33 contract reads it — `a` is declared `const void *` and is passed
  as `*const c_void` for the same read-only reason. The comment records what
  would invalidate it (an `iparm` option making PARDISO write back into `b`) and
  what to do instead if that ever becomes necessary.

- The `blas` module documents why its `Result` is load-bearing (issue #38). The
  issue suspected the `Result<()>`s of `sgemm`/`dgemm`/`saxpy`/… were vestigial
  and proposed dropping them; they had not been vestigial since #20 made every
  routine validate its operands, and dropping them would put the heap
  out-of-bounds accesses back within reach of safe code, because CBLAS does not
  check its own bounds. No code change: the contract is now written down, and
  the issue records why.
