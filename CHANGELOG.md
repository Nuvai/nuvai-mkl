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
  unrelated things depending on who returned it. LAPACK's `info` is decoded
  too, by rule rather than table since it is positional: a negative value is
  reported as the index of the argument that failed (`argument 4 had an illegal
  value`), and a positive one as `U(i,i) is exactly zero` — the singular-factor
  case, which for `?gesv` means no solution was computed.

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

- **Breaking:** `nuvai-mkl-src`'s acquisition dependencies are build-only, and the
  acquisition itself has left the library target (issue #24).
  `ureq`/`zip`/`zstd`/`tar`/`sha2` were declared in `[dependencies]` as well as
  `[build-dependencies]`, which put them — and their trees, `rustls`, `ring`,
  `zstd-sys`, `bzip2-sys`, seven `icu_*` crates and the rest, 114 crates in all —
  in the *runtime* dependency graph of every downstream build, though nothing at
  runtime referenced them. They were compiled only because `lib.rs`
  `include!`d `acquire.rs` wholesale, and no `cfg` distinguishes "I am the
  library target" from "I am a build script" — so a Cargo feature could not have
  fixed it: with the feature off, `nuvai-mkl-src`'s own build script loses
  `locate()` and stops compiling. The acquisition source is therefore split:
  `src/mkl_info.rs` holds `MKL_VERSION`, `MklInfo` and the new
  `MklInfo::from_build_metadata()`, which rebuilds one from the `DEP_MKL_*`
  variables Cargo forwards from this crate's build script, and `src/acquire.rs`
  is `include!`d by `build.rs` alone. `pub fn locate()` is gone from the library
  target, and the build scripts that called it now read the metadata back
  instead. Measured on `aarch64-apple-darwin`, the runtime graph of
  `nuvai-mkl-sys` falls from 114 crates to two — itself and a dependency-free
  `nuvai-mkl-src`. Two consequences worth naming: this is a breaking change to
  `nuvai-mkl-src`'s public API, and acquisition now runs **once per build graph**
  rather than once per dependent. The three `locate()` calls it replaces each
  re-read and re-verified 143 MB of cached conda archives against their pinned
  digests.

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
  compile-time guard, which is fixed rather than suppressed. `deref_nullptr`
  and `unused_imports` are dropped as never-firing;
  `suspicious_runtime_symbol_definitions` moves onto the generated module it
  actually applies to, where bindgen's `malloc`/`realloc` declarations trip it
  by declaring `c_ulong` where the symbol's signature expects `usize`.

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

- The build metadata `nuvai-mkl-src` publishes to dependent build scripts was
  never readable. It emitted `cargo:metadata=KEY=VALUE`, which names a directive
  in neither of Cargo's forms — those are `cargo::metadata=KEY=VALUE` and the
  legacy `cargo:KEY=VALUE` — so Cargo recorded every line as legacy metadata
  under the literal key `metadata`, and the paths arrived as
  `DEP_MKL_METADATA="BACKEND=…"`. None of `DEP_MKL_BACKEND`, `_INCLUDE_DIR`,
  `_LIB_DIR`, `_DLL_DIR` or `_VERSION` existed. It went unnoticed because nothing
  read them: the Windows DLL directories are still re-derived in CI by globbing
  the cache layout, which is a `PATH` change for the test process and so not
  something a build script could do for anyone. Relatedly, Cargo keeps only the
  *last* value for a repeated metadata key, so the `DLL_DIR` loop published
  exactly one directory — whichever happened to be extracted last. The keys are
  indexed now and carry a `DLL_DIR_COUNT` the reader requires, so a list cannot
  be silently truncated into a missing `PATH` entry, and `OMP_LIB_DIR` is
  published for the first time.

- `nuvai-mkl-src` acquired for the host rather than the target. `acquire.rs`
  chose its platform with `#[cfg(target_arch = "aarch64")]` and
  `cfg!(target_os = ...)`, and a build script compiles for and runs on the
  *host*, so both described the wrong machine. Cross-compiling to an Intel target
  from Apple Silicon selected `IntelMkl` from the target triple (correctly) and
  then took `locate()`'s aarch64 *panic* arm; cross-compiling to a Windows target
  from Linux fetched **linux-64** conda packages. Every decision now dispatches
  on `CARGO_CFG_TARGET_OS` / `CARGO_CFG_TARGET_ARCH`, the pattern the three
  `build.rs` files already use to select a backend, and the host `#[cfg]` gates
  are gone. Verified end to end on Apple Silicon:
  `cargo check -p nuvai-mkl-src --target x86_64-unknown-linux-gnu` now completes,
  emitting `mkl_rt`/`iomp5` link directives and the rpath from a real linux-64
  acquisition, where it previously died at this build script. The remaining
  obstacle to a *workspace-wide* cross-check is `nuvai-mkl-sys`, whose bindgen
  invocation is still selected by the host cfg.

- `pardiso.rs`'s `csr_to_csc` no longer narrows `nnz` to `i32`. It compared
  `row_index[n]` against `nnz as i32 + base`, which truncates above `i32::MAX` —
  letting an invalid CSR pass validation and then index `row_indices` out of
  bounds in the transpose loop — and overflows `i32` at exactly `nnz == i32::MAX`,
  a panic under `overflow-checks` and therefore in every debug build. `nnz` is not
  a length handed to C as a 32-bit count, so the comparison moves to `i64` rather
  than being guarded, and the two remaining `usize`→`i64` conversions record why
  they cannot lose anything. The preconditions the function indexes on unchecked
  (`row_index.len() == n + 1`, `values.len() == columns.len()`) are now
  debug-asserted. Flagged as a follow-up at the bottom of PR #59.

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