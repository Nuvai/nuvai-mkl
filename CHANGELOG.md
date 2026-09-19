# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — though
the crate is pre-1.0, so breaking changes are permitted without a major bump.

## [Unreleased]

### Added

- The conda-forge base URL `nuvai-mkl-src` fetches from is overridable through
  `NUVAI_MKL_CONDA_BASE`, so a mirror can stand in when conda-forge's CDN
  refuses a request (issue #19). Every archive is still checked against the
  pinned SHA-256 it is fetched under, so a mirror cannot substitute different
  bytes — it changes where the bytes come from, not what is accepted.

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

- A refused MKL download is reported as a refusal instead of as a corrupt cache,
  and is retried (issue #19). The CDN in front of conda-forge answered CI with
  `2xx` bodies it would not serve — sometimes **0 bytes**, sometimes a few dozen
  bytes of something else entirely — and `download()` wrote whatever arrived
  straight to disk, so the checksum check then reported `checksum mismatch …
  Delete <path> and retry`. That sent the reader to clear a cache that was never
  the problem, discarded the HTTP status, and said nothing about what had
  actually been received. Downloads now stage into a per-process `.part` file,
  are checked against the pinned digest **before** being installed at their final
  name, and the whole attempt — digest included — is retried with backoff: the
  digest is the only check that recognises the second shape of refusal, so
  retrying the transfer alone would accept the garbage. A failure names the
  received length, which is what separates a short body from other content
  altogether.

  The staging file is per-process because `nuvai-mkl-src` is both a dependency
  and a build-dependency of the crates above it: Cargo builds it as two units and
  runs **two** build scripts against this one cache directory, so on CI both
  fetch the same archives at the same time. A shared staging name lets them
  truncate each other mid-download, and removing the final archive after a failed
  attempt lets a unit whose fetch was refused delete the archive the other one
  had just verified — which surfaced as `open .conda: NotFound` during
  extraction, from a process that had itself downloaded successfully. Nothing on
  this path deletes an archive any more; a rejected attempt leaves only its own
  staging file, which it cleans up.

- Enabling both backend features on Apple Silicon is a compile error instead of
  a silent choice (issue #35). `--features accelerate,openblas` resolved to
  OpenBLAS — the `openblas` arm was tested first in both selectors — with no
  diagnostic of any kind, which is the silent selection ADR-0003 forbids and the
  same violation the no-feature case was already rejected for. A `compile_error!`
  now rejects it, and `backend_for_target` returns the equivalent `Err` for
  build scripts, which are host-compiled and cannot see the target. Both
  selectors delegate to one `macos_aarch64_backend`, an exhaustive match over
  the four feature pairs, so the both-features case is now tested for real:
  asserted through `backend_for_target` it could only ever have observed
  whichever pair the test build happened to be compiled with, and the assertion
  would have passed without testing anything.

- `accelerate` and `openblas` were not actually mutually exclusive before
  (issue #35), which is the other half of why the case above went unnoticed.
  `--no-default-features --features openblas` did not produce an OpenBLAS-only
  build: the manifests declared `nuvai-mkl-src`/`nuvai-mkl-sys` without
  `default-features = false`, so every edge back into them re-enabled
  `accelerate`, and features being additive across the graph the build ended up
  with both — leaving the silent preference above to pick the one the command
  line asked for. The OpenBLAS CI job has thus been building a both-features
  workspace and passing only by accident of that bug; its own configuration was
  the defect it exists to catch. The workspace dependency entries now set
  `default-features = false`; each package's `default = ["accelerate"]` names
  the feature explicitly rather than inheriting it, so an ordinary build is
  unchanged. Verified with `cargo tree -e features` in both configurations, and
  by running the job's own command locally — with `OPENBLAS_ROOT` set, since
  Homebrew's OpenBLAS is keg-only — which now resolves to `openblas` alone and
  links and tests clean.

- `nuvai-mkl-sys`'s build script states why it cannot generate bindings, rather
  than failing later on a missing file (issue #34). bindgen is a *host*-resolved
  build-dependency, so Cargo installs it only when the host is an Intel x86_64
  non-macOS platform. Cross-compiling to an Intel *target* from any other host
  (Apple Silicon macOS, aarch64 Linux) still matched the Intel arm of the
  target dispatch, but that arm's body was gated on the host cfg, so it compiled
  nothing and returned successfully having written no `bindings.rs`. The first
  symptom was then the `include!(concat!(env!("OUT_DIR"), "/bindings.rs"))` in
  `src/lib.rs` — `couldn't find file .../bindings.rs`, naming neither the
  host/target split nor the build-dependency gate that caused it. The arm now
  panics with the host, the target and the gate, and points at the
  `x86_64-linux`/`x86_64-windows` CI jobs, which build these targets natively.
  Both behaviours reproduced from an `aarch64-apple-darwin` host with `cargo
  check --target x86_64-unknown-linux-gnu -p nuvai-mkl-sys`. This is a diagnosis
  fix, not a capability one: generating the bindings was never the whole
  problem — linking them still needs an x86_64 Linux linker and that target's
  MKL shared objects — so an Intel cross-check still needs an Intel host or CI.

- PARDISO releases its internal memory after a failed analysis phase (issue
  #23). `Pardiso` gated the phase `-1` release in `Drop` on an `analyzed` flag
  that was set only once phase 11 returned success, so a phase-11 failure (error
  `-2`, out of memory, among others) returned with the flag still false and the
  release skipped — leaking whatever the partial analysis had allocated. The
  flag now records what actually makes a phase `-1` legal, `pardisoinit` having
  run, and is set in `Pardiso::new` before any `pardiso` call can fail, so no
  error path can leave it false. It is true for every Intel handle today and
  stays a field rather than being folded away because the invariant it names is
  what `Drop` is keyed on. An Intel-only unit test pins it at construction —
  precisely the point the old code got wrong — rather than driving a failure
  through `solve`, which would need an input oneMKL rejects at analysis and
  would then be asserting on which phase reported the error instead of on the
  invariant.

- Accelerate's sparse backends reject a structurally empty CSR row instead of
  aborting the process on macOS 14 (issue #58). `_SparseFactorQR_Double` does
  not return an error for such an input there — it aborts with `SIGTRAP`, which
  cannot be caught, so the safe API accepted input that could kill the process
  on a supported platform. macOS 26 returns a documented state-1 error object
  instead, which is why the case passed locally and aborted in CI. The row is
  now rejected in `csr_to_csc`, the single validated CSR→CSC transposition, as
  `ErrorKind::InvalidArgument`. That is not an arbitrary symptom check: an empty
  row is a zero row, so the matrix is singular and no backend has a solution to
  return — the same answer the residual check already gives for a singular
  matrix. Intel PARDISO never reaches this function, so that path's handling of
  an empty row is unchanged: this closes an abort *observed* on Accelerate, and
  deliberately does not propagate the rejection to an Intel path where no abort
  was measured (it would plausibly surface as a zero pivot at phase 22, but that
  is an inference from PARDISO's documented behaviour, not a measurement, and
  the guard does not rest on it).
  The guard covers the DSS Cholesky path as well, since it shares the
  transposition and an empty row of the stored upper triangle likewise leaves
  that row of the full symmetric matrix zero. The test this replaces existed
  only as a comment explaining why it could not be written; it can now, and
  passes on both macOS versions.

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

- `acquire.rs`'s `locate` records why its aarch64 panic is unreachable rather
  than caller-facing (issue #39). The issue asked for it to return `Result`
  because a third-party build script calling it would get a hard panic, but that
  caller no longer exists: `locate` left the library target in #24, and since
  #60 the acquisition is `include!`d by `nuvai-mkl-src`'s `build.rs` alone, so
  `locate` is private to that build script's binary and cannot be imported.
  Within it the panicking arm is unreachable too — `main` dispatches on
  `backend_for_target` first, `emit_intel_mkl` is the sole caller, and no
  aarch64 target selects `IntelMkl`. The doc comment that described a downstream
  caller is corrected and the guard is kept as the check on that invariant.
  Nothing in the library target panics on aarch64 either:
  `MklInfo::from_build_metadata` answers `Option` and `backend_for_target`
  answers `Result`. No code change beyond the comment: the issue is closed as
  overtaken by #24/#60.

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