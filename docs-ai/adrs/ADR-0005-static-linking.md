# ADR-0005: Static linking of Intel oneMKL on x86_64 Linux

- **Status:** Accepted
- **Date:** 2026-09-21
- **Task:** #68 — evaluate/support static linking of oneMKL (Linux x86_64 first)
- **Deciders:** nuvai-mkl maintainers

## Context

`nuvai-mkl` linked Intel oneMKL, Accelerate, and OpenBLAS dynamically only —
`mkl_rt` (the runtime dispatcher `.so`/`.dll`), `-framework Accelerate`, or
`-lopenblas`. There was no static-link option on any platform. Static linking
was raised as a possible enhancement (self-contained binaries — no
`LD_LIBRARY_PATH`/rpath/DLL-next-to-exe distribution step; no build-time vs.
runtime MKL ABI mismatch) but came with open questions that had to be resolved
before committing to it:

- Does conda-forge, the acquisition source `nuvai-mkl-src` already uses, ship
  MKL's static archives at all?
- Is the ISSL redistribution license compatible with shipping a statically
  linked binary, not just a dynamically linked one?
- What does the actual link recipe need to be, given MKL's static archives
  resolve symbols circularly across the interface/threading/core layers?

## Decision

1. **conda-forge ships `mkl-static`, a separate package from `mkl`.**
   Confirmed via `api.anaconda.org` and by downloading and extracting
   `mkl-static-2026.1.0-ha770c72_243.conda`: it contains
   `libmkl_intel_lp64.a`, `libmkl_sequential.a`, `libmkl_intel_thread.a`,
   `libmkl_tbb_thread.a`, `libmkl_core.a`, and the BLACS/ScaLAPACK/BLAS95/
   LAPACK95 archives — `mkl`'s own `lib/` holds only the dynamic dispatcher.
   `mkl-static` is ~130 MB compressed (~700 MB+ uncompressed across the three
   archives this crate links), against a few MB for the dynamic `mkl`
   package — the "large download" concern from the original evaluation is
   real and is accepted as the cost of the feature, which is why it is
   opt-in (`static` feature, off by default) rather than the default.

2. **License is not a blocker.** `mkl-static` carries the same
   `LicenseRef-IntelSimplifiedSoftwareOct2022` license as `mkl`/`mkl-include`,
   which this repository already redistributes under (README, "Third-party
   licenses"). Static linking introduces no new license to evaluate.

3. **Scope: `x86_64-unknown-linux-gnu` only, for now.** Windows has no
   `mkl-static`-equivalent conda package pinned by this crate's acquisition
   path (conda-forge does publish a win-64 `mkl-static`, but wiring it is
   deferred — it needs its own package/SHA-256 pin and Windows uses `.lib`
   import libraries rather than raw archives, a different linking shape).
   Accelerate and OpenBLAS have no static form at all, so `static` is a no-op
   distinction on the aarch64 fallbacks — it is rejected at compile time
   (`backend.rs`) and in the build script (`build.rs::main`) on every target
   but `x86_64-unknown-linux-gnu`, rather than silently linking dynamically.
   Selection is explicit, never silent (ADR-0003's rule, applied here too).

4. **Threading layer: `mkl_sequential`, not `mkl_intel_thread` or
   `mkl_tbb_thread`.** The dynamic path already needs the OpenMP runtime
   (`libiomp5.so`, fetched from conda-forge's `llvm-openmp`) because
   `libmkl_intel_thread.so.3` leaves `omp_*` undefined. Choosing the
   sequential (single-threaded) static archive instead avoids pulling in an
   OpenMP or TBB runtime archive at all — the static link needs nothing
   beyond libc/libm/libpthread/libdl, which every Linux target already has.
   This trades away MKL's internal threading (callers already parallelizing
   at a higher level are unaffected; callers relying on MKL's own threading
   are not the target user for this feature in its first cut).

5. **Link recipe: `-Wl,--start-group,-l:libmkl_intel_lp64.a,-l:libmkl_sequential.a,
   -l:libmkl_core.a,--end-group,-lpthread,-ldl,-lm`**, emitted as a single raw
   `cargo:rustc-link-arg` argument — and emitted from **`nuvai-mkl/build.rs`**
   (`emit_intel_mkl_static`), not from `nuvai-mkl-src`, even though
   `nuvai-mkl-src` is the sole `links = "mkl"` provider for everything else.
   `rustc-link-arg` is scoped to the *emitting* package's own binaries; it is
   not a propagating directive the way `rustc-link-lib`/`rustc-link-search`
   are, and `nuvai-mkl-src` owns no binaries of its own. An earlier revision
   of this feature emitted the group from `nuvai-mkl-src` and passed
   acquisition, caching, `cargo check`, and `cargo clippy` — clippy's default
   invocation never links — while silently failing to link real binaries at
   all (dozens of undefined MKL symbols, caught by remote x86_64 Linux
   validation rather than by any of those checks). `pthread`/`dl`/`m` are
   appended in the *same* argument, immediately after `--end-group`, rather
   than as separate `cargo:rustc-link-lib=dylib=…` directives: `ld`/`mold`
   only resolve a library against symbol references already seen earlier on
   the command line, and Cargo does not guarantee a `rustc-link-lib` directive
   from one crate lands after a `rustc-link-arg` from another on the final
   link line. The three archives also resolve symbols circularly — `mkl_core`
   calls back into the interface and threading layers as well as being called
   by them — which no linear ordering of plain `cargo:rustc-link-lib=static`
   directives can resolve; `--start-group`/`--end-group` tells the linker to
   re-scan the group until nothing new resolves, which is what Intel's own
   link-line advisor prescribes for this exact case.

6. **No rpath, no `force_runtime.c` trick, for the static path.** Both exist
   in `nuvai-mkl/build.rs` only because a *dynamic* `.so` can leave a symbol
   undefined for the loader to resolve at run time (MKL's shared objects
   declare no DT_NEEDED for `libm`/`libiomp5`). A static archive link has no
   "later": every symbol resolves at link time or the link fails outright, so
   `nuvai-mkl/build.rs` skips that whole arm when `MklInfo::static_link` is
   set, and there is no shared object to point an rpath at.

7. **`MklInfo` gained a `static_link: bool` field**, published as
   `cargo::metadata=STATIC_LINK=0|1` and read back by dependent build scripts
   through the existing `DEP_MKL_*` mechanism (#24). This is how
   `nuvai-mkl/build.rs` knows to skip the dynamic-only linker tricks without
   re-deriving the target/feature logic itself.

## Consequences

- `cargo build --features static` on `x86_64-unknown-linux-gnu` produces a
  binary with no runtime dependency on `libmkl_rt.so`/`libiomp5.so` — `ldd`
  should show only libc/libm/libpthread/libdl. The CI job
  `x86_64-linux-static` asserts this on every run (see task #68's acceptance
  criteria).
- The default (no `static`) build path is unaffected: `static_link` defaults
  to `false`, and `acquire.rs`/`build.rs` only branch on it when the feature
  is enabled.
- First `cargo build --features static` on a clean cache downloads ~130 MB —
  slower than the default dynamic acquisition, and worth calling out in
  developer-facing docs so it isn't mistaken for a network problem.
- Windows static linking, and statically linking the threaded layer
  (`mkl_intel_thread`), remain open follow-ups if a concrete need for either
  shows up; neither is blocked by anything decided here.
