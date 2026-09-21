# ADR-0007: Static linking is the default on x86_64 Linux; `dynamic` is the opt-in

- **Status:** Accepted
- **Date:** 2026-09-21
- **Task:** #72 — the default dynamic path still fails at runtime downstream (no
  RUNPATH propagated, `libmkl_rt.so.3` not found)
- **Amends:** ADR-0005 decision 1 (the "off by default" rationale, whose
  download-size premise was wrong by measurement — see Context) and ADR-0006
  decision 6 (the table of what a consumer must write)
- **Deciders:** nuvai-mkl maintainers

## Context

ADR-0006 fixed the *static* half of #70 and accepted the dynamic half as a
documented cost: a consumer selects `static`, or writes two manifest entries and
one line of `build.rs`. #72 is what that acceptance looks like from outside the
repository. The default is dynamic, and a consumer who simply adds the
dependency gets exactly that failure — measured on `x86_64-unknown-linux-gnu`,
`mold`, nightly, against the ADR-0006 revision:

```
$ cargo run --manifest-path consumers/dependency-only/Cargo.toml   # no build.rs, no feature
error while loading shared libraries: libmkl_rt.so.3: cannot open shared object file

$ readelf -d target/debug/…-dependency-only | grep -E 'NEEDED|RUNPATH'
 (NEEDED)  Shared library: [libmkl_rt.so.3]
 (NEEDED)  Shared library: [libgcc_s.so.1]
 (NEEDED)  Shared library: [libc.so.6]
 (NEEDED)  Shared library: [ld-linux-x86-64.so.2]        # no RPATH, no RUNPATH at all
```

Three things about that measurement matter, and each was recorded as a fact
rather than assumed:

1. **`cargo run` fails too**, not just a bare execution of the binary: whatever
   loader path Cargo supplies when it runs a binary did not cover this one, so
   there is no cargo-shaped workaround between the consumer and the failure.
   (This repository's own binaries, by contrast, start — their rpath comes from
   `nuvai-mkl/build.rs` through `emit_binary_link_args`. The gap was always
   specific to a crate that is not the emitting package, which is the shape #70
   and #72 share.)
2. **The second gap #72 flagged is still open.** With `LD_LIBRARY_PATH` supplied
   by hand, the next failure is the one the issue could not re-test on the
   current revision: `libmkl_intel_thread.so.3: undefined symbol:
   __kmpc_global_thread_num`, from `libiomp5` having no DT_NEEDED (its
   `omp_*` references live in a shared object the linker never had to satisfy,
   so `--as-needed` drops what nothing in the *executable* references). Both
   halves are consequences of the same missing per-binary arguments, and one
   build script emitting them fixes both — which is what the
   `build-script-helper` fixture demonstrates.
3. **`static` is not the larger acquisition.** ADR-0005 decision 1 made static
   opt-in because `mkl-static` is a much larger package than `mkl`; measured
   from the pinned conda-forge artifacts, it is *smaller* — 130 MB compressed
   against 143 MB — because the dynamic `mkl` package also ships the four
   threading layers, the ILP64/GNU variants, ScaLAPACK and the BLAS95/LAPACK95
   archives, while `mkl-static` ships the archives. ADR-0005's ~530 MB figure
   was `libmkl_core.a` *uncompressed*. The real costs of static are a larger
   **binary** and the loss of MKL's own threading, and only the second is a
   behavioural difference.

Of #72's three options, the third — "if some Cargo mechanism does let a
`links = "mkl"` crate push an rpath to dependents" — was closed by measurement
rather than by argument: rustc passes a build script's `cargo:rustc-link-search`
to the linker as `-L <dir>` and **never** as `-Wl,-rpath` (probed by compiling a
trivial binary with a fake linker and reading the argument list), so a
propagating search path cannot carry a runtime search path, and no other stable
channel reaches another package's binary. ADR-0006's conclusion stands, and the
choice is therefore between "change what the default *is*" and "keep
documenting a default that does not work".

## Decision

1. **`static` is the default on `x86_64-unknown-linux-gnu`.** `nuvai-mkl` and
   `nuvai-mkl-sys` each carry a target-gated dependency edge
   (`cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))`)
   requesting `nuvai-mkl-src/static`. Cargo merges a `[target.…]` entry with the
   plain `[dependencies]` entry for the same package, and features unify across
   the graph, so a consumer that declares nothing but the dependency gets the
   static link — with no build script, no rpath and no `LD_LIBRARY_PATH`, which
   is #72's repro working.

   The edge exists **twice**, once under `dependencies` and once under
   `build-dependencies`, because Cargo builds two units of `nuvai-mkl-src` (one
   per dependency kind) and resolver "2" does not unify features across them.
   The two tables are matched against *different* triples, which is worth
   recording because it is not obvious: `dependencies` is matched against the
   **target** and `build-dependencies` against the **host**. Verified
   directly — with `--target x86_64-unknown-linux-gnu` from a macOS host, the
   normal unit acquired `static` from the linux table and the build-dependency
   unit did not. Both are present here so that a native Intel-Linux build takes
   one path for both units, and acquires `mkl-static` alone rather than both
   conda packages.

2. **`dynamic` is the explicit opt-in, and the static/dynamic rule is a pure,
   tested function.** Features are additive, so "static by default" cannot be
   expressed as "static unless something subtracts it" — the opt-out has to be a
   feature of its own, and `dynamic` wins over the `static` the edge supplied.
   The rule lives in `nuvai_mkl_src::static_link_selected` (a `const fn` of the
   two booleans, unit-tested for all four pairs) rather than in a `cfg!` read, so
   the one case a build cannot observe about itself — both features on — is
   asserted rather than accidental. `dynamic` is inert on every other target,
   because every other backend already *is* dynamic; `static` remains a compile
   error there, since it asks for something those targets cannot do.

3. **The dynamic path says what it needs, at build time.** `nuvai-mkl-src`'s
   build script emits a `cargo:warning` naming the two ways to satisfy it (a
   `build.rs` calling `emit_binary_link_args`, or `LD_LIBRARY_PATH` where the
   binary runs) and the fact that dropping `dynamic` avoids the question. This is
   #72's actual complaint — the failure "names the dynamic loader rather than
   this crate" — answered at the moment the caller chose the path, rather than at
   the first run.

4. **The threading trade-off is accepted, documented, and is why `dynamic`
   exists.** The static link uses the `mkl_sequential` threading layer (ADR-0005
   decision 4), and it has to: conda-forge's `llvm-openmp` ships `libomp.so` and
   no `libomp.a`, so the threaded layer cannot be linked statically from this
   acquisition. A consumer that needs MKL's own parallelism selects `dynamic`
   and supplies the arguments. This is a **behavioural change for existing
   x86_64 Linux consumers** and is called out as such in the README and the
   changelog.

5. **CI asserts the default rather than assuming it.** `x86_64-linux` (no
   feature) now checks with `ldd` that its test binaries link no dynamic
   MKL/OpenMP runtime at all; a new `x86_64-linux-dynamic` job checks the mirror
   image, that `--features dynamic` really does put `libmkl_rt` in DT_NEEDED;
   and `x86_64-linux-downstream` runs the four-case consumer matrix, including
   `consumers/dynamic-no-plumbing/`, whose expected outcome is a warning plus a
   failure to start. A negative assertion is the only kind that can notice the
   *diagnosis* regressing, which is what #72 was.

## Consequences

- A crate that depends only on `nuvai-mkl` on `x86_64-unknown-linux-gnu` links
  and runs with nothing declared — the case #72 filed. Consumers that were
  relying on the dynamic default get the static link on upgrade, which means
  MKL's internal threading stops being used unless they select `dynamic`; there
  is no flag that preserves the old behaviour *and* zero configuration, because
  the old behaviour is the one that does not work.
- The `dynamic` path's requirements are unchanged from ADR-0006 decision 6 —
  two manifest entries and one line — but they are now the price of a
  *declaration* rather than of using the crate at all, and the build script
  states them. Nothing about `dynamic` became more permissive; what changed is
  who pays when the caller does not read the README.
- One Cargo boundary became load-bearing and is recorded here because it fails
  silently: a build script is compiled for the **host** but carries the features
  of the unit it belongs to. `static`'s guard therefore moved out of
  `backend.rs` — which `build.rs` also `include!`s — and into the crate root,
  because with the feature now arriving from a *target*-keyed edge, a guard in
  the shared file rejected the build script of a cross build whose target is
  supported and whose host is not. Found by reproducing docs.rs's configuration
  locally (`DOCS_RS=1 cargo doc --target x86_64-unknown-linux-gnu`, from macOS);
  without that it would have masked the intentional, well-explained cross-build
  diagnostic in `nuvai-mkl-sys` (`backend.rs`'s sibling panic, #34) with a
  confusing one from `nuvai-mkl-src`. The build script keeps its own check, now
  keyed on `HOST`.
- Windows and the aarch64 fallbacks are untouched: they never had a static form
  (ADR-0005 decision 3), so neither the default nor the opt-out means anything
  there, and the Windows DLL/`PATH` story is still told by the existing
  `cargo:warning`s.
- Verified on `x86_64-unknown-linux-gnu` (Ubuntu 24.04, nightly, conda-forge
  acquisition) under `mold`: the cold-cache workspace build, both feature arms,
  and all three consumer fixtures including the negative one. The VM's cargo
  config forces `-fuse-ld=mold`, so a run that does not override it has not
  tested `ld` — the default workspace build and the dynamic consumer were
  therefore re-run under GNU `ld` via `RUSTFLAGS="-C linker=cc"`, which replaces
  the config's rustflags rather than adding to them.
- What each selection *fetches*, measured on a cold cache rather than reasoned
  about — the first attempt at this paragraph was wrong: the default static link
  acquires `mkl-static` + `mkl-include` (~130 MB compressed), and `dynamic`
  acquires `mkl` + `mkl-include` + `llvm-openmp` (~150 MB). One package each,
  because the rule is evaluated per *unit* and `dynamic` reaches both of
  `nuvai-mkl-src`'s — Cargo applies a forwarded dependency-feature to every edge
  the package appears on. So selecting `dynamic` does not leave the build-script
  unit behind on the static link, and neither selection pays for the other. The
  old default fetched the `dynamic` set, so the default is now the cheaper one
  as well as the one that works.
- Still open, unchanged: Windows static linking; a statically linked *threaded*
  layer, which needs a static OpenMP runtime this acquisition does not have; and
  the cold-cache acquisition race (#64's class), which this change does not
  touch — the default build now acquires fewer packages, so it hits it less
  often, not never.
