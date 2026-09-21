# ADR-0006: Propagating link directives to downstream consumers

- **Status:** Accepted
- **Date:** 2026-09-21
- **Task:** #70 — `rustc-link-arg` directives do not reach downstream consumers (static fails to link, dynamic fails at runtime)
- **Amends:** ADR-0005 decisions 5 and 6; ADR-0003 decision 7 (the aarch64 rpath note)
- **Deciders:** nuvai-mkl maintainers

## Context

ADR-0005 added static linking by emitting Intel's group-line
(`--start-group -l:libmkl_intel_lp64.a … --end-group`) as one raw
`cargo:rustc-link-arg`, from `nuvai-mkl/build.rs`. Every check available at the
time passed — acquisition, caching, `cargo check`, `cargo clippy`, `cargo test`
on x86_64 Linux, and the `x86_64-linux-static` CI job — because all of them link
*this repository's own* targets, and `nuvai-mkl` is the package that emits the
argument.

Issue #70 found both link paths broken for a crate that merely depends on
`nuvai-mkl`, and only there:

- **static:** the consumer's link line carried the archive *directory*
  (`-L …/mkl-static-2026.1.0-…/lib`) and not one `-l` flag, so every MKL symbol
  was undefined (`mold: error: undefined symbol: cblas_sgemm`).
- **dynamic:** the build succeeded and the binary failed at load time
  (`libmkl_rt.so.3: cannot open shared object file`), because nothing ever put a
  DT_RUNPATH on it.

One Cargo rule explains both, and a second one shapes the fix:

1. **Build-script linker output splits in two.** `cargo:rustc-link-lib` and
   `cargo:rustc-link-search` are collected from every build script in the graph
   and applied to each final link. `cargo:rustc-link-arg` is scoped to the
   *emitting package's* own targets. A consumer that is not the emitting package
   receives the first kind and never the second — including when the emitting
   package is a crate the consumer depends on, and including when the argument
   is a file the consumer needs (the compiled `force_runtime.c` object).
2. **`DEP_<links>_*` metadata reaches a build script only through a normal
   `[dependencies]` edge** — never through `[build-dependencies]`. Verified
   directly while writing this (#70): a package with a `links` provider in
   `[build-dependencies]` alone reads `DEP_MKL_LIB_DIR` as unset; the same
   package with it in `[dependencies]` reads it. This is why `nuvai-mkl` and
   `nuvai-mkl-sys` can read the metadata at all: both carry `nuvai-mkl-src` in
   *both* tables.

The first rule is not a bug to route around but a boundary: an executable's
DT_RUNPATH and `--no-as-needed` cannot be expressed as a library or a search
path at all, so no amount of directive-plumbing moves them off the binary's own
build script. (A linker script can carry a group link — `GROUP` is a `-l` input
— but linker scripts have no rpath directive.)

## Decision

1. **Everything expressible as a library or a search path is emitted from
   `nuvai-mkl-src`**, the sole `links = "mkl"` provider: the search paths,
   `-lmkl_rt`/`-liomp5`/`-ldl`/`-lpthread`/`-lm`, and — new here — the static
   group link. Those propagate to every dependent's link, upstream and
   downstream alike (ADR-0003 decision 1 stands).

2. **The static group link is a linker script.** `nuvai-mkl-src/build.rs` writes
   `libnuvai_mkl_static_group.a` into its `OUT_DIR` containing
   `GROUP ( …/libmkl_intel_lp64.a …/libmkl_sequential.a …/libmkl_core.a )`, adds
   that directory to the propagating search path, and requests it as
   `cargo:rustc-link-lib=static:-bundle=nuvai_mkl_static_group`. `GROUP` has the
   same repeated-rescan semantics as `--start-group`/`--end-group` (which is
   what ADR-0005 decision 5 chose it for), but as a `-l` input it propagates
   where a link-argument cannot.

3. **`-bundle` is required**, and is the one non-obvious part of decision 2. A
   `-l static=` directive defaults to *bundle*: rustc folds the library into the
   emitting crate's rlib, and it does that by opening the file and reading it as
   an archive. A linker script is not one, and the build dies with
   `failed to add native library …: Unsupported archive identifier`. `+verbatim`
   does not help — the read is for bundling, not for the name lookup. With
   bundling off, rustc passes the input to the linker unexamined, which is what
   makes a text-file `-l` input (the mechanism glibc's own `libc.so`/`libm.so`
   scripts use) usable here at all.

4. **The per-binary arguments move into one library entry point**,
   `nuvai_mkl_src::emit_binary_link_args()`, which emits the runtime rpath(s),
   `--no-as-needed` around `-lm`/`-liomp5`, and the `force_runtime.c` object for
   the target it is compiled for. `nuvai-mkl/build.rs` calls it for this
   workspace's test/example/bench binaries; a downstream crate calls it from its
   own `build.rs`. One implementation, so the two cannot drift — the divergence
   is what #70 *is*.

5. **`force_runtime.c` moves to `nuvai-mkl-src`**, along with its `cc`
   build-dependency, and its compiled object's path is published as
   `DEP_MKL_FORCE_OBJ`. The object has to be *the same file* for every binary
   that links it, and only the `links = "mkl"` provider can hand a path to a
   dependent's build script (#24). Its emission is now conditioned on
   `omp_lib_dir` being present, i.e. on there being a `-liomp5` on the link line
   for the `omp_*` reference it leaves undefined to resolve against; a system
   oneAPI install publishes no `omp_lib_dir`, and there the object could only
   make the link fail.

6. **What a downstream consumer has to write is therefore split by link mode,
   and this is accepted rather than worked around:**

   | Consumer | Configuration |
   |---|---|
   | static | none. A plain `nuvai-mkl = { …, features = ["static"] }` dependency links and runs |
   | dynamic | `nuvai-mkl-src` in **both** `[dependencies]` (the only edge `DEP_MKL_*` reaches a build script through, context 2) and `[build-dependencies]` (a build script can only use code from that table), plus `fn main() { nuvai_mkl_src::emit_binary_link_args() }` |

   Static linking is consequently the zero-plumbing integration, which ADR-0005
   already made the opt-in for self-contained distribution.

7. **The regression guard is downstream, not local.** `consumers/` holds two
   crates that depend on `crates/nuvai-mkl` from *outside* the workspace —
   `dependency-only` (no `build.rs`, static) and `build-script-helper` (the call
   above, dynamic and static) — and the `x86_64-linux-downstream` CI job builds
   and *runs* all three. A workspace member cannot stand in for them: the
   distinction #70 turns on is between packages, and `cargo test --workspace`
   would both build a member's targets as the wrong package and unify its
   features with the workspace's.

## Consequences

- A crate that depends only on `nuvai-mkl` links and runs under `static`, with
  no build script, no `[build-dependencies]`, no `RUSTFLAGS` and no
  `LD_LIBRARY_PATH` — verified on x86_64 Linux with both `ld` and `mold`
  (`mold` is where #70 was found, and it is the linker whose `-l` handling
  decision 4's object exists for, #44).
- The dynamic path still needs those two manifest entries and one line. It is
  one line rather than the issue's ~12, and it is documented rather than
  discovered — but it is not zero, because no stable Cargo channel can put an
  rpath on another package's binary.
- `nuvai-mkl`'s own build script shrinks to dispatching on the target and
  emitting the macOS deployment-target warning; the link work it used to
  hand-roll is now shared with consumers' build scripts.
- Verified boundaries for decisions 2 and 3, all on x86_64 Linux: `cargo check`
  and `clippy` do **not** link and cannot see any of this; a **library** crate
  is where a `-l static=` directive is read (for bundling), while a binary
  target links it directly — so the failure mode of getting decision 3 wrong is
  a compile error in `nuvai-mkl-src`, and of getting decision 1 wrong is a
  link error in a consumer.
- An earlier revision of #70's fix used `cargo:rustc-link-lib=static:+verbatim=…`
  and failed exactly as decision 3 describes. The `+verbatim` alternative is
  recorded here so it is not tried again.
- Windows and the aarch64 fallbacks are unaffected: the group script is
  `x86_64-unknown-linux-gnu` only (`static` is rejected elsewhere by
  `backend.rs`), and the per-binary helper emits nothing for them — the
  OpenBLAS-aarch64 rpath remains a single `-Wl,-rpath` from the crate that owns
  the binary (ADR-0003 decision 7, amended to name the helper).
