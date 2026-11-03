# ADR-0008: Serialise acquisition with a cache lock rather than merging units

- **Status:** Accepted
- **Date:** 2026-11-03
- **Task:** #94 (parent epic #77)
- **Builds on:** #63/#64 (per-PID staging, atomic extraction), ADR-0005/0006/0007
- **Deciders:** nuvai-mkl maintainers

## Context

`nuvai-mkl-src` is declared as both `[dependencies]` and `[build-dependencies]`
in `nuvai-mkl-sys` and `nuvai-mkl`, so Cargo builds two units and runs its
`build.rs` twice against one `~/.cache/nuvai-mkl`. #63/#64 made that race *safe*
(per-PID `.part` staging, rename-into-place, atomic extraction) but not
*economical*: both units download the same archive concurrently.

## Decision

1. `download_mkl` takes an exclusive advisory lock on
   `<cache>/mkl-<version>/.acquire.lock` (`CacheLock`, `cache_lock.rs`) and holds
   it across every `fetch_and_extract_conda` call. The second unit blocks, then
   finds verified archives and extracted trees and does no network I/O.
2. The lock uses `std::fs::File::lock` (stable; workspace `rust-version` is
   1.99). No `fs2`/`fd-lock` dependency. It works on Linux and Windows.
3. The `system_mkl()` (`MKLROOT`) path stays lock-free; it acquires nothing.
4. Per-PID staging and atomic extraction are kept as defence in depth.

## Why not merge the units

The host (build-dependency) and target (dependency) units deliberately carry
different features: `static` arrives on the target edge only (ADR-0005/0006/0007),
and a cross build needs both the `mkl` and `mkl-static` packages. A single
acquisition unit would have to pick one feature set. Serialisation keeps the
units and removes only the duplicated work.

## Consequences

- A download retrying with backoff holds the lock and stalls the other unit for
  that time; cheaper than both downloading.
- Advisory locks may be unreliable on network filesystems; the staging protocols
  remain the fallback guarantee there.
- `acquire.rs` is `include!`d by `build.rs` only and has no unit tests; the lock
  lives in `cache_lock.rs`, which `lib.rs` compiles under `cfg(test)`.
