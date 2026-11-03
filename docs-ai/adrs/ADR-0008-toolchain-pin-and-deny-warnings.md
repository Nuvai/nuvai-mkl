# ADR-0008: Pin the toolchain to an exact patch and deny warnings

- **Status:** Accepted
- **Date:** 2026-11-03
- **Task:** #96 (parent epic #77)
- **Deciders:** nuvai-mkl maintainers

## Context

CI and the justfile used a floating toolchain, first nightly, then (#98) the
stable `1.99` line. CI deliberately omitted `-D warnings`, because on a floating
toolchain a new lint in a point release breaks unrelated work. The cost is that
warnings accumulate unnoticed.

## Decision

1. **The exact patch is pinned in one file**, `rust-toolchain.toml`
   (currently `1.99.1`, components `clippy` and `rustfmt`). rustup reads it for
   local builds, and `just setup` installs the channel it names. Each CI job also
   passes the same value to `dtolnay/rust-toolchain`; those are kept equal by the
   bump script (decision 3), not by hand.
2. **Warnings are denied** with `RUSTFLAGS="-D warnings"` at the workflow level
   (and in the justfile), which covers `cargo check`, `test` and `clippy`.
   Exceptions:
   - the `x86_64-linux-downstream` job clears `RUSTFLAGS`, because its fixtures
     must build as an outside consumer with no configuration (#72);
   - `cargo:warning=` lines from build scripts (the macOS deployment-target note,
     the `dynamic` disclosure that CI greps for) are not rustc warnings, so they
     are unaffected;
   - dependencies are compiled with cargo's lint cap, so third-party warnings
     cannot fail the build.
3. **The pin moves by a scheduled PR**, never by drift.
   `.github/workflows/toolchain-bump.yml` runs weekly (and on demand), finds the
   newest `1.99.x` tag, rewrites every pin via `.github/scripts/bump-toolchain.sh`
   and opens a PR to `main`. CI on that PR is the validation; any new lint is
   fixed in the PR. Nothing merges automatically. A `toolchain-pins` CI job
   (`.github/scripts/check-toolchain-pins.sh`) fails if any pin in `ci.yml` differs
   from `rust-toolchain.toml`; the bump skips an unpublished tag, reuses an open bump PR
   and closes superseded ones. A new minor (`1.100`) is an MSRV
   decision and is made by hand.
4. `rust-version = "1.99"` in `Cargo.toml` stays as the MSRV floor for consumers;
   the pin applies to this repository's own build only.

## Consequences

- A lint can fail CI only on a bump PR or on a change that introduces it.
- Contributors get 1.99.1 automatically through rustup; the first build downloads it.
- The bump PR needs a `TOOLCHAIN_BUMP_TOKEN` secret to trigger CI: PRs created with
  the default `GITHUB_TOKEN` do not start workflows.
- Intel and Windows warnings cannot be checked on an ARM Mac and surface only in CI.
