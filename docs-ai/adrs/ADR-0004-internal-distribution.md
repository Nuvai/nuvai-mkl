# ADR-0004: Internal distribution for nuvai-mkl (no crates.io publish)

- **Status:** Accepted
- **Date:** 2026-09-20
- **Epic:** #6 — Ship nuvai-mkl as the unified CPU vendor-math backend for the Nuvai compute stack
- **Task:** #12 — Publish nuvai-mkl-src, nuvai-mkl-sys, nuvai-mkl to crates.io (0.1.0)
- **Deciders:** nuvai-mkl maintainers

## Context

Epic #6 was written with acceptance criteria requiring the three crates be
published to crates.io at 0.1.0 and rendered on docs.rs, and task #12 was created
to do it. Neither happened: no crate of this workspace exists on crates.io, no
release tag has ever been cut (`git tag -l` is empty), and no manifest carries
`publish = false` — so nothing recorded an intent either way.

#12 was closed `NOT_PLANNED` on 2026-09-20 **without a comment**, leaving the
decision unrecorded while epic #6, the #1 roadmap, and the README all continued to
assume a registry release. This ADR answers the question durably, so it does not
have to be re-derived from a closed issue.

## Decision

1. **Distribution is internal, not via a registry.** Consumers take a **git** (or
   local **path**) dependency on `github.com/Nuvai/nuvai-mkl`. This workspace is
   not published to crates.io and has no docs.rs presence.

2. **The wrapper is the single entry point.** Consumers depend on `nuvai-mkl`
   only; `nuvai-mkl-sys` and `nuvai-mkl-src` resolve transitively from the same
   checkout. A git dependency traverses the repository to find the package, so
   the repository URL is the correct `git =` target even though the crates live
   under `crates/`. A *path* dependency does not search, so it must name
   `crates/nuvai-mkl` — not the workspace root, which is virtual and has no
   package.

3. **A release is an annotated tag on `main`**, `v0.1.0` first, matching this
   repo's trunk-based policy ("tag main, no release branch"). Cargo treats tags as
   mutable, so `rev = "<commit-sha>"` is the airtight pin and is documented as
   such alongside the tag.

4. **Epic #6's crates.io and docs.rs acceptance criteria are superseded** by this
   decision, and #1's roadmap records the same outcome.

## Rationale

- **There is no external audience to serve.** The crate's stated purpose is to be
  the CPU vendor-math backend for the Nuvai stack, whose consumers all live in
  sibling repositories (`nuvai-ds`, `nuvai-commons`, `Nuvai/cudarc`) that can take
  a git dependency directly. No identified consumer needs a registry artifact.

- **Publishing would commit to a public stability surface the crate does not
  offer.** It is pre-1.0, `edition = 2024`, requires a **nightly** toolchain to
  build at all, and on the Intel targets downloads ~140 MB of Intel oneMKL at
  build time. A crates.io release implies semver and support expectations across
  those axes that we are not prepared to meet. (Redistribution is not the
  obstacle — Intel's binaries are fetched at build time and never redistributed,
  so publishing the wrapper would be permissible; it is simply not warranted.)

- **docs.rs would document a surface it cannot exercise.** That host has no
  network and no MKL; the crates build there only through the `DOCS_RS`
  short-circuits added in #10. Making that generated page authoritative would
  advertise a build configuration nothing validates.

## Consequences

- Consumers must supply a matching **nightly** toolchain themselves — there is no
  `rust-toolchain.toml`, and `rust-version = "1.99"` makes an older one a hard
  error. The README's Installation section states this and gives both the
  tag-pinned and `rev`-pinned forms.

- API documentation comes from the repository (`cargo doc`), not a hosted page.

- The workspace stays free of publish-only metadata obligations. Should a real
  external consumer appear, the prerequisites are already understood: finalize
  `description`/`license`/`repository` on each manifest, `cargo publish --dry-run`
  in dependency order (`-src` → `-sys` → wrapper), and confirm each builds under
  the `DOCS_RS` short-circuit.

- **This decision is reversible, and reversing it is cheap.** A future publish is
  a new ADR superseding this one — it is not a reopening of #12.

## Alternatives considered

- **Publish all three to crates.io** (the original epic plan). Rejected on the
  stability-surface and no-audience grounds above.

- **Publish `nuvai-mkl-src` only**, keeping the wrapper internal. Rejected: the
  wrapper is the API anyone would want, and `links = "mkl"` makes the `-src` crate
  an implementation detail rather than a useful standalone artifact.

- **No tags, `rev`-only distribution.** Rejected as the default: an unpinnable
  `version = "0.1.0"` is already declared in `[workspace.dependencies]`, and
  hand-maintained SHAs do not survive normal `cargo update` workflows. Tags are
  kept as the human-facing identifier with `rev` documented as the hard pin.
