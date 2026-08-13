---
docVer: "1.0"
id: "trunk-based-development"
type: "dev-guidelines"
lastUpdated: "2026-08-13"
---

# Trunk-Based Development — nuvai-mkl

## Overview

This repo uses **trunk-based development (TBD)**. There is a single shared
branch — `main` — and it is the source of truth. All work integrates to `main`
continuously through small, short-lived changes. There is **no `dev` branch and
no long-lived feature branches**.

This overrides the global `dev`-based workflow in `~/.claude/CLAUDE.md`
(see the repo-level `CLAUDE.md`).

## The trunk

- `main` is always releasable. Any commit on `main` must build and pass CI.
- Incomplete work is merged to `main` **behind a feature flag** (off), never on
  a side branch.

## Branch naming

| Prefix | Purpose | Lifetime |
|---|---|---|
| `feature/*` | new capability | < 1 day |
| `fix/*` | bug fix | < 1 day |
| `chore/*` | housekeeping, deps, CI | < 1 day |
| `docs/*` | documentation | < 1 day |
| `hotfix/*` | urgent production fix | hours |

Every branch is created from `main` and merged back to `main` the same day.

## Pull-request workflow

1. Create the branch from `main`.
2. Make one small, focused change (a single logical increment).
3. Open a PR targeting `main` immediately — even while still iterating.
4. CI must pass before merge.
5. Merge the same day; delete the branch.

PRs are **small and reviewable**. A whole feature ships as a series of
flag-guarded PRs, not one large PR.

## Commit conventions

`<type>(<scope>): <subject>` — types: `feat`, `fix`, `refactor`, `docs`,
`perf`, `test`, `chore`, `style`, `ci`, `revert`.

```
feat(blas): add sgemm row-major path
fix(src): resolve MKLROOT detection on Windows
```

## Feature flags

Merge incomplete work by gating it behind a flag that is **off** in production.
The flag — not the branch — controls whether the feature is live.

- Add the flag with the first PR, defaulting to off.
- Flip the flag on in a separate, reversible PR once the feature is complete.

## CI gate

- Every commit to `main` (and every PR) runs `cargo check`, `cargo test`, and
  `cargo clippy`.
- A broken `main` blocks everyone — fix forward, don't revert-and-hide.

## Releases

- There is **no release branch**. A release is a tag on `main`.
- Tag with a semver version: `v1.2.0`.
- Hotfixes: branch `hotfix/*` from the release tag, merge to `main`, tag again.

## Large changes

For work too big for one small PR:

- **Stacked PRs** — a chain of dependent PRs (`a → b → c`), each reviewable.
- **Branch by abstraction** — introduce an interface, build the new
  implementation behind it, then switch over.
- **Feature flags** — land the change incrementally, off by default.

## What this repo does NOT do

- ❌ No `dev` / integration branch.
- ❌ No long-lived `feature/*` branches (weeks).
- ❌ No release branches.
- ❌ No "merge everything at the end" PRs.
