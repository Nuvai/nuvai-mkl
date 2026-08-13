desc: "nuvai-mkl REPO_CLAUDE — inherits USER_CLAUDE (~/.Codex/AGENTS.md). Overrides below take precedence for this repo."

# Repo Identity
repo:
  name: "nuvai-mkl"
  kind: "Rust Cargo workspace — Intel oneMKL 2026.1.0 wrapper (3 crates: -src / -sys / safe wrapper)"
  primaryBranch: "main"

# Development Workflow Override — Trunk-Based Development (TBD)
# Replaces USER_CLAUDE githubEssentials.branchWorkflow (dev-based) for this repo.
githubEssentials:
  branchWorkflow:
    model: "trunk-based development (TBD)"
    policy: "single shared trunk (main) as source of truth; all work integrates to main continuously"
    defaultBase: "main"
    trunk: "main (no dev/integration branch, no long-lived branches)"
    branchLifetime: "< 1 day (branch → PR → merge to main same day)"
    enforcement: "ALL non-main branches target main; long-lived branches prohibited unless user explicitly specifies"
    prTargets: { featureBranches: "epic-*/task-*/feature-*/fix-*/chore-*/docs-* → main", hotfix: "hotfix-* → main (then tag)", release: "tag main, no release branch" }
    enablers: { featureFlags: "merge incomplete code behind off flag", smallCommits: "tiny reviewable commits", ciGate: "every main commit builds+tests", branchByAbstraction: "large refactors behind an interface" }
    rationale: { stability: "main always releasable via feature flags + CI gate", integration: "continuous small merges avoid conflict debt", workflow: "branch → PR → main (same day), no dev hop" }
    reference: "full workflow: docs-team/trunk-based-development.md"
