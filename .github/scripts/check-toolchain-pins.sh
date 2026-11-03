#!/usr/bin/env bash
# Fail unless every `toolchain: "x.y.z"` in ci.yml and every "pinned x.y.z toolchain"
# step name equals the channel in rust-toolchain.toml (ADR-0008). Run by CI and by
# bump-toolchain.sh, so a pin the bump script cannot rewrite is caught, not skipped.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
pin="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$root/rust-toolchain.toml")"
[[ -n "$pin" ]] || { echo "no channel in rust-toolchain.toml" >&2; exit 1; }

ci="$root/.github/workflows/ci.yml"
bad=0
while IFS= read -r line; do
  echo "pin mismatch (want $pin): $line" >&2
  bad=1
done < <(
  grep -nE 'toolchain: *"' "$ci" | grep -vF "toolchain: \"$pin\""
  grep -nE 'pinned [^ ]+ toolchain' "$ci" | grep -vF "pinned $pin toolchain"
)
n="$(grep -cF "toolchain: \"$pin\"" "$ci" || true)"
[[ "$n" -gt 0 ]] || { echo "no toolchain pins found in ci.yml" >&2; bad=1; }
exit "$bad"
