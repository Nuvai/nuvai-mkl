#!/usr/bin/env bash
# CI gate for issue #2498: fails if any crate in this repo could be published to any registry.
# Run from the repo root. Needs python3; cargo + jq for the second check.
set -euo pipefail
python3 -I "$(dirname "$0")/lock-publish.py" . --check
bad=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.publish != []) | .name')
if [ -n "$bad" ]; then echo "ERROR: publishable per cargo: $bad" >&2; exit 1; fi
echo "OK: no crate in this repo can be published."
