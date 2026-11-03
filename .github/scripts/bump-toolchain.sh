#!/usr/bin/env bash
# Move the pinned toolchain to NEW (e.g. 1.99.2) in every place that carries it:
# rust-toolchain.toml and each `toolchain:` / step name in ci.yml. The justfile
# reads rust-toolchain.toml, so it needs no edit. Prints the old pin on stdout.
set -euo pipefail

new="${1:?usage: bump-toolchain.sh <x.y.z>}"
[[ "$new" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "not an exact x.y.z version: $new" >&2; exit 2; }

root="$(git rev-parse --show-toplevel)"
old="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$root/rust-toolchain.toml")"
[[ -n "$old" ]] || { echo "no channel in rust-toolchain.toml" >&2; exit 1; }

if [[ "$old" != "$new" ]]; then
  old_re="${old//./\\.}"
  sed -i.bak "s/\"$old_re\"/\"$new\"/" "$root/rust-toolchain.toml"
  sed -i.bak -e "s/toolchain: \"$old_re\"/toolchain: \"$new\"/" \
             -e "s/pinned $old_re toolchain/pinned $new toolchain/" "$root/.github/workflows/ci.yml"
  rm -f "$root/rust-toolchain.toml.bak" "$root/.github/workflows/ci.yml.bak"
fi
echo "$old"
