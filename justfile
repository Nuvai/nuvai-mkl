# nuvai-mkl developer recipes. Mirrors .github/workflows/ci.yml for the *host*
# platform; CI stays the source of truth (it is the only place Intel/Windows run).
#
#   just            list recipes
#   just setup      install host prerequisites (toolchain + system libs)
#   just ci         what CI runs for this host: check, test, clippy (+ variants)

set shell := ["bash", "-euo", "pipefail", "-c"]

os := os()
arch := arch()

# The macOS deployment target CI pins for vDSP's interleaved-complex DFT (macOS 12+).
export MACOSX_DEPLOYMENT_TARGET := if os == "macos" { "12.0" } else { env("MACOSX_DEPLOYMENT_TARGET", "") }

default:
    @just --list

# Install prerequisites for this host: nightly toolchain + system libraries.
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "host: {{os}}/{{arch}}"
    command -v rustup >/dev/null || { echo "install rustup first: https://rustup.rs"; exit 1; }
    # rust-version = 1.99 (edition 2024) => nightly; there is no rust-toolchain.toml.
    rustup toolchain install nightly --component clippy,rustfmt
    rustup default nightly
    case "{{os}}/{{arch}}" in
      macos/aarch64)
        # Accelerate needs nothing. OpenBLAS is only for the `openblas` feature.
        echo "Accelerate backend: no system packages needed."
        echo "For the openblas feature run: just setup-openblas"
        ;;
      linux/aarch64)
        sudo apt-get update && sudo apt-get install -y libopenblas-dev
        ;;
      linux/x86_64)
        # libclang for the nuvai-mkl-sys bindgen pass; MKL itself is downloaded
        # from conda-forge into ~/.cache/nuvai-mkl on first build (or set MKLROOT).
        sudo apt-get update && sudo apt-get install -y libclang-dev
        ;;
      *)
        echo "unsupported host for scripted setup: {{os}}/{{arch}} (see README Requirements)"
        exit 1
        ;;
    esac

# macOS only: OpenBLAS (keg-only) for the `openblas` feature.
setup-openblas:
    brew install openblas

# Drop the MKL acquisition cache and target/ (reproduces CI's cold-cache build, #64).
clean-cold:
    rm -rf ~/.cache/nuvai-mkl target

check:
    cargo check --workspace

test:
    cargo test --workspace

clippy:
    cargo clippy --workspace --all-targets

fmt:
    cargo fmt --all

# macOS aarch64: the `openblas` feature in place of Accelerate.
[macos]
check-openblas:
    cargo check --workspace --no-default-features --features openblas

[macos]
test-openblas:
    cargo test --workspace --no-default-features --features openblas

[macos]
clippy-openblas:
    cargo clippy --workspace --all-targets --no-default-features --features openblas

# x86_64 Linux: the `dynamic` opt-in (threaded mkl_rt) instead of the static default.
[linux]
test-dynamic:
    cargo test --workspace --features dynamic

[linux]
clippy-dynamic:
    cargo clippy --workspace --all-targets --features dynamic

# x86_64 Linux: separate-workspace downstream consumers (#70/#72). Needs MKL.
[linux]
consumers:
    cargo run --manifest-path consumers/dependency-only/Cargo.toml
    cargo run --manifest-path consumers/build-script-helper/Cargo.toml --features static
    cargo run --manifest-path consumers/build-script-helper/Cargo.toml --features dynamic

# Everything CI runs for this host (no -D warnings, matching CI).
ci: check test clippy
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{os}}/{{arch}}" in
      macos/aarch64) just check-openblas test-openblas clippy-openblas ;;
      linux/x86_64)  just test-dynamic clippy-dynamic consumers ;;
    esac
