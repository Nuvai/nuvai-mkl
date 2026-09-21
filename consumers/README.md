# Downstream consumer fixtures

Three minimal crates that depend on `nuvai-mkl` **from outside** the workspace,
and whose only purpose is to make the difference between "works for this
repository's own test binaries" and "works for a crate that depends on it"
impossible to miss (#70, #72).

Cargo's build-script linker output splits in two. `cargo:rustc-link-lib` /
`cargo:rustc-link-search` are collected from every build script in the graph and
applied to each final link. `cargo:rustc-link-arg` is scoped to the *emitting
package's* own targets. A consumer that is not the emitting package therefore
receives the first kind and never the second — which is why both link paths used
to be broken downstream while every check in this repository passed: the
workspace's test binaries are emitted *by* `nuvai-mkl`, and a consumer's are not.

| Fixture | Link arguments it declares | Paths it proves |
|---|---|---|
| `dependency-only/` | none — no `build.rs`, no `[build-dependencies]`, no feature | the **default** static link, end to end (#70's fix, and #72's default) |
| `build-script-helper/` | `nuvai_mkl_src::emit_binary_link_args()` in `build.rs` | `dynamic` (the opt-in) and `static` (`--features static`) |
| `dynamic-no-plumbing/` | none, with `features = ["dynamic"]` | the documented failure: builds with a warning, **must not** start |

Each is a workspace of its own, so `cargo test --workspace` at the repository
root neither builds them nor unifies their feature selection. All three are
`x86_64-unknown-linux-gnu` fixtures: the static link exists only there, and the
dynamic path's gap is an rpath for conda-forge's MKL shared objects.

The helper fixture carries `nuvai-mkl-src` in **both** `[dependencies]` and
`[build-dependencies]`, which looks redundant and is not:

- Cargo forwards a `links` provider's `DEP_MKL_*` metadata to a build script
  through a normal `[dependencies]` edge **only**. A `[build-dependencies]`
  entry alone receives none of it — the fixture's build script reads
  `DEP_MKL_LIB_DIR` as unset without the first entry, and the binary then links
  and fails to start with `libmkl_rt.so.3: cannot open shared object file`.
- A build script can only *use* code from `[build-dependencies]`, so the call
  itself needs the second entry.

ADR-0006 records both rules. They apply to the `dynamic` path only, which is why
the other two fixtures — `dependency-only` and `dynamic-no-plumbing` — carry
neither entry: under the static default, and under `dynamic` without the
arguments, a build script is not the thing that is missing.

```sh
# The default, declaring nothing: must link and run.
cargo run --manifest-path consumers/dependency-only/Cargo.toml

# The same crate with the static link spelled out, where that call emits nothing.
cargo run --manifest-path consumers/build-script-helper/Cargo.toml --features static

# The dynamic opt-in, with the arguments it needs: must link and run.
cargo run --manifest-path consumers/build-script-helper/Cargo.toml --features dynamic

# The dynamic opt-in without them: must warn at build time and fail to start.
cargo build --manifest-path consumers/dynamic-no-plumbing/Cargo.toml
```

The CI job `x86_64-linux-downstream` runs those four, and asserts the two
outcomes the last one exists for: that the build printed
`linking the dynamic oneMKL runtime`, and that the binary then died with
`libmkl_rt.so.3: cannot open shared object file` rather than for some other
reason. They need `libclang` (the bindgen pass for the Intel targets) and reach
the same `~/.cache/nuvai-mkl` the workspace jobs use; the default static runs acquire
`mkl-static` (~130 MB compressed); the `dynamic` ones acquire `mkl`,
`mkl-include` and `llvm-openmp` instead (~150 MB).

`checks.rs` holds the assertions all of them make — a `vml::exp`, an `sgemm`
and an `hgemm`, the three calls from the issue's report — so the fixtures differ
in link plumbing and nothing else.
