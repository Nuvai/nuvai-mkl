# Downstream consumer fixtures

Two minimal crates that depend on `nuvai-mkl` **from outside** the workspace,
and whose only purpose is to make the difference between "works for this
repository's own test binaries" and "works for a crate that depends on it"
impossible to miss (#70).

Cargo's build-script linker output splits in two. `cargo:rustc-link-lib` /
`cargo:rustc-link-search` are collected from every build script in the graph and
applied to each final link. `cargo:rustc-link-arg` is scoped to the *emitting
package's* own targets. A consumer that is not the emitting package therefore
receives the first kind and never the second — which is why both link paths used
to be broken downstream while every check in this repository passed: the
workspace's test binaries are emitted *by* `nuvai-mkl`, and a consumer's are not.

| Fixture | Link arguments it declares | Paths it proves |
|---|---|---|
| `dependency-only/` | none — no `build.rs`, no `[build-dependencies]` | static, end to end |
| `build-script-helper/` | `nuvai_mkl_src::emit_binary_link_args()` in `build.rs` | dynamic (default) and static (`--features static`) |

Each is a workspace of its own, so `cargo test --workspace` at the repository
root neither builds them nor unifies their feature selection. Both are
`x86_64-unknown-linux-gnu` fixtures: the `static` feature exists only there, and
the dynamic path's gap is an rpath for conda-forge's MKL shared objects.

The helper fixture carries `nuvai-mkl-src` in **both** `[dependencies]` and
`[build-dependencies]`, which looks redundant and is not:

- Cargo forwards a `links` provider's `DEP_MKL_*` metadata to a build script
  through a normal `[dependencies]` edge **only**. A `[build-dependencies]`
  entry alone receives none of it — the fixture's build script reads
  `DEP_MKL_LIB_DIR` as unset without the first entry, and the binary then links
  and fails to start with `libmkl_rt.so.3: cannot open shared object file`.
- A build script can only *use* code from `[build-dependencies]`, so the call
  itself needs the second entry.

(ADR-0006 records both rules, and `checks.rs`'s sibling fixture is what proves
neither entry is needed under `static`.)

```sh
# Static, dependency only: must link and run with nothing configured.
cargo run --manifest-path consumers/dependency-only/Cargo.toml

# Dynamic, with the one-line build script: must link and run.
cargo run --manifest-path consumers/build-script-helper/Cargo.toml
# …and the same crate with static linking, where that call emits nothing.
cargo run --manifest-path consumers/build-script-helper/Cargo.toml --features static
```

The CI job `x86_64-linux-downstream` runs exactly those three commands. They
need `libclang` (the bindgen pass for the Intel targets) and reach the same
`~/.cache/nuvai-mkl` the workspace jobs use; the static runs acquire
`mkl-static` (~130 MB) and the dynamic run the much smaller `mkl` package.

`checks.rs` holds the assertions all of them make — a `vml::exp`, an `sgemm`
and an `hgemm`, the three calls from the issue's report — so the fixtures differ
in link plumbing and nothing else.
