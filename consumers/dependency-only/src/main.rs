//! A downstream consumer of `nuvai-mkl` that declares **no** link plumbing at
//! all (#70).
//!
//! This crate has no `build.rs`, no `[build-dependencies]`, no
//! `.cargo/config.toml`, no `RUSTFLAGS` and no `LD_LIBRARY_PATH`. Running it is
//! therefore the assertion: if this binary links *and* starts, the `nuvai-mkl`
//! dependency on its own was enough to link Intel oneMKL's static archives,
//! pull their members in, and resolve every symbol.
//!
//! Before #70 this was not a possibility. The group link over
//! `libmkl_intel_lp64.a`/`libmkl_sequential.a`/`libmkl_core.a` was a
//! `cargo:rustc-link-arg` from `nuvai-mkl`, and Cargo scopes those to the
//! *emitting package's* own targets — so a consumer received the archive
//! directory on its search path and not one `-l` flag, and failed to link with
//! `undefined symbol: cblas_sgemm`. It is now a linker script, which is a `-l`
//! input, so it propagates the way `rustc-link-lib`/`rustc-link-search` do.
//!
//! `x86_64-unknown-linux-gnu` only, by construction: the static link and the
//! group script it selects exist nowhere else.
//!
//! Since #72 this is also the *default* link mode on that target, which is why
//! this fixture's manifest no longer names the `static` feature: it declares a
//! bare `nuvai-mkl` dependency, exactly as the issue's repro did, and the
//! binary must still link and run. Before #72 the same manifest took the
//! dynamic path and failed at start-up — `libmkl_rt.so.3: cannot open shared
//! object file` — because nothing had put a runtime search path on this
//! binary, and nothing could: `rustc-link-arg` does not propagate to a
//! dependent, and no other Cargo channel carries an rpath (ADR-0006).

// The body is shared with the other fixture, which asserts the same results
// through a different link path — see `consumers/README.md`.
#[path = "../../checks.rs"]
mod checks;

fn main() {
    checks::run();
    println!("dependency-only consumer linked and ran");
}
