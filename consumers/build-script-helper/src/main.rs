//! A downstream consumer of `nuvai-mkl` that supplies the arguments Cargo
//! cannot propagate to its own binary (#70).
//!
//! The other fixture (`consumers/dependency-only`) configures nothing at all,
//! and works because everything it needs is a *propagating* directive. This one
//! covers the dynamic path, where two things are not: an executable's
//! DT_RUNPATH, and the `force_runtime.c` object that keeps `libiomp5`/`libm`
//! in that DT_RUNPATH's DT_NEEDED under mold (#44). Both belong to whoever owns
//! the binary, and `build.rs` here is the entire cost of getting them — one
//! call, plus the two `nuvai-mkl-src` manifest entries `Cargo.toml` explains.
//!
//! Run with `--features dynamic` it is the `libmkl_rt.so.3: cannot open shared
//! object file` half of the issue; run with no feature (or `--features static`,
//! which is the same thing spelled out — see the manifest) the same call emits
//! nothing at all, because a static link has no shared object to point an rpath
//! at. Before #72 the first of those was the *default*, so this fixture's
//! no-feature build was the failing one and the pair read as "the fix" and "the
//! workaround"; now the default is the one that needs nothing, and the fixture
//! exists to prove the `dynamic` opt-in is still usable when its arguments are
//! supplied.

// The body is shared with the other fixture, so the two differ only in link
// plumbing — see `consumers/README.md`.
#[path = "../../checks.rs"]
mod checks;

fn main() {
    checks::run();

    // The crate whose metadata this one's build script reads is also usable
    // from code, for reporting which backend the build selected (see the
    // README's "Installation"). Referencing it here is what keeps Cargo from
    // noting the `nuvai-mkl-src` dependency above as unused.
    println!("backend: {}", nuvai_mkl_src::backend_tag(nuvai_mkl_src::backend()));
    println!("build-script-helper consumer linked and ran");
}
