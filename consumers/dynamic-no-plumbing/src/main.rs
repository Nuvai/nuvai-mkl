//! The `dynamic` opt-in with **no** link plumbing: it builds, and it must not
//! start (#72).
//!
//! This is the negative half of the pair with `../build-script-helper`, and the
//! only fixture whose expected outcome is a failure. It exists because the
//! failure is part of the documented contract now:
//!
//! * `dynamic` is an explicit opt-in, so the caller owns what its own binaries
//!   need — a runtime search path for the acquisition cache, supplied by a
//!   `build.rs` calling `nuvai_mkl_src::emit_binary_link_args()` (as the
//!   `build-script-helper` fixture does), or `LD_LIBRARY_PATH` at run time.
//! * `nuvai-mkl-src`'s build script says exactly that, as a `cargo:warning`, at
//!   *build* time — which is the fix for #72's real complaint: the failure used
//!   to arrive silent, at *load* time, on the *default* path, naming
//!   `libmkl_rt.so.3` and nothing about this crate.
//!
//! So `checks::run()` is never reached: the process dies in the loader before
//! `main`, and the CI step asserts both that the build warned and that the run
//! failed this way. If a later change makes the binary start, that step fails —
//! deliberately — and whoever made it has to come back here and decide whether
//! the warning text, this fixture and the README's dynamic section are all still
//! true.
//!
//! The body is the same `checks.rs` the other two fixtures use, so the three
//! differ in link plumbing and in nothing else — and so this one cannot pass by
//! having failed to compile something.

#[path = "../../checks.rs"]
mod checks;

fn main() {
    checks::run();
    println!("dynamic-no-plumbing consumer linked and ran");
}
