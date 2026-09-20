//! Supply the link arguments Cargo cannot propagate to this crate's binaries
//! (#70).
//!
//! That is the whole of it — the arguments themselves (the runtime rpath, the
//! `--no-as-needed` pair, and the `force_runtime.c` object that keeps
//! `libiomp5`/`libm` in this binary's DT_NEEDED under mold) live in
//! `nuvai_mkl_src::emit_binary_link_args`, which knows the resolved paths
//! because `nuvai-mkl-src` is the crate that acquired them.
//!
//! Without this call the dynamic build of this fixture links and then fails at
//! start-up with `libmkl_rt.so.3: cannot open shared object file`: a build
//! script cannot put a DT_RUNPATH on a *dependent's* executable, and
//! `cargo:rustc-link-arg` is scoped to the emitting package's own targets. See
//! `consumers/README.md` for why the dependency-only fixture needs none of it.

fn main() {
    nuvai_mkl_src::emit_binary_link_args();
}
