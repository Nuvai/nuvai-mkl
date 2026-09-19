// The oneMKL version and the description of a resolved install — the half of
// the acquisition surface that the *library* target compiles.
//
// `lib.rs` includes this file; `build.rs` includes it alongside `acquire.rs`.
// It must therefore stay free of `ureq`/`zip`/`zstd`/`tar`/`sha2`: anything
// added here puts those crates back into `[dependencies]` and back into the
// runtime graph of every downstream build (#24). The machinery that needs them
// lives in `acquire.rs`, which only `build.rs` includes.
//
// Because `build.rs` includes both files into one module, the imports below are
// shared with `acquire.rs`, which relies on `env`/`Path`/`PathBuf` rather than
// declaring them again — a second `use std::env;` in the same module is E0252.
// `acquire.rs` contributes the ones this file has no use for (`fs`,
// `io::Read`, `sha2::Digest`).

use std::env;
use std::path::Path;
use std::path::PathBuf;

/// The oneMKL version this crate acquires and links.
pub const MKL_VERSION: &str = "2026.1.0";

/// Resolved location of an MKL install.
#[derive(Debug, Clone)]
pub struct MklInfo {
    /// Directory containing the `mkl*.h` headers.
    pub include_dir: PathBuf,
    /// Directory containing the MKL libraries.
    pub lib_dir: PathBuf,
    /// Directory containing the OpenMP runtime (`libiomp5.so`, a symlink to
    /// `libomp.so`) that `libmkl_intel_thread.so.3` needs at load time. Linux
    /// conda acquisition only; `None` on Windows (the runtime DLL is surfaced
    /// via [`dll_dirs`]) and on a system oneAPI install (its OpenMP runtime
    /// ships beside MKL and the loader finds it on the standard path).
    pub omp_lib_dir: Option<PathBuf>,
    /// Directories containing the MKL runtime DLLs (Windows only; empty on
    /// platforms where the runtime loader finds them via rpath / system search).
    ///
    /// On `x86_64-pc-windows-msvc` the Windows loader does not search the
    /// link-search path at runtime, so callers that need to load the MKL DLLs
    /// (e.g. `cargo run`/`cargo test` on a conda-forge acquisition) must add
    /// every directory here to `PATH` (or deploy the DLLs beside the
    /// executable). Multiple directories are listed because the conda-forge
    /// `mkl` package depends on runtime DLLs that ship in their own packages
    /// under their own `Library/bin`: `libiomp5md.dll` (OpenMP runtime for the
    /// default `mkl_intel_thread` layer, from `llvm-openmp`) and `tbb12.dll`
    /// (TBB threading layer, from `tbb`).
    pub dll_dirs: Vec<PathBuf>,
}

impl MklInfo {
    /// Primary MKL runtime DLL directory, if any (the first of [`dll_dirs`]).
    ///
    /// Returns `Some` on Windows (conda-forge acquisition or a system oneAPI
    /// install); `None` on platforms where the loader finds the shared objects
    /// via rpath / the system search path. On Windows, prepend [`dll_dirs`] to
    /// `PATH` (or copy the DLLs beside the executable) before `cargo run` /
    /// `cargo test` so the loader can resolve `mkl_rt.3.dll`.
    pub fn dll_dir(&self) -> Option<&Path> {
        self.dll_dirs.first().map(PathBuf::as_path)
    }

    /// All directories that must be on `PATH` for the MKL runtime DLLs to load.
    pub fn dll_dirs(&self) -> &[PathBuf] {
        &self.dll_dirs
    }

    /// Rebuild an [`MklInfo`] from the `DEP_MKL_*` variables Cargo sets for the
    /// build script of a crate that directly depends on this one.
    ///
    /// The library target no longer acquires anything (#24): `build.rs` does,
    /// once, and publishes the result as `cargo::metadata` (this crate declares
    /// `links = "mkl"`, which is what makes Cargo forward it). A dependent build
    /// script reads it back here instead of re-running the acquisition — so the
    /// expensive path, a conda-forge download, happens once for the whole build
    /// graph rather than once per dependent.
    ///
    /// Returns `None` when the variables are absent. That is the case on the
    /// aarch64 backends, where nothing is acquired and only `BACKEND` is
    /// published, and whenever this is called outside a build script.
    pub fn from_build_metadata() -> Option<Self> {
        Self::from_lookup(|key| env::var(key).ok())
    }

    /// [`from_build_metadata`] against an arbitrary lookup, so the parsing is
    /// testable without mutating the process environment — unsafe in edition
    /// 2024 and racy under the test harness's threads.
    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Option<Self> {
        let include_dir = PathBuf::from(lookup("DEP_MKL_INCLUDE_DIR")?);
        let lib_dir = PathBuf::from(lookup("DEP_MKL_LIB_DIR")?);
        // Optional on two different grounds: the OpenMP runtime is a separate
        // package on Linux conda only, and a system oneAPI install keeps it
        // beside MKL where the loader finds it unaided.
        let omp_lib_dir = lookup("DEP_MKL_OMP_LIB_DIR").map(PathBuf::from);
        // Indexed rather than a repeated `DLL_DIR`: Cargo keeps only the *last*
        // value for a repeated metadata key, so emitting one key in a loop
        // publishes exactly the final directory. Scan until the first gap, which
        // is why the emitter writes contiguous indices from zero.
        let mut dll_dirs = Vec::new();
        while let Some(dir) = lookup(&format!("DEP_MKL_DLL_DIR_{}", dll_dirs.len())) {
            dll_dirs.push(PathBuf::from(dir));
        }
        Some(Self {
            include_dir,
            lib_dir,
            omp_lib_dir,
            dll_dirs,
        })
    }
}

// Named rather than a plain `mod tests`: this file and `backend.rs` are both
// included into the *same* module (`lib.rs`'s crate root), so two `mod tests`
// would be E0428 there.
#[cfg(test)]
mod mkl_info_tests {
    use super::*;
    use std::collections::HashMap;

    /// Build an `MklInfo` from a fixed set of "environment" entries. The map
    /// lives here rather than in the process environment, so these run in
    /// parallel with every other test in the crate.
    fn from_pairs(pairs: &[(&str, &str)]) -> Option<MklInfo> {
        let map: HashMap<&str, &str> = pairs.iter().copied().collect();
        MklInfo::from_lookup(|key| map.get(key).map(|v| (*v).to_string()))
    }

    #[test]
    fn reads_every_published_field() {
        let info = from_pairs(&[
            ("DEP_MKL_INCLUDE_DIR", "/mkl/include"),
            ("DEP_MKL_LIB_DIR", "/mkl/lib"),
            ("DEP_MKL_OMP_LIB_DIR", "/omp/lib"),
            ("DEP_MKL_DLL_DIR_0", "/mkl/Library/bin"),
            ("DEP_MKL_DLL_DIR_1", "/omp/Library/bin"),
        ])
        .expect("all required variables present");

        assert_eq!(info.include_dir, PathBuf::from("/mkl/include"));
        assert_eq!(info.lib_dir, PathBuf::from("/mkl/lib"));
        assert_eq!(info.omp_lib_dir, Some(PathBuf::from("/omp/lib")));
        // Order matters: the Windows loader is given these in sequence, and
        // `dll_dir()` promises the first one.
        assert_eq!(
            info.dll_dirs(),
            [
                PathBuf::from("/mkl/Library/bin"),
                PathBuf::from("/omp/Library/bin")
            ]
        );
        assert_eq!(
            info.dll_dir(),
            Some(Path::new("/mkl/Library/bin")),
            "the first published directory is the primary one"
        );
    }

    /// The two required keys are required: a build script that saw only part of
    /// the metadata must fail loudly rather than proceed with a bogus `"."`.
    #[test]
    fn required_fields_are_required() {
        assert!(from_pairs(&[("DEP_MKL_LIB_DIR", "/mkl/lib")]).is_none());
        assert!(from_pairs(&[("DEP_MKL_INCLUDE_DIR", "/mkl/include")]).is_none());
        assert!(from_pairs(&[]).is_none());
    }

    /// The aarch64 backends acquire nothing, so a caller reaching this on an
    /// unsupported-target build gets `None` — never a partly-filled struct.
    #[test]
    fn the_optional_fields_may_be_absent() {
        let info = from_pairs(&[
            ("DEP_MKL_INCLUDE_DIR", "/mkl/include"),
            ("DEP_MKL_LIB_DIR", "/mkl/lib"),
        ])
        .expect("both required variables present");

        assert_eq!(info.omp_lib_dir, None);
        assert!(info.dll_dirs().is_empty());
        assert_eq!(info.dll_dir(), None);
    }

    /// A gap ends the scan. The emitter writes contiguous indices, so a gap
    /// means the remaining entries are from some other source and must not be
    /// guessed at.
    #[test]
    fn the_dll_scan_stops_at_the_first_gap() {
        let info = from_pairs(&[
            ("DEP_MKL_INCLUDE_DIR", "/mkl/include"),
            ("DEP_MKL_LIB_DIR", "/mkl/lib"),
            ("DEP_MKL_DLL_DIR_0", "/first"),
            // no _1
            ("DEP_MKL_DLL_DIR_2", "/skipped"),
        ])
        .expect("both required variables present");

        assert_eq!(info.dll_dirs(), [PathBuf::from("/first")]);
    }

    #[test]
    fn the_version_is_pinned_to_the_headers_this_crate_binds() {
        assert_eq!(MKL_VERSION, "2026.1.0");
    }
}
