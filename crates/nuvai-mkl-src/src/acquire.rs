// The oneMKL acquisition machinery: locate the install on the system, or
// download and extract it from conda-forge into a shared cache.
//
// Every decision here dispatches on the **target** (`CARGO_CFG_TARGET_*`), never
// the host. A build script compiles for and runs on the host, so the
// `#[cfg(target_arch = "aarch64")]` / `cfg!(target_os = ...)` gates this file
// used to carry described the host instead — the trap the module docs of the
// three `build.rs` files describe for backend selection, but in the code that
// acquires. Two consequences, both fixed: cross-compiling to an Intel target
// from Apple Silicon took `locate()`'s aarch64 panic arm instead of acquiring,
// and cross-compiling to a Windows target from Linux fetched **linux-64** conda
// packages.
//
// `build.rs` includes this file, and **only** `build.rs` (#24). It needs
// `ureq`/`zip`/`zstd`/`tar`/`sha2`, and the library target must not: the two are
// built from the same source through `include!`, and no `cfg` distinguishes "I
// am the library target" from "I am a build script" — so leaving this in the
// library's source keeps those crates in `[dependencies]`, and therefore in the
// runtime graph of every downstream build, however it is gated.
//
// `mkl_info.rs` is the half the library *does* compile: `MKL_VERSION`,
// `MklInfo`, and the reader that rebuilds one from the metadata published below.
// `build.rs` includes it immediately before this file, which is where the
// `env`/`Path`/`PathBuf` names this file uses come from.

use std::fs;
use std::io::Read;
use sha2::Digest;

const CONDA_BASE: &str = "https://conda.anaconda.org/conda-forge";
const LINUX_MKL: &str = "mkl-2026.1.0-hecca717_243.conda";
const LINUX_INCLUDE: &str = "mkl-include-2026.1.0-ha770c72_243.conda";
const WIN_MKL: &str = "mkl-2026.1.0-hac47afa_233.conda";
const WIN_INCLUDE: &str = "mkl-include-2026.1.0-h57928b3_233.conda";
const WIN_DEVEL: &str = "mkl-devel-2026.1.0-h57928b3_233.conda";
// The win-64 `mkl` package declares (at the conda level) dependencies on
// `llvm-openmp` and `tbb` for its threading layers. Those runtime DLLs do not
// ship in `mkl` itself, so a faithful conda-forge acquisition must fetch them
// too: `libiomp5md.dll` (OpenMP runtime used by the default `mkl_intel_thread`
// layer) and `tbb12.dll` (TBB threading layer).
const WIN_LLVM_OPENMP: &str = "llvm-openmp-22.1.8-h4fa8253_0.conda";
const WIN_TBB: &str = "tbb-2021.10.0-h91493d7_2.conda";

// SHA-256 of each pinned conda-forge package (from api.anaconda.org/dist).
// Pinning the digest lets `download()` reject a tampered or corrupted archive
// before it is extracted into the linker search path.
//
// The pinned filenames and digests must stay in *this* file: CI keys its
// conda-download cache on `.github/workflows/ci.yml`'s `hashFiles` of this
// path, so moving them elsewhere would leave that cache silently stale.
const LINUX_MKL_SHA256: &str = "c68967a13488684d87fb7ac77b73c6f3f825f2da403707a14e75374c0ce3629f";
const LINUX_INCLUDE_SHA256: &str = "6a8869386f70c5b9d49d02872cf172d2b2a84687509be54f40a5a1c4eddafa97";
const WIN_MKL_SHA256: &str = "ff355522fb0b6e33841167d9ca749147c8734d8be07b63b2ce25b0db043f42ed";
const WIN_INCLUDE_SHA256: &str = "b8809ceb7ad6a48392dcfdc806959a5cbd7bd906c2a996c5650096694f3694e4";
const WIN_DEVEL_SHA256: &str = "102bcfa02484432086f72180e826cbca5db0203267871f1bf37a40e8080d8891";
const WIN_LLVM_OPENMP_SHA256: &str =
    "50c02902bb516eeb56680358f052be38b5bf74b40e78ea4b2a675e84957e7307";
const WIN_TBB_SHA256: &str = "e55a2f1324f0fc8916ab8d590a3944ba1af62de727bb66e3019cf2744d26e679";
// The linux-64 `mkl` package's `libmkl_intel_thread.so.3` leaves its OpenMP
// `omp_*` symbols undefined (no DT_NEEDED on the runtime), exactly as the
// win-64 build does — so the OpenMP runtime must be fetched and linked
// explicitly. conda-forge ships it as `llvm-openmp` on both platforms (the
// `intel-openmp` package is win-64 only); on linux-64 it provides
// `libiomp5.so` (a symlink to `libomp.so`).
const LINUX_LLVM_OPENMP: &str = "llvm-openmp-22.1.8-h4922eb0_0.conda";
const LINUX_LLVM_OPENMP_SHA256: &str =
    "a37aba21b85800af1e7c5b04ba76abab96b6e591eedf99dc6e4df83b0fefd7a5";

/// The target OS Cargo is building for — `CARGO_CFG_TARGET_OS`, which Cargo
/// sets for every build script.
///
/// Read rather than `cfg!`-ed: a build script's `cfg!` describes the host. See
/// the module docs.
fn target_os() -> String {
    env::var("CARGO_CFG_TARGET_OS").expect("Cargo sets CARGO_CFG_TARGET_OS for build scripts")
}

/// The target architecture Cargo is building for — `CARGO_CFG_TARGET_ARCH`.
/// See [`target_os`].
fn target_arch() -> String {
    env::var("CARGO_CFG_TARGET_ARCH").expect("Cargo sets CARGO_CFG_TARGET_ARCH for build scripts")
}

/// Locate MKL: a system oneAPI install first, then download from conda-forge.
///
/// Intel ships no oneMKL for *any* aarch64 target (Apple Silicon or Linux/ARM),
/// so for an aarch64 *target* this panics with a clear pointer to the fallback
/// path. The build script never calls it there (it dispatches on [`backend`]
/// first), so the guard exists for a downstream build script calling `locate()`
/// directly.
///
/// This is not in the library target (#24) — build scripts read the metadata
/// this crate's own build script published, via [`MklInfo::from_build_metadata`],
/// rather than acquiring a second time.
pub fn locate() -> MklInfo {
    // Keyed on the *target* arch. This was `#[cfg(target_arch = ...)]`, which in
    // a build script describes the host — so an Apple Silicon host cross-building
    // for Intel took this panic instead of acquiring, which is what blocked local
    // Intel cross-checks.
    if target_arch() == "aarch64" {
        panic!(
            "Intel oneMKL is unavailable on aarch64 (Intel ships x86_64 builds only); \
             select the Accelerate (macOS) or OpenBLAS (Linux/macOS) fallback via \
             nuvai_mkl_src::backend() instead of nuvai_mkl_src::locate()"
        );
    }
    if let Some(info) = system_mkl() {
        return info;
    }
    download_mkl()
}

/// Detect a system oneAPI install via `MKLROOT` or a well-known path.
fn system_mkl() -> Option<MklInfo> {
    let root = env::var("MKLROOT")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            ["/opt/intel/oneapi/mkl/latest", "/opt/intel/oneapi/mkl/2026.1.0"]
                .iter()
                .map(PathBuf::from)
                .find(|p| p.join("include").join("mkl.h").exists())
        })?;

    if !root.join("include").join("mkl.h").exists() {
        return None;
    }
    let include_dir = root.join("include");
    let lib_dir = if root.join("lib/intel64").exists() {
        root.join("lib/intel64")
    } else {
        root.join("lib")
    };
    // Windows oneAPI installs keep the runtime DLLs under `bin` (or the
    // conda-style `Library/bin`); the Unix loader finds them via rpath.
    let dll_dirs: Vec<PathBuf> = if target_os() == "windows" {
        [root.join("Library").join("bin"), root.join("bin")]
            .into_iter()
            .filter(|p| p.exists())
            .collect()
    } else {
        Vec::new()
    };
    Some(MklInfo { include_dir, lib_dir, omp_lib_dir: None, dll_dirs })
}

/// One downloadable conda package: `(filename, sha256)`.
type Pkg<'a> = (&'a str, &'a str);

/// The package set for one platform: base MKL, headers, an optional `devel`
/// package (Windows import libs), and the runtime packages.
type PkgSet<'a> = (Pkg<'a>, Pkg<'a>, Option<Pkg<'a>>, &'a [Pkg<'a>]);

/// Download + extract MKL into the shared cache, returning its paths.
fn download_mkl() -> MklInfo {
    let pkg_dir = cache_dir().join(format!("mkl-{MKL_VERSION}"));

    let ((mkl_file, mkl_sha), (include_file, include_sha), devel, runtime): PkgSet<'_> =
        match target_os().as_str() {
        "linux" => (
            (LINUX_MKL, LINUX_MKL_SHA256),
            (LINUX_INCLUDE, LINUX_INCLUDE_SHA256),
            None,
            &[(LINUX_LLVM_OPENMP, LINUX_LLVM_OPENMP_SHA256)][..],
        ),
        // The win-64 `mkl` package is 26 DLLs with no `.lib` — import libs ship
        // in a third package, `mkl-devel`. Its threading layers also need the
        // OpenMP runtime (`libiomp5md.dll` from `llvm-openmp`) and the TBB
        // threading layer (`tbb12.dll` from `tbb`), which `mkl` declares as
        // conda dependencies but which do not ship inside `mkl` itself.
        "windows" => (
            (WIN_MKL, WIN_MKL_SHA256),
            (WIN_INCLUDE, WIN_INCLUDE_SHA256),
            Some((WIN_DEVEL, WIN_DEVEL_SHA256)),
            &[(WIN_LLVM_OPENMP, WIN_LLVM_OPENMP_SHA256), (WIN_TBB, WIN_TBB_SHA256)][..],
        ),
        other => panic!(
            "unsupported target for Intel oneMKL {MKL_VERSION}: {other} is not x86_64 \
             Linux/Windows. On aarch64 use the `accelerate`/`openblas` fallback."
        ),
    };

    let mkl_root = fetch_and_extract_conda(mkl_file, mkl_sha, &pkg_dir);
    let include_root = fetch_and_extract_conda(include_file, include_sha, &pkg_dir);

    if let Some((devel_file, devel_sha)) = devel {
        // Windows conda packages use a `Library/` prefix (`Library/include`,
        // `Library/lib`, `Library/bin`); import libs come from `mkl-devel`.
        let devel_root = fetch_and_extract_conda(devel_file, devel_sha, &pkg_dir);
        let mut dll_dirs = vec![mkl_root.join("Library").join("bin")];
        for (file, sha) in runtime {
            dll_dirs.push(fetch_and_extract_conda(file, sha, &pkg_dir).join("Library").join("bin"));
        }
        MklInfo {
            include_dir: include_root.join("Library").join("include"),
            lib_dir: devel_root.join("Library").join("lib"),
            omp_lib_dir: None,
            dll_dirs,
        }
    } else {
        // Linux conda packages lay out headers under `include/` and libs under
        // `lib/`; the runtime loader finds the shared objects via rpath.
        let mut omp_lib_dir = None;
        for (file, sha) in runtime {
            // The runtime entries are OpenMP runtime packages (`llvm-openmp`)
            // whose `lib/` holds `libiomp5.so` (a symlink to `libomp.so`).
            omp_lib_dir = Some(fetch_and_extract_conda(file, sha, &pkg_dir).join("lib"));
        }
        MklInfo {
            include_dir: include_root.join("include"),
            lib_dir: mkl_root.join("lib"),
            omp_lib_dir,
            dll_dirs: Vec::new(),
        }
    }
}

fn fetch_and_extract_conda(file: &str, sha256: &str, pkg_dir: &Path) -> PathBuf {
    let url = format!("{CONDA_BASE}/{}/{}", conda_subdir(), file);
    let dest = pkg_dir.join(file);
    if !dest.exists() {
        download(&url, &dest);
    }
    // Verify both freshly downloaded and cached archives: a corrupted or
    // tampered cache file must fail loudly rather than reach the linker path.
    verify_sha256(&dest, sha256, file);
    let out = pkg_dir.join(file.trim_end_matches(".conda"));
    if !out.exists() {
        fs::create_dir_all(&out).expect("create conda extract dir");
        extract_conda(&dest, &out);
    }
    out
}

/// Confirm `path` hashes to `expected` (the pinned conda-forge digest),
/// panicking with a clear pointer to clear the cache if it does not.
fn verify_sha256(path: &Path, expected: &str, file: &str) {
    let mut archive = fs::File::open(path)
        .unwrap_or_else(|e| panic!("open downloaded archive {}: {e}", path.display()));
    let mut hasher = sha2::Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = archive
            .read(&mut buf)
            .unwrap_or_else(|e| panic!("hash downloaded archive {}: {e}", path.display()));
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected {
        panic!(
            "checksum mismatch for {file}: expected {expected}, got {actual}. \
             Delete {} and retry.",
            path.display()
        );
    }
}

fn conda_subdir() -> &'static str {
    if target_os() == "windows" {
        "win-64"
    } else {
        "linux-64"
    }
}

fn cache_dir() -> PathBuf {
    // `HOME` is commonly unset in stock Windows shells; fall back to
    // `USERPROFILE` there so the cache does not silently land in `.`.
    let base = env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|_| env::var("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .or_else(|_| env::var("USERPROFILE").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|_| PathBuf::from("."));
    let dir = base.join("nuvai-mkl");
    fs::create_dir_all(&dir).expect("create MKL cache dir");
    dir
}

fn download(url: &str, dest: &Path) {
    eprintln!("[nuvai-mkl-src] downloading {url}");
    let resp = ureq::get(url)
        .call()
        .unwrap_or_else(|e| panic!("failed to download {url}: {e}"));
    let mut body = Vec::new();
    resp.into_reader()
        .read_to_end(&mut body)
        .expect("read download body");
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).expect("create download parent dir");
    }
    fs::write(dest, body).expect("write downloaded archive");
}

/// A `.conda` file is a ZIP containing `info-*.tar.zst` and `pkg-*.tar.zst`.
fn extract_conda(conda_path: &Path, dest: &Path) {
    let file = fs::File::open(conda_path).expect("open .conda");
    let mut zip = zip::ZipArchive::new(file).expect("open .conda as zip");
    for i in 0..zip.len() {
        let entry = zip.by_index(i).expect("read zip entry");
        let name = entry.name().to_string();
        if name.starts_with("pkg-") && name.ends_with(".tar.zst") {
            let decoder = zstd::stream::read::Decoder::new(entry).expect("zstd decoder");
            let mut tar = tar::Archive::new(decoder);
            tar.unpack(dest).expect("extract conda payload");
            return;
        }
    }
    panic!("no pkg-*.tar.zst found in {}", conda_path.display());
}
