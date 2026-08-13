// Shared MKL acquisition logic.
//
// This module is `include!`d by both `build.rs` (to emit linker directives)
// and `lib.rs` (so downstream build scripts can call [`locate`] to find the
// headers for bindgen). It locates oneMKL 2026.1.0 on the system, or
// downloads + extracts it from conda-forge into a shared cache.

#[cfg(not(target_arch = "aarch64"))]
use std::env;
#[cfg(not(target_arch = "aarch64"))]
use std::fs;
#[cfg(not(target_arch = "aarch64"))]
use std::io::Read;
use std::path::PathBuf;
#[cfg(not(target_arch = "aarch64"))]
use std::path::Path;

/// The oneMKL version this crate acquires and links.
pub const MKL_VERSION: &str = "2026.1.0";

#[cfg(not(target_arch = "aarch64"))]
const CONDA_BASE: &str = "https://conda.anaconda.org/conda-forge";
#[cfg(not(target_arch = "aarch64"))]
const LINUX_MKL: &str = "mkl-2026.1.0-hecca717_243.conda";
#[cfg(not(target_arch = "aarch64"))]
const LINUX_INCLUDE: &str = "mkl-include-2026.1.0-ha770c72_243.conda";
#[cfg(not(target_arch = "aarch64"))]
const WIN_MKL: &str = "mkl-2026.1.0-hac47afa_233.conda";
#[cfg(not(target_arch = "aarch64"))]
const WIN_INCLUDE: &str = "mkl-include-2026.1.0-h57928b3_233.conda";

/// Resolved location of an MKL install.
#[derive(Debug, Clone)]
pub struct MklInfo {
    /// Directory containing the `mkl*.h` headers.
    pub include_dir: PathBuf,
    /// Directory containing the MKL libraries.
    pub lib_dir: PathBuf,
}

/// Locate MKL: a system oneAPI install first, then download from conda-forge.
///
/// Intel ships no oneMKL for Apple Silicon, so on `aarch64-apple-darwin` this
/// panics with a clear pointer to the fallback path — the build script never
/// calls it there (it dispatches on [`backend`] instead), so this guard only
/// fires if a downstream build script calls `locate()` directly on aarch64.
pub fn locate() -> MklInfo {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        panic!(
            "Intel oneMKL is unavailable on Apple Silicon; select the Accelerate/OpenBLAS \
             fallback via nuvai_mkl_src::backend() instead of nuvai_mkl_src::locate()"
        );
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        if let Some(info) = system_mkl() {
            return info;
        }
        download_mkl()
    }
}

/// Detect a system oneAPI install via `MKLROOT` or a well-known path.
#[cfg(not(target_arch = "aarch64"))]
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
    Some(MklInfo { include_dir, lib_dir })
}

/// Download + extract MKL into the shared cache, returning its paths.
#[cfg(not(target_arch = "aarch64"))]
fn download_mkl() -> MklInfo {
    let pkg_dir = cache_dir().join(format!("mkl-{MKL_VERSION}"));

    let (mkl_file, include_file) =
        match (cfg!(target_os = "linux"), cfg!(target_os = "windows"), cfg!(target_os = "macos")) {
            (true, _, _) => (LINUX_MKL, LINUX_INCLUDE),
            (_, true, _) => (WIN_MKL, WIN_INCLUDE),
            (_, _, true) => panic!(
                "macOS NuGet acquisition is not yet wired; install oneAPI and set MKLROOT."
            ),
            _ => panic!(
                "unsupported target for Intel oneMKL {MKL_VERSION}: MKL is x86_64 \
                 Linux/Windows/macOS only. On aarch64 use the `accelerate`/`openblas` fallback."
            ),
        };

    let mkl_root = fetch_and_extract_conda(mkl_file, &pkg_dir);
    let include_root = fetch_and_extract_conda(include_file, &pkg_dir);

    MklInfo {
        // conda packages lay out headers under `include/` and libs under `lib/`.
        include_dir: include_root.join("include"),
        lib_dir: mkl_root.join("lib"),
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn fetch_and_extract_conda(file: &str, pkg_dir: &Path) -> PathBuf {
    let url = format!("{CONDA_BASE}/{}/{}", conda_subdir(), file);
    let dest = pkg_dir.join(file);
    if !dest.exists() {
        download(&url, &dest);
    }
    let out = pkg_dir.join(file.trim_end_matches(".conda"));
    if !out.exists() {
        fs::create_dir_all(&out).expect("create conda extract dir");
        extract_conda(&dest, &out);
    }
    out
}

#[cfg(not(target_arch = "aarch64"))]
fn conda_subdir() -> &'static str {
    if cfg!(target_os = "windows") {
        "win-64"
    } else {
        "linux-64"
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn cache_dir() -> PathBuf {
    let base = env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|_| env::var("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|_| PathBuf::from("."));
    let dir = base.join("nuvai-mkl");
    fs::create_dir_all(&dir).expect("create MKL cache dir");
    dir
}

#[cfg(not(target_arch = "aarch64"))]
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
#[cfg(not(target_arch = "aarch64"))]
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
