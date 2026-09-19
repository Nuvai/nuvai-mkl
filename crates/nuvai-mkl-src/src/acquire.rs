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
use std::io::{self, Read};
use std::time::Duration;
use sha2::Digest;

/// How many times to attempt one package download before giving up.
///
/// The CDN in front of conda-forge refuses transiently. Epic #19's CI run saw
/// it answer `2xx` with an **empty body**, which the old `download()` wrote to
/// disk and then reported as a *checksum* mismatch — i.e. as a corrupt cache,
/// sending the reader to delete a file that was never the problem.
const DOWNLOAD_ATTEMPTS: u32 = 3;

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
/// path. In the build script the arm is unreachable — `main` dispatches on
/// [`backend_for_target`] first and only `Backend::IntelMkl` reaches
/// `emit_intel_mkl`, which is the sole caller, and that backend is never
/// selected for an aarch64 target. The guard is kept as the check for that
/// invariant rather than as a caller-facing one: it used to be described as
/// protection for "a downstream build script calling `locate()` directly", but
/// since #24 there is no such caller to protect. This file is `include!`d by
/// `build.rs` alone, so `locate` is private to that build script's binary —
/// it is not a library item, cannot be imported, and no dependent can reach it.
/// A dependent build script reads what this one published instead, through
/// [`MklInfo::from_build_metadata`], and never acquires a second time.
///
/// #39 asked for this to return `Result` rather than panic, on the premise that
/// a third-party build script could call it and get a hard panic. That premise
/// is gone: the removal of `locate` from the library target (#24, #60) removed
/// the public entry point the issue described, and the build-script-internal
/// caller cannot reach the panicking arm. Nothing in the library target panics
/// on aarch64 either — [`MklInfo::from_build_metadata`] answers `Option`, and
/// [`backend_for_target`] answers `Result`.
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

/// Base URL the `.conda` packages are fetched from.
///
/// Overridable by `NUVAI_MKL_CONDA_BASE` for two reasons: a mirror can stand in
/// when conda-forge's CDN refuses a request (epic #19 — every archive is still
/// checked against its pinned digest, so a mirror cannot substitute different
/// bytes), and the refusal path can be exercised against a server that returns
/// it on demand. That last part matters because `acquire.rs` is included only
/// by `build.rs`: nothing in this file is reachable from `cargo test`, so an
/// end-to-end check of the download path has to run a real build.
fn conda_base() -> String {
    env::var("NUVAI_MKL_CONDA_BASE").unwrap_or_else(|_| CONDA_BASE.to_string())
}

fn fetch_and_extract_conda(file: &str, sha256: &str, pkg_dir: &Path) -> PathBuf {
    let url = format!("{}/{}/{}", conda_base(), conda_subdir(), file);
    let dest = pkg_dir.join(file);
    if dest.exists() {
        // Check a cached archive too — a corrupted or tampered cache file must
        // fail loudly rather than reach the linker path. It is *not* replaced
        // here: re-reading the same bytes cannot change the answer, and deleting
        // it would race the second build script that shares this directory (see
        // `staging_path`). Fail loudly and let the re-run clear it.
        if let Err(e) = verify_sha256(&dest, sha256, file) {
            panic!("{e}. Delete {} and re-run.", dest.display());
        }
    } else {
        download_verified(&url, &dest, sha256, file);
    }
    let out = pkg_dir.join(file.trim_end_matches(".conda"));
    if !out.exists() {
        fs::create_dir_all(&out).expect("create conda extract dir");
        extract_conda(&dest, &out);
    }
    out
}

/// Hash `path` and compare it to `expected`.
///
/// The received length is part of the error because it is what separates the
/// two ways this fails: a short or empty body is a refused transfer, while a
/// plausible length that still misses the digest is *other content* altogether —
/// a proxy's error page, say. The digest alone can only say "not this".
fn verify_sha256(path: &Path, expected: &str, file: &str) -> Result<(), String> {
    let mut archive = fs::File::open(path)
        .map_err(|e| format!("open downloaded archive {}: {e}", path.display()))?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut len = 0u64;
    loop {
        let n = archive
            .read(&mut buf)
            .map_err(|e| format!("hash downloaded archive {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        len += n as u64;
        hasher.update(&buf[..n]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected {
        return Err(format!(
            "checksum mismatch for {file}: expected {expected}, got {actual} ({len} bytes)"
        ));
    }
    Ok(())
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

/// Download `url` and install it at `dest`, retrying the whole attempt before
/// giving up.
///
/// The retry has to cover the digest, not just the transfer. The CDN in front of
/// conda-forge answers a request it will not serve with *either* an empty body
/// *or* a body of some other content entirely — CI has seen both, from the same
/// URL minutes apart — and only the digest recognises the second shape. Retrying
/// the transfer alone would accept the garbage and report it as a corrupt cache.
fn download_verified(url: &str, dest: &Path, sha256: &str, file: &str) {
    let mut last = String::new();
    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        if attempt > 1 {
            // Back off between attempts: a CDN that is rate-limiting wants the
            // gap, not another request.
            let pause = Duration::from_secs(1 << (attempt - 1));
            eprintln!(
                "[nuvai-mkl-src] retrying in {pause:?} \
                 (attempt {attempt}/{DOWNLOAD_ATTEMPTS}): {last}"
            );
            std::thread::sleep(pause);
        }
        match fetch_verified(url, dest, sha256, file) {
            Ok(()) => return,
            Err(e) => last = e,
        }
    }
    panic!("failed to download {url} after {DOWNLOAD_ATTEMPTS} attempts: {last}");
}

/// Where one attempt stages its bytes before they are known to be good.
///
/// The name is per-process, and `dest` is only ever written by a rename of a
/// file that already hashed correctly — so a *rejected* attempt leaves nothing
/// behind but its own staging file.
///
/// That matters because `nuvai-mkl-src` is both a dependency and a
/// build-dependency of the crates above it: Cargo builds it as two units and
/// runs **two** build scripts, aimed at this one cache directory. On CI both
/// download the same archives at the same time. Sharing a staging name would let
/// them truncate each other mid-download, and removing `dest` on a failed
/// attempt — which an earlier revision of this code did — lets a unit whose
/// fetch was refused delete the archive the other one had just verified, taking
/// it down with `NotFound` somewhere later. Both run on the same machine with no
/// coordination beyond these filenames, so neither may touch the other's work.
fn staging_path(dest: &Path) -> PathBuf {
    dest.with_extension(format!("{}.part", std::process::id()))
}

/// One attempt: stage the bytes, check them, then install them at `dest`.
fn fetch_verified(url: &str, dest: &Path, sha256: &str, file: &str) -> Result<(), String> {
    let tmp = staging_path(dest);
    let _ = fs::remove_file(&tmp); // leftovers from a build that was interrupted
    let outcome = download_to(url, &tmp).and_then(|()| verify_sha256(&tmp, sha256, file));
    if let Err(e) = outcome {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    install(&tmp, dest, sha256, file)
}

/// Move a verified staging file into place, without disturbing a good archive
/// another build script may have installed first.
///
/// Both copies are checked against the same pinned digest, so there is nothing
/// to choose between them and no reason to overwrite one with the other.
fn install(tmp: &Path, dest: &Path, sha256: &str, file: &str) -> Result<(), String> {
    if dest.exists() && verify_sha256(dest, sha256, file).is_ok() {
        let _ = fs::remove_file(tmp);
        return Ok(());
    }
    match fs::rename(tmp, dest) {
        Ok(()) => Ok(()),
        // Windows refuses to rename onto an existing file, so a failure here
        // usually means the other build script won the race. Re-check before
        // treating it as an error.
        Err(e) => {
            let installed = verify_sha256(dest, sha256, file).is_ok();
            let _ = fs::remove_file(tmp);
            if installed {
                Ok(())
            } else {
                Err(format!("install {}: {e}", dest.display()))
            }
        }
    }
}

/// One transfer, staged at `path`. Never removes `path`; the caller owns it.
fn download_to(url: &str, path: &Path) -> Result<(), String> {
    eprintln!("[nuvai-mkl-src] downloading {url}");
    let resp = ureq::get(url).call().map_err(|e| format!("{e}"))?;
    let status = resp.status();
    let declared = resp
        .header("Content-Length")
        .and_then(|v| v.trim().parse::<u64>().ok());

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let mut file = fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let written = io::copy(&mut resp.into_reader(), &mut file)
        .map_err(|e| format!("read {url} body: {e}"))?;
    drop(file);

    match unusable_body(written, declared) {
        Some(reason) => Err(format!("HTTP {status} from {url}: {reason}")),
        None => Ok(()),
    }
}

/// Why a downloaded body cannot be used, or `None` when it can.
///
/// A `2xx` whose body is empty — or shorter than the `Content-Length` it
/// declared — is how the CDN in front of conda-forge refuses a request it will
/// not serve. That is a defence aimed at the caller, not evidence of a bad
/// archive, and the two want different responses: a retry and a diagnostic
/// here, versus clearing the cache there. A non-empty body is accepted when the
/// response declared no length (chunked transfer); the pinned digest in
/// [`verify_sha256`] stays the only gate in that case.
fn unusable_body(written: u64, declared: Option<u64>) -> Option<String> {
    if written == 0 {
        return Some("empty body".to_string());
    }
    declared
        .filter(|declared| *declared != written)
        .map(|declared| format!("truncated body: {written} of {declared} bytes"))
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
