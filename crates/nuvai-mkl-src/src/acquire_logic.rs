// The pure, `std`-only decisions behind the acquisition code in `acquire.rs`,
// split out so that they can be unit-tested (#95).
//
// `acquire.rs` is `include!`d by `build.rs` alone (#24) because it needs
// `ureq`/`zip`/`zstd`/`tar`/`sha2`, so nothing in it is compiled by `cargo test`.
// Everything here takes its inputs as arguments — the process id, the target OS,
// the cache environment variables, the digest, the verifier — and touches no
// network, no environment and no crate outside `std`, so it can be included from
// both places:
//
// * `build.rs` includes it before `acquire.rs`, which calls it. This is the only
//   place it is *used*.
// * `lib.rs` includes it under `#[cfg(test)]` only, so `cargo test` runs the
//   tests below. It is not compiled into the library proper: nothing there
//   calls it, and including it unconditionally would raise `dead_code` in that
//   target.
//
// It must stay free of the acquisition crates (#24), and it declares **no**
// `use` items: `build.rs` shares one module between this file, `mkl_info.rs`
// and `acquire.rs`, where a repeated import is E0252. Paths are spelled out.

/// Why a downloaded body cannot be used, or `None` when it can.
///
/// A `2xx` whose body is empty — or shorter than the `Content-Length` it
/// declared — is how the CDN in front of conda-forge refuses a request it will
/// not serve. That is a defence aimed at the caller, not evidence of a bad
/// archive, and the two want different responses: a retry and a diagnostic
/// here, versus clearing the cache there. A non-empty body is accepted when the
/// response declared no length (chunked transfer); the pinned digest checked by
/// [`check_digest`] stays the only gate in that case.
fn unusable_body(written: u64, declared: Option<u64>) -> Option<String> {
    if written == 0 {
        return Some("empty body".to_string());
    }
    declared
        .filter(|declared| *declared != written)
        .map(|declared| format!("truncated body: {written} of {declared} bytes"))
}

/// Where one download attempt stages its bytes, for the process `pid`.
///
/// Per-process so that the two build scripts sharing one cache directory never
/// truncate each other's staging file; see `staging_path` in `acquire.rs` for
/// the race this prevents.
fn staging_path_for(dest: &std::path::Path, pid: u32) -> std::path::PathBuf {
    dest.with_extension(format!("{pid}.part"))
}

/// The conda-forge subdirectory serving a *target* OS.
fn conda_subdir_for(target_os: &str) -> &'static str {
    if target_os == "windows" {
        "win-64"
    } else {
        "linux-64"
    }
}

/// The directory the MKL cache lives under, given the three environment
/// variables that can place it — `None` meaning unset.
///
/// Precedence: `XDG_CACHE_HOME` (used as is), then `HOME` and then
/// `USERPROFILE` (each with `.cache` appended), then `.`. `HOME` is commonly
/// unset in stock Windows shells, hence the `USERPROFILE` fallback: the cache
/// must not silently land in the working directory there. A variable that is
/// set but empty counts as set, as `std::env::var` reports it.
fn cache_base(
    xdg_cache_home: Option<&str>,
    home: Option<&str>,
    userprofile: Option<&str>,
) -> std::path::PathBuf {
    xdg_cache_home
        .map(std::path::PathBuf::from)
        .or_else(|| home.map(|h| std::path::PathBuf::from(h).join(".cache")))
        .or_else(|| userprofile.map(|h| std::path::PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| std::path::PathBuf::from("."))
}

/// Compare a computed digest with the pinned one.
///
/// The received length is part of the error because it is what separates the
/// two ways this fails: a short or empty body is a refused transfer, while a
/// plausible length that still misses the digest is *other content* altogether —
/// a proxy's error page, say. The digest alone can only say "not this".
fn check_digest(actual: &str, expected: &str, file: &str, len: u64) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "checksum mismatch for {file}: expected {expected}, got {actual} ({len} bytes)"
        ));
    }
    Ok(())
}

/// How long to wait before `attempt` (1-based) of a download. Attempt 1 is not
/// preceded by a pause (nor is attempt 0, which does not exist); the schedule
/// doubles from there: 2 s, then 4 s. The exponent is capped so a large
/// `attempt` cannot overflow the shift.
///
/// A CDN that is rate-limiting wants the gap, not another request.
fn retry_pause(attempt: u32) -> std::time::Duration {
    if attempt <= 1 {
        return std::time::Duration::ZERO;
    }
    std::time::Duration::from_secs(1u64 << (attempt - 1).min(32))
}

/// Move a verified staging file into place, without disturbing a good archive
/// another build script may have installed first.
///
/// `verify` checks a path against the pinned digest. Both copies are checked
/// against the same one, so there is nothing to choose between them and no
/// reason to overwrite one with the other.
fn install_with<V>(
    tmp: &std::path::Path,
    dest: &std::path::Path,
    verify: V,
) -> Result<(), String>
where
    V: Fn(&std::path::Path) -> Result<(), String>,
{
    if dest.exists() && verify(dest).is_ok() {
        let _ = std::fs::remove_file(tmp);
        return Ok(());
    }
    match std::fs::rename(tmp, dest) {
        Ok(()) => Ok(()),
        // `rename` replaces an existing file on every platform we build for, but
        // it can still fail — on Windows when the other build script has `dest`
        // open, or when `tmp` is gone — and that usually means the other build
        // script won the race. Re-check before treating it as an error.
        Err(e) => {
            let installed = verify(dest).is_ok();
            let _ = std::fs::remove_file(tmp);
            if installed {
                Ok(())
            } else {
                Err(format!("install {}: {e}", dest.display()))
            }
        }
    }
}

#[cfg(test)]
mod acquire_logic_tests {
    use super::*;
    use std::cell::Cell;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    // ---- unusable_body ----------------------------------------------------

    #[test]
    fn empty_body_is_unusable_whatever_was_declared() {
        assert_eq!(unusable_body(0, None).as_deref(), Some("empty body"));
        assert_eq!(unusable_body(0, Some(0)).as_deref(), Some("empty body"));
        assert_eq!(unusable_body(0, Some(10)).as_deref(), Some("empty body"));
    }

    #[test]
    fn short_body_reports_both_lengths() {
        assert_eq!(
            unusable_body(5, Some(10)).as_deref(),
            Some("truncated body: 5 of 10 bytes")
        );
    }

    #[test]
    fn long_body_is_also_a_length_mismatch() {
        assert_eq!(
            unusable_body(11, Some(10)).as_deref(),
            Some("truncated body: 11 of 10 bytes")
        );
    }

    #[test]
    fn matching_or_undeclared_length_is_accepted() {
        assert_eq!(unusable_body(10, Some(10)), None);
        assert_eq!(unusable_body(10, None), None);
    }

    // ---- staging_path_for -------------------------------------------------

    #[test]
    fn staging_path_appends_pid_and_part() {
        assert_eq!(
            staging_path_for(Path::new("/c/mkl-2026.1.0-x_243.conda"), 4242),
            PathBuf::from("/c/mkl-2026.1.0-x_243.4242.part")
        );
    }

    #[test]
    fn staging_path_differs_per_process() {
        let dest = Path::new("/c/a.conda");
        assert_ne!(staging_path_for(dest, 1), staging_path_for(dest, 2));
        assert_ne!(staging_path_for(dest, 1), dest);
    }

    // ---- conda_subdir_for -------------------------------------------------

    #[test]
    fn subdir_follows_the_target_os() {
        assert_eq!(conda_subdir_for("windows"), "win-64");
        assert_eq!(conda_subdir_for("linux"), "linux-64");
        // Anything else falls through to linux-64, as before; `download_mkl`
        // rejects unsupported targets before a URL is built.
        assert_eq!(conda_subdir_for("macos"), "linux-64");
    }

    // ---- cache_base -------------------------------------------------------

    #[test]
    fn cache_base_prefers_xdg_unmodified() {
        assert_eq!(
            cache_base(Some("/xdg"), Some("/home/u"), Some("C:\\u")),
            PathBuf::from("/xdg")
        );
    }

    #[test]
    fn cache_base_then_home_then_userprofile() {
        assert_eq!(
            cache_base(None, Some("/home/u"), Some("C:\\u")),
            PathBuf::from("/home/u").join(".cache")
        );
        assert_eq!(
            cache_base(None, None, Some("C:\\u")),
            PathBuf::from("C:\\u").join(".cache")
        );
    }

    #[test]
    fn cache_base_falls_back_to_cwd() {
        assert_eq!(cache_base(None, None, None), PathBuf::from("."));
    }

    #[test]
    fn cache_base_treats_empty_as_set() {
        // `env::var` returns `Ok("")` for an empty variable, so it wins.
        assert_eq!(cache_base(Some(""), Some("/h"), None), PathBuf::from(""));
        assert_eq!(cache_base(None, Some(""), None), PathBuf::from("").join(".cache"));
    }

    // ---- check_digest -----------------------------------------------------

    #[test]
    fn matching_digest_is_ok() {
        assert_eq!(check_digest("abc", "abc", "f.conda", 3), Ok(()));
    }

    #[test]
    fn digest_mismatch_names_file_both_digests_and_length() {
        assert_eq!(
            check_digest("beef", "cafe", "f.conda", 1234),
            Err("checksum mismatch for f.conda: expected cafe, got beef (1234 bytes)".to_string())
        );
    }

    #[test]
    fn digest_mismatch_reports_a_zero_length() {
        let e = check_digest("e3b0", "cafe", "f.conda", 0).unwrap_err();
        assert!(e.ends_with("(0 bytes)"), "{e}");
    }

    // ---- retry_pause ------------------------------------------------------

    #[test]
    fn retry_pause_doubles_from_two_seconds() {
        assert_eq!(retry_pause(2), Duration::from_secs(2));
        assert_eq!(retry_pause(3), Duration::from_secs(4));
        assert_eq!(retry_pause(4), Duration::from_secs(8));
    }

    #[test]
    fn retry_pause_is_total() {
        // No pause before the first attempt, and no panic for 0 or a huge value.
        assert_eq!(retry_pause(0), Duration::ZERO);
        assert_eq!(retry_pause(1), Duration::ZERO);
        assert_eq!(retry_pause(u32::MAX), Duration::from_secs(1 << 32));
    }

    // ---- install_with -----------------------------------------------------

    /// A fresh scratch directory under the system temp dir, removed on drop.
    /// `std` only: no `tempfile` dev-dependency.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("nuvai-mkl-src-test-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }

        fn path(&self, file: &str) -> PathBuf {
            self.0.join(file)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A verifier that accepts a file iff it holds exactly `good`.
    fn accepts(good: &'static [u8]) -> impl Fn(&Path) -> Result<(), String> {
        move |p| match std::fs::read(p) {
            Ok(b) if b == good => Ok(()),
            Ok(_) => Err("bad content".to_string()),
            Err(e) => Err(e.to_string()),
        }
    }

    #[test]
    fn install_moves_staging_file_into_place() {
        let s = Scratch::new("fresh");
        let (tmp, dest) = (s.path("a.part"), s.path("a.conda"));
        std::fs::write(&tmp, b"good").unwrap();

        assert_eq!(install_with(&tmp, &dest, accepts(b"good")), Ok(()));
        assert_eq!(std::fs::read(&dest).unwrap(), b"good");
        assert!(!tmp.exists());
    }

    #[test]
    fn install_keeps_a_good_dest_and_drops_staging() {
        let s = Scratch::new("dest-good");
        let (tmp, dest) = (s.path("a.part"), s.path("a.conda"));
        std::fs::write(&tmp, b"good").unwrap();
        std::fs::write(&dest, b"good").unwrap();
        let before = std::fs::metadata(&dest).unwrap().modified().unwrap();

        assert_eq!(install_with(&tmp, &dest, accepts(b"good")), Ok(()));
        assert!(!tmp.exists(), "staging file must be cleaned up");
        // Untouched, not replaced by an equal copy.
        assert_eq!(std::fs::metadata(&dest).unwrap().modified().unwrap(), before);
    }

    #[test]
    fn install_replaces_a_bad_dest_with_the_verified_staging_file() {
        let s = Scratch::new("dest-bad");
        let (tmp, dest) = (s.path("a.part"), s.path("a.conda"));
        std::fs::write(&tmp, b"good").unwrap();
        std::fs::write(&dest, b"corrupt").unwrap();

        assert_eq!(install_with(&tmp, &dest, accepts(b"good")), Ok(()));
        assert_eq!(std::fs::read(&dest).unwrap(), b"good");
    }

    #[test]
    fn install_rename_fails_but_dest_became_good() {
        // The other build script wins the race between our first look at `dest`
        // and our rename: the first verification fails, the rename fails (no
        // staging file to move — the loser's own copy is gone), and the re-check
        // finds the archive the winner installed.
        let s = Scratch::new("rename-fail-dest-good");
        let (tmp, dest) = (s.path("missing.part"), s.path("a.conda"));
        std::fs::write(&dest, b"good").unwrap();
        let calls = Cell::new(0u32);
        let verify = |_: &Path| {
            calls.set(calls.get() + 1);
            if calls.get() == 1 { Err("not yet".to_string()) } else { Ok(()) }
        };

        assert_eq!(install_with(&tmp, &dest, verify), Ok(()));
        assert_eq!(calls.get(), 2, "verified once before the rename, once after it failed");
        assert_eq!(std::fs::read(&dest).unwrap(), b"good");
    }

    #[test]
    fn install_rename_fails_and_dest_is_bad() {
        let s = Scratch::new("rename-fail-dest-bad");
        let (tmp, dest) = (s.path("missing.part"), s.path("a.conda"));
        std::fs::write(&dest, b"corrupt").unwrap();

        let e = install_with(&tmp, &dest, accepts(b"good")).unwrap_err();
        assert!(e.starts_with(&format!("install {}:", dest.display())), "{e}");
    }

    #[test]
    fn install_rename_fails_and_dest_is_absent() {
        let s = Scratch::new("rename-fail-no-dest");
        let (tmp, dest) = (s.path("missing.part"), s.path("a.conda"));

        let e = install_with(&tmp, &dest, accepts(b"good")).unwrap_err();
        assert!(e.starts_with("install "), "{e}");
        assert!(!dest.exists());
    }
}
