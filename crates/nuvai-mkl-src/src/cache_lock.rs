// Cross-process lock serialising acquisition into the shared cache (#94).
//
// `nuvai-mkl-src` is built as two units (a `[dependencies]` and a
// `[build-dependencies]` edge), so a `--workspace` build runs two build scripts
// against one `~/.cache/nuvai-mkl`. The per-PID staging and rename-into-place
// protocols (#63/#64) keep that race *safe*, but both units still download the
// same archives. Holding this lock across the download makes the second unit
// wait, then find verified archives and extracted trees already in place.
//
// Merging the two units instead is not viable: they carry different features
// (ADR-0005/0006/0007), and a cross build needs both. See ADR-0009.
//
// `build.rs` includes this file; `lib.rs` includes it under `cfg(test)` so the
// unit tests below run (`acquire.rs` itself has no test harness). It declares
// no imports — `build.rs` shares one module with `mkl_info.rs`/`acquire.rs`,
// and a repeated `use` there is E0252 — so every path below is fully qualified.

/// Name of the advisory lock file, created inside the cache package directory.
const CACHE_LOCK_FILE: &str = ".acquire.lock";

/// An exclusive advisory lock on `<dir>/.acquire.lock`, released on drop.
///
/// The lock belongs to the open file description, so it is also released if the
/// process dies. It is advisory: it only coordinates cooperating build scripts.
#[derive(Debug)]
struct CacheLock {
    file: std::fs::File,
}

impl CacheLock {
    /// Create `dir` if needed, then block until the exclusive lock is held.
    ///
    /// Panics (failing the build with a message naming the lock path) if the
    /// directory or lock file cannot be created or the lock cannot be taken.
    fn acquire(dir: &std::path::Path) -> CacheLock {
        let path = dir.join(CACHE_LOCK_FILE);
        std::fs::create_dir_all(dir)
            .unwrap_or_else(|e| panic!("create cache dir for lock {}: {e}", path.display()));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .unwrap_or_else(|e| panic!("open cache lock {}: {e}", path.display()));
        file.lock()
            .unwrap_or_else(|e| panic!("lock cache lock {}: {e}", path.display()));
        CacheLock { file }
    }
}

impl Drop for CacheLock {
    fn drop(&mut self) {
        // Closing the handle releases the lock regardless; the explicit unlock
        // just makes the release prompt. A failure here has no caller to tell.
        let _ = self.file.unlock();
    }
}

#[cfg(test)]
mod cache_lock_tests {
    use super::{CacheLock, CACHE_LOCK_FILE};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("nuvai-mkl-lock-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn probe(dir: &std::path::Path) -> std::fs::File {
        std::fs::OpenOptions::new()
            .write(true)
            .open(dir.join(CACHE_LOCK_FILE))
            .expect("open lock file for probing")
    }

    #[test]
    fn second_handle_is_refused_while_held_and_admitted_after_drop() {
        let dir = scratch("held");
        let held = CacheLock::acquire(&dir);
        assert!(probe(&dir).try_lock().is_err(), "lock must be exclusive while held");
        drop(held);
        assert!(probe(&dir).try_lock().is_ok(), "lock must be free after drop");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn creates_lock_file_in_missing_directory() {
        let dir = scratch("create").join("nested").join("mkl");
        assert!(!dir.exists());
        let _lock = CacheLock::acquire(&dir);
        assert!(dir.join(CACHE_LOCK_FILE).is_file());
        std::fs::remove_dir_all(dir.parent().unwrap().parent().unwrap()).unwrap();
    }

    #[test]
    fn blocking_acquire_in_another_thread_waits_for_release() {
        let dir = scratch("block");
        let held = CacheLock::acquire(&dir);
        let entered = Arc::new(AtomicUsize::new(0));
        let (d, e) = (dir.clone(), Arc::clone(&entered));
        let waiter = std::thread::spawn(move || {
            let _lock = CacheLock::acquire(&d);
            e.fetch_add(1, Ordering::SeqCst);
        });
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(entered.load(Ordering::SeqCst), 0, "waiter got in while lock was held");
        drop(held);
        waiter.join().expect("waiter thread");
        assert_eq!(entered.load(Ordering::SeqCst), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
