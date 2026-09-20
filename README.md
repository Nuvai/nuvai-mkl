# nuvai-mkl

A modern Rust wrapper over **Intel oneMKL 2026.1.0** on x86_64 Linux/Windows, over
Apple Silicon-native replacements (Accelerate / `rand`) on `aarch64-apple-darwin`,
and over **OpenBLAS** on `aarch64-unknown-linux-gnu`.

`nuvai-mkl` is the successor to the abandoned [`intel-mkl-src`](https://crates.io/crates/intel-mkl-src)
(frozen at MKL 2020.1, last released 2022). It acquires and links the current
oneMKL release (2026.1.0) and exposes it through a safe, idiomatic API. Where
Intel ships no oneMKL build (Apple Silicon, ARM64 Linux), the same typed API is
backed by Accelerate's vecLib/vDSP/vForce/Sparse frameworks and the `rand` stack
on macOS, or by OpenBLAS (BLAS/LAPACK only) on Linux, selected by `cfg` (never
silently).

## Architecture

A three-crate Cargo workspace, mirroring the proven `-src`/`-sys`/wrapper split:

| Crate | Role |
|---|---|
| `nuvai-mkl-src` | Acquire + link. Build script detects `MKLROOT`/oneAPI, or downloads 2026.1.0 from conda-forge (Linux/Windows; Windows also pulls `mkl-devel` for the `mkl_rt.lib` import lib and the `llvm-openmp`/`tbb` runtime DLLs), then emits linker directives. `links = "mkl"`. On `aarch64-apple-darwin` it emits Accelerate (`-framework Accelerate`) or OpenBLAS (`-lopenblas`) directives, and on `aarch64-unknown-linux-gnu` it emits OpenBLAS — see [Backend selection](#backend-selection). |
| `nuvai-mkl-sys` | Raw FFI bindings to the full C interface, generated with `bindgen`. On `aarch64-apple-darwin` a hand-written `extern "C"` surface replaces the bindgen pass (Accelerate cblas, Fortran LAPACK `_`, vDSP DFT, vForce, Sparse/SparseSolve); on `aarch64-unknown-linux-gnu` a hand-written surface covers OpenBLAS (netlib CBLAS + Fortran LAPACK `_`). |
| `nuvai-mkl` | Safe, typed wrapper over all MKL domains. |

```
crates/
├── nuvai-mkl-src/   acquisition + linking
├── nuvai-mkl-sys/   raw FFI (bindgen)
└── nuvai-mkl/       safe wrapper
```

## Domain coverage

`nuvai-mkl` targets the full oneMKL surface. On x86_64 Linux/Windows every domain runs on
Intel oneMKL; on the ARM64 targets where Intel ships no oneMKL each domain maps to a
native backend behind the same typed API (ADR-0003) — or returns
`ErrorKind::Unsupported` where no backend exists.

| MKL domain | x86_64 Linux/Windows (Intel MKL) | aarch64-apple-darwin backend | aarch64-unknown-linux-gnu backend |
|---|---|---|---|
| **BLAS** | `cblas_*` / `?gemm` (incl. fp16 `cblas_hgemm`), `?dot`, `?axpy`, `?scal` | Accelerate **vecLib** (`cblas_*`, symbol-aliased) or OpenBLAS — no fp16 GEMM (`ErrorKind::Unsupported`) | OpenBLAS (`cblas_*`, symbol-aliased) — no fp16 GEMM (`ErrorKind::Unsupported`) |
| **LAPACK** | `LAPACKE_*` (`?gesv`, `?getrf`, `?syev`, …) | Accelerate Fortran `_` entry points (`?gesv_`, …) + RowMajor shim, or OpenBLAS | OpenBLAS Fortran `_` entry points + RowMajor shim |
| **FFT (DFTI)** | `DftiCreateDescriptor*` / `DftiCompute*` | Accelerate **vDSP** DFT (forward/inverse setups, `1/n` applied on inverse) | `ErrorKind::Unsupported` |
| **VML** (vector math) | unary `vsExp`, `vsLn`, `vsSin`, `vsTanh`, `vsSqr`, … + binary `vsAdd`, `vsMul`, `vsSub`, `vsDiv`, `vsFmax`, `vsFmin` | Accelerate **vForce** for the unary transcendentals (`vvexpf`, `vvlogf`, `vvtanhf`, …, `(dst, src, n)` order), **vDSP** for `sqr` and all six binary ops (`vDSP_vsq`, `vDSP_vadd`, …, `(A, IA, B, IB, C, IC, N)` order) — vForce carries neither | `ErrorKind::Unsupported` |
| **Sparse direct solvers (PARDISO/DSS)** | `pardisoinit`/`pardiso`, `dss_*` | Accelerate **Sparse/SparseSolve** (CSR→CSC transpose; QR for PARDISO, Cholesky for DSS) | `ErrorKind::Unsupported` |
| **VSL** (RNG) | `vslNewStream` / `vsRngUniform` / `vsRngGaussian` | `rand` / `rand_chacha` / `rand_distr` (ChaCha20; statistically valid, not sequence-identical) | `ErrorKind::Unsupported` |

## Notes on the coverage table

- **fp16 GEMM takes `u16`.** `blas::hgemm` mirrors oneMKL's `MKL_F16`, which the
  C interface defines as `unsigned short` — so the buffers are IEEE-754 binary16
  bit patterns (`0x3C00` is `1.0`, `0x0000` is `0.0`) and this crate does not
  convert for you. The symbol (`cblas_hgemm`) exists in oneMKL 2026.1.0 but in
  neither ARM64 backend, hence the `Unsupported` above.
- **`vml::fmax`/`vml::fmin` disagree about `NaN`.** oneMKL's `vsFmax`/`vsFmin`
  return the non-`NaN` operand when exactly one of a pair is `NaN`; Accelerate's
  `vDSP_vmax`/`vDSP_vmin` propagate it (measured against the framework). Apply
  your own `NaN` policy if the difference matters to you.
- **VSL RNG sequences are not identical to Intel's** on Apple Silicon — the
  generators are statistically valid (`rand_chacha`), not sequence-compatible
  (ADR-0003 decision 7).

## Backend selection

Selection is **explicit, never silent** (ADR-0003 decision 2):

- On `x86_64` Linux/Windows targets the backend is always Intel oneMKL; no feature changes that. On `x86_64-unknown-linux-gnu` the `static` feature changes *how* it links — see [Static linking](#static-linking).
- On `aarch64-unknown-linux-gnu` the backend is always **OpenBLAS** (`cfg(target_arch = "aarch64")` + `target_os = "linux"`); the `accelerate`/`openblas` features are no-ops there because there is no second backend to choose from.
- On `aarch64-apple-darwin` the non-MKL path is mandatory (`cfg(target_arch = "aarch64")`); the *choice* of backend is a Cargo feature on `nuvai-mkl`:

| Feature | Default? | Effect on aarch64-apple-darwin |
|---|---|---|
| `accelerate` | ✅ | Use Accelerate for every domain (BLAS/LAPACK via vecLib, FFT via vDSP, VML via vForce and vDSP, sparse via Sparse/SparseSolve). |
| `openblas` | — | Use OpenBLAS for BLAS/LAPACK instead of vecLib (opt-in; FFT/VML/sparse/VSL still use Accelerate/`rand`). |

- The active backend is queryable at build time via `nuvai_mkl_src::backend()` / `nuvai_mkl_src::backend_tag()`.
- Enabling **both** features is a compile error, not a silent pick — and so is enabling neither (`--no-default-features` with no replacement). The backend is never implicitly resolved (ADR-0003, #35).
- A domain with no selected backend fails to compile or returns `ErrorKind::Unsupported` — it never degrades silently.

## Platform support

| Target | Backend | Status |
|---|---|---|
| `x86_64-unknown-linux-gnu` | Intel oneMKL — conda-forge `mkl` + `mkl-include`, or system oneAPI (`MKLROOT`) | ✅ |
| `x86_64-pc-windows-msvc` | Intel oneMKL — conda-forge `mkl` + `mkl-include` + `mkl-devel` + `llvm-openmp` + `tbb` (links `mkl_rt` → `mkl_rt.3.dll`; runtime DLLs `libiomp5md.dll`/`tbb12.dll` on `PATH`), or system oneAPI (`MKLROOT`) | ✅ |
| `x86_64-apple-darwin` | — | ❌ unsupported (Intel ended macOS oneMKL after 2023.2.0) |
| `aarch64-apple-darwin` (Apple Silicon, macOS 12.0+) | Accelerate + `rand` (`accelerate` feature, default) | ✅ |
| `aarch64-unknown-linux-gnu` | OpenBLAS — system `libopenblas-dev` (BLAS/LAPACK only; FFT/VML/VSL/sparse return `ErrorKind::Unsupported`) | ✅ |

On `x86_64-pc-windows-msvc` the Windows loader resolves the MKL runtime
(`mkl_rt.3.dll`, plus the OpenMP `libiomp5md.dll` and TBB `tbb12.dll` its
threading layers depend on) from `PATH` at process start, not from the
link-search path. When acquiring from conda-forge, the DLLs are extracted under
`~/.cache/nuvai-mkl/mkl-2026.1.0/<pkg>/Library/bin` — one directory per package
(`mkl`, `llvm-openmp`, `tbb`); add all of them to `PATH` (or copy the DLLs
beside the executable) before `cargo run` / `cargo test`. The system oneAPI path
on Windows is `MKLROOT`-only (the well-known `/opt/intel/oneapi/…` Unix paths do
not exist there).

## Static linking

`x86_64-unknown-linux-gnu` only, opt-in (ADR-0005):

```toml
nuvai-mkl = { git = "https://github.com/Nuvai/nuvai-mkl", features = ["static"] }
```

This links Intel oneMKL's static archives (`libmkl_intel_lp64.a` +
`libmkl_sequential.a` + `libmkl_core.a`, the single-threaded/sequential
threading layer) instead of the dynamic runtime dispatcher (`mkl_rt`). The
resulting binary has no runtime dependency on `libmkl_rt.so`/`libiomp5.so` —
useful for container/Lambda-style distribution where a self-contained binary
avoids `LD_LIBRARY_PATH`/rpath setup, at the cost of a larger download
(conda-forge's `mkl-static` package is ~130 MB compressed, versus a few MB for
the dynamic `mkl` package used by default) and a larger binary. `static` is
unsupported (compile error) on every other target — Windows has no
static-archive package wired up yet, and Accelerate/OpenBLAS have no static
form to switch to.

## Requirements

- Rust (built against **1.99 nightly**, edition 2024).
- First build downloads ~140 MB of MKL into `~/.cache/nuvai-mkl/` (cached thereafter; on Windows the cache falls back to `%USERPROFILE%\.cache\nuvai-mkl` since `HOME` is often unset, and the acquisition also fetches `mkl-devel`, `llvm-openmp` and `tbb`).
- `libclang` + `bindgen` for regenerating FFI bindings on Intel targets (LLVM on Windows, `libclang-dev` on Linux). The ARM64 aarch64 targets use a hand-written FFI surface and need no libclang.
- On `aarch64-apple-darwin`, **macOS 12.0+** is required: the FFT backend uses vDSP's interleaved-complex DFT (`vDSP_DFT_Interleaved_*`), which is `API_AVAILABLE(macos(12.0))`.
- On `aarch64-unknown-linux-gnu`, OpenBLAS is the sole backend: install `libopenblas-dev` (system default search path), or point `OPENBLAS_ROOT` at a conda/pip install — `nuvai-mkl-src` adds its `lib` dir to the propagated link-search path, and `nuvai_mkl_src::emit_binary_link_args()` supplies the matching runtime rpath wherever the binary is linked from (see [Linking from another crate](#linking-from-another-crate)).

## Linking from another crate

Everything `nuvai-mkl-src` can express as a link *library* or a *search path*
propagates to a dependent's binaries on its own — on `static` that includes the
group link over Intel's archives, which is why a plain dependency is enough.
What cannot propagate is an executable's **runtime rpath** and the object that
keeps the OpenMP runtime in its `DT_NEEDED` under `mold` (#44): Cargo scopes
`cargo:rustc-link-arg` to the targets of the package that emits it, and no
directive can put an rpath on another package's binary (ADR-0006). The two link
modes therefore differ:

| Link mode | What a depending crate needs |
|---|---|
| `static` | **nothing.** `nuvai-mkl = { …, features = ["static"] }` links and runs |
| dynamic (default) | the two manifest entries below, plus one line of `build.rs` |

```toml
[dependencies]
nuvai-mkl = { git = "https://github.com/Nuvai/nuvai-mkl", tag = "v0.1.0" }
# For the DEP_MKL_* metadata the build script below reads. Cargo forwards a
# `links` provider's metadata through this table only — a [build-dependencies]
# entry on its own receives none of it (ADR-0006).
nuvai-mkl-src = { git = "https://github.com/Nuvai/nuvai-mkl", tag = "v0.1.0" }

[build-dependencies]
# The same crate again: a build script can only use code from this table.
nuvai-mkl-src = { git = "https://github.com/Nuvai/nuvai-mkl", tag = "v0.1.0" }
```

```rust
// build.rs
fn main() {
    nuvai_mkl_src::emit_binary_link_args();
}
```

Without that call the binary links and then fails at start-up with
`libmkl_rt.so.3: cannot open shared object file`. Putting the MKL directories on
`LD_LIBRARY_PATH` (or `PATH` on Windows) wherever the binary runs is the
alternative, and is what consumers had to do before.

Both modes are exercised end to end by the crates in
[`consumers/`](consumers/README.md) — a *separate* workspace, because the
distinction is between packages — and by the `x86_64-linux-downstream` CI job,
which builds and runs a downstream binary against this repository's crates.

## Installation

`nuvai-mkl` is consumed **from this repository**, not from crates.io — it is not
published to a registry (ADR-0004). Depend on the wrapper crate only:
`nuvai-mkl-sys` and `nuvai-mkl-src` resolve through it. `nuvai_mkl_src::backend()`
and `backend_tag()` are public if you want to report the active backend yourself.

The workspace declares `rust-version = "1.99"` (edition 2024), so Cargo refuses an
older toolchain; today that means a **nightly** toolchain. Nothing pins one for
you — there is no `rust-toolchain.toml` — so pin it in your own project. See
[Requirements](#requirements) for the per-target prerequisites, and expect a
~140 MB oneMKL download into `~/.cache/nuvai-mkl/` on the Intel targets the first
time you build.

### Git dependency (recommended)

```toml
[dependencies]
nuvai-mkl = { git = "https://github.com/Nuvai/nuvai-mkl", tag = "v0.1.0" }
```

Cargo fetches the repository and locates `nuvai-mkl` in `crates/nuvai-mkl` by
package name — the crate is not at the repository root, which is fine for a git
dependency. The inner crates come from the same checkout.

Cargo treats tags as mutable, so to pin something that cannot move under you, name
the commit instead:

```toml
nuvai-mkl = { git = "https://github.com/Nuvai/nuvai-mkl", rev = "<commit-sha>" }
```

Omitting both takes the tip of the default branch, re-resolved on `cargo update`.

If your environment keeps git credentials in a CLI helper (SSO, Keychain,
`gh auth`), set `CARGO_NET_GIT_FETCH_WITH_CLI=true`; otherwise use the SSH URL,
`git = "ssh://git@github.com/Nuvai/nuvai-mkl.git"`.

### Path dependency (sibling checkout)

```toml
[dependencies]
nuvai-mkl = { path = "../nuvai-mkl/crates/nuvai-mkl" }
```

The path must name the directory holding the `Cargo.toml` — here
`crates/nuvai-mkl`. Unlike a git dependency, Cargo does **not** search a path for
the package, and the repository root is a virtual workspace with no package to
depend on, so pointing at the root does not work.

### Choosing a backend

`accelerate` is the default everywhere, but it only *matters* on
`aarch64-apple-darwin`: on x86_64 Linux/Windows the backend is always Intel oneMKL,
and on `aarch64-unknown-linux-gnu` it is always OpenBLAS, with both features inert.

```toml
# Apple Silicon: OpenBLAS for BLAS/LAPACK instead of vecLib (FFT, VML, sparse and
# VSL still use Accelerate and `rand`).
nuvai-mkl = { git = "https://github.com/Nuvai/nuvai-mkl", tag = "v0.1.0",
              default-features = false, features = ["openblas"] }
```

On `aarch64-apple-darwin` exactly one of the two must be enabled. A build with
neither (`default-features = false` and no replacement) or with both is a compile
error by design — the backend is never selected silently (ADR-0003). Cargo
features are additive across the whole graph, so `default-features = false` is
what turns `accelerate` off; a feature list that merely omits it does not.

## Usage

```rust
use nuvai_mkl::blas;
use nuvai_mkl::layout::{Layout, Transpose};

fn main() -> nuvai_mkl::error::Result<()> {
    // BLAS Level 3: C = A·B  (row-major)
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let mut c = [0.0f32; 4];
    blas::sgemm(
        Layout::RowMajor,
        Transpose::NoTrans,
        Transpose::NoTrans,
        2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2,
    )?;
    Ok(())
}
```

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option.

Intel oneMKL is distributed separately under the Intel Simplified Software
License; this repository downloads it at build time and does not redistribute
Intel's binaries.
