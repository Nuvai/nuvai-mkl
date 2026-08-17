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
| **BLAS** | `cblas_*` / `?gemm`, `?gemv`, `?dot`, `?axpy` | Accelerate **vecLib** (`cblas_*`, symbol-aliased) or OpenBLAS | OpenBLAS (`cblas_*`, symbol-aliased) |
| **LAPACK** | `LAPACKE_*` (`?gesv`, `?getrf`, `?syev`, …) | Accelerate Fortran `_` entry points (`?gesv_`, …) + RowMajor shim, or OpenBLAS | OpenBLAS Fortran `_` entry points + RowMajor shim |
| **FFT (DFTI)** | `DftiCreateDescriptor*` / `DftiCompute*` | Accelerate **vDSP** DFT (forward/inverse setups, `1/n` applied on inverse) | `ErrorKind::Unsupported` |
| **VML** (vector math) | `vsExp`, `vsLn`, `vsSin`, `vsSqrt`, … | Accelerate **vForce** (`vvexpf`, `vvlogf`, …, `(dst, src, n)` order) | `ErrorKind::Unsupported` |
| **Sparse direct solvers (PARDISO/DSS)** | `pardisoinit`/`pardiso`, `dss_*` | Accelerate **Sparse/SparseSolve** (CSR→CSC transpose; QR for PARDISO, Cholesky for DSS) | `ErrorKind::Unsupported` |
| **VSL** (RNG) | `vslNewStream` / `vsRngUniform` / `vsRngGaussian` | `rand` / `rand_chacha` / `rand_distr` (ChaCha20; statistically valid, not sequence-identical) | `ErrorKind::Unsupported` |

## Backend selection

Selection is **explicit, never silent** (ADR-0003 decision 2):

- On `x86_64` Linux/Windows targets the backend is always Intel oneMKL; no feature changes that.
- On `aarch64-unknown-linux-gnu` the backend is always **OpenBLAS** (`cfg(target_arch = "aarch64")` + `target_os = "linux"`); the `accelerate`/`openblas` features are no-ops there because there is no second backend to choose from.
- On `aarch64-apple-darwin` the non-MKL path is mandatory (`cfg(target_arch = "aarch64")`); the *choice* of backend is a Cargo feature on `nuvai-mkl`:

| Feature | Default? | Effect on aarch64-apple-darwin |
|---|---|---|
| `accelerate` | ✅ | Use Accelerate for every domain (BLAS/LAPACK via vecLib, FFT via vDSP, VML via vForce, sparse via Sparse/SparseSolve). |
| `openblas` | — | Use OpenBLAS for BLAS/LAPACK instead of vecLib (opt-in; FFT/VML/sparse/VSL still use Accelerate/`rand`). |

- The active backend is queryable at build time via `nuvai_mkl_src::backend()` / `nuvai_mkl_src::backend_tag()`.
- If both features are enabled the selection is still explicit (never silently picked): the first matching backend wins and is reported.
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

## Requirements

- Rust (built against **1.99 nightly**, edition 2024).
- First build downloads ~140 MB of MKL into `~/.cache/nuvai-mkl/` (cached thereafter; on Windows the cache falls back to `%USERPROFILE%\.cache\nuvai-mkl` since `HOME` is often unset, and the acquisition also fetches `mkl-devel`, `llvm-openmp` and `tbb`).
- `libclang` + `bindgen` for regenerating FFI bindings on Intel targets (LLVM on Windows, `libclang-dev` on Linux). The ARM64 aarch64 targets use a hand-written FFI surface and need no libclang.
- On `aarch64-apple-darwin`, **macOS 12.0+** is required: the FFT backend uses vDSP's interleaved-complex DFT (`vDSP_DFT_Interleaved_*`), which is `API_AVAILABLE(macos(12.0))`.
- On `aarch64-unknown-linux-gnu`, OpenBLAS is the sole backend: install `libopenblas-dev` (system default search path), or point `OPENBLAS_ROOT` at a conda/pip install — `nuvai-mkl-src` adds its `lib` dir to the propagated link-search path, and the workspace's own test/example binaries get the matching runtime rpath via `nuvai-mkl`'s build script (`cargo:rustc-link-arg` only applies to the emitting crate's own targets, so downstream crates linking a non-system OpenBLAS must set their own rpath).

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
