# nuvai-mkl

A modern Rust wrapper over **Intel oneMKL 2026.1.0** on x86_64, and over
Apple Silicon-native replacements (Accelerate / `rand`) on `aarch64-apple-darwin`.

`nuvai-mkl` is the successor to the abandoned [`intel-mkl-src`](https://crates.io/crates/intel-mkl-src)
(frozen at MKL 2020.1, last released 2022). It acquires and links the current
oneMKL release (2026.1.0) and exposes it through a safe, idiomatic API. Where
Intel ships no oneMKL build (Apple Silicon), the same typed API is backed by
Accelerate's vecLib/vDSP/vForce/Sparse frameworks and the `rand` stack, selected
by `cfg` (never silently).

## Architecture

A three-crate Cargo workspace, mirroring the proven `-src`/`-sys`/wrapper split:

| Crate | Role |
|---|---|
| `nuvai-mkl-src` | Acquire + link. Build script detects `MKLROOT`/oneAPI, or downloads 2026.1.0 from conda-forge (Linux/Windows), then emits linker directives. `links = "mkl"`. On `aarch64-apple-darwin` it emits Accelerate (`-framework Accelerate`) or OpenBLAS (`-lopenblas`) directives instead — see [Backend selection](#backend-selection). |
| `nuvai-mkl-sys` | Raw FFI bindings to the full C interface, generated with `bindgen`. On `aarch64-apple-darwin` a hand-written `extern "C"` surface replaces the bindgen pass (Accelerate cblas, Fortran LAPACK `_`, vDSP DFT, vForce, Sparse/SparseSolve). |
| `nuvai-mkl` | Safe, typed wrapper over all MKL domains. |

```
crates/
├── nuvai-mkl-src/   acquisition + linking
├── nuvai-mkl-sys/   raw FFI (bindgen)
└── nuvai-mkl/       safe wrapper
```

## Domain coverage

`nuvai-mkl` targets the full oneMKL surface. On x86_64 every domain runs on
Intel oneMKL; on Apple Silicon each domain maps to a native backend behind the
same typed API (ADR-0003).

| MKL domain | x86_64 (Intel MKL) | aarch64-apple-darwin backend |
|---|---|---|
| **BLAS** | `cblas_*` / `?gemm`, `?gemv`, `?dot`, `?axpy` | Accelerate **vecLib** (`cblas_*`, symbol-aliased) or OpenBLAS |
| **LAPACK** | `LAPACKE_*` (`?gesv`, `?getrf`, `?syev`, …) | Accelerate Fortran `_` entry points (`?gesv_`, …) + RowMajor shim, or OpenBLAS |
| **FFT (DFTI)** | `DftiCreateDescriptor*` / `DftiCompute*` | Accelerate **vDSP** DFT (forward/inverse setups, `1/n` applied on inverse) |
| **VML** (vector math) | `vsExp`, `vsLn`, `vsSin`, `vsSqrt`, … | Accelerate **vForce** (`vvExpf`, `vvLogf`, …, `(dst, src, n)` order) |
| **Sparse direct solvers (PARDISO/DSS)** | `pardisoinit`/`pardiso`, `dss_*` | Accelerate **Sparse/SparseSolve** (CSR→CSC transpose; QR for PARDISO, Cholesky for DSS) |
| **VSL** (RNG) | `vslNewStream` / `vsRngUniform` / `vsRngGaussian` | `rand` / `rand_chacha` / `rand_distr` (ChaCha20; statistically valid, not sequence-identical) |

## Backend selection

Selection is **explicit, never silent** (ADR-0003 decision 2):

- On `x86_64` targets the backend is always Intel oneMKL; no feature changes that.
- On `aarch64-apple-darwin` the non-MKL path is mandatory (`cfg(target_arch = "aarch64")`); the *choice* of backend is a Cargo feature on `nuvai-mkl`:

| Feature | Default? | Effect on aarch64 |
|---|---|---|
| `accelerate` | ✅ | Use Accelerate for every domain (BLAS/LAPACK via vecLib, FFT via vDSP, VML via vForce, sparse via Sparse/SparseSolve). |
| `openblas` | — | Use OpenBLAS for BLAS/LAPACK instead of vecLib (opt-in; FFT/VML/sparse/VSL still use Accelerate/`rand`). |

- The active backend is queryable at build time via `nuvai_mkl_src::backend()` / `nuvai_mkl_src::backend_tag()`.
- If both features are enabled the selection is still explicit (never silently picked): the first matching backend wins and is reported.
- A domain with no selected backend fails to compile or returns `ErrorKind::Unsupported` — it never degrades silently.

## Platform support

| Target | Backend | Status |
|---|---|---|
| `x86_64-unknown-linux-gnu` | Intel oneMKL (download conda-forge or system oneAPI) | ✅ |
| `x86_64-pc-windows-msvc` | Intel oneMKL (download conda-forge or system oneAPI) | ✅ |
| `x86_64-apple-darwin` | Intel oneMKL (system oneAPI) | ✅ |
| `aarch64-apple-darwin` (Apple Silicon) | Accelerate + `rand` (`accelerate` feature, default) | ✅ |
| `aarch64-unknown-linux-gnu` | `openblas` feature (planned) | 🚧 planned |

## Requirements

- Rust (built against **1.99 nightly**, edition 2024).
- First build downloads ~140 MB of MKL into `~/.cache/nuvai-mkl/` (cached thereafter).
- `libclang` + `bindgen` for regenerating FFI bindings.

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
