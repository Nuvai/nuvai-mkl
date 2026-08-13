# nuvai-mkl

A modern Rust wrapper over **Intel oneMKL 2026.1.0**.

`nuvai-mkl` is the successor to the abandoned [`intel-mkl-src`](https://crates.io/crates/intel-mkl-src)
(frozen at MKL 2020.1, last released 2022). It acquires and links the current
oneMKL release (2026.1.0) and exposes it through a safe, idiomatic API.

## Architecture

A three-crate Cargo workspace, mirroring the proven `-src`/`-sys`/wrapper split:

| Crate | Role |
|---|---|
| `nuvai-mkl-src` | Acquire + link. Build script detects `MKLROOT`/oneAPI, or downloads 2026.1.0 from conda-forge (Linux/Windows), then emits linker directives. `links = "mkl"`. |
| `nuvai-mkl-sys` | Raw FFI bindings to the full C interface, generated with `bindgen`. |
| `nuvai-mkl` | Safe, typed wrapper over all MKL domains. |

```
crates/
├── nuvai-mkl-src/   acquisition + linking
├── nuvai-mkl-sys/   raw FFI (bindgen)
└── nuvai-mkl/       safe wrapper
```

## Domain coverage

`nuvai-mkl` targets the full oneMKL surface. For each domain we also record its
native Apple Silicon replacement, which the backend abstraction will map to.

| MKL domain | x86_64 (Intel MKL) | Apple Silicon native replacement |
|---|---|---|
| **BLAS** | `cblas_*` / `?gemm`, `?gemv`, `?dot`, `?axpy` | Accelerate **vecLib** (`cblas_*`) or OpenBLAS |
| **LAPACK** | `LAPACKE_*` (`?gesv`, `?getrf`, `?syev`, …) | Accelerate (`?gesv`, `?gemm`, …) or OpenBLAS |
| **FFT (DFTI)** | `DftiCreateDescriptor*` / `DftiCompute*` | Accelerate **vDSP** or RustFFT / FFTW |
| **VML** (vector math) | `vsExp`, `vsLn`, `vsSin`, `vsSqrt`, … | Accelerate **vForce** |
| **Sparse direct solvers (PARDISO/DSS)** | `pardisoinit`/`pardiso`, `dss_*` | Accelerate **Sparse/SparseSolve** or MUMPS / SuperLU |
| **VSL** (RNG) | `vslNewStream` / `vsRngUniform` / `vsRngGaussian` | `rand` / `rand_chacha` |

## Platform support

| Target | Intel MKL availability |
|---|---|
| `x86_64-unknown-linux-gnu` | ✅ download (conda-forge) or system oneAPI |
| `x86_64-pc-windows-msvc` | ✅ download (conda-forge) or system oneAPI |
| `x86_64-apple-darwin` | system oneAPI (NuGet wiring pending) |
| `aarch64-apple-darwin` (Apple Silicon) | ❌ Intel ships no MKL — `accelerate` fallback (planned) |
| `aarch64-unknown-linux-gnu` | ❌ no 2026.1.0 build — `openblas` fallback (planned) |

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
