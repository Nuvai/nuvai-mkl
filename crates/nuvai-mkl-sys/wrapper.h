// Bindgen input: the full oneMKL C interface.
// MKL_LP64 selects the 32-bit `MKL_INT` interface (matches mkl_rt's default layer).
#define MKL_LP64 1

#include "mkl.h"          // types, BLAS, CBLAS, service, vector math decls
#include "mkl_cblas.h"    // CBLAS
#include "mkl_lapacke.h"  // LAPACKE (C LAPACK)
#include "mkl_dfti.h"     // FFT
#include "mkl_vml.h"      // Vector Math Library
#include "mkl_vsl.h"      // Vector Statistics (RNG, summary stats)
#include "mkl_spblas.h"   // Inspector-executor sparse BLAS
#include "mkl_pardiso.h"  // PARDISO direct sparse solver
#include "mkl_dss.h"      // Direct Sparse Solver (DSS)
#include "mkl_service.h"  // threading / memory / verbosity service
