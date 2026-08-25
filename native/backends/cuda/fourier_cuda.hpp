#pragma once
// Only compiled when SCL_WITH_CUDA is defined (see native/CMakeLists.txt).

#include <complex>
#include <vector>

#include "scl/fourier.hpp"

namespace scl {

//: cuFFT-backed implementation of the SAME contract as
//: compute_fourier_transform_1d_cpu (scl/fourier.hpp). Uses cuFFT rather
//: than a hand-written GPU FFT deliberately: cuFFT is an established,
//: vendor-verified primitive that fits the existing backend seam, and
//: writing a custom FFT kernel would add a large unvalidated numerical
//: surface for no measured reason.
//:
//: STATUS: compiled and linked against a real CUDA 12 toolchain; NEVER
//: GPU-EXECUTED -- no CUDA device has been available in any session so far
//: (docs/PHASE4_AUDIT.md). No numerical claim is made for this path.
FourierResult compute_fourier_transform_1d_cuda(const std::vector<std::complex<double>>& signal,
                                                 const FourierParameters& params);

}  // namespace scl
