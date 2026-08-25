// cuFFT backend for `fourier_transform_1d`.
//
// STATUS: written against the CUDA 12 runtime + cuFFT API and COMPILED AND
// LINKED against them, but NEVER GPU-EXECUTED -- no CUDA device has been
// visible in any session to date (docs/PHASE4_AUDIT.md). It therefore
// carries none of the numerical validation the CPU path has. Do not treat
// this as a verified computational path until it has run on real hardware
// and been compared against the CPU reference within a stated tolerance.
//
// Convention mapping (the one thing most likely to be silently wrong, so
// it is stated explicitly):
//   scl::FourierDirection::Forward  == exp(-2*pi*i*k*n/N) == CUFFT_FORWARD
//   scl::FourierDirection::Inverse  == exp(+2*pi*i*k*n/N) == CUFFT_INVERSE
// cuFFT applies NO normalization in either direction, so the scale factor
// is applied here on the host exactly as the CPU path applies it -- both
// backends therefore answer the same normalization contract.

#include "backends/cuda/fourier_cuda.hpp"

#include <cufft.h>
#include <cuda_runtime.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace scl {
namespace {

#define SCL_CUDA_CHECK(expr)                                                             \
    do {                                                                                 \
        cudaError_t _scl_err = (expr);                                                   \
        if (_scl_err != cudaSuccess) {                                                   \
            std::ostringstream _scl_os;                                                  \
            _scl_os << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "           \
                     << cudaGetErrorString(_scl_err);                                     \
            throw std::runtime_error(_scl_os.str());                                     \
        }                                                                                 \
    } while (0)

#define SCL_CUFFT_CHECK(expr)                                                            \
    do {                                                                                 \
        cufftResult _scl_res = (expr);                                                    \
        if (_scl_res != CUFFT_SUCCESS) {                                                  \
            std::ostringstream _scl_os;                                                  \
            _scl_os << "cuFFT error at " << __FILE__ << ":" << __LINE__ << ": code "     \
                     << static_cast<int>(_scl_res);                                        \
            throw std::runtime_error(_scl_os.str());                                     \
        }                                                                                 \
    } while (0)

}  // namespace

FourierResult compute_fourier_transform_1d_cuda(const std::vector<std::complex<double>>& signal,
                                                 const FourierParameters& params) {
    FourierResult result;
    const int n = static_cast<int>(signal.size());
    result.spectrum.assign(signal.size(), std::complex<double>(0.0, 0.0));

    // std::complex<double> is layout-compatible with cufftDoubleComplex
    // (two contiguous doubles, real then imag) -- guaranteed by the C++
    // standard for std::complex specializations.
    std::vector<cufftDoubleComplex> host(signal.size());
    for (std::size_t i = 0; i < signal.size(); ++i) {
        host[i].x = signal[i].real();
        host[i].y = signal[i].imag();
    }

    cufftDoubleComplex* device = nullptr;
    SCL_CUDA_CHECK(cudaMalloc(&device, sizeof(cufftDoubleComplex) * signal.size()));
    SCL_CUDA_CHECK(cudaMemcpy(device, host.data(), sizeof(cufftDoubleComplex) * signal.size(),
                               cudaMemcpyHostToDevice));

    cufftHandle plan;
    SCL_CUFFT_CHECK(cufftPlan1d(&plan, n, CUFFT_Z2Z, 1));
    const int direction =
        (params.direction == FourierDirection::Forward) ? CUFFT_FORWARD : CUFFT_INVERSE;
    SCL_CUFFT_CHECK(cufftExecZ2Z(plan, device, device, direction));
    SCL_CUDA_CHECK(cudaDeviceSynchronize());

    SCL_CUDA_CHECK(cudaMemcpy(host.data(), device, sizeof(cufftDoubleComplex) * signal.size(),
                               cudaMemcpyDeviceToHost));
    cufftDestroy(plan);
    cudaFree(device);

    const double scale = fourier_normalization_scale(params.normalization, signal.size());
    for (std::size_t i = 0; i < signal.size(); ++i) {
        const double re = host[i].x * scale;
        const double im = host[i].y * scale;
        if (!std::isfinite(re) || !std::isfinite(im)) {
            result.ok = false;
            result.fault = FourierFault::NonFinite;
            return result;
        }
        result.spectrum[i] = std::complex<double>(re, im);
    }

    result.ok = true;
    result.fault = FourierFault::None;
    return result;
}

}  // namespace scl
