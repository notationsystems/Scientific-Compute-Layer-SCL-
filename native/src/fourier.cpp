#include "scl/fourier.hpp"

#ifdef SCL_WITH_CUDA
#include "backends/cuda/fourier_cuda.hpp"
#endif

#include <cmath>
#include <sstream>

namespace scl {

namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
}

double fourier_normalization_scale(FourierNormalization normalization, std::size_t n) {
    switch (normalization) {
        case FourierNormalization::None:
            return 1.0;
        case FourierNormalization::OneOverN:
            return 1.0 / static_cast<double>(n);
        case FourierNormalization::OneOverSqrtN:
            return 1.0 / std::sqrt(static_cast<double>(n));
    }
    return 1.0;
}

std::string validate_fourier_input(const std::vector<std::complex<double>>& signal,
                                    const FourierParameters& params) {
    if (signal.empty()) {
        return "at least one sample is required, got 0";
    }
    for (std::size_t i = 0; i < signal.size(); ++i) {
        if (!std::isfinite(signal[i].real()) || !std::isfinite(signal[i].imag())) {
            std::ostringstream os;
            os << "sample[" << i << "] is not finite";
            return os.str();
        }
    }
    if (params.has_sample_spacing) {
        if (!std::isfinite(params.sample_spacing_seconds) || params.sample_spacing_seconds <= 0.0) {
            std::ostringstream os;
            os << "sample_spacing_seconds must be finite and > 0 when supplied, got "
               << params.sample_spacing_seconds;
            return os.str();
        }
    }
    return "";
}

FourierResult compute_fourier_transform_1d_cpu(const std::vector<std::complex<double>>& signal,
                                                const FourierParameters& params) {
    FourierResult result;
    const std::size_t n = signal.size();
    result.spectrum.assign(n, std::complex<double>(0.0, 0.0));

    // Forward carries the negative exponent, per the contract in the header.
    const double exponent_sign = (params.direction == FourierDirection::Forward) ? -1.0 : 1.0;
    const double scale = fourier_normalization_scale(params.normalization, n);

    for (std::size_t k = 0; k < n; ++k) {
        double sum_real = 0.0;
        double sum_imag = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            // (k*j) mod N keeps the trigonometric argument small for large
            // k*j, which loses far less precision than reducing a huge
            // angle inside sin/cos would. The Python oracle
            // (tests/test_fourier_contract.py) reduces identically, so the
            // two agree to round-off rather than to argument-reduction
            // noise.
            const std::size_t kj = static_cast<std::size_t>(
                (static_cast<unsigned long long>(k) * static_cast<unsigned long long>(j)) % n);
            const double angle = exponent_sign * kTwoPi * static_cast<double>(kj) / static_cast<double>(n);
            const double c = std::cos(angle);
            const double s = std::sin(angle);
            // (a+bi)(c+si) = (ac - bs) + (as + bc)i
            sum_real += signal[j].real() * c - signal[j].imag() * s;
            sum_imag += signal[j].real() * s + signal[j].imag() * c;
        }
        result.spectrum[k] = std::complex<double>(sum_real * scale, sum_imag * scale);
    }

    for (const auto& value : result.spectrum) {
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
            result.ok = false;
            result.fault = FourierFault::NonFinite;
            return result;
        }
    }

    result.ok = true;
    result.fault = FourierFault::None;
    return result;
}

FourierResult compute_fourier_transform_1d(Backend backend,
                                            const std::vector<std::complex<double>>& signal,
                                            const FourierParameters& params) {
    if (backend == Backend::Cpu) {
        return compute_fourier_transform_1d_cpu(signal, params);
    }
    // Same single-source-of-truth availability reason the LJ path uses --
    // never a second, drifting message for the same condition.
    std::string reason = backend_unavailable_reason(backend);
    if (!reason.empty()) {
        throw BackendUnavailableError(reason);
    }
#ifdef SCL_WITH_CUDA
    return compute_fourier_transform_1d_cuda(signal, params);
#else
    throw BackendUnavailableError("backend cuda: unreachable -- backend_unavailable_reason "
                                   "should have already reported this build has no CUDA support");
#endif
}

}  // namespace scl
