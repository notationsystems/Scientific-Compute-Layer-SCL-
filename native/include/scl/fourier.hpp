#pragma once
// The one-dimensional discrete Fourier transform -- the MATHEMATICAL
// operation, deliberately named for the transform and not for any
// algorithm that computes it (a direct O(N^2) sum here on CPU, cuFFT on
// the CUDA backend; both answer the same contract).
//
// Contract (docs/SCL_CONTRACT.md has the wire-level form):
//
//   forward  (direction = +1):  X_k = s * SUM_{n=0}^{N-1} x_n exp(-2*pi*i*k*n/N)
//   inverse  (direction = -1):  X_k = s * SUM_{n=0}^{N-1} x_n exp(+2*pi*i*k*n/N)
//
//   k = 0 .. N-1, in increasing order; no fftshift, no reordering.
//   s = the normalization scale factor:
//         None         -> s = 1            (matches the bare definition above)
//         OneOverN     -> s = 1/N
//         OneOverSqrtN -> s = 1/sqrt(N)    (unitary convention)
//
// Input and output are BOTH complex, N in and N out. A real-valued signal
// is supplied with zero imaginary parts: complex-in/complex-out is the
// simplest uniform contract, with none of the Hermitian-packing subtleties
// a real-input-only variant would force into the wire format.
//
// Precision is IEEE-754 float64 throughout, fixed for this contract (not a
// configuration knob -- a single-precision transform would be a different
// contract, not a parameter of this one).
//
// Any N >= 1 is supported; there is no power-of-two restriction.
//
// NOT part of this contract: sampling interval, frequency axis, units.
// Those are metadata ABOUT a signal, not part of the transform -- see
// docs/SCL_CONTRACT.md and python/scl/fourier.py::frequency_bins for the
// deliberate separation between the mathematical output and its physical
// interpretation.

#include <complex>
#include <string>
#include <vector>

#include "scl/backend.hpp"

namespace scl {

enum class FourierDirection { Forward = 1, Inverse = -1 };

enum class FourierNormalization { None = 0, OneOverN = 1, OneOverSqrtN = 2 };

struct FourierParameters {
    FourierDirection direction;
    FourierNormalization normalization;
    //: Sampling interval, carried so it participates in the request's
    //: parameter identity -- NOT used by the transform itself (see the
    //: header comment). When absent, the result is interpretable only in
    //: bin/index terms.
    bool has_sample_spacing;
    double sample_spacing_seconds;
};

enum class FourierFault {
    None,
    NonFinite,  // a transformed value evaluated to NaN/Inf
};

struct FourierResult {
    bool ok = false;
    FourierFault fault = FourierFault::None;
    std::vector<std::complex<double>> spectrum;
};

//: Structural validation: non-empty signal, all-finite samples, and (when
//: supplied) a finite, strictly positive sample spacing. Returns a
//: human-readable reason, or an empty string when the input is valid.
std::string validate_fourier_input(const std::vector<std::complex<double>>& signal,
                                    const FourierParameters& params);

//: The CPU reference implementation: the direct O(N^2) sum, straight from
//: the definition above. Deliberately NOT an optimized FFT -- correctness
//: precedes optimization, and a transform written from the definition is
//: the thing an optimized implementation would later be validated against.
FourierResult compute_fourier_transform_1d_cpu(const std::vector<std::complex<double>>& signal,
                                                const FourierParameters& params);

//: Backend dispatch. Throws BackendUnavailableError for a backend this
//: build/host cannot run, exactly as compute_lj_pairwise does.
FourierResult compute_fourier_transform_1d(Backend backend,
                                            const std::vector<std::complex<double>>& signal,
                                            const FourierParameters& params);

//: The scale factor s for a given normalization and length.
double fourier_normalization_scale(FourierNormalization normalization, std::size_t n);

}  // namespace scl
