// The `fourier_transform_1d` operation: wire decoding/encoding and fault
// mapping only. The mathematics lives in scl/fourier.hpp + fourier.cpp so
// it stays independently testable (native/tests) without any process or
// protocol concern.
//
// configuration: exactly 24 bytes, little-endian
//     offset  0  int32    direction              +1 forward, -1 inverse
//     offset  4  int32    normalization           0 none, 1 1/N, 2 1/sqrt(N)
//     offset  8  int32    has_sample_spacing      0 or 1
//     offset 12  int32    reserved                must be 0
//     offset 16  float64  sample_spacing_seconds  ignored when has_sample_spacing == 0
//
// input:  N * 16 bytes -- N complex samples as (real float64, imag float64)
// output: N * 16 bytes -- N complex bins,  same layout, k = 0..N-1 ascending

#include "scl/bytes.hpp"
#include "scl/fourier.hpp"
#include "scl/operation.hpp"
#include "scl/protocol.hpp"

#include <chrono>
#include <sstream>

namespace scl {
namespace {

FourierParameters decode_configuration(const std::vector<uint8_t>& bytes) {
    if (bytes.size() != 24) {
        std::ostringstream os;
        os << "configuration must be exactly 24 bytes (int32 direction, int32 normalization, "
           << "int32 has_sample_spacing, int32 reserved, float64 sample_spacing_seconds), got "
           << bytes.size();
        throw OperationValidationError(os.str());
    }

    const int32_t direction = read_int32_le(bytes, 0);
    const int32_t normalization = read_int32_le(bytes, 4);
    const int32_t has_spacing = read_int32_le(bytes, 8);
    const int32_t reserved = read_int32_le(bytes, 12);

    FourierParameters params;
    if (direction == 1) {
        params.direction = FourierDirection::Forward;
    } else if (direction == -1) {
        params.direction = FourierDirection::Inverse;
    } else {
        std::ostringstream os;
        os << "direction must be +1 (forward) or -1 (inverse), got " << direction;
        throw OperationValidationError(os.str());
    }

    switch (normalization) {
        case 0:
            params.normalization = FourierNormalization::None;
            break;
        case 1:
            params.normalization = FourierNormalization::OneOverN;
            break;
        case 2:
            params.normalization = FourierNormalization::OneOverSqrtN;
            break;
        default: {
            std::ostringstream os;
            os << "normalization must be 0 (none), 1 (1/N) or 2 (1/sqrt(N)), got " << normalization;
            throw OperationValidationError(os.str());
        }
    }

    if (has_spacing != 0 && has_spacing != 1) {
        std::ostringstream os;
        os << "has_sample_spacing must be 0 or 1, got " << has_spacing;
        throw OperationValidationError(os.str());
    }
    if (reserved != 0) {
        std::ostringstream os;
        os << "reserved configuration field must be 0, got " << reserved;
        throw OperationValidationError(os.str());
    }

    params.has_sample_spacing = (has_spacing == 1);
    params.sample_spacing_seconds = read_double_le(bytes, 16);
    return params;
}

std::vector<std::complex<double>> decode_signal(const std::vector<uint8_t>& bytes) {
    if (bytes.size() % 16 != 0) {
        std::ostringstream os;
        os << "input must be a whole number of 16-byte complex samples (2 little-endian "
           << "float64 each: real, imag), got " << bytes.size() << " bytes";
        throw OperationValidationError(os.str());
    }
    std::vector<std::complex<double>> signal;
    signal.reserve(bytes.size() / 16);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 16) {
        signal.emplace_back(read_double_le(bytes, offset), read_double_le(bytes, offset + 8));
    }
    return signal;
}

std::vector<uint8_t> encode_spectrum(const std::vector<std::complex<double>>& spectrum) {
    std::vector<uint8_t> out;
    out.reserve(spectrum.size() * 16);
    for (const auto& value : spectrum) {
        write_double_le(out, value.real());
        write_double_le(out, value.imag());
    }
    return out;
}

}  // namespace

OperationOutcome run_fourier_transform_1d(const OperationRequest& request) {
    try {
        FourierParameters params = decode_configuration(request.configuration);
        std::vector<std::complex<double>> signal = decode_signal(request.input);

        std::string reason = validate_fourier_input(signal, params);
        if (!reason.empty()) {
            return OperationOutcome::halted(kFaultValidation, reason);
        }

        auto compute_start = std::chrono::steady_clock::now();
        FourierResult result = compute_fourier_transform_1d(request.backend, signal, params);
        auto compute_end = std::chrono::steady_clock::now();
        double compute_seconds = std::chrono::duration<double>(compute_end - compute_start).count();

        std::vector<Metric> metrics = {
            {"native_compute_seconds", compute_seconds},
            {"n_samples", static_cast<double>(signal.size())},
            {"transform_size", static_cast<double>(signal.size())},
        };

        if (!result.ok) {
            return OperationOutcome::halted(
                kFaultComputation, "a transformed value evaluated to a non-finite value", metrics);
        }

        return OperationOutcome::completed(encode_spectrum(result.spectrum), metrics);
    } catch (const OperationValidationError& e) {
        return OperationOutcome::halted(kFaultValidation, e.what());
    } catch (const BackendUnavailableError& e) {
        return OperationOutcome::halted(kFaultBackendUnavailable, e.what());
    }
}

}  // namespace scl
