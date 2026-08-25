// The `lj_pairwise_energy_forces` operation.
//
// Every function here was MOVED VERBATIM out of native/src/main.cpp when the
// operation boundary was generalized -- same buffer-length rules, same
// error message text (several are asserted on by name in
// tests/test_failure_paths.py), same fault codes, same metrics, same
// output encoding. Nothing about this operation's behavior changed; it
// only stopped being hardcoded into the CLI.

#include "scl/bytes.hpp"
#include "scl/lj_pairwise.hpp"
#include "scl/operation.hpp"
#include "scl/protocol.hpp"

#include <chrono>
#include <sstream>

namespace scl {
namespace {

LJParameters decode_configuration(const std::vector<uint8_t>& bytes) {
    if (bytes.size() != 24) {
        std::ostringstream os;
        os << "configuration must be exactly 24 bytes (3 little-endian float64: "
           << "epsilon, sigma, cutoff), got " << bytes.size();
        throw OperationValidationError(os.str());
    }
    LJParameters params;
    params.epsilon = read_double_le(bytes, 0);
    params.sigma = read_double_le(bytes, 8);
    params.cutoff = read_double_le(bytes, 16);
    return params;
}

std::vector<Vec3> decode_positions(const std::vector<uint8_t>& bytes) {
    if (bytes.size() % 24 != 0) {
        std::ostringstream os;
        os << "input must be a whole number of 24-byte particles (3 little-endian "
           << "float64 each: x, y, z), got " << bytes.size() << " bytes";
        throw OperationValidationError(os.str());
    }
    std::vector<Vec3> positions;
    positions.reserve(bytes.size() / 24);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 24) {
        Vec3 p;
        p.x = read_double_le(bytes, offset);
        p.y = read_double_le(bytes, offset + 8);
        p.z = read_double_le(bytes, offset + 16);
        positions.push_back(p);
    }
    return positions;
}

std::vector<uint8_t> encode_output(double total_energy, const std::vector<Vec3>& forces) {
    std::vector<uint8_t> out;
    out.reserve(8 + forces.size() * 24);
    write_double_le(out, total_energy);
    for (const auto& f : forces) {
        write_double_le(out, f.x);
        write_double_le(out, f.y);
        write_double_le(out, f.z);
    }
    return out;
}

}  // namespace

OperationOutcome run_lj_pairwise_energy_forces(const OperationRequest& request) {
    try {
        LJParameters params = decode_configuration(request.configuration);
        std::vector<Vec3> positions = decode_positions(request.input);

        std::string reason = validate_lj_input(positions, params);
        if (!reason.empty()) {
            return OperationOutcome::halted(kFaultValidation, reason);
        }

        auto compute_start = std::chrono::steady_clock::now();
        LJResult result = compute_lj_pairwise(request.backend, positions, params);
        auto compute_end = std::chrono::steady_clock::now();
        double compute_seconds = std::chrono::duration<double>(compute_end - compute_start).count();

        std::vector<Metric> metrics = {
            {"native_compute_seconds", compute_seconds},
            {"n_particles", static_cast<double>(positions.size())},
        };

        if (!result.ok) {
            std::string detail = (result.fault == ComputeFault::CoincidentParticles)
                                      ? "two particles at zero separation: the potential is singular"
                                      : "energy or force evaluated to a non-finite value";
            return OperationOutcome::halted(kFaultComputation, detail, metrics);
        }

        return OperationOutcome::completed(encode_output(result.total_energy, result.forces), metrics);
    } catch (const OperationValidationError& e) {
        return OperationOutcome::halted(kFaultValidation, e.what());
    } catch (const BackendUnavailableError& e) {
        return OperationOutcome::halted(kFaultBackendUnavailable, e.what());
    }
}

}  // namespace scl
