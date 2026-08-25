// scl_cli: the SCL native compute boundary's process entry point.
//
// Reads one JSON request from stdin, runs one computation, writes one
// JSON response to stdout, exits. See docs/SCL_CONTRACT.md for the full
// wire format. This file owns ONLY protocol marshaling (JSON <-> the
// pure LJParameters/Vec3 types) and fault mapping; the actual physics
// lives in lj_pairwise.cpp/backend.cpp and is unit-tested independently
// of this process boundary (native/tests/).

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "scl/backend.hpp"
#include "scl/lj_pairwise.hpp"
#include "scl/protocol.hpp"
#include "scl/version.hpp"

using nlohmann::json;

namespace {

class ProtocolError : public std::runtime_error {
public:
    explicit ProtocolError(const std::string& what) : std::runtime_error(what) {}
};
class ValidationError : public std::runtime_error {
public:
    explicit ValidationError(const std::string& what) : std::runtime_error(what) {}
};

std::string to_hex(const std::vector<uint8_t>& bytes) {
    static const char* digits = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (uint8_t b : bytes) {
        out.push_back(digits[(b >> 4) & 0xF]);
        out.push_back(digits[b & 0xF]);
    }
    return out;
}

std::vector<uint8_t> from_hex(const std::string& hex) {
    if (hex.size() % 2 != 0) {
        throw ProtocolError("hex field has odd length");
    }
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        throw ProtocolError(std::string("invalid hex digit: ") + c);
    };
    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    for (std::size_t i = 0; i < hex.size(); i += 2) {
        out.push_back(static_cast<uint8_t>((nibble(hex[i]) << 4) | nibble(hex[i + 1])));
    }
    return out;
}

double read_double_le(const std::vector<uint8_t>& bytes, std::size_t offset) {
    // x86_64 is little-endian with IEEE-754 doubles, matching Python's
    // struct.pack("<d", ...) byte-for-byte -- a documented environment
    // assumption (docs/SCL_CONTRACT.md), not a portable deserializer.
    double value;
    std::memcpy(&value, bytes.data() + offset, sizeof(double));
    return value;
}

void write_double_le(std::vector<uint8_t>& out, double value) {
    uint8_t buf[sizeof(double)];
    std::memcpy(buf, &value, sizeof(double));
    out.insert(out.end(), buf, buf + sizeof(double));
}

scl::LJParameters decode_configuration(const std::vector<uint8_t>& bytes) {
    if (bytes.size() != 24) {
        std::ostringstream os;
        os << "configuration must be exactly 24 bytes (3 little-endian float64: "
           << "epsilon, sigma, cutoff), got " << bytes.size();
        throw ValidationError(os.str());
    }
    scl::LJParameters params;
    params.epsilon = read_double_le(bytes, 0);
    params.sigma = read_double_le(bytes, 8);
    params.cutoff = read_double_le(bytes, 16);
    return params;
}

std::vector<scl::Vec3> decode_positions(const std::vector<uint8_t>& bytes) {
    if (bytes.size() % 24 != 0) {
        std::ostringstream os;
        os << "input must be a whole number of 24-byte particles (3 little-endian "
           << "float64 each: x, y, z), got " << bytes.size() << " bytes";
        throw ValidationError(os.str());
    }
    std::vector<scl::Vec3> positions;
    positions.reserve(bytes.size() / 24);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 24) {
        scl::Vec3 p;
        p.x = read_double_le(bytes, offset);
        p.y = read_double_le(bytes, offset + 8);
        p.z = read_double_le(bytes, offset + 16);
        positions.push_back(p);
    }
    return positions;
}

std::string encode_output(double total_energy, const std::vector<scl::Vec3>& forces) {
    std::vector<uint8_t> out;
    out.reserve(8 + forces.size() * 24);
    write_double_le(out, total_energy);
    for (const auto& f : forces) {
        write_double_le(out, f.x);
        write_double_le(out, f.y);
        write_double_le(out, f.z);
    }
    return to_hex(out);
}

json make_response(const std::string& status, int exit_code, const std::string& backend_used,
                    std::optional<std::string> output_hex, std::optional<std::string> detail,
                    json metrics) {
    json response;
    response["status"] = status;
    response["exit_code"] = exit_code;
    response["backend_used"] = backend_used;
    response["backend_version"] = scl::kVersionString;
    response["output_hex"] = output_hex.has_value() ? json(*output_hex) : json(nullptr);
    response["detail"] = detail.has_value() ? json(*detail) : json(nullptr);
    response["metrics"] = std::move(metrics);
    return response;
}

int run(std::istream& in, std::ostream& out) {
    std::ostringstream buffer;
    buffer << in.rdbuf();
    const std::string raw = buffer.str();

    json request;
    std::string operation;
    std::string backend_field;
    scl::Backend backend = scl::Backend::Cpu;

    try {
        request = json::parse(raw);
        if (!request.is_object()) {
            throw ProtocolError("request is not a JSON object");
        }
        if (!request.contains("operation") || !request["operation"].is_string()) {
            throw ProtocolError("request missing string field 'operation'");
        }
        operation = request["operation"].get<std::string>();
        if (!request.contains("backend") || !request["backend"].is_string()) {
            throw ProtocolError("request missing string field 'backend'");
        }
        backend_field = request["backend"].get<std::string>();
        if (backend_field == "cpu") {
            backend = scl::Backend::Cpu;
        } else if (backend_field == "cuda") {
            backend = scl::Backend::Cuda;
        } else {
            throw ProtocolError("unknown backend '" + backend_field + "' (expected cpu|cuda)");
        }
        if (!request.contains("configuration_hex") || !request["configuration_hex"].is_string()) {
            throw ProtocolError("request missing string field 'configuration_hex'");
        }
        if (!request.contains("input_hex") || !request["input_hex"].is_string()) {
            throw ProtocolError("request missing string field 'input_hex'");
        }
    } catch (const json::parse_error& e) {
        out << make_response("halted", scl::kFaultProtocol, "unknown", std::nullopt,
                              std::string("JSON parse error: ") + e.what(), json::object())
            << "\n";
        return scl::kProcessExitOk;
    } catch (const ProtocolError& e) {
        out << make_response("halted", scl::kFaultProtocol, "unknown", std::nullopt, e.what(),
                              json::object())
            << "\n";
        return scl::kProcessExitOk;
    }

    if (operation != "lj_pairwise_energy_forces") {
        out << make_response("halted", scl::kFaultProtocol, backend_field, std::nullopt,
                              "unknown operation '" + operation +
                                  "' (this build supports 'lj_pairwise_energy_forces' only)",
                              json::object())
            << "\n";
        return scl::kProcessExitOk;
    }

    std::string unavailable_reason = scl::backend_unavailable_reason(backend);
    if (!unavailable_reason.empty()) {
        out << make_response("halted", scl::kFaultBackendUnavailable, backend_field, std::nullopt,
                              unavailable_reason, json::object())
            << "\n";
        return scl::kProcessExitOk;
    }

    try {
        std::vector<uint8_t> configuration_bytes = from_hex(request["configuration_hex"].get<std::string>());
        std::vector<uint8_t> input_bytes = from_hex(request["input_hex"].get<std::string>());

        scl::LJParameters params = decode_configuration(configuration_bytes);
        std::vector<scl::Vec3> positions = decode_positions(input_bytes);

        std::string reason = scl::validate_lj_input(positions, params);
        if (!reason.empty()) {
            out << make_response("halted", scl::kFaultValidation, backend_field, std::nullopt,
                                  reason, json::object())
                << "\n";
            return scl::kProcessExitOk;
        }

        auto compute_start = std::chrono::steady_clock::now();
        scl::LJResult result = scl::compute_lj_pairwise(backend, positions, params);
        auto compute_end = std::chrono::steady_clock::now();
        double compute_seconds =
            std::chrono::duration<double>(compute_end - compute_start).count();

        json metrics;
        metrics["native_compute_seconds"] = compute_seconds;
        metrics["n_particles"] = positions.size();

        if (!result.ok) {
            std::string detail = (result.fault == scl::ComputeFault::CoincidentParticles)
                                      ? "two particles at zero separation: the potential is singular"
                                      : "energy or force evaluated to a non-finite value";
            out << make_response("halted", scl::kFaultComputation, backend_field, std::nullopt,
                                  detail, metrics)
                << "\n";
            return scl::kProcessExitOk;
        }

        std::string output_hex = encode_output(result.total_energy, result.forces);
        out << make_response("completed", scl::kFaultNone, backend_field, output_hex, std::nullopt,
                              metrics)
            << "\n";
        return scl::kProcessExitOk;
    } catch (const ProtocolError& e) {
        out << make_response("halted", scl::kFaultProtocol, backend_field, std::nullopt, e.what(),
                              json::object())
            << "\n";
        return scl::kProcessExitOk;
    } catch (const ValidationError& e) {
        out << make_response("halted", scl::kFaultValidation, backend_field, std::nullopt, e.what(),
                              json::object())
            << "\n";
        return scl::kProcessExitOk;
    } catch (const scl::BackendUnavailableError& e) {
        out << make_response("halted", scl::kFaultBackendUnavailable, backend_field, std::nullopt,
                              e.what(), json::object())
            << "\n";
        return scl::kProcessExitOk;
    } catch (const std::exception& e) {
        out << make_response("halted", scl::kFaultInternal, backend_field, std::nullopt,
                              std::string("internal error: ") + e.what(), json::object())
            << "\n";
        return scl::kProcessExitOk;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--version") {
        std::cout << scl::kVersionString << "\n";
        return 0;
    }
    // A truly catastrophic failure (e.g. bad_alloc escaping `run`, which
    // already catches std::exception) is the ONE case where the process
    // exit code itself carries the fault, because there was no way left
    // to write well-formed JSON.
    try {
        return run(std::cin, std::cout);
    } catch (...) {
        std::cerr << "scl_cli: catastrophic failure producing no JSON response\n";
        return scl::kProcessExitCatastrophic;
    }
}
