// scl_cli: the SCL native compute boundary's process entry point.
//
// Reads one JSON request from stdin, runs one computation, writes one
// JSON response to stdout, exits. See docs/SCL_CONTRACT.md for the full
// wire format. This file owns ONLY the operation-agnostic boundary:
// envelope parsing, hex framing, backend selection and its
// availability/fault ordering, and the response shape. Each operation
// owns its own configuration/input decoding, validation, backend
// dispatch, output encoding and metrics behind `scl::Operation`
// (scl/operation.hpp) -- so adding an operation touches a registry row
// and one op_*.cpp, never this file.

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "scl/backend.hpp"
#include "scl/operation.hpp"
#include "scl/protocol.hpp"
#include "scl/version.hpp"

using nlohmann::json;

namespace {

class ProtocolError : public std::runtime_error {
public:
    explicit ProtocolError(const std::string& what) : std::runtime_error(what) {}
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

json metrics_to_json(const std::vector<scl::Metric>& metrics) {
    json out = json::object();
    for (const scl::Metric& metric : metrics) {
        out[metric.name] = metric.value;
    }
    return out;
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

    const scl::Operation* selected = scl::find_operation(operation);
    if (selected == nullptr) {
        std::ostringstream os;
        os << "unknown operation '" << operation << "' (this build supports";
        const std::vector<std::string> names = scl::supported_operation_names();
        for (std::size_t i = 0; i < names.size(); ++i) {
            os << (i == 0 ? " '" : ", '") << names[i] << "'";
        }
        os << ")";
        out << make_response("halted", scl::kFaultProtocol, backend_field, std::nullopt, os.str(),
                              json::object())
            << "\n";
        return scl::kProcessExitOk;
    }

    // Backend availability is deliberately checked BEFORE the operation
    // decodes anything: a request for a backend this build/host cannot run
    // is answered as BACKEND_UNAVAILABLE even when it is ALSO malformed
    // (locked by tests/test_cpu_cuda_equivalence.py). Single source of
    // truth for the reason text, shared with BackendUnavailableError.
    std::string unavailable_reason = scl::backend_unavailable_reason(backend);
    if (!unavailable_reason.empty()) {
        out << make_response("halted", scl::kFaultBackendUnavailable, backend_field, std::nullopt,
                              unavailable_reason, json::object())
            << "\n";
        return scl::kProcessExitOk;
    }

    try {
        scl::OperationRequest op_request;
        op_request.backend = backend;
        op_request.configuration = from_hex(request["configuration_hex"].get<std::string>());
        op_request.input = from_hex(request["input_hex"].get<std::string>());

        scl::OperationOutcome outcome = selected->run(op_request);

        std::optional<std::string> output_hex;
        if (outcome.has_output) {
            output_hex = to_hex(outcome.output);
        }
        std::optional<std::string> detail;
        if (!outcome.detail.empty()) {
            detail = outcome.detail;
        }
        out << make_response(outcome.status, outcome.exit_code, backend_field, output_hex, detail,
                              metrics_to_json(outcome.metrics))
            << "\n";
        return scl::kProcessExitOk;
    } catch (const ProtocolError& e) {
        out << make_response("halted", scl::kFaultProtocol, backend_field, std::nullopt, e.what(),
                              json::object())
            << "\n";
        return scl::kProcessExitOk;
    } catch (const scl::OperationValidationError& e) {
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
