#pragma once
// The SCL operation boundary.
//
// Before the second operation existed, `scl_cli` compared the requested
// operation name against one hardcoded string and inlined that operation's
// decoders, validation, compute call, output encoder, and metrics directly
// in `main.cpp`. That was correct for exactly one operation and blocked a
// second one.
//
// This header is the smallest generalization that unblocks it: an
// operation is a NAME plus ONE entry point that owns its own
// configuration decoding, input decoding, validation, backend dispatch,
// output encoding, and metrics. Everything the operations share -- the
// process/JSON envelope, hex framing, backend-availability checking and
// its fault ordering, the fault vocabulary, the response shape -- stays in
// `main.cpp`, unchanged and operation-agnostic.
//
// Deliberately NOT a plugin framework: no dynamic loading, no
// registration side effects, no virtual class hierarchy. A fixed table of
// function pointers (`operation_registry.cpp`), mirroring the fixed
// kernel array STE's own Rust `execution-cli` uses.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "scl/backend.hpp"

namespace scl {

//: One named numeric metric. A vector of these (rather than a JSON object)
//: keeps `scl_core` free of the nlohmann_json dependency -- only the CLI
//: layer knows about JSON -- and keeps metric ordering deterministic
//: rather than dependent on a hash container's iteration order.
struct Metric {
    std::string name;
    double value;
};

//: Thrown by an operation's decoders when the request is structurally
//: invalid for THAT operation (wrong buffer length, out-of-range
//: parameter). Mapped to kFaultValidation (11).
class OperationValidationError : public std::runtime_error {
public:
    explicit OperationValidationError(const std::string& what) : std::runtime_error(what) {}
};

//: What an operation answers. `output` is meaningful only when
//: `status == "completed"`; a halted outcome carries no output at all,
//: never an empty-but-present one -- the same absence discipline
//: SCLResult/ExecutionResult keep on the Python side.
struct OperationOutcome {
    std::string status;  // "completed" | "halted"
    int exit_code;
    bool has_output;
    std::vector<uint8_t> output;
    std::string detail;  // empty when completed
    std::vector<Metric> metrics;

    static OperationOutcome completed(std::vector<uint8_t> output, std::vector<Metric> metrics) {
        OperationOutcome outcome;
        outcome.status = "completed";
        outcome.exit_code = 0;
        outcome.has_output = true;
        outcome.output = std::move(output);
        outcome.metrics = std::move(metrics);
        return outcome;
    }

    static OperationOutcome halted(int exit_code, const std::string& detail,
                                    std::vector<Metric> metrics = {}) {
        OperationOutcome outcome;
        outcome.status = "halted";
        outcome.exit_code = exit_code;
        outcome.has_output = false;
        outcome.detail = detail;
        outcome.metrics = std::move(metrics);
        return outcome;
    }
};

//: The already-hex-decoded request an operation receives. The CLI owns
//: envelope parsing and hex framing; an operation never sees JSON.
struct OperationRequest {
    Backend backend;
    std::vector<uint8_t> configuration;
    std::vector<uint8_t> input;
};

struct Operation {
    const char* name;
    OperationOutcome (*run)(const OperationRequest&);
};

//: nullptr when no operation of that name is built into this binary.
const Operation* find_operation(const std::string& name);

//: Every operation this build supports, in registry order -- used to make
//: the "unknown operation" fault message name the real alternatives
//: rather than a hardcoded list that could drift.
std::vector<std::string> supported_operation_names();

// The operations themselves (one entry point each; see the matching
// native/src/op_*.cpp for the decoders/validation/encoders each one owns).
OperationOutcome run_lj_pairwise_energy_forces(const OperationRequest& request);
OperationOutcome run_fourier_transform_1d(const OperationRequest& request);

}  // namespace scl
