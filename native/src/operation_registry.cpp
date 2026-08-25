#include "scl/operation.hpp"

namespace scl {
namespace {

//: The complete, fixed set of operations this binary can run. Adding an
//: operation means adding one row here and one `op_*.cpp` -- there is no
//: dynamic registration, no load order to reason about, and no way for a
//: half-registered operation to exist.
const Operation kOperations[] = {
    {"lj_pairwise_energy_forces", &run_lj_pairwise_energy_forces},
    {"fourier_transform_1d", &run_fourier_transform_1d},
};

}  // namespace

const Operation* find_operation(const std::string& name) {
    for (const Operation& op : kOperations) {
        if (name == op.name) {
            return &op;
        }
    }
    return nullptr;
}

std::vector<std::string> supported_operation_names() {
    std::vector<std::string> names;
    names.reserve(sizeof(kOperations) / sizeof(kOperations[0]));
    for (const Operation& op : kOperations) {
        names.emplace_back(op.name);
    }
    return names;
}

}  // namespace scl
