#pragma once
// Backend selection: the ONE seam where "which physical engine ran"
// enters the computation. Mirrors how STE folds engine identity into
// program bytes (see docs/SCL_CONTRACT.md) -- SCL folds backend choice
// into the CLI request the same way, and this header is where a future
// backend (a second CUDA compute capability tier, an OpenCL backend,
// whatever) gets added, without touching lj_pairwise.hpp's pure math.

#include <stdexcept>
#include <string>

#include "scl/lj_pairwise.hpp"

namespace scl {

enum class Backend { Cpu, Cuda };

// Thrown when a caller asks for a backend this BUILD does not contain
// (e.g. backend=cuda in a binary compiled without SCL_WITH_CUDA, or a
// binary compiled with it but with no CUDA device visible at runtime).
// Deliberately its own type -- never collapsed into a generic runtime
// error -- so the CLI can map it to SCL_FAULT_BACKEND_UNAVAILABLE (12)
// distinctly from a validation or computation fault.
class BackendUnavailableError : public std::runtime_error {
public:
    explicit BackendUnavailableError(const std::string& what) : std::runtime_error(what) {}
};

// True if this binary was compiled with support for `backend` AND (for
// backends that need one) a usable device is present right now. Cpu is
// always true.
bool backend_available(Backend backend);

std::string backend_name(Backend backend);

// Empty optional-by-exception convention (see BackendUnavailableError):
// returns nullopt-equivalent never happens -- either it computes, or it
// throws BackendUnavailableError, or lj_pairwise's own fault reporting
// (LJResult::ok == false) surfaces a computation-level fault.
LJResult compute_lj_pairwise(Backend backend, const std::vector<Vec3>& positions,
                              const LJParameters& params);

}  // namespace scl
