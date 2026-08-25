#pragma once
// Only included/compiled when SCL_WITH_CUDA is defined (see
// native/CMakeLists.txt) -- this header does not appear in the default
// CPU-only build at all, so it never becomes a dead include on a machine
// without the CUDA toolkit.

#include "scl/lj_pairwise.hpp"

namespace scl {

// True iff a CUDA device is visible right now (cudaGetDeviceCount() > 0
// and cudaGetLastError() is clean). Distinct from "was this binary built
// with CUDA support" (that's the SCL_WITH_CUDA compile guard in
// backend.cpp) -- a binary can be CUDA-capable yet find no device on a
// given machine, and that must fault the same way: BACKEND_UNAVAILABLE,
// never a silent fallback to CPU.
bool cuda_device_available();

// Same contract as compute_lj_pairwise_cpu (scl/lj_pairwise.hpp): one
// thread per particle i, each summing its N-1 pair interactions. Reduces
// per-particle energy contributions on the host after copy-back rather
// than with an on-device reduction, for a first implementation that is
// easy to audit; a device-side reduction is the obvious Phase 2
// optimization once real hardware is available to benchmark it against.
LJResult compute_lj_pairwise_cuda(const std::vector<Vec3>& positions, const LJParameters& params);

}  // namespace scl
