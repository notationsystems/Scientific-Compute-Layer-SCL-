#pragma once
// Pure computational core: truncated Lennard-Jones pairwise energy and
// forces for an N-particle system. Deliberately free of I/O, JSON, and
// process concerns -- those live in the CLI layer (src/main.cpp) so this
// header can be unit-tested and, later, re-implemented by a CUDA backend
// behind the exact same signature (see backends/cuda/).
//
// V(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6], plain truncation at
// `cutoff` (V=0 beyond cutoff; not shifted to zero at the boundary -- a
// stated modeling simplification, not a bug).

#include <cstddef>
#include <string>
#include <vector>

namespace scl {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct LJParameters {
    double epsilon;
    double sigma;
    double cutoff;
};

enum class ComputeFault {
    None,
    CoincidentParticles,  // r == 0 between some pair: potential is singular
    NonFinite,             // energy or force evaluated to NaN/Inf
};

struct LJResult {
    bool ok = false;
    ComputeFault fault = ComputeFault::None;
    double total_energy = 0.0;
    std::vector<Vec3> forces;  // forces[i] corresponds to positions[i]
};

// Validates parameters/positions structurally (finite, sigma>0, epsilon>=0,
// cutoff>0, at least one particle). Returns a human-readable reason on
// failure, empty string on success. This is the VALIDATION fault surface;
// distinct from ComputeFault, which covers failures only detectable during
// the O(N^2) sweep itself (e.g. coincident particles).
std::string validate_lj_input(const std::vector<Vec3>& positions, const LJParameters& params);

// The reference CPU implementation: O(N^2) double loop, IEEE-754 double
// precision throughout, no SIMD/fast-math (see CMakeLists.txt) so results
// are reproducible bit-for-bit on the same binary/machine.
LJResult compute_lj_pairwise_cpu(const std::vector<Vec3>& positions, const LJParameters& params);

}  // namespace scl
