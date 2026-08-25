// CUDA backend for the SCL Lennard-Jones pairwise kernel.
//
// STATUS: written against the CUDA 12 runtime API, mirrors
// native/src/lj_pairwise.cpp's math term-for-term, but has NEVER BEEN
// COMPILED OR RUN -- this development sandbox has no nvcc and no GPU
// (verified: `which nvcc` and `nvidia-smi` both fail; see
// docs/PHASE1_AUDIT.md). It is included as a real, reviewable first
// implementation for Phase 2, not a placeholder -- but it carries none of
// the verification the CPU backend has (no build, no run, no numerical
// validation). Do not treat this file as a proven computational path
// until it has been built and validated on real CUDA hardware.
//
// Only compiled when native/CMakeLists.txt's SCL_WITH_CUDA option is ON
// and CMake's FindCUDAToolkit succeeds.

#include "backends/cuda/lj_pairwise_cuda.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace scl {
namespace {

struct Double3 {
    double x, y, z;
};

// One thread per particle i; each thread sweeps all N-1 partners. Not the
// fastest possible layout (no tiling/shared-memory reuse, no Newton's-
// third-law halving across threads -- every pair is computed twice, once
// from each side, matching the CPU reference's per-particle force sum
// exactly so the two backends are checkable against each other) but it is
// the simplest layout that is straightforwardly correct, which matters
// more for an unverified first CUDA path than raw throughput.
__global__ void lj_kernel(const Double3* positions, int n, double epsilon, double sigma,
                           double cutoff2, Double3* forces, double* pair_energy_twice) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Double3 pi = positions[i];
    Double3 f{0.0, 0.0, 0.0};
    double e = 0.0;

    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        double dx = pi.x - positions[j].x;
        double dy = pi.y - positions[j].y;
        double dz = pi.z - positions[j].z;
        double r2 = dx * dx + dy * dy + dz * dz;
        // KNOWN GAP (see file header): unlike the CPU backend, a
        // coincident pair (r2==0) is silently skipped here rather than
        // raising ComputeFault::CoincidentParticles -- flagging it would
        // need a device-wide atomic and this path has no hardware to
        // validate that against yet. Phase 2 must close this before the
        // CUDA backend is trusted for the same fault contract as CPU.
        if (r2 == 0.0 || r2 > cutoff2) continue;

        double inv_r2 = 1.0 / r2;
        double sr2 = (sigma * sigma) * inv_r2;
        double sr6 = sr2 * sr2 * sr2;
        double sr12 = sr6 * sr6;

        // Halved: this thread and thread j both traverse the (i,j) pair,
        // so summing pair_energy_twice[i] over all i double-counts every
        // pair exactly once too many; the host divides by 2 after copy-
        // back (see compute_lj_pairwise_cuda below).
        e += 4.0 * epsilon * (sr12 - sr6);

        double fscalar = 24.0 * epsilon * inv_r2 * (2.0 * sr12 - sr6);
        f.x += fscalar * dx;
        f.y += fscalar * dy;
        f.z += fscalar * dz;
    }

    forces[i] = f;
    pair_energy_twice[i] = e;
}

#define SCL_CUDA_CHECK(expr)                                                                \
    do {                                                                                    \
        cudaError_t _scl_err = (expr);                                                      \
        if (_scl_err != cudaSuccess) {                                                      \
            std::ostringstream _scl_os;                                                     \
            _scl_os << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "              \
                     << cudaGetErrorString(_scl_err);                                        \
            throw std::runtime_error(_scl_os.str());                                        \
        }                                                                                    \
    } while (0)

}  // namespace

bool cuda_device_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    // A clean "no device"/"no driver" answer is NOT a thrown error here:
    // it is exactly what this function exists to report, honestly, as
    // `false` -- distinct from a real CUDA runtime malfunction, which
    // would be a bug in Phase 2, not an expected BACKEND_UNAVAILABLE path.
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver) {
        return false;
    }
    if (err != cudaSuccess) {
        return false;
    }
    return count > 0;
}

LJResult compute_lj_pairwise_cuda(const std::vector<Vec3>& positions, const LJParameters& params) {
    const int n = static_cast<int>(positions.size());
    LJResult result;
    result.forces.assign(positions.size(), Vec3{0.0, 0.0, 0.0});

    std::vector<Double3> host_positions(n);
    for (int i = 0; i < n; ++i) {
        host_positions[i] = Double3{positions[i].x, positions[i].y, positions[i].z};
    }

    Double3* device_positions = nullptr;
    Double3* device_forces = nullptr;
    double* device_energy = nullptr;

    SCL_CUDA_CHECK(cudaMalloc(&device_positions, sizeof(Double3) * n));
    SCL_CUDA_CHECK(cudaMalloc(&device_forces, sizeof(Double3) * n));
    SCL_CUDA_CHECK(cudaMalloc(&device_energy, sizeof(double) * n));
    SCL_CUDA_CHECK(cudaMemcpy(device_positions, host_positions.data(), sizeof(Double3) * n,
                               cudaMemcpyHostToDevice));

    const int threads_per_block = 128;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    lj_kernel<<<blocks, threads_per_block>>>(device_positions, n, params.epsilon, params.sigma,
                                              params.cutoff * params.cutoff, device_forces,
                                              device_energy);
    SCL_CUDA_CHECK(cudaGetLastError());
    SCL_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Double3> host_forces(n);
    std::vector<double> host_energy(n);
    SCL_CUDA_CHECK(cudaMemcpy(host_forces.data(), device_forces, sizeof(Double3) * n,
                               cudaMemcpyDeviceToHost));
    SCL_CUDA_CHECK(cudaMemcpy(host_energy.data(), device_energy, sizeof(double) * n,
                               cudaMemcpyDeviceToHost));

    cudaFree(device_positions);
    cudaFree(device_forces);
    cudaFree(device_energy);

    double total_twice = 0.0;
    for (int i = 0; i < n; ++i) {
        result.forces[i] = Vec3{host_forces[i].x, host_forces[i].y, host_forces[i].z};
        total_twice += host_energy[i];
        if (!std::isfinite(host_forces[i].x) || !std::isfinite(host_forces[i].y) ||
            !std::isfinite(host_forces[i].z)) {
            result.ok = false;
            result.fault = ComputeFault::NonFinite;
            return result;
        }
    }
    double total_energy = total_twice / 2.0;  // every pair counted from both sides
    if (!std::isfinite(total_energy)) {
        result.ok = false;
        result.fault = ComputeFault::NonFinite;
        return result;
    }

    result.ok = true;
    result.fault = ComputeFault::None;
    result.total_energy = total_energy;
    return result;
}

}  // namespace scl
