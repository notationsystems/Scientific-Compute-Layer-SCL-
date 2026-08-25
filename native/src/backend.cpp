#include "scl/backend.hpp"

#ifdef SCL_WITH_CUDA
#include "scl/backends/cuda/lj_pairwise_cuda.hpp"
#endif

namespace scl {

std::string backend_name(Backend backend) {
    switch (backend) {
        case Backend::Cpu:
            return "cpu";
        case Backend::Cuda:
            return "cuda";
    }
    return "unknown";
}

bool backend_available(Backend backend) {
    if (backend == Backend::Cpu) {
        return true;
    }
#ifdef SCL_WITH_CUDA
    return scl::cuda_device_available();
#else
    return false;
#endif
}

LJResult compute_lj_pairwise(Backend backend, const std::vector<Vec3>& positions,
                              const LJParameters& params) {
    if (backend == Backend::Cpu) {
        return compute_lj_pairwise_cpu(positions, params);
    }
#ifdef SCL_WITH_CUDA
    if (!scl::cuda_device_available()) {
        throw BackendUnavailableError(
            "backend cuda: binary was built with SCL_WITH_CUDA but no CUDA device is visible "
            "at runtime (cudaGetDeviceCount() found none)");
    }
    return scl::compute_lj_pairwise_cuda(positions, params);
#else
    throw BackendUnavailableError(
        "backend cuda: this scl_cli binary was built without SCL_WITH_CUDA "
        "(no nvcc/CUDA toolkit was available at build time) -- rebuild with "
        "-DSCL_WITH_CUDA=ON on a machine with the CUDA toolkit installed");
#endif
}

}  // namespace scl
