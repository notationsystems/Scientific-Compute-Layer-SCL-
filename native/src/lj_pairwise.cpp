#include "scl/lj_pairwise.hpp"

#include <cmath>
#include <sstream>

namespace scl {

std::string validate_lj_input(const std::vector<Vec3>& positions, const LJParameters& params) {
    if (positions.empty()) {
        return "at least one particle is required, got 0";
    }
    if (!(params.epsilon >= 0.0) || !std::isfinite(params.epsilon)) {
        std::ostringstream os;
        os << "epsilon must be finite and >= 0, got " << params.epsilon;
        return os.str();
    }
    if (!(params.sigma > 0.0) || !std::isfinite(params.sigma)) {
        std::ostringstream os;
        os << "sigma must be finite and > 0, got " << params.sigma;
        return os.str();
    }
    if (!(params.cutoff > 0.0) || !std::isfinite(params.cutoff)) {
        std::ostringstream os;
        os << "cutoff must be finite and > 0, got " << params.cutoff;
        return os.str();
    }
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const Vec3& p = positions[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            std::ostringstream os;
            os << "position[" << i << "] is not finite";
            return os.str();
        }
    }
    return "";
}

LJResult compute_lj_pairwise_cpu(const std::vector<Vec3>& positions, const LJParameters& params) {
    LJResult result;
    const std::size_t n = positions.size();
    result.forces.assign(n, Vec3{0.0, 0.0, 0.0});

    const double cutoff2 = params.cutoff * params.cutoff;
    double energy = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double dx = positions[i].x - positions[j].x;
            const double dy = positions[i].y - positions[j].y;
            const double dz = positions[i].z - positions[j].z;
            const double r2 = dx * dx + dy * dy + dz * dz;

            if (r2 == 0.0) {
                result.ok = false;
                result.fault = ComputeFault::CoincidentParticles;
                return result;
            }
            if (r2 > cutoff2) {
                continue;  // beyond cutoff: zero contribution, plain truncation
            }

            const double inv_r2 = 1.0 / r2;
            const double sr2 = (params.sigma * params.sigma) * inv_r2;
            const double sr6 = sr2 * sr2 * sr2;
            const double sr12 = sr6 * sr6;

            const double pair_energy = 4.0 * params.epsilon * (sr12 - sr6);
            const double fscalar = 24.0 * params.epsilon * inv_r2 * (2.0 * sr12 - sr6);

            energy += pair_energy;

            result.forces[i].x += fscalar * dx;
            result.forces[i].y += fscalar * dy;
            result.forces[i].z += fscalar * dz;
            result.forces[j].x -= fscalar * dx;
            result.forces[j].y -= fscalar * dy;
            result.forces[j].z -= fscalar * dz;
        }
    }

    if (!std::isfinite(energy)) {
        result.ok = false;
        result.fault = ComputeFault::NonFinite;
        return result;
    }
    for (const auto& f : result.forces) {
        if (!std::isfinite(f.x) || !std::isfinite(f.y) || !std::isfinite(f.z)) {
            result.ok = false;
            result.fault = ComputeFault::NonFinite;
            return result;
        }
    }

    result.ok = true;
    result.fault = ComputeFault::None;
    result.total_energy = energy;
    return result;
}

}  // namespace scl
