// Native unit tests for the SCL LJ pairwise kernel (Task 9.1 / Task 5 of
// the SCL Phase 1 brief). Hand-rolled harness -- no gtest dependency, so
// the native build stays hermetic (only nlohmann_json is required, and
// only by the CLI, not by this test binary). Run via `ctest` or directly.

#include "scl/backend.hpp"
#include "scl/lj_pairwise.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", expr, file, line);
    }
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

// --- Test 1: two-particle analytic energy -----------------------------
// V(r) = 4*eps*((sigma/r)^12 - (sigma/r)^6) has a closed form for N=2;
// this checks the kernel against that closed form independently of the
// kernel's own code path.
void test_two_particle_analytic_energy() {
    scl::LJParameters params{/*epsilon=*/1.0, /*sigma=*/1.0, /*cutoff=*/5.0};
    double r = 1.5;
    std::vector<scl::Vec3> positions = {{0, 0, 0}, {r, 0, 0}};

    scl::LJResult result = scl::compute_lj_pairwise_cpu(positions, params);
    CHECK(result.ok);

    double sr6 = std::pow(params.sigma / r, 6);
    double sr12 = sr6 * sr6;
    double expected_energy = 4.0 * params.epsilon * (sr12 - sr6);
    CHECK(close(result.total_energy, expected_energy, 1e-12));
}

// --- Test 2: Newton's third law (sum of forces is zero) ---------------
// A pairwise central potential is translationally invariant: total force
// on the system is zero for ANY configuration, not just symmetric ones.
void test_newton_third_law_five_particles() {
    scl::LJParameters params{1.0, 1.0, 4.0};
    std::vector<scl::Vec3> positions = {
        {0.0, 0.0, 0.0}, {1.3, 0.2, -0.4}, {-0.7, 1.1, 0.3},
        {2.1, -0.9, 0.6}, {0.4, 0.4, 1.8},
    };
    scl::LJResult result = scl::compute_lj_pairwise_cpu(positions, params);
    CHECK(result.ok);

    scl::Vec3 total{0, 0, 0};
    for (const auto& f : result.forces) {
        total.x += f.x;
        total.y += f.y;
        total.z += f.z;
    }
    CHECK(close(total.x, 0.0, 1e-9));
    CHECK(close(total.y, 0.0, 1e-9));
    CHECK(close(total.z, 0.0, 1e-9));
}

// --- Test 3: cutoff truncation -----------------------------------------
void test_cutoff_zeroes_distant_pairs() {
    scl::LJParameters params{1.0, 1.0, 2.0};
    std::vector<scl::Vec3> positions = {{0, 0, 0}, {10.0, 0, 0}};  // far beyond cutoff
    scl::LJResult result = scl::compute_lj_pairwise_cpu(positions, params);
    CHECK(result.ok);
    CHECK(result.total_energy == 0.0);
    CHECK(result.forces[0].x == 0.0 && result.forces[0].y == 0.0 && result.forces[0].z == 0.0);
    CHECK(result.forces[1].x == 0.0 && result.forces[1].y == 0.0 && result.forces[1].z == 0.0);
}

// --- Test 4: coincident particles fault, not a crash or a fabricated value
void test_coincident_particles_faults() {
    scl::LJParameters params{1.0, 1.0, 5.0};
    std::vector<scl::Vec3> positions = {{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}};
    scl::LJResult result = scl::compute_lj_pairwise_cpu(positions, params);
    CHECK(!result.ok);
    CHECK(result.fault == scl::ComputeFault::CoincidentParticles);
}

// --- Test 5: force = -dV/dr, checked by central finite difference -----
// Cross-validates the analytically-derived force formula against the
// energy function alone, independent of test 1's closed form: perturb
// one particle along x by +-delta and use (E+ - E-) / (2*delta) as a
// numerical estimate of dE/dx_i, which must equal -forces[i].x.
void test_force_matches_finite_difference_gradient() {
    scl::LJParameters params{0.8, 1.1, 6.0};
    std::vector<scl::Vec3> base = {
        {0.0, 0.0, 0.0}, {1.6, 0.3, -0.2}, {-1.1, 0.9, 0.5},
    };
    scl::LJResult result = scl::compute_lj_pairwise_cpu(base, params);
    CHECK(result.ok);

    const double delta = 1e-6;
    for (std::size_t i = 0; i < base.size(); ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<scl::Vec3> plus = base;
            std::vector<scl::Vec3> minus = base;
            double* p_plus = (axis == 0) ? &plus[i].x : (axis == 1) ? &plus[i].y : &plus[i].z;
            double* p_minus = (axis == 0) ? &minus[i].x : (axis == 1) ? &minus[i].y : &minus[i].z;
            *p_plus += delta;
            *p_minus -= delta;

            scl::LJResult r_plus = scl::compute_lj_pairwise_cpu(plus, params);
            scl::LJResult r_minus = scl::compute_lj_pairwise_cpu(minus, params);
            CHECK(r_plus.ok && r_minus.ok);

            double numeric_dE = (r_plus.total_energy - r_minus.total_energy) / (2.0 * delta);
            double analytic_force_component =
                (axis == 0) ? result.forces[i].x : (axis == 1) ? result.forces[i].y : result.forces[i].z;
            // F = -dE/dx, so analytic_force_component should equal -numeric_dE.
            CHECK(close(analytic_force_component, -numeric_dE, 1e-4));
        }
    }
}

// --- Test 6: parameter validation --------------------------------------
void test_validation_rejects_bad_parameters() {
    std::vector<scl::Vec3> positions = {{0, 0, 0}, {1, 0, 0}};
    CHECK(!scl::validate_lj_input({}, scl::LJParameters{1, 1, 1}).empty());  // no particles
    CHECK(!scl::validate_lj_input(positions, scl::LJParameters{1, -1, 1}).empty());  // sigma<=0
    CHECK(!scl::validate_lj_input(positions, scl::LJParameters{-1, 1, 1}).empty());  // epsilon<0
    CHECK(!scl::validate_lj_input(positions, scl::LJParameters{1, 1, 0}).empty());  // cutoff<=0
    CHECK(scl::validate_lj_input(positions, scl::LJParameters{1, 1, 1}).empty());  // valid
}

// --- Test 7: backend dispatch and availability -------------------------
void test_cuda_backend_unavailable_in_cpu_only_build() {
#ifndef SCL_WITH_CUDA
    CHECK(!scl::backend_available(scl::Backend::Cuda));
    std::vector<scl::Vec3> positions = {{0, 0, 0}, {1, 0, 0}};
    scl::LJParameters params{1, 1, 5};
    bool threw = false;
    try {
        scl::compute_lj_pairwise(scl::Backend::Cuda, positions, params);
    } catch (const scl::BackendUnavailableError&) {
        threw = true;
    }
    CHECK(threw);
#endif
    CHECK(scl::backend_available(scl::Backend::Cpu));
}

// --- Test 8: reproducibility (same input, same binary, same machine) --
void test_bitwise_reproducible_same_process() {
    scl::LJParameters params{1.0, 1.0, 5.0};
    std::vector<scl::Vec3> positions = {{0.1, 0.2, 0.3}, {1.4, -0.5, 0.2}, {-0.6, 0.9, -1.1}};
    scl::LJResult a = scl::compute_lj_pairwise_cpu(positions, params);
    scl::LJResult b = scl::compute_lj_pairwise_cpu(positions, params);
    CHECK(a.ok && b.ok);
    CHECK(a.total_energy == b.total_energy);  // bit-identical, not just close
    for (std::size_t i = 0; i < a.forces.size(); ++i) {
        CHECK(a.forces[i].x == b.forces[i].x);
        CHECK(a.forces[i].y == b.forces[i].y);
        CHECK(a.forces[i].z == b.forces[i].z);
    }
}

}  // namespace

int main() {
    test_two_particle_analytic_energy();
    test_newton_third_law_five_particles();
    test_cutoff_zeroes_distant_pairs();
    test_coincident_particles_faults();
    test_force_matches_finite_difference_gradient();
    test_validation_rejects_bad_parameters();
    test_cuda_backend_unavailable_in_cpu_only_build();
    test_bitwise_reproducible_same_process();

    std::printf("%d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
