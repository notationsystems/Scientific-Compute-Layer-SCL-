// Runs the pre-registered Kalman validation scenarios and PRINTS the
// statistics. It applies no thresholds and reaches no verdict -- that is
// deliberate. The thresholds live in
// architecture/kalman_validation_preregistration.yaml and are applied by
// tests/test_kalman_validation.py, which READS them. A binary that both
// computed a statistic and judged it could have the judgement adjusted in
// the commit that makes it pass.
//
// Usage: kalman_validate <seed> <steps>

#include "scl/kalman.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

namespace {

// A stated, self-contained generator: the seed must reproduce the run
// exactly, and depending on a library RNG whose stream may change between
// implementations would make "seed 20260826" mean different things in
// different places. splitmix64 + Box-Muller, both written out.
struct Rng {
    std::uint64_t state;
    explicit Rng(std::uint64_t seed) : state(seed) {}
    std::uint64_t next_u64() {
        state += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uniform() {   // (0, 1)
        return (static_cast<double>(next_u64() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }
    double normal() {
        const double u1 = uniform(), u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: kalman_validate <seed> <steps>\n");
        return 2;
    }
    const std::uint64_t seed = std::strtoull(argv[1], nullptr, 10);
    const std::size_t steps = std::strtoull(argv[2], nullptr, 10);

    // A 2-state, 2-measurement constant-velocity-ish model. The point is
    // that the SIMULATION uses exactly the F, H, Q, R the filter is given,
    // so the innovation statistics test the filter and not the modelling.
    const std::size_t n = 2, m = 2;
    scl::KalmanProblem problem;
    problem.state_dimension = n;
    problem.measurement_dimension = m;
    problem.transition = {1.0, 0.1,
                          0.0, 1.0};
    problem.observation = {1.0, 0.0,
                           0.0, 1.0};
    problem.process_noise.dimension = n;
    problem.process_noise.matrix = {0.01, 0.0,
                                    0.0,  0.04};
    problem.process_noise.provenance = scl::NoiseProvenance::kAsserted;
    problem.measurement_noise.dimension = m;
    problem.measurement_noise.matrix = {0.25, 0.0,
                                        0.0,  0.16};
    problem.measurement_noise.provenance = scl::NoiseProvenance::kAsserted;
    problem.initial_state = {0.0, 0.0};
    problem.initial_covariance = {1.0, 0.0,
                                  0.0, 1.0};
    problem.steps = steps;

    // Simulate the TRUE process with the same Q and R the filter assumes.
    Rng rng(seed);
    std::vector<double> truth = {0.0, 1.0};
    problem.measurements.resize(steps * m);
    const double qs[2] = {std::sqrt(0.01), std::sqrt(0.04)};
    const double rs[2] = {std::sqrt(0.25), std::sqrt(0.16)};
    for (std::size_t k = 0; k < steps; ++k) {
        const double x0 = truth[0] + 0.1 * truth[1] + qs[0] * rng.normal();
        const double x1 = truth[1] + qs[1] * rng.normal();
        truth[0] = x0;
        truth[1] = x1;
        problem.measurements[k * m + 0] = truth[0] + rs[0] * rng.normal();
        problem.measurements[k * m + 1] = truth[1] + rs[1] * rng.normal();
    }

    scl::KalmanParameters params;
    const scl::KalmanResult r = scl::run_kalman_filter(problem, params);

    // NORMALISED innovations: v_k,i / sqrt(S_k,ii). Everything below is a
    // statistic of these, so the filter's own predicted covariance is what
    // normalises its own error -- which is the whole content of the test.
    std::vector<double> nu(steps * m);
    for (std::size_t k = 0; k < steps; ++k)
        for (std::size_t i = 0; i < m; ++i)
            nu[k * m + i] = r.innovations[k * m + i] /
                            std::sqrt(r.innovation_covariances[k * m * m + i * m + i]);

    // 1. mean, per component
    std::printf("steps %zu\n", steps);
    for (std::size_t i = 0; i < m; ++i) {
        double sum = 0.0;
        for (std::size_t k = 0; k < steps; ++k) sum += nu[k * m + i];
        std::printf("mean %zu %.17g\n", i, sum / static_cast<double>(steps));
    }

    // 2. NIS = (1/N) sum v^T S^-1 v.  S is diagonal in this model, so the
    // inverse is exact and needs no solve -- deliberate, so the statistic
    // does not inherit the inverter's error.
    double nis = 0.0;
    for (std::size_t k = 0; k < steps; ++k)
        for (std::size_t i = 0; i < m; ++i) nis += nu[k * m + i] * nu[k * m + i];
    std::printf("nis %.17g\n", nis / static_cast<double>(steps));

    // 3. whiteness: autocorrelation of the scalar normalised innovations
    for (int lag = 1; lag <= 10; ++lag) {
        double num = 0.0, den = 0.0;
        for (std::size_t k = 0; k < steps * m; ++k) den += nu[k] * nu[k];
        for (std::size_t k = 0; k + static_cast<std::size_t>(lag) * m < steps * m; ++k)
            num += nu[k] * nu[k + static_cast<std::size_t>(lag) * m];
        std::printf("rho %d %.17g\n", lag, num / den);
    }

    std::printf("min_posterior_eigenvalue %.17g\n", r.smallest_posterior_eigenvalue);
    return 0;
}
