#pragma once
// kalman_filter_linear: the discrete linear Kalman filter.
//
// WHAT IS ASSERTED AND WHAT IS MEASURED. Every matrix here is supplied by
// the caller; none is estimated. F, H, Q and R are the MODEL, and the
// filter is only correct relative to a model it is given. That is why the
// validation criteria are about the INNOVATION SEQUENCE rather than about
// agreement with a truth: comparing a filtered estimate to simulated
// ground truth conflates the filter, the model and the simulator, and
// passes when two errors cancel. See
// architecture/kalman_validation_preregistration.yaml, whose thresholds
// were fixed before this file existed.
//
// COVARIANCE RULES ARE NOT INHERITED. DAQ's aligned-observation gate
// formally declined to supply numeric-entry and shape rules for matrix
// cells -- both admissions are pinned as tests on that side. So P0, Q and
// R each pass validate_covariance (scl/covariance.hpp) before a single
// step runs. A covariance that is ragged, asymmetric or indefinite is a
// VALIDATION fault, not a computation that produces a strange answer.
//
// JOSEPH FORM for the covariance update:
//
//     P = (I - K H) P (I - K H)^T + K R K^T
//
// rather than the shorter P = (I - K H) P.
//
// THE JUSTIFICATION FIRST WRITTEN HERE WAS WRONG, AND THE CORRECTION IS
// KEPT VISIBLE because the shape of the error matters more than the fix.
// It claimed the short form "quietly loses PSD after a few hundred
// steps". Replacing Joseph with the short form was then mutation-tested
// and SURVIVED every test in this operation -- a property chosen for a
// stated reason with nothing testing it.
//
// MEASURED, four variants x seven regimes, verdicts from this project's
// own covariance gate rather than from a threshold invented for the
// occasion (n = 3, diffuse P0):
//
//   optimal K, r=0.25 q=0.01, 2000 steps
//       joseph  lambda_min  1.540e-02   accepted
//       short   lambda_min  1.540e-02   accepted     IDENTICAL
//
//   K perturbed by 1e-3, same regime
//       joseph  lambda_min  1.540e-02   accepted
//       short   lambda_min -1.994e+06   NOT-PSD
//
//   K perturbed by 1e-1                 short: -1.282e+08   NOT-PSD
//   K perturbed by 0.5                  short: -9.549e+08   NOT-PSD
//
// So the real guarantee is NOT about step count and NOT about roundoff.
// It is that Joseph holds for ANY GAIN, optimal or not, while
// (I - K H) P is only a valid covariance when K is exactly optimal. With
// the optimal K in float64 the two are indistinguishable, which is
// precisely why the mutant survived: nothing in the suite ran a regime
// where the choice can matter.
//
// AND WITHIN THIS OPERATION, AS WRITTEN, IT CANNOT. K is computed from an
// exact inverse of S, so the gain is optimal to roundoff -- a perturbation
// of 1e-16, twelve orders below where the forms diverge. Joseph is
// therefore a ROBUSTNESS MARGIN here, not a fix for a live defect, and it
// is kept deliberately on that basis: a fixed or steady-state gain, a
// replayed gain, a truncated inverse, or a reduced-precision backend all
// produce a suboptimal K, and each is a change to how K is made rather
// than to the update -- so the update should not be the thing that breaks.
// Stated as a margin rather than as a defect avoided, because claiming
// more than that is what the first version of this comment did.
//
// SYMMETRISATION IS LOAD-BEARING, NOT BELT-AND-BRACES. Measured in the
// same sweep: Joseph WITHOUT the explicit symmetrise step reaches an
// asymmetry of 7.45e-09 and is REFUSED by this project's own covariance
// gate as NOT-SYMMETRIC. Joseph is symmetry-preserving in exact
// arithmetic and not quite in floating point. The two mechanisms are
// independent and both are required.
//
// THE THIRD IDENTITY AXIS (docs/SCL_CONTRACT.md 6.3). Q and R both
// participate, and they are NOT the same kind of asserted: R may be
// measurement-derived, Q never can be. Each carries a provenance
// discriminant, `supplied` on Q is a validation fault, and because the
// discriminant guards a payload, operation-contract clause 2 governs it --
// a source identity supplied under `asserted` is refused, never accepted
// and ignored.

#include <cstddef>
#include <string>
#include <vector>

namespace scl {

//: Whether a measurement stands behind a noise matrix. Participating:
//: `asserted` and `supplied` describe different computations at identical
//: numeric values, because they license different conclusions.
enum class NoiseProvenance {
    kAsserted = 0,   //: a modelling choice; no measurement behind it
    kSupplied = 1,   //: derived from measurement, with a source identity
};

struct NoiseModel {
    std::vector<double> matrix;      //: row-major, dim * dim
    std::size_t dimension = 0;
    NoiseProvenance provenance = NoiseProvenance::kAsserted;
    //: GUARDED PAYLOAD. Non-empty only when provenance is kSupplied;
    //: a value here under kAsserted is REFUSED (clause 2).
    std::string source_identity;
};

struct KalmanProblem {
    std::size_t state_dimension = 0;        //: n
    std::size_t measurement_dimension = 0;  //: m

    std::vector<double> transition;         //: F, n x n
    std::vector<double> observation;        //: H, m x n
    NoiseModel process_noise;               //: Q, n x n -- kAsserted always
    NoiseModel measurement_noise;           //: R, m x m

    std::vector<double> initial_state;          //: x0, n
    std::vector<double> initial_covariance;     //: P0, n x n

    //: Row-major, steps * m. Ordering is REQUIRED AND SIGNIFICANT for this
    //: modality -- unlike least_squares, where it is explicitly not.
    std::vector<double> measurements;
    std::size_t steps = 0;
};

struct KalmanParameters {
    //: Applied to P0, Q and R alike. Participating: they change which
    //: inputs are accepted, so they change the answer.
    double symmetry_tolerance = 1e-10;
    double psd_tolerance = 1e-10;
};

struct KalmanResult {
    std::vector<double> states;         //: steps * n, posterior x_k|k
    std::vector<double> covariances;    //: steps * n * n, posterior P_k|k
    std::vector<double> innovations;    //: steps * m, v_k = z_k - H x_k|k-1
    //: S_k = H P_k|k-1 H^T + R, the covariance the filter PREDICTS for its
    //: own innovation. Retained because every validation statistic
    //: normalises by it; a filter that reports innovations without S has
    //: reported a sequence nobody can test.
    std::vector<double> innovation_covariances;   //: steps * m * m
    std::vector<double> gains;          //: steps * n * m
    std::size_t steps = 0;
    double smallest_posterior_eigenvalue = 0.0;   //: min over all steps
};

class KalmanValidationError : public std::exception {
public:
    explicit KalmanValidationError(std::string what) : what_(std::move(what)) {}
    const char* what() const noexcept override { return what_.c_str(); }
private:
    std::string what_;
};

//: The covariance update, as its own function so its CONTRACT can be
//: stated and tested over any gain:
//:
//:     FOR ANY K, the result is a valid covariance whenever P_pred is.
//:
//: That is the property Joseph actually provides, and it is what this
//: returns. It is deliberately NOT a test that "Joseph is used" -- pinning
//: an implementation is not testing a property, and a later equivalent
//: form should be free to replace it while the contract holds.
//:
//: Exposed rather than inlined because the operation itself only ever
//: supplies the OPTIMAL K, under which every update form agrees. Testing
//: the guarantee therefore requires a suboptimal gain, and the honest way
//: to supply one is to call the real function with it -- not to add a
//: perturbation knob to the operation so a test can reach it.
//:
//: Includes the explicit symmetrisation, which the same measurement showed
//: is load-bearing: Joseph alone leaves asymmetry around 7e-09, which this
//: project's own covariance gate refuses.
std::vector<double> covariance_update(const std::vector<double>& p_predicted,
                                      const std::vector<double>& gain,
                                      const std::vector<double>& observation,
                                      const std::vector<double>& measurement_noise,
                                      std::size_t state_dimension,
                                      std::size_t measurement_dimension);

//: Runs the filter. Throws KalmanValidationError on any contract breach --
//: a shape mismatch, a covariance that fails any of the five rules, or a
//: provenance claim that cannot be true.
KalmanResult run_kalman_filter(const KalmanProblem& problem,
                               const KalmanParameters& params);

}  // namespace scl
