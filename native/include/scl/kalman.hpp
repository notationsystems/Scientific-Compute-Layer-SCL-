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
// rather than the shorter P = (I - K H) P. Algebraically identical in
// exact arithmetic; different in floating point.
//
// THE FIRST VERSION OF THIS COMMENT OVERSTATED THE CASE, and the
// correction is kept visible because the overstatement is the more
// instructive half. It said the short form "quietly loses PSD after a few
// hundred steps". Measured, across six regimes, n = 2 with a scalar
// measurement and P0 = 1e10:
//
//   r      q      steps    JOSEPH lambda_min   SHORT lambda_min
//   0.25   0.01    2000    2.125e-02           2.125e-02   IDENTICAL
//   1e-16  0       5000    2.401e-27           0.000e+00
//   1e-18  0       5000    2.401e-29           0.000e+00
//   1e-20  0      20000    3.750e-33           0.000e+00
//   1e-22  0      50000    2.400e-36           0.000e+00
//   1e-30  0      50000    2.400e-44           0.000e+00
//
// So: in any benign regime the two forms are INDISTINGUISHABLE, and the
// "few hundred steps" claim was wrong. What the short form actually does,
// consistently, is drive the covariance SINGULAR -- lambda_min reaches
// exactly zero -- where Joseph retains a small positive value. No regime
// was found in which it goes NEGATIVE. Loss of rank, not indefiniteness.
//
// That is still worth the extra n x n multiply: a singular P means the
// filter has concluded it knows one direction of the state perfectly, and
// from there the gain in that direction is identically zero and no further
// measurement can correct it. The failure is silent and permanent, which
// is the same argument that elected SVD for least_squares -- applied to
// the recursion rather than the solve -- but it is a smaller claim than
// the one first written here.
//
// THE CHOICE IS ENFORCED, NOT ASSERTED. Replacing Joseph with the short
// form was mutation-tested and SURVIVED every test in this operation,
// because none of them ran a regime where the two differ. That is a
// stated-but-unenforced property, in code written the same hour as the
// rule against them (architecture/proof_integrity.yaml). The regime above
// is now a test.
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

//: Runs the filter. Throws KalmanValidationError on any contract breach --
//: a shape mismatch, a covariance that fails any of the five rules, or a
//: provenance claim that cannot be true.
KalmanResult run_kalman_filter(const KalmanProblem& problem,
                               const KalmanParameters& params);

}  // namespace scl
