"""The precondition the whitening prescription did not state.

WHAT THIS COMPOSES. native/include/scl/least_squares.hpp records two
boundaries side by side and calls them the same family:

  * rank and conditioning are properties of the design matrix's COLUMNS,
    so an exact dependence among the DATA ROWS is invisible to every
    diagnostic the operation reports;
  * a caller holding a covariance whitens with its Cholesky factor, and
    the result IS the generalized least-squares estimate.

They were never put together. Put together, the second one has a
precondition the first one violates.

THE WORD DOING THE WORK IS `KNOWN`. The whitening identity holds for a
known Sigma = L L^T. A caller who MEASURES Sigma from replicate runs --
now the expected route, since the acquisition layer's
science/replicate_pairing.py recovers one and the core's
materials/replicate_join.py joins on it -- holds a SAMPLE covariance. If
the variables carry an exact linear relation, that sample covariance is
positive SEMI-definite by construction, and a Cholesky of it has no
defined result.

THE CASE IS THE ONE THE FIRST BOUNDARY ALREADY USES. A GPC instrument
reports Mn, Mw and PDI = Mw/Mn together. In logs, log PDI is identically
log Mw - log Mn, so the third row of the covariance is the second minus
the first. The row dependence the conditioning cannot see, arriving on
the other side as a singular Sigma.

WHAT THIS FILE ASSERTS, AND WHAT IT DELIBERATELY DOES NOT. It does not
assert that either outcome is correct, or that the operation should
change. It asserts that THE OUTCOME IS NOT DETERMINED BY THE SITUATION --
the same physical case both refuses and silently succeeds, decided by
where the last pivot lands relative to zero -- and that applying the
operation's own rank_tolerance to Sigma makes it deterministic.

MEASURED 2026-09-03 over 2000 five-run sets: 828 accepted, 1172 refused,
and every accepted case had a pivot ratio below the 1e-12 default. The
exact counts are recorded in the header as a dated measurement; what is
ASSERTED here is the property, because a count pinned to one interpreter
would go stale on a float or RNG change while the property would not.
"""

from __future__ import annotations

import math
import random

#: Five is what a replicate set of GPC runs actually looks like, and it
#: is above science/replicate_pairing.py's floor for a covariance. The
#: singularity here is structural, not a small-sample artefact: it holds
#: for any number of runs, because it comes from the relation between
#: the variables rather than from how many times they were measured.
RUNS_PER_SET = 5
SETS = 2000

#: The operation's own default, from LeastSquaresParameters.
RANK_TOLERANCE = 1e-12


def _replicate_covariance(seed, runs=RUNS_PER_SET, exact_relation=True):
    """One replicate set's sample covariance of (log Mn, log Mw, log PDI).

    `exact_relation=False` breaks the relation by measuring PDI
    independently rather than deriving it -- the control, so that a
    covariance that is singular for ANY reason cannot be mistaken for
    one that is singular for THIS reason."""
    random.seed(seed)
    rows = []
    for _ in range(runs):
        mn = random.gauss(3300.0, 150.0)
        mw = mn * random.gauss(2.45, 0.10)
        pdi = (mw / mn) if exact_relation else random.gauss(2.45, 0.10)
        rows.append((math.log(mn), math.log(mw), math.log(pdi)))
    width = len(rows[0])
    mean = [sum(row[j] for row in rows) / runs for j in range(width)]
    return [
        [
            sum((row[i] - mean[i]) * (row[j] - mean[j]) for row in rows) / (runs - 1)
            for j in range(width)
        ]
        for i in range(width)
    ]


def _plain_cholesky(matrix):
    """The route the header prescribes, written the way anyone would
    write it: refuse on a non-positive pivot, otherwise proceed."""
    size = len(matrix)
    factor = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            carried = sum(factor[i][k] * factor[j][k] for k in range(j))
            if i == j:
                pivot = matrix[i][i] - carried
                if pivot <= 0.0:
                    return None
                factor[i][j] = math.sqrt(pivot)
            else:
                factor[i][j] = (matrix[i][j] - carried) / factor[j][j]
    return factor


def _pivots(matrix):
    """Every pivot, without refusing -- so the deficient direction can be
    measured rather than merely tripped over."""
    size = len(matrix)
    factor = [[0.0] * size for _ in range(size)]
    pivots = []
    for i in range(size):
        for j in range(i + 1):
            carried = sum(factor[i][k] * factor[j][k] for k in range(j))
            if i == j:
                pivot = matrix[i][i] - carried
                pivots.append(pivot)
                factor[i][j] = math.sqrt(pivot) if pivot > 0.0 else 0.0
            else:
                factor[i][j] = (
                    (matrix[i][j] - carried) / factor[j][j] if factor[j][j] > 0.0 else 0.0
                )
    return pivots


def _effective_rank(matrix, rank_tolerance=RANK_TOLERANCE):
    """The operation's own cutoff, applied to the covariance instead of
    the design. `sigma_j <= rank_tolerance * sigma_max` becomes
    `pivot <= rank_tolerance * largest_pivot`, which is the same rule
    about the same kind of object."""
    pivots = _pivots(matrix)
    largest = max(pivots)
    return sum(1 for pivot in pivots if pivot > rank_tolerance * largest)


def test_the_structural_relation_makes_the_covariance_exactly_singular():
    """The premise, checked before anything is concluded from it. If the
    third row were not the second minus the first, everything below
    would be measuring floating-point noise about nothing."""
    covariance = _replicate_covariance(20260903)
    for column in range(3):
        residual = covariance[2][column] - (
            covariance[1][column] - covariance[0][column]
        )
        scale = max(abs(covariance[i][column]) for i in range(3))
        assert abs(residual) < 1e-12 * scale, (
            f"column {column}: the exact relation does not hold, residual {residual:.3e}"
        )


def test_the_control_is_not_singular_when_the_relation_is_broken():
    """PLANT THE ABSENCE OF THE DEFECT. Measuring PDI independently
    instead of deriving it removes the relation and nothing else. If this
    were also singular, the finding would be about small samples rather
    than about the relation, and every conclusion below would be
    misattributed."""
    for seed in range(50):
        covariance = _replicate_covariance(seed, exact_relation=False)
        assert _effective_rank(covariance) == 3, (
            f"seed {seed}: an independently measured PDI still gives a "
            "rank-deficient covariance, so the relation is not the cause"
        )
        assert _plain_cholesky(covariance) is not None


def test_the_prescribed_route_is_not_determined_by_the_situation():
    """THE FINDING. Both outcomes occur across sets of the SAME physical
    case, so a caller following the header cannot know which they will
    get -- and one of them is silent.

    The property is asserted rather than the count. Pinning 828 and 1172
    would bind this to one interpreter's float and RNG behaviour, and a
    change in either would fail the test while the property it exists for
    was still true."""
    accepted = refused = 0
    for seed in range(SETS):
        covariance = _replicate_covariance(seed)
        if _plain_cholesky(covariance) is None:
            refused += 1
        else:
            accepted += 1
    assert accepted + refused == SETS
    assert accepted > SETS // 10, (
        f"only {accepted} of {SETS} succeeded; the silent branch may have "
        "stopped being reachable, which would change the finding"
    )
    assert refused > SETS // 10, (
        f"only {refused} of {SETS} refused; the loud branch may have "
        "stopped being reachable, which would change the finding"
    )


def test_every_silent_success_is_below_the_operations_own_cutoff():
    """The sharper half. The cases that succeed are not marginal calls a
    caller might reasonably accept -- every one of them is deficient by
    the operation's OWN standard, applied to the wrong matrix. The
    threshold that would have caught them already exists."""
    ratios = []
    for seed in range(SETS):
        covariance = _replicate_covariance(seed)
        if _plain_cholesky(covariance) is None:
            continue
        pivots = _pivots(covariance)
        ratios.append(min(pivots) / max(pivots))
    assert ratios, "no case succeeded; there is nothing to characterise"
    assert max(ratios) < RANK_TOLERANCE, (
        f"a case succeeded with pivot ratio {max(ratios):.3e}, at or above "
        f"the operation's own rank_tolerance of {RANK_TOLERANCE:.0e} -- the "
        "claim that every silent success is deficient by that standard no "
        "longer holds"
    )


def test_the_operations_own_tolerance_makes_the_answer_deterministic():
    """The repair, measured. Applying `rank_tolerance` to Sigma gives one
    answer for every set: effective rank 2, the deficient direction named
    rather than survived."""
    observed = {_effective_rank(_replicate_covariance(seed)) for seed in range(SETS)}
    assert observed == {2}, (
        f"the tolerant rank is not deterministic across {SETS} sets: {sorted(observed)}"
    )


def test_the_deficiency_is_structural_rather_than_a_small_sample_effect():
    """More runs do not fix it, which is what distinguishes a relation
    between the variables from a shortage of observations. A caller who
    responds to a singular Sigma by collecting more replicates is
    treating the wrong cause."""
    for runs in (5, 10, 40):
        observed = {
            _effective_rank(_replicate_covariance(seed, runs=runs))
            for seed in range(100)
        }
        assert observed == {2}, (
            f"at {runs} runs the tolerant rank is {sorted(observed)}; the "
            "deficiency is behaving like a sample-size effect"
        )
