#!/usr/bin/env python3
"""Mutation-checks kalman_filter_linear.

WHY THIS EXISTS. The Joseph-form mutant survived the whole suite on its
first run -- not because an assertion was missing, but because no test
operated in the regime where the choice can matter. Measuring that showed
the STATED REASON for choosing Joseph was also false. A suite that passes
is not evidence that its subject is right; this is what turns "we tested
it" into a list of specific defects each caught by a specific test.

A MUTATION SET IS A LOWER BOUND, NEVER A PROOF OF COMPLETENESS. It is an
enumeration -- the exact shape architecture/proof_integrity.yaml names as
coverage_specified_by_enumeration -- and it is correct only until someone
writes a defect nobody listed. That limitation cannot be engineered away
here, because there is no derivable set of "all wrong filters". So it is
stated rather than hidden, and the report says LOWER BOUND rather than
COVERED.

Not a pytest module: it edits tracked source and rebuilds. Run as
`python3 tests/mutation_check_kalman.py`. Always restores the tree.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
BUILD = REPO / "native" / "build"
BINARY = BUILD / "tests" / "scl_kalman_tests"
TEST_SOURCE = REPO / "native" / "tests" / "test_kalman_analytic.cpp"

#: (label, file, old, new, the test that MUST catch it)
MUTATIONS = [
    # ---- the recursion itself
    ("innovation sign: z + Hx instead of z - Hx", "native/src/kalman.cpp",
     "v[i] = problem.measurements[k * m + i] - Hx[i];",
     "v[i] = problem.measurements[k * m + i] + Hx[i];",
     # NOT the steady-state test, which the harness proved cannot see it:
     # that case drives all-zero measurements from a zero initial state, so
     # v = z - Hx and v = z + Hx are both zero. The expectation was wrong,
     # not the test.
     "test_zero_process_noise_estimate_is_the_sample_mean"),
    ("process noise dropped from predict", "native/src/kalman.cpp",
     "for (std::size_t i = 0; i < n * n; ++i) P_pred[i] += Q[i];",
     "for (std::size_t i = 0; i < n * n; ++i) P_pred[i] += 0.0 * Q[i];",
     "test_steady_state_gain_matches_closed_form"),
    ("measurement noise halved in S", "native/src/kalman.cpp",
     "for (std::size_t i = 0; i < m * m; ++i) S[i] += R[i];",
     "for (std::size_t i = 0; i < m * m; ++i) S[i] += 0.5 * R[i];",
     "test_steady_state_gain_matches_closed_form"),
    ("short-form covariance update, not Joseph", "native/src/kalman.cpp",
     """    std::vector<double> P =
        matmul(matmul(IKH, n, n, p_predicted, n), n, n, transpose(IKH, n, n), n);
    const std::vector<double> KRKt =
        matmul(matmul(gain, n, m, measurement_noise, m), n, m, transpose(gain, n, m), n);
    for (std::size_t i = 0; i < n * n; ++i) P[i] += KRKt[i];""",
     "    std::vector<double> P = matmul(IKH, n, n, p_predicted, n);",
     "test_the_covariance_update_holds_for_any_gain"),
    ("symmetrisation removed", "native/src/kalman.cpp",
     """    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double mean = 0.5 * (P[i * n + j] + P[j * n + i]);
            P[i * n + j] = mean;
            P[j * n + i] = mean;
        }
    }
    return P;""",
     "    return P;",
     "test_the_contract_also_holds_at_the_optimal_gain_over_many_steps"),
    ("prediction uses F P F, not F P F^T", "native/src/kalman.cpp",
     "std::vector<double> P_pred = matmul(matmul(F, n, n, P, n), n, n, Ft, n);",
     "std::vector<double> P_pred = matmul(matmul(F, n, n, P, n), n, n, F, n);",
     "test_a_non_symmetric_transition_is_propagated_with_the_transpose"),
    # THIS ENTRY WAS ITSELF A DEFECTIVE MUTATION and is kept as a corrected
    # one rather than deleted. The first version declared an unused K_mut
    # variable: a real statement, so `reaches_executable_semantics` passed
    # it, but dead code, so it changed nothing and reported SURVIVED. The
    # malformed check is TEXTUAL -- it catches comment-only diffs, not
    # dead-code ones -- and that limit is now on the record.
    ("gain scaled by 1.5", "native/src/kalman.cpp",
     "const std::vector<double> K = matmul(matmul(P_pred, n, n, Ht, m), n, m, S_inv, m);",
     "std::vector<double> K = matmul(matmul(P_pred, n, n, Ht, m), n, m, S_inv, m);\n"
     "        for (double& g : K) g *= 1.5;",
     "test_steady_state_gain_matches_closed_form"),
    ("state update drops the prediction: x = K v", "native/src/kalman.cpp",
     "for (std::size_t i = 0; i < n; ++i) x[i] = x_pred[i] + Kv[i];",
     "for (std::size_t i = 0; i < n; ++i) x[i] = Kv[i];",
     "test_zero_process_noise_estimate_is_the_sample_mean"),
    ("state update ignores the measurement: x = x_pred", "native/src/kalman.cpp",
     "for (std::size_t i = 0; i < n; ++i) x[i] = x_pred[i] + Kv[i];",
     "for (std::size_t i = 0; i < n; ++i) x[i] = x_pred[i];",
     "test_zero_process_noise_estimate_is_the_sample_mean"),
    # ---- the contract gates
    ("empty stream accepted as an empty success", "native/src/kalman.cpp",
     'require(problem.steps > 0,',
     'require(true || problem.steps > 0,',
     "test_the_contract_faults"),
    ("non-finite measurements tolerated", "native/src/kalman.cpp",
     '        require(std::isfinite(problem.measurements[i]),',
     '        require(true || std::isfinite(problem.measurements[i]),',
     "test_the_contract_faults"),
    ("R accepted without validating it as a covariance", "native/src/kalman.cpp",
     'require_valid_covariance(problem.measurement_noise.matrix, m, "measurement noise R", cp);',
     '// MUTANT: R unvalidated',
     "test_the_contract_faults"),
    ("Q may be declared `supplied`", "native/src/kalman.cpp",
     "    require_provenance_is_coherent(problem.process_noise, \"process noise Q\",\n"
     "                                   /*may_be_supplied=*/false);",
     "    require_provenance_is_coherent(problem.process_noise, \"process noise Q\",\n"
     "                                   /*may_be_supplied=*/true);",
     "test_the_contract_faults"),
    ("an ignored source_identity is tolerated (clause 2)", "native/src/kalman.cpp",
     "        require(noise.source_identity.empty(),",
     "        require(true || noise.source_identity.empty(),",
     "test_the_contract_faults"),
]


def _strip_cxx_comments(text: str) -> str:
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            quote, start = c, i
            i += 1
            while i < n and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
            i += 1
            out.append(text[start:i])
        elif text.startswith("//", i):
            i = text.find("\n", i)
            i = n if i == -1 else i
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def reaches_executable_semantics(before: str, after: str) -> bool:
    """Same rule as the operation-contract harness: a diff that cannot
    reach execution is MALFORMED, not surviving."""
    return (" ".join(_strip_cxx_comments(before).split())
            != " ".join(_strip_cxx_comments(after).split()))


def _function_at(line: int, source_lines) -> str:
    """The test function containing a given 1-indexed line, so a mutant is
    attributed to the test that caught it rather than to the binary."""
    name = "?"
    for index in range(line - 1, -1, -1):
        match = re.match(r"^(?:void|static void)\s+(test_[a-z_0-9]+)\s*\(", source_lines[index])
        if match:
            return match.group(1)
    return name


def failing_tests() -> set:
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    if result.returncode == 0:
        return set()
    source_lines = TEST_SOURCE.read_text().splitlines()
    names = set()
    for line in result.stderr.splitlines():
        match = re.search(r"test_kalman_analytic\.cpp:(\d+)", line)
        if match:
            names.add(_function_at(int(match.group(1)), source_lines))
    return names


def rebuild() -> bool:
    return subprocess.run(["cmake", "--build", str(BUILD), "-j4"],
                          capture_output=True).returncode == 0


def main() -> int:
    backup = pathlib.Path(tempfile.mkdtemp(prefix="scl-kalman-mutation-"))
    touched = {path for _, path, _, _, _ in MUTATIONS}
    for relative in touched:
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, target)

    verdicts = []
    try:
        assert rebuild(), "the tree does not build before any mutation"
        assert not failing_tests(), "the suite is already failing before any mutation"
        print("baseline: builds clean, suite green\n")

        for label, relative, old, new, expected in MUTATIONS:
            path = REPO / relative
            source = path.read_text()
            if old not in source:
                verdicts.append((label, "PATCH-STALE", expected))
                print(f"  PATCH STALE   {label}")
                continue
            mutated = source.replace(old, new, 1)
            if not reaches_executable_semantics(source, mutated):
                verdicts.append((label, "MALFORMED", expected))
                print(f"  MALFORMED     {label} -- the diff does not reach executable "
                      "semantics, so nothing was tested")
                continue
            path.write_text(mutated)

            if not rebuild():
                verdicts.append((label, "NO-COMPILE", expected))
                print(f"  NO-COMPILE    {label} -- inconclusive")
            else:
                failed = failing_tests()
                if expected in failed:
                    verdicts.append((label, "CAUGHT", expected))
                    print(f"  CAUGHT        {label}\n                by {expected}")
                elif failed:
                    verdicts.append((label, "WRONG-TEST", expected))
                    print(f"  WRONG TEST    {label}\n                caught by "
                          f"{sorted(failed)}, expected {expected}")
                else:
                    verdicts.append((label, "SURVIVED", expected))
                    print(f"  SURVIVED      {label}")
            shutil.copy(backup / relative, path)
            path.touch()
    finally:
        for relative in touched:
            shutil.copy(backup / relative, REPO / relative)
            (REPO / relative).touch()
        rebuild()
        shutil.rmtree(backup, ignore_errors=True)

    caught = [v for v in verdicts if v[1] == "CAUGHT"]
    print("\n" + "=" * 70)
    print(f"{len(caught)}/{len(MUTATIONS)} mutations caught by the claiming test")
    for label, verdict, expected in verdicts:
        if verdict != "CAUGHT":
            print(f"  {verdict}: {label}")
    print("\nThis is a LOWER BOUND on coverage, not a completeness proof: the set is "
          "\nan enumeration, and no derivable set of 'all wrong filters' exists to "
          "\nreplace it with. See architecture/proof_integrity.yaml, "
          "coverage_specified_by_enumeration.")
    print("\nrestored tree still green:", not failing_tests())
    return 0 if len(caught) == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
