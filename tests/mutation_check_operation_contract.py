#!/usr/bin/env python3
"""Mutation-checks every clause of the operation contract that
tests/test_operation_registry_contract.py claims to enforce.

WHY THIS EXISTS. That conformance suite passed on its first run, which is
the suspicious case: a test that has never failed has not been shown to
be capable of failing. Each clause below is deliberately BROKEN in the
real source, the binary is rebuilt, and the suite is required to catch
it -- by the specific test that claims the clause, not merely by
something going red somewhere.

Not a pytest module. It edits tracked source and rebuilds, so it is run
deliberately (`python3 tests/mutation_check_operation_contract.py`) and
always restores the tree, including after a failure or Ctrl-C.

Clauses 2 and 3 now HAVE dedicated tests -- they did not when this script
was written, and it reported that gap rather than hiding it. Writing them
found a live defect (see the clause-2 entry below), which is why the gap
was worth reporting instead of tolerating.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
BUILD = REPO / "native" / "build"
SUITE = "tests/test_operation_registry_contract.py"

#: (clause, description, file, old, new, test that MUST fail)
MUTATIONS = [
    (
        1, "name the algorithm rather than the mathematical operation",
        "native/src/operation_registry.cpp",
        '{"fourier_transform_1d", &run_fourier_transform_1d},',
        '{"fft", &run_fourier_transform_1d},',
        "test_clause1_operation_names_are_snake_case_and_not_algorithm_names",
    ),
    (
        2, "a conditionally-unused field accepted and ignored rather than refused",
        "native/src/op_fourier.cpp",
        "    if (!params.has_sample_spacing && params.sample_spacing_seconds != 0.0) {",
        "    if (false) {",
        "test_clause2_an_ignored_configuration_field_is_refused_not_tolerated",
    ),
    (
        3, "an empty input accepted as a silently-empty success",
        "native/src/fourier.cpp",
        "    if (signal.empty()) {",
        "    if (false) {  // MUTANT: empty input becomes an empty success",
        "test_clause3_empty_input_is_a_validation_fault_not_an_empty_success",
    ),
    (
        4, "a validation fault that names nothing the caller can act on",
        "native/src/op_fourier.cpp",
        'os << "configuration must be exactly 24 bytes (int32 direction, int32 normalization, "',
        'os << "bad"; if (false) os << "(int32 direction, int32 normalization, "',
        "test_clause4_validation_faults_are_actionable",
    ),
    (
        5, "a per-operation backend-availability message instead of one source of truth",
        "native/src/main.cpp",
        "    std::string unavailable_reason = scl::backend_unavailable_reason(backend);",
        '    std::string unavailable_reason = scl::backend_unavailable_reason(backend);\n    if (!unavailable_reason.empty() && operation == "fourier_transform_1d") unavailable_reason = "cuda is not here";',
        "test_clause5_backend_unavailability_is_identical_across_operations",
    ),
    (
        6, "a halted outcome carrying an empty-but-present output",
        "native/include/scl/operation.hpp",
        "        outcome.has_output = false;\n        outcome.detail = detail;",
        "        outcome.has_output = true;\n        outcome.detail = detail;",
        "test_clause6_the_wire_response_omits_output_entirely_when_halted",
    ),
    (
        8, "minting a fault code outside the shared vocabulary",
        "native/src/op_fourier.cpp",
        "return OperationOutcome::halted(kFaultValidation, e.what());",
        "return OperationOutcome::halted(15, e.what());",
        "test_clause8_malformed_requests_only_use_the_shared_vocabulary",
    ),
    (
        9, "an operation that escapes instead of answering",
        "native/src/op_fourier.cpp",
        "FourierParameters decode_configuration(const std::vector<uint8_t>& bytes) {",
        "FourierParameters decode_configuration(const std::vector<uint8_t>& bytes) {\n    if (bytes.empty()) std::exit(3);",
        "test_clause9_totality_holds_for_each_registered_operation",
    ),
    (
        11, "two operations sharing one program identity",
        "python/scl/ste_adapter.py",
        'return b"ste.scl." + operation.replace("_", "-").encode("utf-8") + b".v1"',
        'return b"ste.scl.shared.v1"',
        "test_clause11_every_operation_has_its_own_program_identity",
    ),
]

#: Empty, and kept as a named slot rather than deleted: the moment a new
#: clause is added to the contract without a test, it belongs here where
#: the script REPORTS it. Clauses 2 and 3 lived here until writing their
#: tests found the canonical-encoding defect.
CLAUSES_WITHOUT_A_DEDICATED_TEST = {}


def _restore(source: pathlib.Path, target: pathlib.Path) -> None:
    """Restore WITHOUT preserving mtime, then bump it.

    shutil.copy2 preserves the backup's older mtime, so make considers the
    object file current and silently skips the recompile -- leaving a
    MUTATED BINARY behind a clean `git status`. That happened once; it is
    the reason this helper exists rather than a bare copy2."""
    shutil.copy(source, target)
    target.touch()


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def rebuild() -> bool:
    return run(["cmake", "--build", str(BUILD), "-j4"]).returncode == 0


def suite_failures() -> set:
    result = run([sys.executable, "-m", "pytest", SUITE, "-q", "--no-header",
                  "-p", "no:cacheprovider"], cwd=str(REPO))
    names = set()
    for line in result.stdout.splitlines():
        if line.startswith("FAILED"):
            # strip pytest's [param] suffix: a parametrized test is still
            # the test that claims the clause.
            names.add(line.split("::")[-1].split()[0].split("[")[0])
    return names


def main() -> int:
    backup = pathlib.Path(tempfile.mkdtemp(prefix="scl-mutation-"))
    touched = {path for _, _, path, _, _, _ in MUTATIONS}
    for relative in touched:
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, target)

    verdicts = []
    try:
        assert rebuild(), "the tree does not build before any mutation"
        assert not suite_failures(), "the suite is already failing before any mutation"
        print("baseline: builds clean, suite green\n")

        for clause, description, relative, old, new, expected in MUTATIONS:
            path = REPO / relative
            source = path.read_text()
            if old not in source:
                verdicts.append((clause, "PATCH-STALE", description))
                print(f"clause {clause:>2}: PATCH STALE -- anchor not found in {relative}")
                continue
            path.write_text(source.replace(old, new, 1))

            if not rebuild():
                # a mutation that will not compile proves nothing about the
                # test suite, so it is reported as such rather than counted.
                verdicts.append((clause, "NO-COMPILE", description))
                print(f"clause {clause:>2}: mutation did not compile -- inconclusive")
            else:
                failed = suite_failures()
                if expected in failed:
                    verdicts.append((clause, "CAUGHT", description))
                    print(f"clause {clause:>2}: CAUGHT by {expected}")
                elif failed:
                    verdicts.append((clause, "WRONG-TEST", description))
                    print(f"clause {clause:>2}: caught, but by {sorted(failed)}, "
                          f"not {expected}")
                else:
                    verdicts.append((clause, "SURVIVED", description))
                    print(f"clause {clause:>2}: SURVIVED -- '{description}' is not caught")

            _restore(backup / relative, path)
    finally:
        for relative in touched:
            _restore(backup / relative, REPO / relative)
        rebuild()
        shutil.rmtree(backup, ignore_errors=True)

    print("\n" + "=" * 70)
    caught = [v for v in verdicts if v[1] == "CAUGHT"]
    print(f"{len(caught)}/{len(MUTATIONS)} mutations caught by the claiming test")
    for clause, verdict, description in verdicts:
        if verdict != "CAUGHT":
            print(f"  clause {clause}: {verdict} -- {description}")

    if CLAUSES_WITHOUT_A_DEDICATED_TEST:
        print("\nCLAUSES WITH NO DEDICATED TEST (reported, not hidden):")
        for clause, why in sorted(CLAUSES_WITHOUT_A_DEDICATED_TEST.items()):
            print(f"  clause {clause}: {why}")
    else:
        print("\nevery clause has a dedicated test")

    print("\nrestored tree still green:", not suite_failures())
    return 0 if len(caught) == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
