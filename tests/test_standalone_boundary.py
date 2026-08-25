"""SCL Core Independence, proven mechanically, not by convention.

Five of SCL's existing eight test files (test_client_subprocess.py,
test_contract_identity.py, test_failure_paths.py,
test_numerical_validation.py, test_performance_baseline.py) already only
ever import `scl.client`/`scl.errors`/`scl.quantity` -- never `scl.ste_
adapter`, never anything from STE (`execution.*`). That was true from
Phase 1 onward; it was simply never checked as an explicit property or
labeled as what it proves. This file makes that property explicit,
mechanically checked, and impossible to regress silently.

Two things this file must get right that an in-process pytest assertion
cannot:

1. By the time this test FILE runs, other test files in the same pytest
   session (test_ste_integration.py, test_phase2_evidence_conformance.py,
   test_cpu_cuda_equivalence.py) may already have imported `execution`,
   `evidence`, `materials` into THIS PROCESS's `sys.modules` -- checking
   `sys.modules` in-process would then report contamination that has
   nothing to do with whether `scl.client` itself imports those things.
2. The real claim under test is "an external consumer, in a FRESH
   process, with only SCL's own `python/` directory on `sys.path` (no
   STE checkout, no Notations siblings at all), can import and use SCL."
   That is a fresh-interpreter question, not an already-warm-process one.

So this test spawns a genuinely separate Python subprocess with a
minimal `PYTHONPATH` (SCL's `python/` directory only -- explicitly NOT
including whatever `STE_REPO` conftest.py resolved for the rest of this
suite) and inspects that subprocess's own `sys.modules` from the inside.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _run_in_fresh_interpreter(script: str, cli_path: pathlib.Path) -> subprocess.CompletedProcess:
    """Run `script` in a brand-new Python process whose sys.path contains
    ONLY this repo's python/ directory -- not tests/, not STE_REPO, not
    anything conftest.py set up for the rest of this suite."""
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT / "python"), "PATH": "/usr/bin:/bin", "SCL_CLI_PATH": str(cli_path)},
        capture_output=True, text=True, timeout=30,
    )


_FORBIDDEN_MODULE_PREFIXES = ("execution", "evidence", "materials", "core.canonical", "experiment", "campaign")


def test_scl_client_imports_and_computes_in_a_fresh_interpreter_with_no_notations_on_path(cli_path):
    """The decisive standalone-consumer proof (Task 6 of this phase):
    external application -> SCL API -> scientific computation -> result,
    in a process that has never heard of STE, DAQ, EvidencePool, or
    CanonicalState -- because they are not importable at all (not on
    PYTHONPATH), so any accidental dependency would raise ImportError,
    not silently succeed because conftest.py happened to have STE_REPO
    on the path for the REST of this suite."""
    script = textwrap.dedent(
        """
        import os, pathlib, struct, sys

        # 1. import -- must succeed with ONLY scl's own python/ on sys.path
        from scl.client import (
            SCLRequest, decode_lj_output, encode_lj_configuration,
            encode_lj_positions, run_scl_request,
        )
        from scl.quantity import Quantity, absent_uncertainty

        # 2. compute -- a real external-consumer-shaped call, no Notations
        #    concept (candidate/session/pool/specification/dispatcher)
        #    anywhere in sight.
        request = SCLRequest(
            operation="lj_pairwise_energy_forces", backend="cpu",
            parameters=encode_lj_configuration(epsilon=1.0, sigma=1.0, cutoff=5.0),
            input_payload=encode_lj_positions([(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]),
        )
        result = run_scl_request(request, cli_path=pathlib.Path(os.environ["SCL_CLI_PATH"]))
        assert result.status == "completed", result.detail
        total_energy, forces = decode_lj_output(result.output)
        assert isinstance(total_energy, float)
        assert len(forces) == 2

        # 3. a standalone consumer can even use the typed-quantity helper
        #    without any STE concept of "evidence class" or "method block".
        q = absent_uncertainty(total_energy, unit="epsilon")
        assert q.uncertainty_kind == "absent"

        # 4. the decisive check: nothing Notations-specific ever loaded,
        #    because it was never even importable.
        forbidden = """ + repr(_FORBIDDEN_MODULE_PREFIXES) + """
        loaded = sorted(sys.modules.keys())
        violations = [m for m in loaded if any(m == f or m.startswith(f + ".") for f in forbidden)]
        assert not violations, f"standalone SCL import pulled in Notations modules: {violations}"

        print("STANDALONE_OK")
        """
    )
    proc = _run_in_fresh_interpreter(script, cli_path)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "STANDALONE_OK" in proc.stdout


def test_scl_client_module_itself_declares_no_notations_import(cli_path):
    """A static companion to the dynamic proof above: read
    python/scl/client.py's own source and confirm, textually, that it
    contains no `execution`/`evidence`/`materials`/`core.canonical`
    import statement. Catches the case where a future edit adds such an
    import but happens not to exercise the code path the dynamic test
    above calls."""
    source = (REPO_ROOT / "python" / "scl" / "client.py").read_text()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import execution") or stripped.startswith("from execution"):
            pytest.fail(f"scl/client.py imports STE's execution package: {stripped!r}")
        if stripped.startswith("import evidence") or stripped.startswith("from evidence"):
            pytest.fail(f"scl/client.py imports STE's evidence package: {stripped!r}")
        if stripped.startswith("import materials") or stripped.startswith("from materials"):
            pytest.fail(f"scl/client.py imports STE's materials package: {stripped!r}")


def test_ste_adapter_is_the_only_module_importing_ste(cli_path):
    """The inverse check: exactly ONE file in python/scl/ is allowed to
    import STE's `execution` package -- ste_adapter.py, by design
    (package docstring, python/scl/__init__.py). If a second file starts
    importing STE, the standalone/integration boundary has silently
    widened and this test must catch it."""
    scl_dir = REPO_ROOT / "python" / "scl"
    ste_importing_files = []
    for path in sorted(scl_dir.glob("*.py")):
        text = path.read_text()
        if any(
            line.strip().startswith(("import execution", "from execution"))
            for line in text.splitlines()
        ):
            ste_importing_files.append(path.name)
    assert ste_importing_files == ["ste_adapter.py"], (
        f"expected exactly ste_adapter.py to import STE's execution package, found: {ste_importing_files}"
    )
