"""Test setup shared by the whole SCL Phase 1 suite.

Puts `python/` (this repo's `scl` package) and, if present, a local clone
of the Scientific Transformer Engine (STE) repo on sys.path so
tests/test_ste_integration.py can import STE's REAL `execution.*` types
and exercise a genuine STE<->SCL round trip, not a fake/mocked stand-in.

The STE clone location is resolved via the STE_REPO env var if set,
falling back to the sibling checkout this session used for reconnaissance
(`/home/user/notationsystems/scientific-transformer-engine`). Tests that
need it skip cleanly (not fail) when it is not present -- an environment
gap, not an architectural failure, exactly the posture
tests/test_execution_gromacs.py takes in STE itself for a missing `gmx`
binary."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

#: NOTE: deliberately NOT nested under a directory literally named
#: "notationsystems" -- STE's own tests/test_phase120_dispatch_default_
#: witness.py excludes any path containing that substring from its
#: construction-site census (a filter aimed at something else that
#: coincidentally matches a checkout directory of that name), so a clone
#: at .../notationsystems/scientific-transformer-engine makes an
#: unrelated, pre-existing STE test fail for path reasons alone -- not
#: anything this SCL integration touches. Verified independently: same
#: STE commit, cloned to a path without that substring, passes clean
#: (1914 passed, 0 failed; see docs/PHASE1_AUDIT.md).
_DEFAULT_STE_PATH = pathlib.Path("/home/user/ste-clean")
STE_REPO = pathlib.Path(os.environ.get("STE_REPO", str(_DEFAULT_STE_PATH)))
STE_AVAILABLE = (STE_REPO / "execution" / "specification.py").exists()
if STE_AVAILABLE:
    sys.path.insert(0, str(STE_REPO))

requires_ste = pytest.mark.skipif(
    not STE_AVAILABLE,
    reason=f"no STE checkout at {STE_REPO} (set STE_REPO); environment gap, not an architectural pass",
)


def _cli_path() -> pathlib.Path:
    return REPO_ROOT / "native" / "build" / "scl_cli"


@pytest.fixture(scope="session", autouse=True)
def _ensure_native_build():
    """Configure+build the native CLI once per test session if it is not
    already built, so a clean checkout can run `pytest` with no manual
    build step. Mirrors execution/engine.py's expectation that
    crates/target/release/execution-cli already exists -- here we just
    build it ourselves rather than requiring a separate `cargo build`."""
    cli = _cli_path()
    if cli.exists():
        return
    build_dir = REPO_ROOT / "native" / "build"
    subprocess.run(
        ["cmake", "-S", str(REPO_ROOT / "native"), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", str(os.cpu_count() or 2)],
        check=True,
        capture_output=True,
    )
    assert cli.exists(), f"native build completed but {cli} is still missing"


@pytest.fixture()
def cli_path() -> pathlib.Path:
    return _cli_path()
