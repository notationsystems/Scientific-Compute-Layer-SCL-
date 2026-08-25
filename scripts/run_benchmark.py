#!/usr/bin/env python3
"""Performance baseline for the SCL Phase 1 substrate (Task 8).

Sweeps particle count N and records, per run:
  wall_clock_seconds        the FULL round trip from Python's point of
                             view: subprocess spawn + JSON marshaling +
                             native compute + JSON marshaling back
  native_compute_seconds    the O(N^2) kernel loop alone, timed inside
                             scl_cli with std::chrono, EXCLUDING process
                             startup and JSON (de)serialization
  ste_overhead_seconds      wall_clock_seconds - native_compute_seconds
                             -- everything the process/subprocess/JSON
                             boundary costs on top of the computation
                             itself
  input_bytes / output_bytes  the exact request/response payload sizes

Purpose (Task 8): establish that the SCL boundary introduces measurable,
UNDERSTOOD overhead -- not to optimize it. Memory is not measured here
(N up to a few thousand keeps peak RSS well under a megabyte of doubles;
detailed RSS profiling was judged out of scope for a Phase 1 baseline).

Run: python3 scripts/run_benchmark.py [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "python"))

from scl.client import SCLRequest, encode_lj_configuration, encode_lj_positions, run_scl_request  # noqa: E402

PARTICLE_COUNTS = [10, 50, 100, 250, 500, 1000, 2000]
REPEATS = 3


def make_positions(n: int):
    # A simple cubic-ish spread, spaced comfortably beyond sigma so the
    # potential stays well-behaved (no coincident/near-singular pairs) at
    # every N in the sweep.
    side = max(1, int(round(n ** (1 / 3))) + 1)
    positions = []
    spacing = 1.5
    for i in range(n):
        x = (i % side) * spacing
        y = ((i // side) % side) * spacing
        z = (i // (side * side)) * spacing
        positions.append((x, y, z))
    return positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument(
        "--cli-path",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "native" / "build" / "scl_cli",
    )
    args = parser.parse_args()

    rows = []
    print(f"{'N':>6}  {'wall_clock(ms)':>15}  {'native(ms)':>12}  {'overhead(ms)':>13}  {'input(B)':>9}  {'output(B)':>10}")
    for n in PARTICLE_COUNTS:
        positions = make_positions(n)
        request = SCLRequest(
            operation="lj_pairwise_energy_forces",
            backend="cpu",
            parameters=encode_lj_configuration(1.0, 1.0, 5.0),
            input_payload=encode_lj_positions(positions),
        )
        wall_samples = []
        native_samples = []
        for _ in range(REPEATS):
            result = run_scl_request(request, cli_path=args.cli_path, timeout=120.0)
            assert result.status == "completed", result.detail
            wall_samples.append(result.wall_clock_seconds)
            native_samples.append(result.native_compute_seconds)

        wall_ms = statistics.median(wall_samples) * 1000
        native_ms = statistics.median(native_samples) * 1000
        overhead_ms = wall_ms - native_ms
        input_bytes = len(request.input_payload)
        output_bytes = 8 + n * 24

        print(f"{n:>6}  {wall_ms:>15.3f}  {native_ms:>12.4f}  {overhead_ms:>13.3f}  {input_bytes:>9}  {output_bytes:>10}")
        rows.append(
            {
                "n_particles": n,
                "wall_clock_ms_median": wall_ms,
                "native_compute_ms_median": native_ms,
                "ste_overhead_ms_median": overhead_ms,
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "repeats": REPEATS,
            }
        )

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
