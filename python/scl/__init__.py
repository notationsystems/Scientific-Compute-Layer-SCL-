"""SCL: the Scientific Compute Layer's Python-side client and STE adapter.

This package is the STE-facing half of the SCL boundary described in
docs/SCL_ARCHITECTURE.md. It does two, deliberately separate, things:

  scl.client       -- SCLRequest/SCLResult: a standalone, STE-agnostic
                       contract for invoking the native `scl_cli` binary.
                       Usable and testable with no STE installed at all.
  scl.ste_adapter   -- translates STE's own ExecutionSpecification /
                       ExecutionResult (imported from the real `execution`
                       package) into scl.client calls and back. THIS is
                       the module STE's dispatcher would import, exactly
                       parallel to how it imports execution.gromacs today.

SCL never imports evidence/, materials/, core.canonical, or any other STE
admission machinery -- only execution.specification and execution.engine's
plain data types, the same two modules execution/gromacs.py depends on.
"""
