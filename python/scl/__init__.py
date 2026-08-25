"""SCL: the Scientific Compute Layer.

SCL is a reusable native scientific-computation layer (`native/` — a C++17
computational core, a CUDA backend, and `scl_cli`, a process-boundary
entry point) with a thin Python client. It is designed to be used two
ways, and this top-level package only ever imports the code the FIRST way
needs:

  1. STANDALONE -- any external application, research script, or
     computational workflow. Needs only what this module (`scl`)
     re-exports below: `SCLRequest`/`SCLResult` (a plain request/response
     contract), `run_scl_request` (drives the native `scl_cli` process),
     `Quantity` (a typed-value-with-uncertainty shape), and the SCL-level
     error types. No Notations-specific concept (evidence, canonical
     state, campaigns, execution identity beyond SCL's own) is reachable
     through this import path -- `tests/test_standalone_boundary.py`
     checks this mechanically, in a fresh interpreter, not by convention.

  2. INTEGRATED into the Scientific Transformer Engine (STE) or a similar
     Notations system -- via `scl.ste_adapter`, a SEPARATE module this
     package does NOT import for you. `ste_adapter` translates STE's own
     `execution.specification.ExecutionSpecification`/`execution.engine
     .ExecutionResult` to/from the plain `scl.client` contract; it is the
     only file under `python/scl/` that imports anything from STE's
     `execution` package (see `docs/SCL_STANDALONE_BOUNDARY.md`). Import
     it explicitly -- `from scl import ste_adapter` or
     `import scl.ste_adapter` -- when you need that integration; it is
     never imported implicitly just because you imported `scl`.

SCL never imports evidence/, materials/, core.canonical, or any other STE
admission machinery -- not even from ste_adapter, which touches only
execution.specification and execution.engine's plain data types, the
same two modules execution/gromacs.py depends on.
"""

from __future__ import annotations

from .client import (
    SCLRequest,
    SCLResult,
    decode_lj_configuration,
    decode_lj_output,
    default_cli_path,
    encode_lj_configuration,
    encode_lj_positions,
    raise_for_result,
    run_scl_request,
)
from .errors import (
    SCLBackendUnavailableError,
    SCLComputationError,
    SCLError,
    SCLInternalError,
    SCLProtocolError,
    SCLTimeoutError,
    SCLValidationError,
)
from .method_block import LJMethodBlock, lj_method_block_for
from .quantity import Quantity, absent_uncertainty

__all__ = [
    # request/response contract
    "SCLRequest",
    "SCLResult",
    "run_scl_request",
    "raise_for_result",
    "default_cli_path",
    # lj_pairwise_energy_forces wire encoding
    "encode_lj_configuration",
    "encode_lj_positions",
    "decode_lj_configuration",
    "decode_lj_output",
    # typed quantities
    "Quantity",
    "absent_uncertainty",
    # method metadata
    "LJMethodBlock",
    "lj_method_block_for",
    # errors
    "SCLError",
    "SCLProtocolError",
    "SCLTimeoutError",
    "SCLValidationError",
    "SCLBackendUnavailableError",
    "SCLComputationError",
    "SCLInternalError",
]
