"""SCL exception types.

Two tiers, matching execution/engine.py's own EngineProtocolError /
EngineIdentityMismatch / ExecutionRefused split:

  * Client/protocol-layer failures (SCLProtocolError, SCLTimeoutError) --
    something is wrong with the ENVIRONMENT or CHANNEL: the binary is
    missing, the process hung, the response could not be parsed at all.
    scl.client.run_scl_request RAISES these; they are not part of the
    normal computational outcome space.

  * Computation-outcome faults (SCLValidationError, SCLBackendUnavailable
    Error, SCLComputationError, SCLInternalError) -- the native CLI
    answered cleanly with status="halted" and a specific exit_code. These
    are NOT raised by run_scl_request itself (an SCLResult with
    status="halted" is a normal, fully-described return value, exactly as
    execution/gromacs.py returns a halted ExecutionResult rather than
    raising). They exist here so a caller that prefers exceptions can use
    `scl.client.raise_for_result(result)` to convert a halted SCLResult
    into the matching typed exception, distinguishing every fault stage
    (Task 7 of the SCL Phase 1 brief: never collapse to one generic
    'execution failed')."""

from __future__ import annotations


class SCLError(RuntimeError):
    """Base class for every SCL-specific exception."""


class SCLProtocolError(SCLError):
    """The scl_cli process or channel misbehaved: missing binary, non-zero
    catastrophic exit, unparseable stdout, or a response missing required
    fields. Never raised for a normal computational halt."""


class SCLTimeoutError(SCLError):
    """scl_cli did not answer within the caller's timeout budget."""


class SCLValidationError(SCLError):
    """exit_code 11: the request's parameters or input were structurally
    invalid (e.g. sigma <= 0, zero particles)."""


class SCLBackendUnavailableError(SCLError):
    """exit_code 12: the requested backend is not usable in this
    build/host (e.g. backend=cuda with no CUDA toolkit at build time, or
    no device visible at run time)."""


class SCLComputationError(SCLError):
    """exit_code 13: the algorithm itself faulted (e.g. coincident
    particles, a non-finite result) -- distinct from bad input, which
    validation would have already caught."""


class SCLInternalError(SCLError):
    """exit_code 14: an unexpected exception was caught inside scl_cli.
    Should not normally occur; if it does, it is a defect in the native
    binary, not a modeling or input problem."""
