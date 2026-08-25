"""SCL's own content-commitment scheme -- the SAME pattern STE's
`execution.commitments` uses (length-prefixed tag + fields, SHA-256,
lowercase hex), independently implemented for SCL's own identity space.

This is deliberately NOT an import of `execution.commitments`: per the
DAF/STE reconnaissance behind this design (docs/SCL_ARCHITECTURE.md), an
identity SCHEME may be reused freely, but a specific identity NAMESPACE
must not be -- SCL's job-level identities (operation/parameters/input at
the SCL-request level) are a different kind of thing from STE's execution
identities (program/input/output/computation at the ExecutionSpecification
level), even for the fields that end up meaning almost the same thing, and
conflating the two namespaces would let a coincidental hash collision
across unrelated systems look like a real relationship. Where SCL produces
values that DO need to live in STE's own execution-identity space (an
ExecutionResult's output_identity/computation_identity), scl/ste_adapter.py
imports and uses `execution.commitments` directly, exactly as
execution/gromacs.py does -- see that module for the real reuse point.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Sequence

#: Domain tags for SCL's own request-level identity space. Distinct string
#: space from STE's `scout.execution.*` tags by construction (different
#: prefix), so even an accidental byte collision in the hashed content
#: could never be mistaken for an STE execution identity.
OPERATION_TAG = "scl.request.operation.v1"
PARAMETERS_TAG = "scl.request.parameters.v1"
INPUT_TAG = "scl.request.input.v1"
REQUEST_TAG = "scl.request.v1"
OUTPUT_TAG = "scl.result.output.v1"
COMPUTATION_TAG = "scl.result.computation.v1"


def canonical(tag: str, fields: Sequence[bytes]) -> bytes:
    """`len(tag) u64 LE | tag | count(fields) u64 LE | (len u64 LE | bytes)*`
    -- byte-identical framing to STE's `execution.commitments.canonical`,
    reused as a documented pattern (see module docstring), not imported."""
    tag_bytes = tag.encode("utf-8")
    out = struct.pack("<Q", len(tag_bytes)) + tag_bytes + struct.pack("<Q", len(fields))
    for field in fields:
        out += struct.pack("<Q", len(field)) + field
    return out


def commit_hex(tag: str, fields: Sequence[bytes]) -> str:
    return hashlib.sha256(canonical(tag, fields)).hexdigest()
