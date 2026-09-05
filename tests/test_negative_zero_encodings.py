"""Negative zero is a second byte encoding of a value the contract already has.

WHY THIS FILE EXISTS. Two guards in this repository were written specifically
to stop one meaning from having two encodings, and both were defeated by the
one value that is byte-different from, and numerically equal to, the constant
they test against.

    native/src/op_fourier.cpp   !has_sample_spacing && spacing != 0.0
    native/src/op_kalman.cpp    !(symmetry_tolerance >= 0.0)

IEEE-754 says -0.0 == +0.0. So `-0.0 != 0.0` is FALSE and `-0.0 >= 0.0` is
TRUE, and negative zero walks through both. The fourier guard's own comment
says the failure it prevents is "two byte-different configurations mean
exactly the same thing while producing different parameters_identity values"
-- which is exactly, and only, what negative zero does to it.

WHY THE EXISTING SUITE COULD NOT REACH IT. `encode_fourier_configuration`
writes `0.0` for the absent case and `encode_kalman_configuration` calls
`float()` on its tolerances; neither can emit -0.0 from its public API. A test
that goes through the encoder therefore cannot construct the payload that
breaks the decoder. These tests HAND-PACK the bytes, which is the only way to
probe a decoder's acceptance set rather than the encoder's output set.

THE GENERAL FORM WORTH CARRYING. A canonical-encoding guard needs two laws,
not one: decode(encode(m)) == m for every meaning, AND encode(decode(b)) == b
for every ACCEPTED byte string. The first was tested and holds. The second is
the one -0.0 breaks, and nothing tested it. `!=` compares magnitudes;
`std::signbit` reads the sign bit. Only the second can separate the two zeros.
"""

from __future__ import annotations

import math
import struct

import pytest

from scl.client import SCLRequest, run_scl_request
from scl.fourier import encode_real_signal

FORWARD, NORM_NONE = 1, 0


def _fourier_config(spacing_slot: float, has_spacing: int = 0) -> bytes:
    """Hand-packed, so the ignored slot can carry a value the encoder never
    emits. Layout copied from encode_fourier_configuration."""
    return struct.pack("<iiiid", FORWARD, NORM_NONE, has_spacing, 0, spacing_slot)


def test_the_encoder_cannot_produce_the_payload_these_tests_probe():
    """The reason this file hand-packs. If the encoder could emit -0.0, the
    existing suite would already have covered this and these tests would be
    redundant."""
    from scl.fourier import encode_fourier_configuration
    emitted = encode_fourier_configuration(sample_spacing_seconds=None)
    slot = struct.unpack("<d", emitted[16:24])[0]
    assert slot == 0.0 and not math.copysign(1.0, slot) < 0, (
        "encoder emits +0.0; a test routed through it cannot reach the defect"
    )


@pytest.mark.parametrize("slot,should_be_accepted", [(0.0, True), (-0.0, False)])
def test_only_positive_zero_is_accepted_in_the_ignored_spacing_slot(
        cli_path, slot, should_be_accepted):
    """+0.0 is the canonical encoding of 'absent' and must work. -0.0 means
    the same thing in different bytes and must be refused, or one meaning has
    two parameter identities."""
    result = run_scl_request(
        SCLRequest("fourier_transform_1d", "cpu",
                   _fourier_config(slot),
                   encode_real_signal([1.0, 2.0, 3.0, 4.0])),
        cli_path=cli_path)
    if should_be_accepted:
        assert result.status == "completed", result.detail
    else:
        assert result.status == "halted", (
            "-0.0 in the ignored slot was ACCEPTED. It decodes to the same "
            "meaning as +0.0 and hashes to a different parameters_identity, "
            "which is the exact failure the guard beside it names."
        )
        assert "negative zero" in (result.detail or "").lower() or \
               "sample_spacing_seconds must be 0" in (result.detail or ""), result.detail


def test_the_two_zeros_really_do_give_different_parameter_identities(cli_path):
    """The discriminating fact. If the two payloads hashed the same, refusing
    -0.0 would be pedantry rather than a fix."""
    a = _fourier_config(0.0)
    b = _fourier_config(-0.0)
    assert a != b, "the two payloads are byte-identical; there is nothing to fix"
    ra = run_scl_request(SCLRequest("fourier_transform_1d", "cpu", a,
                                    encode_real_signal([1.0, 2.0])), cli_path=cli_path)
    assert ra.status == "completed"
    # the identity is over the bytes, so differing bytes -> differing identity
    from scl.identity import commit_hex, PARAMETERS_TAG
    assert commit_hex(PARAMETERS_TAG, [a]) != commit_hex(PARAMETERS_TAG, [b]), (
        "two encodings of one meaning would share an identity, which would "
        "make this a non-issue -- they do not"
    )


def _kalman_pair(symmetry_tolerance: float):
    """A request that COMPLETES with a canonical tolerance, so that a halt on
    -0.0 is attributable to the tolerance and to nothing else."""
    from scl.kalman import encode_kalman_configuration, encode_kalman_input
    return (
        encode_kalman_configuration(
            transition=[[1.0, 0.1], [0.0, 1.0]],
            observation=[[1.0, 0.0], [0.0, 1.0]],
            process_noise=[[0.01, 0.0], [0.0, 0.04]],
            measurement_noise=[[0.25, 0.0], [0.0, 0.16]],
            symmetry_tolerance=symmetry_tolerance,
        ),
        encode_kalman_input(
            initial_state=[0.0, 0.0],
            initial_covariance=[[1.0, 0.0], [0.0, 1.0]],
            measurements=[[0.1, 0.9], [0.2, 1.1], [0.35, 0.95]],
        ),
    )


def test_the_kalman_control_case_actually_completes(cli_path):
    """THE HALF THAT WAS MISSING, AND ITS ABSENCE MADE THE NEXT TEST VACUOUS.

    The first version of the test below sent an empty input payload. The
    request halted -- for the empty payload, not for the tolerance -- and the
    assertion `status == halted` passed. Planting the defect (removing the
    signbit guard) did NOT make it fail, which is the only reason this was
    caught. A refusal test with no completing control asserts nothing."""
    cfg, inp = _kalman_pair(0.0)
    result = run_scl_request(SCLRequest("kalman_filter_linear", "cpu", cfg, inp),
                             cli_path=cli_path)
    assert result.status == "completed", (
        f"the control must succeed or the refusal below is unattributable: "
        f"{result.detail}")


def test_negative_zero_tolerances_are_refused_by_kalman(cli_path):
    """Same class as the fourier slot, second site. `!(x >= 0.0)` rejects NaN
    and accepts -0.0, because -0.0 >= 0.0 is true."""
    good_cfg, inp = _kalman_pair(0.0)
    bad_cfg, _ = _kalman_pair(-0.0)
    assert bad_cfg != good_cfg, "the two encodings are byte-identical"
    assert struct.unpack("<d", bad_cfg[bad_cfg.find(struct.pack("<d", -0.0)):][:8])[0] == 0.0

    result = run_scl_request(SCLRequest("kalman_filter_linear", "cpu", bad_cfg, inp),
                             cli_path=cli_path)
    assert result.status == "halted", (
        "a -0.0 symmetry_tolerance was accepted. It is numerically identical "
        "to +0.0 everywhere it is used, so this is one meaning with two "
        "parameter identities."
    )
    assert "negative zero" in (result.detail or "").lower(), (
        f"halted for the wrong reason -- the message must name the defect, "
        f"or this test passes on any unrelated failure: {result.detail}"
    )
