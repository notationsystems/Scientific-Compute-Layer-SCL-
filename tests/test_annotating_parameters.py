"""Every configuration field is either PARTICIPATING or ANNOTATING, and
the identity contract treats the two differently. See docs/SCL_CONTRACT.md
section 6.1.

Delta-t surfaced this as a Fourier property; it is stated as a general
rule because the same shape recurs whenever a parameter annotates a result
without participating in it. So the check here is deliberately NOT a
Fourier test: it is a table of declared classifications covering every
configuration field of every registered operation, each one asserted
against the running binary rather than trusted from its declaration.

The four clauses being enforced:

  1. both kinds change `parameters_identity` and `request_identity`;
  2. only PARTICIPATING fields may change `output_identity` /
     `computation_identity`;
  3. equal computation identity + differing request identity is the
     SIGNATURE of an annotating field (asserted positively, so that
     accidentally making an annotating field participate FAILS rather than
     quietly strengthening the coupling);
  4. an annotating field is never silently defaulted -- absent and
     present-with-the-conventional-value stay distinguishable.

Clause 4 is the one that will matter for units, the next instance of this
shape: a wrong unit is invisibly present in a way a missing frequency axis
is not.
"""

from __future__ import annotations

import pytest

from scl.client import (
    SCLRequest,
    encode_lj_configuration,
    encode_lj_positions,
    run_scl_request,
)
from scl.fourier import encode_fourier_configuration, encode_real_signal

FORWARD, INVERSE = 1, -1
NORM_NONE, NORM_ONE_OVER_N = 0, 1

_LJ_INPUT = encode_lj_positions([(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)])
_FOURIER_INPUT = encode_real_signal([1.0, 2.0, 3.0, 4.0])


def _lj(epsilon=1.0, sigma=1.0, cutoff=5.0):
    return SCLRequest("lj_pairwise_energy_forces", "cpu",
                      encode_lj_configuration(epsilon, sigma, cutoff), _LJ_INPUT)


def _fourier(direction=FORWARD, normalization=NORM_NONE, dt=None):
    return SCLRequest("fourier_transform_1d", "cpu",
                      encode_fourier_configuration(direction, normalization, dt),
                      _FOURIER_INPUT)


#: (label, classification, base request, request differing ONLY in that field).
#: Every configuration field of every registered operation appears exactly
#: once; `test_the_table_covers_every_configuration_field` holds it to that.
CLASSIFIED_FIELDS = [
    ("lj.epsilon",                "participating", _lj(),        _lj(epsilon=2.0)),
    ("lj.sigma",                  "participating", _lj(),        _lj(sigma=1.2)),
    ("lj.cutoff",                 "participating", _lj(),        _lj(cutoff=1.4)),
    ("fourier.direction",         "participating", _fourier(),   _fourier(direction=INVERSE)),
    ("fourier.normalization",     "participating", _fourier(),   _fourier(normalization=NORM_ONE_OVER_N)),
    ("fourier.sample_spacing",    "annotating",    _fourier(),   _fourier(dt=0.01)),
]

PARTICIPATING = [row for row in CLASSIFIED_FIELDS if row[1] == "participating"]
ANNOTATING = [row for row in CLASSIFIED_FIELDS if row[1] == "annotating"]


def _completed(request, cli_path):
    result = run_scl_request(request, cli_path=cli_path)
    assert result.status == "completed", f"{request.operation}: {result.detail}"
    return result


# --- clause 1: both kinds are part of what was asked for -----------------

@pytest.mark.parametrize("label,classification,base,other", CLASSIFIED_FIELDS,
                         ids=[row[0] for row in CLASSIFIED_FIELDS])
def test_clause1_every_field_changes_the_request_identity(label, classification, base, other):
    """An annotating field is still part of the request. It is carried in
    the configuration bytes for exactly this reason -- no side channel, no
    separate metadata identity."""
    assert base.parameters_identity() != other.parameters_identity(), label
    assert base.identity() != other.identity(), label
    assert base.input_identity() == other.input_identity(), f"{label} varied more than one field"
    assert base.operation_identity() == other.operation_identity(), label


# --- clause 2/3: only participating fields reach the computation ---------

@pytest.mark.parametrize("label,classification,base,other", PARTICIPATING,
                         ids=[row[0] for row in PARTICIPATING])
def test_clause2_participating_fields_change_the_computed_result(label, classification,
                                                                  base, other, cli_path):
    """Declared participating, so the mathematics must actually read it.
    A field declared participating that does not change the output is
    either misclassified or silently ignored by the kernel."""
    first, second = _completed(base, cli_path), _completed(other, cli_path)
    assert first.output != second.output, f"{label} declared participating but changed nothing"
    assert first.output_identity != second.output_identity, label
    assert first.computation_identity != second.computation_identity, label


@pytest.mark.parametrize("label,classification,base,other", ANNOTATING,
                         ids=[row[0] for row in ANNOTATING])
def test_clause3_annotating_fields_leave_the_computation_identical(label, classification,
                                                                    base, other, cli_path):
    """The signature of an annotating field, asserted POSITIVELY: byte-equal
    output and equal computation identity, with a differing request
    identity. If someone later makes dt participate in the transform, this
    fails -- which is the point. A cache keyed on computation_identity may
    reuse these output bytes and must not reuse their interpretation."""
    first, second = _completed(base, cli_path), _completed(other, cli_path)
    assert first.output == second.output, f"{label} declared annotating but changed the output"
    assert first.output_identity == second.output_identity, label
    assert first.computation_identity == second.computation_identity, label
    assert first.request_identity != second.request_identity, label


# --- clause 4: absence is never a default --------------------------------

@pytest.mark.parametrize("supplied_value", [1.0, 0.5, 1e-9, 0.0])
def test_clause4_absent_is_distinguishable_from_any_supplied_value(supplied_value):
    """SCL never assumes dt = 1. Absent and present-with-the-obvious-value
    are different facts and stay different in `parameters_identity` --
    including for dt = 0, which the operation goes on to REJECT: absence
    and an invalid value are also distinct, so absence cannot be smuggled
    in as a sentinel.

    This is the clause that will matter for units: a missing frequency axis
    is visibly absent, but a wrongly-assumed unit is invisibly present."""
    absent = _fourier(dt=None)
    supplied = _fourier(dt=supplied_value)
    assert absent.parameters_identity() != supplied.parameters_identity()
    assert absent.identity() != supplied.identity()


def test_clause4_the_conventional_default_computes_identically_to_absence(cli_path):
    """The distinction above is not paid for with a behavioural difference:
    dt = 1 (the value an implicit default would have assumed) yields
    byte-identical output to no dt at all. The whole difference lives in
    the request identity and the method block, which is where it belongs --
    absence changes what may be CONCLUDED, never what was COMPUTED."""
    assert _completed(_fourier(dt=None), cli_path).output == \
        _completed(_fourier(dt=1.0), cli_path).output


def test_clause4_zero_sample_spacing_is_rejected_not_defaulted(cli_path):
    """A zero spacing is not "unspecified": it is an invalid spacing, and
    it names its own field in the fault. Silently treating it as absence
    would be the exact defaulting clause 4 forbids."""
    result = run_scl_request(_fourier(dt=0.0), cli_path=cli_path)
    assert result.status == "halted" and result.exit_code == 11
    assert "sample_spacing" in result.detail


def test_clause4_absence_is_representable_at_all():
    """A layout with no way to SAY 'absent' would force a default. The
    24-byte Fourier configuration carries an explicit has_sample_spacing
    flag rather than overloading a sentinel value."""
    assert encode_fourier_configuration(FORWARD, NORM_NONE, None) != \
        encode_fourier_configuration(FORWARD, NORM_NONE, 0.0)


# --- the table itself ----------------------------------------------------

def test_the_table_covers_every_configuration_field(cli_path):
    """A classification table that silently misses a field proves nothing
    about that field. Both configurations are 24 bytes with a documented
    layout; the declared field counts must account for all of it."""
    from scl.client import SCLRequest as _R  # noqa: F401  (kept explicit for readers)

    declared = {}
    for label, classification, _, _ in CLASSIFIED_FIELDS:
        operation, _, field = label.partition(".")
        declared.setdefault(operation, set()).add(field)

    assert declared["lj"] == {"epsilon", "sigma", "cutoff"}
    # `reserved` is not classified: it must be zero, so it has no values to
    # vary. It is covered by the operation contract's clause 2 instead.
    assert declared["fourier"] == {"direction", "normalization", "sample_spacing"}

    assert len(encode_lj_configuration(1.0, 1.0, 5.0)) == 24
    assert len(encode_fourier_configuration(FORWARD, NORM_NONE, None)) == 24


def test_every_classification_is_one_of_the_two_kinds():
    for label, classification, _, _ in CLASSIFIED_FIELDS:
        assert classification in ("participating", "annotating"), label
    assert PARTICIPATING and ANNOTATING, "a table with only one kind proves nothing"
