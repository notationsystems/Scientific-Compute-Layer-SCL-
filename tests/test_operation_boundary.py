"""The multi-operation boundary, and a hard lock proving the LJ operation
was preserved byte-for-byte across the dispatch generalization.

LJ_GOLDEN below was captured from the binary BUILT BEFORE the refactor
(single hardcoded operation, decoders inlined in main.cpp) and is embedded
here as literals. It is not a snapshot of current behavior regenerated
after the fact -- it is the pre-refactor answer, so any drift in output
bytes, any identity, any fault code, or any fault message fails
immediately. Per the phase rule: "Do not update expected values merely
because the implementation moved."
"""

from __future__ import annotations

import json
import subprocess

import pytest

from conftest import requires_ste
from scl.client import (
    SCLRequest,
    encode_lj_configuration,
    encode_lj_positions,
    run_scl_request,
)

LJ_GOLDEN = {
    'two_particle': {
        'status': 'completed',
        'exit_code': 0,
        'output_hex': '2add080f6580d4bf8db1523d4987f23f000000000000000000000000000000008db1523d4987f2bf00000000000000000000000000000000',
        'output_identity': '259a5d81a23211a2c702b74dc288c693b021c1d6f5e2e48caeb93f36515c2f91',
        'computation_identity': 'd3062f519d75e6d5fa4436bc9f31947962e6aff8ca1ca22de3fbd248a67bb684',
        'detail': None,
        'operation_identity': '6f0a99827cc6b27dc6a77cd71bdd85423fd15cded3b795a51b4f5e95b62f5ddb',
        'parameters_identity': '21c6e646286983f898255ae36c8185d6bf9a5db4b1d54ad7e6bfdb891193209d',
        'input_identity': 'e3ad51ed29bc9573675689a4e8c877a6e5e1af1535d96509470d59342358365c',
        'request_identity': '8ccc8e292d6a96a248170c86ac240caebfd79bebf8c5a9f0091c940b5db811b0',
    },
    'three_particle_asym': {
        'status': 'completed',
        'exit_code': 0,
        'output_hex': 'e901b2a1cdebe2bfe9e53132413aef3f26c28f4cae76dabff14939d19a86e2bfb5843c43a5c3f3bf7f9e8f73155fe53f62724241ce32b53f01478ea8129ad03fd87a8f9a7c47d0bf4af7211282c0df3f',
        'output_identity': '0c819bb35a1677cfef7a5540afe9159a1c6190c7389cebdd808a746b4b3d7476',
        'computation_identity': 'ec5ad25c82efe681d815a297480a17f0f39d3eae5df347f77fefafaa263abb1c',
        'detail': None,
        'operation_identity': '6f0a99827cc6b27dc6a77cd71bdd85423fd15cded3b795a51b4f5e95b62f5ddb',
        'parameters_identity': 'd385928aefab105da17c1ff7c829aca79c89d0fef5a7a96dcbfc6c2ca5e0f326',
        'input_identity': '8553aa8f219ae17d7e694690855580be821c4e5fd35bbb56bb50c1dce2633069',
        'request_identity': '074a49ffa346a2d86800841ab4241cb8019a26f4dfc5e2fcae2adf9f3fde4c3b',
    },
    'cutoff_truncated': {
        'status': 'completed',
        'exit_code': 0,
        'output_hex': '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000',
        'output_identity': '154bb9367255f5a53749ab00c8da77844607696466aa36317d85fd81c396b90a',
        'computation_identity': 'c6be81d59c935da4fe4a33a56083098b5914bcadba502df478a474a8bee91863',
        'detail': None,
        'operation_identity': '6f0a99827cc6b27dc6a77cd71bdd85423fd15cded3b795a51b4f5e95b62f5ddb',
        'parameters_identity': '99414522b69e3d254ed4d93484c503c3400311196461a59901a02cbca92e1834',
        'input_identity': '2977cecf90f6074250425fd5b0e8459402a646ded1d501b42c5e07e848e866bf',
        'request_identity': 'd9231eca9a25f839d8b78b438865bcdefc17387c2bb69dfcb4828ebc0302089a',
    },
    'fault_sigma': {
        'status': 'halted',
        'exit_code': 11,
        'output_hex': None,
        'output_identity': None,
        'computation_identity': None,
        'detail': 'sigma must be finite and > 0, got -1',
        'operation_identity': '6f0a99827cc6b27dc6a77cd71bdd85423fd15cded3b795a51b4f5e95b62f5ddb',
        'parameters_identity': '4418da69b5d1e35a00a5c98801600f3f57c30c3767b5f0c5016ed2c740af87b1',
        'input_identity': 'e3ad51ed29bc9573675689a4e8c877a6e5e1af1535d96509470d59342358365c',
        'request_identity': 'bb4255843a0c1363923ef09f1a529c3683936096017fce61eb35a78015f867f2',
    },
    'fault_coincident': {
        'status': 'halted',
        'exit_code': 13,
        'output_hex': None,
        'output_identity': None,
        'computation_identity': None,
        'detail': 'two particles at zero separation: the potential is singular',
        'operation_identity': '6f0a99827cc6b27dc6a77cd71bdd85423fd15cded3b795a51b4f5e95b62f5ddb',
        'parameters_identity': '21c6e646286983f898255ae36c8185d6bf9a5db4b1d54ad7e6bfdb891193209d',
        'input_identity': 'ea6e2cceb53600e2a7b4ca5a32f239b98345e93bd58237bb786acdce97ff19c4',
        'request_identity': 'b85dd8afd5a1e6b27c6ebe4aaf70d636996e080d3b673f2dda26b131ef630a4d',
    },
    'fault_empty': {
        'status': 'halted',
        'exit_code': 11,
        'output_hex': None,
        'output_identity': None,
        'computation_identity': None,
        'detail': 'at least one particle is required, got 0',
        'operation_identity': '6f0a99827cc6b27dc6a77cd71bdd85423fd15cded3b795a51b4f5e95b62f5ddb',
        'parameters_identity': '21c6e646286983f898255ae36c8185d6bf9a5db4b1d54ad7e6bfdb891193209d',
        'input_identity': '2400aafc058b3994af0ba259e73ad5f2078d5a6745cd0777a170250c91f2a343',
        'request_identity': '82e2d5b0d7e11f2295afe6bdbf0435102a5eff0f04f4d085bc9b165f84f823b2',
    },
    'fault_badcfg': {
        'status': 'halted',
        'exit_code': 11,
        'output_hex': None,
        'output_identity': None,
        'computation_identity': None,
        'detail': 'configuration must be exactly 24 bytes (3 little-endian float64: epsilon, sigma, cutoff), got 5',
        'operation_identity': '6f0a99827cc6b27dc6a77cd71bdd85423fd15cded3b795a51b4f5e95b62f5ddb',
        'parameters_identity': '0d138f3f1b0110bf735e1ca665b4970da6d5fd19fd04325659bf137488429f25',
        'input_identity': 'e3ad51ed29bc9573675689a4e8c877a6e5e1af1535d96509470d59342358365c',
        'request_identity': '1f9590910b391a2662c30d8e7bed21b8f4436246fcbd1adab2d7e8d7399d3380',
    },
    'fault_badinput': {
        'status': 'halted',
        'exit_code': 11,
        'output_hex': None,
        'output_identity': None,
        'computation_identity': None,
        'detail': 'input must be a whole number of 24-byte particles (3 little-endian float64 each: x, y, z), got 10 bytes',
        'operation_identity': '6f0a99827cc6b27dc6a77cd71bdd85423fd15cded3b795a51b4f5e95b62f5ddb',
        'parameters_identity': '21c6e646286983f898255ae36c8185d6bf9a5db4b1d54ad7e6bfdb891193209d',
        'input_identity': '8b3010eec628f90827094f95b570c57011a6486fba29de9dfe183772dfde8a4a',
        'request_identity': 'd03d206bbc788fc3d1f8d86781ce8dcd54f4fd92465674c7908b77c7d68e5804',
    },
}


_CASES = {
    "two_particle": (encode_lj_configuration(1.0, 1.0, 5.0),
                      encode_lj_positions([(0, 0, 0), (1.5, 0, 0)])),
    "three_particle_asym": (encode_lj_configuration(0.9, 1.05, 6.0),
                             encode_lj_positions([(0.1, 0.2, 0.3), (1.4, -0.5, 0.2), (-0.6, 0.9, -1.1)])),
    "cutoff_truncated": (encode_lj_configuration(1.0, 1.0, 2.0),
                          encode_lj_positions([(0, 0, 0), (10, 0, 0)])),
    "fault_sigma": (encode_lj_configuration(1.0, -1.0, 5.0),
                     encode_lj_positions([(0, 0, 0), (1.5, 0, 0)])),
    "fault_coincident": (encode_lj_configuration(1.0, 1.0, 5.0),
                          encode_lj_positions([(1, 1, 1), (1, 1, 1)])),
    "fault_empty": (encode_lj_configuration(1.0, 1.0, 5.0), b""),
    "fault_badcfg": (b"\x00" * 5, encode_lj_positions([(0, 0, 0), (1.5, 0, 0)])),
    "fault_badinput": (encode_lj_configuration(1.0, 1.0, 5.0), b"\x00" * 10),
}


@pytest.mark.parametrize("case_name", sorted(LJ_GOLDEN))
def test_lj_is_byte_identical_to_the_pre_refactor_baseline(case_name, cli_path):
    params, payload = _CASES[case_name]
    request = SCLRequest("lj_pairwise_energy_forces", "cpu", params, payload)
    result = run_scl_request(request, cli_path=cli_path)
    expected = LJ_GOLDEN[case_name]

    assert result.status == expected["status"]
    assert result.exit_code == expected["exit_code"]
    assert (result.output.hex() if result.output else None) == expected["output_hex"]
    assert result.output_identity == expected["output_identity"]
    assert result.computation_identity == expected["computation_identity"]
    assert result.detail == expected["detail"]
    assert request.operation_identity() == expected["operation_identity"]
    assert request.parameters_identity() == expected["parameters_identity"]
    assert request.input_identity() == expected["input_identity"]
    assert request.identity() == expected["request_identity"]


# --- the dispatch mechanism itself --------------------------------------

def test_both_operations_are_dispatchable(cli_path):
    """The generalization's actual point: two different operations answer
    from the same binary, through the same envelope, with no change to the
    protocol."""
    from scl.fourier import encode_fourier_configuration, encode_real_signal

    lj = run_scl_request(SCLRequest(
        "lj_pairwise_energy_forces", "cpu",
        encode_lj_configuration(1.0, 1.0, 5.0), encode_lj_positions([(0, 0, 0), (1.5, 0, 0)])),
        cli_path=cli_path)
    ft = run_scl_request(SCLRequest(
        "fourier_transform_1d", "cpu",
        encode_fourier_configuration(), encode_real_signal([1.0, 0.0, 0.0, 0.0])),
        cli_path=cli_path)
    assert lj.status == "completed"
    assert ft.status == "completed"


def test_unknown_operation_names_the_real_alternatives(cli_path):
    """The unknown-operation fault stays PROTOCOL (10) and now enumerates
    what the build actually supports, rather than a hardcoded sentence
    that could drift from the registry."""
    result = run_scl_request(
        SCLRequest("not_a_real_operation", "cpu", b"", b""), cli_path=cli_path)
    assert result.status == "halted"
    assert result.exit_code == 10
    assert "unknown operation" in result.detail
    assert "lj_pairwise_energy_forces" in result.detail
    assert "fourier_transform_1d" in result.detail


def test_operation_participates_in_identity(cli_path):
    """Two operations over identical bytes are different computations --
    the existing identity model already carries this, with no
    per-operation identity mechanism."""
    params, payload = encode_lj_configuration(1.0, 1.0, 5.0), encode_lj_positions([(0, 0, 0), (1.5, 0, 0)])
    a = SCLRequest("lj_pairwise_energy_forces", "cpu", params, payload)
    b = SCLRequest("fourier_transform_1d", "cpu", params, payload)
    assert a.operation_identity() != b.operation_identity()
    assert a.identity() != b.identity()
    assert a.parameters_identity() == b.parameters_identity()  # same bytes, by construction
    assert a.input_identity() == b.input_identity()


def test_metrics_are_generic_and_operation_shaped(cli_path):
    """`SCLResult.metrics` carries whatever the operation measured;
    `n_particles` is a compat convenience that is simply absent for an
    operation with no particles -- never faked."""
    from scl.fourier import encode_fourier_configuration, encode_real_signal

    lj = run_scl_request(SCLRequest(
        "lj_pairwise_energy_forces", "cpu",
        encode_lj_configuration(1.0, 1.0, 5.0), encode_lj_positions([(0, 0, 0), (1.5, 0, 0)])),
        cli_path=cli_path)
    assert lj.metrics["n_particles"] == 2
    assert lj.n_particles == 2
    assert "native_compute_seconds" in lj.metrics

    ft = run_scl_request(SCLRequest(
        "fourier_transform_1d", "cpu",
        encode_fourier_configuration(), encode_real_signal([1.0, 2.0, 3.0])),
        cli_path=cli_path)
    assert ft.metrics["n_samples"] == 3
    assert ft.metrics["transform_size"] == 3
    assert "n_particles" not in ft.metrics
    assert ft.n_particles is None


# --- per-operation STE descriptor ---------------------------------------

@requires_ste
def test_lj_descriptor_header_is_unchanged_and_fourier_differs():
    """The descriptor generalization must reproduce LJ's header
    BYTE-FOR-BYTE (so every existing LJ program_identity is untouched)
    while giving Fourier its own -- Fourier can never inherit LJ's
    program identity."""
    from scl.ste_adapter import (
        FOURIER_DESCRIPTOR_HEADER,
        SCL_DESCRIPTOR_HEADER,
        descriptor_header,
        operation_from_descriptor_header,
    )

    assert descriptor_header("lj_pairwise_energy_forces") == SCL_DESCRIPTOR_HEADER
    assert SCL_DESCRIPTOR_HEADER == b"ste.scl.lj-pairwise-energy-forces.v1"
    assert FOURIER_DESCRIPTOR_HEADER == b"ste.scl.fourier-transform-1d.v1"
    assert FOURIER_DESCRIPTOR_HEADER != SCL_DESCRIPTOR_HEADER
    for operation in ("lj_pairwise_energy_forces", "fourier_transform_1d"):
        assert operation_from_descriptor_header(descriptor_header(operation)) == operation


@requires_ste
def test_operation_changes_program_identity(cli_path):
    from execution.specification import ExecutionSpecification
    from scl.ste_adapter import scl_program_descriptor

    lj = ExecutionSpecification(scl_program_descriptor("scl-cli/0.1.0", "cpu",
                                                       "lj_pairwise_energy_forces"), b"", b"")
    ft = ExecutionSpecification(scl_program_descriptor("scl-cli/0.1.0", "cpu",
                                                        "fourier_transform_1d"), b"", b"")
    assert lj.program_identity() != ft.program_identity()
