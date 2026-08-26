"""The shared serializer's own source digest.

The fixture agreement check catches any emitter change that alters what
the fixture SERIALIZES TO. It cannot catch an edit that leaves the
fixture's output unchanged -- a comment, a refactor, or behaviour for a
shape the fixture does not cover -- and those edits still make the two
repositories' copies differ, which is the thing byte-identity-by-agreement
exists to prevent.

So the file's own digest is pinned, identically in both repositories.
Either side's suite now catches a local edit WITHOUT needing the other
tree present, which is what makes this runnable in both CI paths rather
than only in the cross-repo check.

If this fails legitimately -- because the serializer is being changed on
purpose -- the change is a COORDINATED REISSUE: update both repositories,
regenerate every artifact, reissue every record carrying a digest, and
verify with architecture/exchange/verify_pair_landed.py that BOTH remotes
landed it. Updating this digest alone is the one thing that is never the
right fix.
"""

from __future__ import annotations

import hashlib
import pathlib

EXCHANGE = pathlib.Path(__file__).resolve().parent.parent / "architecture" / "exchange"


def test_the_shared_serializer_matches_its_pinned_digest():
    source = EXCHANGE / "canonical_yaml.py"
    recorded = (EXCHANGE / "canonical_yaml.sha256").read_text().strip()
    actual = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual == recorded, (
        "the shared serializer has been edited on one side. This is a coordinated "
        "reissue, not a local fix: change both repositories, regenerate every "
        "artifact, reissue every record carrying a digest, and confirm with "
        "verify_pair_landed.py that both REMOTES landed it."
    )


def test_the_pin_is_not_self_referential():
    """The digest file must not be inside what it digests, or it could
    never be satisfied -- and a check that cannot pass is as useless as
    one that cannot fail."""
    source = (EXCHANGE / "canonical_yaml.py").read_text()
    assert "canonical_yaml.sha256" not in source


def test_the_agreement_fixture_is_still_the_primary_check():
    """The pin is a SECOND line, not a replacement. The fixture is what
    proves the two encodings agree in behaviour; this only proves the
    source has not drifted."""
    assert (EXCHANGE / "canonicalization_fixture.yaml").exists()
    assert (EXCHANGE / "canonicalization_fixture.sha256").exists()
