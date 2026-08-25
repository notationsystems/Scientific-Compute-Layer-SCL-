"""Canonical YAML serialization for cross-repository exchange artifacts.

Pinned by the joint prompt's §2.1. Exchange artifacts are content-addressed
and YAML has no canonical serialization, so without a byte-exact encoding
the joint decision record's hashes are not reproducible and its whole
traceability guarantee fails.

    serialization: YAML 1.2, block style only
    keys:          sorted lexicographically at every level
    anchors/aliases: forbidden (this emitter cannot produce them)
    floats:        shortest round-trip repr; no padded zeros; exponent only
                   when |x| < 1e-4 or |x| >= 1e16
    strings:       double-quoted only where the spec requires it
    encoding:      UTF-8, LF line endings, single trailing newline
    hash:          sha256 over the serialized bytes
    reference:     "sha256:<hex>"

STDLIB ONLY, deliberately. PyYAML is installed here but is NOT used: its
output depends on the installed version's emitter, so two repositories
could hold the same data and disagree on bytes. This module is ~120 lines
with no dependencies precisely so the DAQ repository can vendor the
identical file. `canonicalization_fixture.yaml` + `.sha256` in this
directory are the shared agreement fixture -- run
`python3 canonical_yaml.py --verify-fixture` in either repository; both
must print the same digest before Phase 2 is considered complete.

ONE DOCUMENTED EXCEPTION to "block style only": an empty mapping or
sequence has no block form in YAML, so `{}` / `[]` are emitted. Prefer
omitting an empty field, or stating an explicit sentinel, over relying on
this.
"""

from __future__ import annotations

import hashlib
import math
import pathlib
import sys
from typing import Any, List, Mapping, Sequence

_INDICATORS = set("-?:,[]{}#&*!|>'\"%@`")
_RESERVED_PLAIN = {
    "true", "false", "null", "yes", "no", "on", "off", "~", "y", "n",
    "True", "False", "Null", "NULL", "TRUE", "FALSE", "None",
}


def _is_numeric_looking(text: str) -> bool:
    try:
        int(text)
        return True
    except ValueError:
        pass
    try:
        float(text)
        return True
    except ValueError:
        return False


def _needs_quoting(text: str) -> bool:
    """True when plain (unquoted) style would be unsafe or ambiguous."""
    if text == "":
        return True
    if text.strip() != text:
        return True
    if text in _RESERVED_PLAIN or text.lower() in _RESERVED_PLAIN:
        return True
    if _is_numeric_looking(text):
        return True
    if text[0] in _INDICATORS:
        return True
    if ": " in text or " #" in text or text.endswith(":"):
        return True
    if any(ch in text for ch in "\n\r\t\x00"):
        return True
    return False


def _quote(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _format_float(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"non-finite float is not canonically representable: {value!r}")
    magnitude = abs(value)
    use_exponent = (magnitude != 0.0 and magnitude < 1e-4) or magnitude >= 1e16
    if use_exponent:
        mantissa, _, exponent = repr(value).partition("e")
        if not exponent:  # repr chose positional; force exponent form
            mantissa, _, exponent = f"{value:e}".partition("e")
            mantissa = mantissa.rstrip("0").rstrip(".") or "0"
        # YAML 1.1's float grammar (which PyYAML's safe_load implements)
        # requires BOTH a decimal point in the mantissa and an explicit
        # exponent sign; without them `1e+16` and `1e-5` resolve as
        # STRINGS, silently turning a number into text in a
        # content-addressed artifact. Verified by the round-trip check in
        # tests/test_exchange_artifact.py, which caught exactly that.
        if "." not in mantissa:
            mantissa = f"{mantissa}.0"
        sign = "-" if exponent.startswith("-") else "+"
        digits = exponent.lstrip("+-").lstrip("0") or "0"
        return f"{mantissa}e{sign}{digits}"
    text = repr(float(value))
    if "e" in text or "E" in text:  # repr chose exponent but the rule says positional
        text = f"{value:.17f}".rstrip("0")
        if text.endswith("."):
            text += "0"
    return text


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, str):
        return _quote(value) if _needs_quoting(value) else value
    raise TypeError(f"unsupported scalar type for canonical YAML: {type(value).__name__}")


def _emit(value: Any, indent: int, lines: List[str]) -> None:
    pad = "  " * indent
    if isinstance(value, Mapping):
        if not value:
            lines[-1] += " {}"
            return
        for key in sorted(value.keys()):
            if not isinstance(key, str):
                raise TypeError(f"canonical YAML requires string keys, got {type(key).__name__}")
            child = value[key]
            key_text = _quote(key) if _needs_quoting(key) else key
            if isinstance(child, (Mapping, list, tuple)) and child:
                lines.append(f"{pad}{key_text}:")
                _emit(child, indent + 1, lines)
            elif isinstance(child, (Mapping, list, tuple)):
                lines.append(f"{pad}{key_text}:")
                _emit(child, indent + 1, lines)
            else:
                lines.append(f"{pad}{key_text}: {_format_scalar(child)}")
    elif isinstance(value, (list, tuple)):
        if not value:
            lines[-1] += " []"
            return
        for item in value:
            if isinstance(item, (Mapping, list, tuple)) and item:
                lines.append(f"{pad}-")
                _emit(item, indent + 1, lines)
                # collapse "-\n  key:" into "- key:" for the common case
                _collapse_dash(lines, indent)
            else:
                lines.append(f"{pad}- {_format_scalar(item)}")
    else:
        raise TypeError(f"unsupported node type: {type(value).__name__}")


def _collapse_dash(lines: List[str], indent: int) -> None:
    """Turn the two-line `-` / `  key: v` form into `- key: v`, which is the
    conventional block rendering and keeps the output stable."""
    pad = "  " * indent
    marker = f"{pad}-"
    for position in range(len(lines) - 1, -1, -1):
        if lines[position] == marker:
            if position + 1 < len(lines):
                following = lines[position + 1]
                stripped = following[len(pad) + 2:]
                lines[position] = f"{pad}- {stripped}"
                del lines[position + 1]
            return


def canonical_dump(document: Any) -> str:
    lines: List[str] = []
    _emit(document, 0, lines)
    return "\n".join(lines) + "\n"


def canonical_bytes(document: Any) -> bytes:
    return canonical_dump(document).encode("utf-8")


def canonical_sha256(document: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(document)).hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


FIXTURE = {
    "booleans": {"no": False, "yes": True},
    "empty_map_exception": {},
    "empty_seq_exception": [],
    "floats": {
        "large_exponent": 1e16,
        "one": 1.0,
        "small_exponent": 1e-05,
        "third": 0.3333333333333333,
        "threshold_positional": 0.0001,
    },
    "integers": {"negative": -7, "zero": 0},
    "nested": {"b": {"d": ["x", "y"], "c": 1}, "a": "first"},
    "nulls": {"absent": None},
    "sequence_of_maps": [{"k": 1}, {"k": 2}],
    "strings": {
        "colon_space": "needs: quoting",
        "hash": "trailing #comment-like",
        "numeric_looking": "1.5",
        "plain": "plain_scalar_is_unquoted",
        "reserved": "true",
        "unicode": "ångström",
    },
    "zzz_key_sorting_proof": "sorted last",
}


def main(argv: Sequence[str]) -> int:
    here = pathlib.Path(__file__).resolve().parent
    fixture_path = here / "canonicalization_fixture.yaml"
    digest_path = here / "canonicalization_fixture.sha256"
    if "--write-fixture" in argv:
        fixture_path.write_bytes(canonical_bytes(FIXTURE))
        digest_path.write_text(canonical_sha256(FIXTURE) + "\n")
        print(f"wrote {fixture_path.name} {canonical_sha256(FIXTURE)}")
        return 0
    produced = canonical_sha256(FIXTURE)
    print(f"produced: {produced}")
    if fixture_path.exists():
        on_disk = file_sha256(fixture_path)
        recorded = digest_path.read_text().strip() if digest_path.exists() else "<absent>"
        print(f"on disk:  {on_disk}")
        print(f"recorded: {recorded}")
        ok = produced == on_disk == recorded
        print("AGREEMENT" if ok else "DISAGREEMENT")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
