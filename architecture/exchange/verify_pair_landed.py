#!/usr/bin/env python3
"""Did a coordinated reissue land in BOTH repositories, or only one?

WHY THIS EXISTS. It happened. The SCL half of the canonicalization reissue
reached its remote while the DAQ half did not, and for that window the two
repositories disagreed on the encoding their shared agreement fixture
pins. The push had been reported as failed when it had succeeded, because
the check tested for a FAILURE STRING in command output rather than for
the resulting STATE, and matched a stale `fatal` from earlier output.

So this asks the REMOTES, not the locals. A local commit proves
authorship; only the remote HEADs prove the pair landed together. And it
compares CONTENT, because two repositories can both be pushed and still
disagree on the bytes that matter.

Usage:  python3 verify_pair_landed.py <repo-a> <repo-b>
Exit 0 only if both branches are in sync with their remotes AND every
shared file is byte-identical across the two.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

#: Files that MUST be byte-identical across the pair. The serializer is
#: byte-identical by agreement, and that agreement is what makes any
#: artifact digest mean anything.
SHARED = (
    "architecture/decisions/2026-08-26-joint-workload-decision.sha256",
    "architecture/decisions/2026-08-26-joint-workload-decision.yaml",
    "architecture/kalman_validation_preregistration.yaml",
    "architecture/proof_integrity.yaml",
    "architecture/exchange/canonical_yaml.py",
    "architecture/exchange/canonical_yaml.sha256",
    "architecture/exchange/canonicalization_fixture.yaml",
    "architecture/exchange/canonicalization_fixture.sha256",
    "architecture/exchange/daq_requirement_response.yaml",
    "architecture/exchange/daq_requirement_response.sha256",
    "architecture/exchange/scl_requirements.yaml",
    "architecture/exchange/scl_requirements.sha256",
)
#: Three entries were added in the second coordinated reissue, and each was
#: missing for the same reason: this list was written from the files the
#: FIRST reissue happened to touch, not from the files the pair actually
#: shares. The serializer's own source pin is shared by construction. The
#: joint decision record is byte-identical by agreement and binds the
#: fixture hash -- so a fixture change reissues it, and a check that did
#: not look at it would have called the pair landed while the two records
#: disagreed. Measured: they HAD diverged once already, when one clone was
#: three commits stale and held the pre-correction record.


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True).stdout.strip()


def in_sync_with_remote(repo):
    branch = _git(repo, "branch", "--show-current")
    local = _git(repo, "rev-parse", "HEAD")
    line = _git(repo, "ls-remote", "origin", f"refs/heads/{branch}")
    remote = line.split("\t")[0] if line else ""
    return branch, local, remote, bool(remote) and local == remote


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    repos = [pathlib.Path(a).resolve() for a in argv]
    ok = True

    print("=== remote sync (asks the remote, not the local) ===")
    for repo in repos:
        branch, local, remote, synced = in_sync_with_remote(repo)
        state = "IN-SYNC" if synced else ("UNPUSHED/DIVERGED" if remote else "NO REMOTE BRANCH")
        print(f"  {repo.name:34} {branch}\n      local={local[:12]} remote={remote[:12] or '-'}  {state}")
        ok &= synced

    print("\n=== shared files byte-identical across the pair ===")
    for relative in SHARED:
        digests = []
        for repo in repos:
            path = repo / relative
            digests.append(hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None)
        same = len(set(digests)) == 1 and digests[0] is not None
        missing = [r.name for r, d in zip(repos, digests) if d is None]
        note = f"  (absent in {', '.join(missing)})" if missing else ""
        print(f"  {'OK  ' if same else 'DIFF'} {relative}{note}")
        ok &= same

    print("\n" + ("PAIR LANDED" if ok else "PAIR NOT LANDED -- do not treat the reissue as complete"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
