#pragma once
// Single source of truth for the SCL native binary's version string,
// echoed in every CLI response ("backend_version") and by `scl_cli
// --version`. Bump SCL_KERNEL_VERSION whenever lj_pairwise.cpp's
// algorithm changes in any way that could change its numerical output --
// the Python-side adapter folds this string into STE's `program` bytes
// (docs/SCL_CONTRACT.md), exactly as execution/gromacs.py folds
// `gmx --version` into its own program descriptor, so a kernel version
// bump is a NEW program identity, never a silent behavior change under
// an unchanged identity.

namespace scl {

inline constexpr const char* kVersionString = "scl-cli/0.1.0";

}  // namespace scl
