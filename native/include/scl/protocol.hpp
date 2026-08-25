#pragma once
// The scl_cli wire protocol: one JSON object read from stdin, one JSON
// object written to stdout, one process exit per invocation (subprocess
// boundary, matching how STE's Rust execution-cli and execution/gromacs.py
// are both invoked -- see docs/SCL_ARCHITECTURE.md "why a process boundary
// and not FFI"). This header names the fault codes as a single source of
// truth shared by main.cpp and the native tests.

namespace scl {

// Process exit code stays 0 whenever the CLI produced a well-formed JSON
// answer -- including a "halted" computation. A non-zero PROCESS exit
// code means the CLI could not even answer (crash, out-of-memory before
// JSON could be written); callers must not confuse that with a "halted"
// JSON status, which is a normal, fully-described outcome living in the
// "exit_code" JSON field below, never the process's own return code.
constexpr int kProcessExitOk = 0;
constexpr int kProcessExitCatastrophic = 1;

// JSON "exit_code" field values (the SCL-level fault vocabulary). 0 is
// success; every fault below is a distinct, distinguishable stage --
// never collapsed into one generic "execution failed" (Task 7 of the
// SCL Phase 1 brief).
constexpr int kFaultNone = 0;
constexpr int kFaultProtocol = 10;              // malformed/unreadable request envelope
constexpr int kFaultValidation = 11;             // structurally invalid parameters or input
constexpr int kFaultBackendUnavailable = 12;     // requested backend not usable in this build/host
constexpr int kFaultComputation = 13;            // the algorithm itself faulted (e.g. coincident particles)
constexpr int kFaultInternal = 14;               // unexpected exception, caught at the top level

}  // namespace scl
