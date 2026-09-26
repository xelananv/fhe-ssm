#!/usr/bin/env python3
"""
patch_peerutils.py — compat shim for FIDESlib's src/PeerUtils.cu.

Root cause (confirmed against NVIDIA's own CUDA 12.4 release notes and
FIDESlib's GitHub history, 2026-07-11): `launch()` unconditionally uses
CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE, a CUDA-Graphs attribute
introduced in CUDA 12.4 (NVIDIA blog: "CUDA Toolkit 12.4 Enhances Support for
NVIDIA Grace Hopper and Confidential Computing" — device-side kernel node
update). FIDESlib's PR #20 (merged 2026-05-12) added CUDART_VERSION-gated
compat shims for CUDA-13-only graph-API signature changes elsewhere in this
same file, but did not guard this specific CUDA-12.4-only attribute. Present
in both `main` and the latest tag `v2.1.2` as of 2026-07-11 — confirmed via
GitHub raw fetch of both refs.

Symptom: build fails with "identifier CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_
KERNEL_NODE is undefined" on any toolkit < 12.4 (observed: CUDA 12.2.140 on
a 4x A100 RunPod instance).

Fix: wrap the attribute-array declaration in `#if CUDART_VERSION >= 12040`,
falling back to the 1-element array (just CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN)
on older toolkits — mirrors the versioning style of FIDESlib's own PR #20.
The omitted attribute is an opt-in performance path for repeated graph
kernel-node updates; omitting it changes the launch to the ordinary
non-device-updatable path, which is functionally equivalent for a
single-shot launch.

Usage: python3 patch_peerutils.py <path-to-FIDESlib-checkout>/src/PeerUtils.cu
Idempotent: skips (exit 0) if already patched or the source no longer
contains the target pattern (upstream may have fixed it — check first).
"""
import re
import sys
import pathlib

MARKER = "CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE"
GUARD = "CUDART_VERSION >= 12040"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_peerutils.py <PeerUtils.cu path>", file=sys.stderr)
        return 2
    path = pathlib.Path(sys.argv[1])
    src = path.read_text()

    if GUARD in src:
        print(f"{path}: already patched, skipping")
        return 0
    if MARKER not in src:
        print(f"{path}: marker not found — upstream may have fixed this "
              f"already; nothing to do")
        return 0

    # Bound the match to the `CUlaunchAttribute attr[] = { ... };` statement
    # containing the marker. Non-greedy + DOTALL is robust to whitespace/
    # formatting drift between FIDESlib versions; it does not require an
    # exact literal match of the surrounding source.
    pattern = re.compile(
        r"([ \t]*)CUlaunchAttribute attr\[\] = \{.*?" + re.escape(MARKER)
        + r".*?\}\s*\};\n",
        re.DOTALL,
    )
    m = pattern.search(src)
    if not m:
        print(f"{path}: marker present but the enclosing statement didn't "
              f"match the expected shape — manual patch needed", file=sys.stderr)
        return 1

    indent = m.group(1)
    guarded = (
        f"{indent}#if {GUARD}\n"
        f"{m.group(0)}"
        f"{indent}#else\n"
        f"{indent}// {MARKER} requires CUDA >= 12.4 (NVIDIA CUDA 12.4 release notes,\n"
        f"{indent}// device-side kernel node update). Omitted on older toolkits\n"
        f"{indent}// (fhe-ssm-research compat patch, 2026-07-11) — this is an opt-in\n"
        f"{indent}// perf path for repeated graph kernel-node updates; launch is\n"
        f"{indent}// functionally equivalent without it.\n"
        f"{indent}CUlaunchAttribute attr[] = {{ {{ .id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN, "
        f".value = {{ .memSyncDomain = CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE }} }} }};\n"
        f"{indent}#endif\n"
    )
    path.write_text(src[: m.start()] + guarded + src[m.end():])
    print(f"{path}: patched ({GUARD} guard added around the device-updatable "
          f"kernel-node attribute)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
