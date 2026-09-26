#!/usr/bin/env python3
"""
patch_mgpu_key_decomp.py — stop FIDESlib from allocating a full, replicated,
never-read copy of every key-switching key's DECOMP limbs on EVERY device
under multi-GPU (2026-09-03, demo-pod prep, two-card route).

What the source does (FIDESlib main @786c760):
  * KeySwitchingKey::Initialize (src/CKKS/KeySwitchingKey.cu:20-35) calls
    a.generateDecompAndDigit(true) / b.generateDecompAndDigit(true) on every
    partition, then under GPUid.size()>1 grows the main `limb` set and calls
    loadDecompDigit.
  * LimbPartition::generateAllDecompAndDigit(iskey=true)
    (src/CKKS/LimbPartition.cu:863-919) allocates DECOMPlimb[i][j] for EVERY
    entry of DECOMPmeta -- and DECOMPmeta[d] holds every Q limb with
    digit==d across ALL devices (Context.cu:145-171) -- i.e. L+1 constant
    limbs of N*8 bytes per polynomial per device, replicated.
  * LimbPartition::loadDecompDigit (:1050-1107): the single-GPU branch loads
    DECOMPlimb; the multi-GPU branch loads only the main `limb` set and
    DIGITlimb. The key's DECOMP limbs are therefore never written.
  * The multi-GPU key switch (src/CKKS/LimbPartitionMGPU.cu:428-432) reads
    ksk.DIGITlimbptr and ksk.limbptr only; the DECOMPlimbptr it reads at
    :1176-1205 belongs to the OPERAND ciphertext (`this`), not to the key.
  So under multi-GPU each key carries 2 * (L+1) * N * 8 bytes of dead
  allocation per device: at N=2^17, L+1=42 that is 84 MiB per key per card,
  143 keys -> 12 GB per card of the 96 GB the demo has to fit in.

The patch: in generateAllDecompAndDigit, skip the DECOMP generation loop when
`iskey && cc.GPUid.size() > 1` (DECOMPlimb is still resized so every
consumer that only sizes it keeps working). Single-GPU behaviour is
byte-for-byte unchanged; the multi-GPU numerics are unchanged because the
skipped limbs were never loaded or read. VERIFY ON THE BOX: the fit test's
fidelity fields (top1Agree, relErrRms) under --devices 0,1 must match the
single-device values, and `Rotation keys loaded` MB per card must fall.

Usage: python3 patch_mgpu_key_decomp.py <FIDESlib checkout>/src/CKKS/LimbPartition.cu
Idempotent: exit 0 if already patched; exit 1 (loud) if the anchor is not
found, because a silently unpatched library would be indistinguishable from
a patched one until the OOM.
"""
import pathlib
import sys

MARKER = "FHE_SSM_MGPU_KEY_DECOMP_SKIP"
ANCHOR = "\t\t// generateAllDecompLimb(bufferDECOMPandDIGIT, 0);\n\t\tgenerateGatherLimb(iskey);\n\t\tDECOMPlimb.resize(DECOMPmeta.size());\n\t\tfor (size_t i = 0; i < DECOMPmeta.size(); ++i) {\n"
REPLACEMENT = (
    "\t\t// generateAllDecompLimb(bufferDECOMPandDIGIT, 0);\n"
    "\t\tgenerateGatherLimb(iskey);\n"
    "\t\tDECOMPlimb.resize(DECOMPmeta.size());\n"
    "\t\t// " + MARKER + " (fhe-ssm, 2026-09-03): under multi-GPU a KEY's DECOMP limbs\n"
    "\t\t// are never loaded (loadDecompDigit mgpu branch) nor read (the mgpu key\n"
    "\t\t// switch uses DIGITlimbptr/limbptr); allocating them replicated L+1\n"
    "\t\t// limbs per polynomial per device (12 GB/card at N=2^17, 143 keys).\n"
    "\t\tconst bool skipKeyDecomp = iskey && cc.GPUid.size() > 1;\n"
    "\t\tfor (size_t i = 0; !skipKeyDecomp && i < DECOMPmeta.size(); ++i) {\n"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_mgpu_key_decomp.py <LimbPartition.cu path>", file=sys.stderr)
        return 2
    path = pathlib.Path(sys.argv[1])
    src = path.read_text()
    if MARKER in src:
        print(f"{path}: already patched, skipping")
        return 0
    n = src.count(ANCHOR)
    if n != 1:
        print(f"{path}: anchor found {n} times (want 1) -- FIDESlib drifted; refusing to guess. "
              f"Re-derive the patch against this checkout (see the docstring).", file=sys.stderr)
        return 1
    path.write_text(src.replace(ANCHOR, REPLACEMENT))
    print(f"{path}: patched ({MARKER})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
