#!/usr/bin/env python3
"""
rescale_bundle.py -- produce an EXACTLY-EQUIVALENT bundle whose residual stream
is scaled by a constant c, to lift the trained model off the CKKS bootstrap
noise floor.

WHY
---
Measured on the pod 2026-07-29 (results/gpu-verify-20260728/exp7_amp.jsonl):
CKKS bootstrap error is ABSOLUTE (~1.3e-3 at ring 2^16), not relative. So the
RELATIVE damage a bootstrap does is set entirely by the message magnitude:

    amp   0.028   0.1    0.3    1.5    5      20     100
    relErr@50 boots  127%  23.8%  6.05%  1.89%  0.96%  0.70%  13.9%

The trained model's layer-0 activations have RMS 0.0283 (emb.weight std 0.0309
=> ms ~8e-4, matching native_seeds.json L0.tm ms_range [5.1e-4, 1.5e-3]). That
sits at the far-left, worst column: the ciphertext is destroyed inside ~35
bootstraps. The synthetic gpu_full_layer harness runs at ~0.29 and survives.
This is the whole "synthetic works, real doesn't" gap.

WHY THIS IS EXACTLY EQUIVALENT
------------------------------
RMSNorm is scale-invariant: RMSNorm(c*h) == RMSNorm(h). The residual stream is
written by exactly three things -- the embedding, tm.wout and cm.wv -- so
scaling all three by c makes h_l -> c*h_l at every layer while every normalized
quantity is untouched:

    u  = RMSNorm(h)*g        unchanged
    x  = win(u)              unchanged
    s  = decay*s + b*x       unchanged   (b = 1-decay acts on normalized x)
    z  = c_tm*s + dd*x       unchanged
    h += wout(gate(z))       scaled by c  <- scale wout
    h += wv(poly(k,r))       scaled by c  <- scale wv
    logits = head(RMSNorm(h)*g_out)       unchanged (norm kills c; head untouched)

head.weight is TIED to emb.weight in the checkpoint but stored as a SEPARATE
tensor in the bundle, so scaling the embedding does not disturb the output
layer. Verified: np.array_equal(head.weight, emb.weight) is True, and the
bundle index lists them separately.

The calibrated rsqrt seed must follow the change of variable. With ms' = c^2*ms
we need y' = rsqrt(ms') = y/c, and the seed is linear in ms:
    y0' = a' + b'*ms' = a/c + (b/c^3)*(c^2 ms) = (a + b*ms)/c = y0/c
so  a -> a/c,  b -> b/c^3,  iters unchanged.

BUNDLE FORMAT
-------------
index line: "<name> <byte_offset> <numel> [dims...]"; payload is float64.
(Confirmed: L0.decay at 48 with 768 elems, L0.b at 6192 => 6144 = 768*8.)
"""
import argparse, json, math, os, shutil, sys
import numpy as np

# The bundle carries NO embedding matrix: the client embeds, and the bundle ships
# the already-embedded rows as "inputs" (T x d). That tensor IS h_0, so it is what
# gets scaled. "head" is a separate tensor and must NOT be touched (it is tied to
# emb.weight in the checkpoint, but the bundle stores them independently).
# "logits_ref" also stays untouched -- the whole point is that logits are invariant.
SCALE_EXACT = ("inputs",)
SCALE_SUFFIXES = (".tm.wout", ".cm.wv")


def read_index(path):
    rows = []
    for ln in open(path):
        p = ln.split()
        if len(p) < 3:
            continue
        rows.append((p[0], int(p[1]), int(p[2]), [int(x) for x in p[3:]]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="ml-eval/artifacts")
    ap.add_argument("--in-tag", default="native")
    ap.add_argument("--out-tag", default="scaled")
    ap.add_argument("--c", type=float, required=True, help="residual-stream scale")
    a = ap.parse_args()

    src_bin = os.path.join(a.in_dir, f"bundle_{a.in_tag}.bin")
    src_idx = os.path.join(a.in_dir, f"bundle_{a.in_tag}.index.txt")
    dst_bin = os.path.join(a.in_dir, f"bundle_{a.out_tag}.bin")
    dst_idx = os.path.join(a.in_dir, f"bundle_{a.out_tag}.index.txt")

    rows = read_index(src_idx)
    print(f"index: {len(rows)} tensors")
    print(f"copying {src_bin} -> {dst_bin} ...", flush=True)
    shutil.copyfile(src_bin, dst_bin)
    shutil.copyfile(src_idx, dst_idx)

    c = a.c
    mm = np.memmap(dst_bin, dtype=np.float64, mode="r+")
    scaled, seeds = [], []
    for name, off, numel, dims in rows:
        assert off % 8 == 0, f"{name}: offset {off} not 8-byte aligned"
        i0 = off // 8
        if name in SCALE_EXACT or name.endswith(SCALE_SUFFIXES):
            mm[i0:i0 + numel] *= c
            scaled.append((name, numel))
        elif name.endswith(".rsqrt"):
            blk = mm[i0:i0 + numel]
            a0, b0 = float(blk[0]), float(blk[1])
            blk[0] = a0 / c
            blk[1] = b0 / (c ** 3)
            seeds.append((name, a0, b0, float(blk[0]), float(blk[1]), float(blk[2])))
    mm.flush()
    del mm

    print(f"\nscaled by c={c} ({len(scaled)} tensors):")
    for n, k in scaled[:6]:
        print(f"   {n:34s} numel={k}")
    if len(scaled) > 6:
        print(f"   ... and {len(scaled)-6} more")
    print(f"\nrsqrt seeds rewritten ({len(seeds)}):  a->a/c, b->b/c^3")
    for n, a0, b0, a1, b1, it in seeds[:4]:
        print(f"   {n:14s} a {a0:12.4f} -> {a1:10.5f}   b {b0:14.2f} -> {b1:12.6f}  iters={it:.0f}")
    print(f"\nwrote {dst_bin}")
    print(f"      {dst_idx}")

    # report the new expected activation magnitudes
    seeds_json = os.path.join(a.in_dir, f"{a.in_tag}_seeds.json")
    if os.path.exists(seeds_json):
        d = json.load(open(seeds_json))
        lo = min(v["ms_range"][0] for v in d.values() if isinstance(v, dict) and "ms_range" in v)
        hi = max(v["ms_range"][1] for v in d.values() if isinstance(v, dict) and "ms_range" in v)
        print(f"\nactivation RMS across all norm sites:")
        print(f"   before: {math.sqrt(lo):.4f} .. {math.sqrt(hi):.4f}")
        print(f"   after : {c*math.sqrt(lo):.4f} .. {c*math.sqrt(hi):.4f}   "
              f"(measured good window ~0.3 .. 20)")


if __name__ == "__main__":
    main()
