"""identify_bundle_checkpoint.py -- which checkpoint is actually INSIDE a bundle?

The FHE harness executes `bundle_<tag>.bin`, not `<tag>_weights.npz`. Those are
separate artifacts written at separate times, and on 2026-08-02 three of eight
groups were found desynchronised. Matching a bundle's *recorded perplexity*
identifies the eval, NOT the bundle -- so a run's quality label is only sound if
the bundle's actual bytes are traced to a specific .npz.

This reads a tensor straight out of the bundle at the offset its index gives and
compares it against every candidate .npz on disk. Exact identification, no
inference, no GPU, seconds per bundle (it seeks; it does not load the file).

    python ml-eval/identify_bundle_checkpoint.py                # all bundles
    python ml-eval/identify_bundle_checkpoint.py agnd768        # one
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_DIRS = [os.path.join(ROOT, "results/a100-secure-20260730/artifacts"),
            os.path.join(ROOT, "ml-eval/artifacts")]

# a small tensor present in every bundle, big enough to be a fingerprint
PROBE = "blocks.0.tm.win.weight"
BUNDLE_KEY = "l0.tm.win"          # the bundle's own naming for it


def read_index(idx_path):
    out = {}
    for line in open(idx_path, errors="ignore"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue                      # provenance header (2026-08-02+)
        parts = line.split()
        if len(parts) < 3:
            continue
        name, off, nel = parts[0], parts[1], parts[2]
        try:
            out[name] = (int(off), int(nel), [int(x) for x in parts[3:]])
        except ValueError:
            continue
    return out


def read_tensor(bin_path, off_bytes, nel):
    with open(bin_path, "rb") as f:
        f.seek(off_bytes)
        return np.frombuffer(f.read(nel * 8), dtype=np.float64, count=nel)


def candidates(tag):
    """every .npz that could plausibly be this bundle's source"""
    base = tag.split("_")[0]
    out = []
    for d in ART_DIRS:
        for p in glob.glob(os.path.join(d, f"{base}*_weights*.npz")):
            if not os.path.islink(p):
                out.append(p)
    return sorted(set(out))


def identify(tag):
    bin_path = idx_path = None
    for d in ART_DIRS:
        b, i = os.path.join(d, f"bundle_{tag}.bin"), os.path.join(d, f"bundle_{tag}.index.txt")
        if os.path.exists(b) and os.path.exists(i):
            bin_path, idx_path = b, i
            break
    if not bin_path:
        print(f"{tag}: no bundle on disk"); return

    idx = read_index(idx_path)
    key = BUNDLE_KEY if BUNDLE_KEY in idx else next(
        (k for k in idx if k.endswith("tm.win") or k.endswith("win")), None)
    if key is None:
        print(f"{tag}: no win tensor in index (keys e.g. {list(idx)[:4]})"); return
    off, nel, shape = idx[key]
    ref = read_tensor(bin_path, off, nel)

    print(f"\n=== bundle_{tag}.bin  [{key}] {shape}  {nel} elements ===")
    best = None
    for c in candidates(tag):
        try:
            with np.load(c) as z:
                if PROBE not in z:
                    continue
                w = np.asarray(z[PROBE], dtype=np.float64).ravel()
        except Exception as e:
            print(f"  {os.path.basename(c):<38} unreadable ({type(e).__name__})"); continue
        if w.size != ref.size:
            print(f"  {os.path.basename(c):<38} shape mismatch ({w.size} vs {ref.size})"); continue
        denom = max(float(np.abs(ref).max()), 1e-30)
        rel = float(np.abs(w - ref).max()) / denom
        verdict = "*** MATCH ***" if rel < 1e-6 else ""
        print(f"  {os.path.basename(c):<38} max rel diff {rel:.3e}  {verdict}")
        if best is None or rel < best[1]:
            best = (c, rel)
    if best and best[1] < 1e-6:
        print(f"  -> bundle contains {os.path.basename(best[0])}")
    else:
        print("  -> NO candidate matches. The source .npz is not on disk.")


if __name__ == "__main__":
    tags = sys.argv[1:] or ["agnd768", "agnd2048", "agnd256", "agnd768b", "agnd768w", "agnd768q2"]
    for t in tags:
        identify(t)
