"""Host-RAM probe (2026-11-05): the bytes of numpy arrays reachable from Params, a twin (make_model capture=True) and
the mirror, for a checkpoint truncated to n blocks. Usage: probe_weights.py n_blocks [batch K | prefill T]"""
import gc, json, sys
import numpy as np

from perseus.examples.rwkv7 import ref
from perseus.examples.rwkv7.model import load_coeffs, Mirror
from perseus.examples.rwkv7.session import make_model
from perseus.examples.rwkv7.tests.conftest import CKPT, COEFFS


def reach_bytes(root, seen):
    """numpy bytes reachable from root through __dict__/__slots__/containers, each base buffer counted once."""
    tot, stack, vis = 0, [root], set()
    while stack:
        o = stack.pop()
        if id(o) in vis:
            continue
        vis.add(id(o))
        if isinstance(o, np.ndarray):
            b = o
            while isinstance(b.base, np.ndarray):
                b = b.base
            if id(b) not in seen:
                seen.add(id(b)); tot += b.nbytes
            continue
        if isinstance(o, (str, bytes, int, float, complex, bool, type(None), type)) or callable(o) and not hasattr(o, "__dict__"):
            continue
        if isinstance(o, dict):
            stack.extend(o.values()); continue
        if isinstance(o, (list, tuple, set, frozenset)):
            stack.extend(o); continue
        d = getattr(o, "__dict__", None)
        if d is not None:
            stack.extend(d.values())
        for s in getattr(type(o), "__slots__", ()) or ():
            if hasattr(o, s):
                stack.append(getattr(o, s))
    return tot


def main():
    nb = int(sys.argv[1])
    kw = {}
    if len(sys.argv) > 3:
        kw = {"batch": dict(batch=int(sys.argv[3]), batch_carry="lane"), "prefill": dict(chunk=int(sys.argv[3]))}[sys.argv[2]]
    P = ref.load_params(CKPT, nb)
    co = load_coeffs(COEFFS)
    seen = set()
    p_b = reach_bytes(P, seen)
    _f, _i, model = make_model(P, co, capture=True, **kw)
    twin_b = reach_bytes(model, seen)          # beyond what Params already holds
    lins = model.linears()
    wd = sum(np.asarray(l.Wd).nbytes for l in lins.values())
    mir = Mirror(P, co, model.L.n, lanes=0)
    mir_b = reach_bytes(mir, seen)
    print(json.dumps({"blocks": nb, "kw": {k: v for k, v in kw.items()}, "params_bytes": p_b,
                      "twin_extra_bytes": twin_b, "twin_pcmm_Wd_bytes": wd, "n_linears": len(lins),
                      "mirror_extra_bytes": mir_b, "C": P.C, "emb_bytes": int(np.asarray(P.emb).nbytes)}), flush=True)


if __name__ == "__main__":
    main()
