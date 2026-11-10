"""Host-RAM probe (2026-11-05): what a PlannedRun twin's recorded probe tokens hold, per block, on the CPU.

Builds the 0.1B checkpoint's twin with ONE block (+ the tail stage) exactly as run_planned.py does
(make_model(capture=True, ...), Recorder values on), records the two probe tokens (decode: probe_rows; batched K:
batch_probe; prefill T: two T-token public chunks), and accounts every byte the BlockRec objects keep:
objs (the fake ciphertexts' slot vectors), pts (plaintext vectors), nodes (JSON-able op dicts, sized by pickling).
Prints one JSON line per (mode, token, stage). Usage: probe_host_ram.py decode | batch K | prefill T  [shed]  (shed: Recorder(shed=True), RWKV7_SHED_TWIN)
"""
import gc, json, os, pickle, subprocess, sys, time
import numpy as np

from perseus.examples.rwkv7 import common, ref
from perseus.examples.rwkv7.model import load_coeffs
from perseus.examples.rwkv7.run_planned import Recorder, probe_rows
from perseus.examples.rwkv7.run_capture import batch_probe
from perseus.examples.rwkv7.session import make_model
from perseus.examples.rwkv7.tests.conftest import CKPT, COEFFS
from perseus.examples.rwkv7.tests.test_ref import tokenizer


def rss_mb():
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode()) / 1024


def arr_bytes(a, seen):
    if not isinstance(a, np.ndarray):
        a = getattr(a, "vec", None)
        if not isinstance(a, np.ndarray):
            return 0
    base = a if a.base is None else a.base
    if id(base) in seen:
        return 0
    seen.add(id(base))
    return int(getattr(base, "nbytes", a.nbytes))


def account(r, seen):
    vb = sum(arr_bytes(o.vec, seen) for o in r.objs.values() if hasattr(o, "vec"))
    dtypes = sorted({str(o.vec.dtype) + str(o.vec.shape) for o in r.objs.values() if hasattr(o, "vec")})[:3]
    pb = sum(arr_bytes(v, seen) for v in r.pts.values())
    nb = len(pickle.dumps(r.nodes, protocol=5))
    return {"n_objs": len(r.objs), "obj_vec_bytes": vb, "obj_dtypes": dtypes, "n_pts": len(r.pts),
            "pt_bytes": pb, "n_nodes": len(r.nodes), "nodes_pickle_bytes": nb}


def main():
    mode = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 1
    P = ref.load_params(CKPT, int(os.environ.get('PROBE_BLOCKS', '1')))
    tok = tokenizer()
    ids = tok.encode(common.DEFAULT_TEXT)
    _rf, st = common.prefix_state(P, COEFFS, ids, 1)
    r0 = rss_mb()
    kw = {}
    if mode == "batch":
        kw = dict(batch=n, batch_carry="lane")
    elif mode == "prefill":
        kw = dict(chunk=n)
    _fhe, _inf, model = make_model(P, load_coeffs(COEFFS), capture=True, **kw)
    model.init_state = st
    r1 = rss_mb()
    print(json.dumps({"mode": mode, "n": n, "rss_before_model_mb": r0, "rss_after_model_mb": r1}), flush=True)
    shed = "shed" in sys.argv[3:] or (len(sys.argv) > 2 and sys.argv[2] == "shed")
    rec = Recorder(model, model.n_layers, shed=shed)
    label = mode + ("+shed" if shed else "")
    long_ids = tok.encode(common.CALIB_TEXT_LONG)[1:]
    if mode == "decode":
        rows = probe_rows(P, tok.encode(common.CALIB_TEXT), ids[:1])
    elif mode == "batch":
        rows, _t = batch_probe(P, long_ids, n, 2)
    else:
        rows = [[P.emb[t] for t in long_ids[i * n:(i + 1) * n]] for i in range(2)]
    recs, seen = {}, set()
    for t in (0, 1):
        gc.collect()
        a = rss_mb()
        t0 = time.time()
        recs[t] = rec.token(t, rows[t])
        gc.collect()
        b = rss_mb()
        for s, r in recs[t].items():
            out = {"mode": label, "n": n, "token": t, "stage": s, **account(r, seen)}
            print(json.dumps(out), flush=True)
        print(json.dumps({"mode": label, "n": n, "token": t, "rss_delta_mb": round(b - a, 1), "rss_mb": round(b, 1),
                          "record_s": round(time.time() - t0, 1)}), flush=True)
    from perseus.examples.rwkv7 import run_planned as _rp
    print(json.dumps({"mode": label, "n": n, "shed_stats": rec.shed_stats, "pt_intern_distinct": len(_rp._PT_INTERN),
                      "pt_intern_bytes": sum(v.nbytes for v in _rp._PT_INTERN.values())}), flush=True)
    fp = subprocess.check_output([os.path.join(os.path.dirname(__file__), "../../../../tools/memguard.sh"), "--probe",
                                  str(os.getpid())]).decode().strip()
    print(json.dumps({"mode": label, "n": n, "blocks": model.n_layers, "footprint_after_two_tokens": json.loads(fp)}),
          flush=True)
    # the same drop the cut would make: objs (and their vectors) and nodes released, pts kept
    a = rss_mb()
    for rr in recs.values():
        for r in rr.values():
            r.objs = None
            r.nodes = None
    rec.blocks = {}
    gc.collect()
    print(json.dumps({"mode": label, "n": n, "after_drop_objs_nodes_rss_mb": round(rss_mb(), 1),
                      "freed_mb": round(a - rss_mb(), 1)}), flush=True)


if __name__ == "__main__":
    main()
