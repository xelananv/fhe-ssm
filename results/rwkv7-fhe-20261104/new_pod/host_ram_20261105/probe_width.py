"""Host-RAM probe at other widths (2026-11-05): a synthetic 1-block checkpoint (synth.py, as test_anysize) of width C,
its twin recorded with the shed ON; the unshed holding is read from the shed's own count (vectors shed + vectors kept
+ plaintext bytes replaced), so the process never holds it. Usage: probe_width.py C decode|batch K|prefill T"""
import json, sys
import numpy as np

from perseus.examples.rwkv7 import common
from perseus.examples.rwkv7.model import load_coeffs
from perseus.examples.rwkv7.run_planned import Recorder
from perseus.examples.rwkv7.run_capture import batch_probe
from perseus.examples.rwkv7.session import make_model
from perseus.examples.rwkv7.tests.test_anysize import synthetic

C, mode = int(sys.argv[1]), sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
P, cj, ids = synthetic(C, 1)
co = load_coeffs(cj)
_rf, st = common.prefix_state(P, cj, ids, 1)
kw = {"batch": dict(batch=n, batch_carry="lane"), "prefill": dict(chunk=n)}.get(mode, {})
_f, _i, model = make_model(P, co, capture=True, **kw)
model.init_state = st
rec = Recorder(model, model.n_layers, shed=True)
body = list(ids[1:]) * 8
if mode == "decode":
    rows = [P.emb[t] for t in body[:2]]
elif mode == "batch":
    rows, _t = batch_probe(P, body, n, 2)
else:
    rows = [[P.emb[t] for t in body[i * n:(i + 1) * n]] for i in range(2)]
blocks = {}
for t in (0, 1):
    for b, r in rec.token(t, rows[t]).items():
        blocks.setdefault(b, []).append((len(r.objs), len(r.pts), len(r.nodes)))
kept = 0
for t_recs in [rec.blocks]:
    pass
st_ = rec.shed_stats
from perseus.examples.rwkv7 import run_planned as rp
print(json.dumps({"width": C, "mode": mode, "n": n, "lanes": model.L.t, "shards": model.L.S, "blocks_objs_pts_nodes": blocks,
                  "shed_stats": st_, "pt_intern_distinct": len(rp._PT_INTERN),
                  "unshed_two_tokens_bytes_lower": st_["bytes"] + st_["pt_bytes"]}), flush=True)
