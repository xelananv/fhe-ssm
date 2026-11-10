"""Which recorded ciphertexts keep their vectors after the shed (prefill T, 1 block): names and who references them."""
import collections, gc, re, sys
import numpy as np
from perseus.examples.rwkv7 import common, ref
from perseus.examples.rwkv7.model import load_coeffs
from perseus.examples.rwkv7.run_planned import Recorder
from perseus.examples.rwkv7.session import make_model
from perseus.examples.rwkv7.tests.conftest import CKPT, COEFFS
from perseus.examples.rwkv7.tests.test_ref import tokenizer

T = int(sys.argv[1])
P = ref.load_params(CKPT, 1)
tok = tokenizer()
ids = tok.encode(common.DEFAULT_TEXT)
_rf, st = common.prefix_state(P, COEFFS, ids, 1)
_f, _i, model = make_model(P, load_coeffs(COEFFS), capture=True, chunk=T)
model.init_state = st
rec = Recorder(model, model.n_layers, shed=True)
long_ids = tok.encode(common.CALIB_TEXT_LONG)[1:]
rows = [[P.emb[t] for t in long_ids[i * T:(i + 1) * T]] for i in range(2)]
recs = {t: rec.token(t, rows[t]) for t in (0, 1)}
def count(r):
    return sum(1 for o in r.objs.values() if getattr(o, "vec", None) is not None and o.vec.shape[0] > 1)


print("kept before re-shed: token0", count(recs[0][0]), "token1", count(recs[1][0]))
gc.collect()
for t in (0, 1):
    for rr in recs[t].values():
        rec._shed(rr, final=True)
print("kept after re-shed: token0", count(recs[0][0]), "token1", count(recs[1][0]))


def holders_of(r):
    kept = [n for n, o in r.objs.items() if getattr(o, "vec", None) is not None and o.vec.shape[0] > 1]
    print("kept", len(kept), "of", len(r.objs), collections.Counter(re.sub(r"\d+", "#", n) for n in kept).most_common(8))
    hc = collections.Counter()
    for n in kept:
        for h in gc.get_referrers(r.objs[n]):
            if h is r.objs:
                continue
            if isinstance(h, dict):
                ks = [str(k)[:24] for k, v in h.items() if v is r.objs[n]][:1]
                hc[("dict", len(h), tuple(re.sub(r"\d+", "#", k) for k in ks))] += 1
            elif isinstance(h, (list, tuple)):
                hc[(type(h).__name__, len(h))] += 1
            else:
                hc[(type(h).__name__,)] += 1
    for k, v in hc.most_common(12):
        print(" ", v, k)


holders_of(recs[1][0])
holders_of(recs[0][0])
r = recs[1][0]
kn = [n for n, o in r.objs.items() if getattr(o, "vec", None) is not None and o.vec.shape[0] > 1]
print("shapes", collections.Counter((r.objs[n].vec.shape, r.objs[n].vec.dtype.str, type(r.objs[n]).__name__) for n in kn).most_common(5))
print("refcounts", collections.Counter(sys.getrefcount(r.objs[n]) for n in kn).most_common(6))
