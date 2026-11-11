# SPDX-License-Identifier: MIT
"""Plaintext range scan (P6): run the torch poly model UNCLAMPED (as a ciphertext evaluates it) over N tokens of a
text file and count, per site kind, the tokens whose site input left the site's valid range, with the worst excursion.

Valid range: [lo, hi] of the coefficients JSON -- the Chebyshev interval of a polynomial site, the accuracy range of a
`newton_const` site (whose convergence region (conv_lo, conv_hi) is counted separately: leaving THAT diverges), the
fitted interval of an old `newton` site. A token counts once per site kind if any channel / head of any layer of that
kind left the range at that position. Non-finite inputs count as out and are reported.

  python scan_ranges.py --model M.pth --coeffs coeffs_mixed_nc.json --text wiki.test.raw --tokens 65536 \
      --ctx 4096 --out scan.json [--feed exact]

--feed poly (default): every site evaluated by its polynomial, unclamped -- what the circuit sees, excursions compound.
--feed exact: sites evaluated exactly, inputs only checked -- the model's own activations, no compounding.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from common import batches, tokenize_file, windows
from rwkv7_model import Sites, load_rwkv7, pick_device, site_func, site_short
from rwkv_tokenizer import RWKVTokenizer


class ScanSites(Sites):
    def __init__(self, feed: str = "poly"):
        super().__init__("exact")
        self.feed = feed
        self.kinds: dict[str, dict] = {}
        self.tok_bad: dict[str, torch.Tensor] = {}      # kind -> (B, T) bool, for the current batch
        self.tok_conv: dict[str, torch.Tensor] = {}

    def __call__(self, name, x):
        if self.mode == "poly" and name in self.polys:
            self._check(name, x)
            if self.feed == "exact":
                with torch.autocast(device_type=x.device.type, enabled=False):
                    return site_func(name)(x.float())
        return super().__call__(name, x)

    def _check(self, name, x):
        p = self.polys[name]
        k = site_short(name)
        with torch.no_grad():
            v = x.detach().double()
            B, T = (v.shape[0], v.shape[1]) if v.dim() >= 2 else (1, v.numel())
            v = v.reshape(B, T, -1)
            fin = torch.isfinite(v)
            bad = ~fin | (v < p["lo"]) | (v > p["hi"])
            tb = bad.any(-1)
            self.tok_bad[k] = tb | self.tok_bad.get(k, torch.zeros_like(tb))
            d = self.kinds.setdefault(k, {"lo": p["lo"], "hi": p["hi"], "method": p.get("method", "cheb"),
                                          "min": math.inf, "max": -math.inf, "nonfinite": 0, "tokens_out": 0,
                                          "tokens_conv_out": 0, "worst_site": None, "worst_excursion": 0.0})
            d["lo"], d["hi"] = min(d["lo"], p["lo"]), max(d["hi"], p["hi"])
            d["nonfinite"] += int((~fin).sum())
            vf = v[fin]
            if vf.numel():
                d["min"], d["max"] = min(d["min"], vf.min().item()), max(d["max"], vf.max().item())
                w = p["hi"] - p["lo"]
                exc = max((vf.max().item() - p["hi"]) / w, (p["lo"] - vf.min().item()) / w, 0.0)
                if exc > d["worst_excursion"]:
                    d["worst_excursion"], d["worst_site"] = exc, name
            if p.get("method") in ("newton_const", "newton_scaled"):
                cb = (~fin | (v <= p["conv_lo"]) | (v >= p["conv_hi"])).any(-1)
                self.tok_conv[k] = cb | self.tok_conv.get(k, torch.zeros_like(cb))

    def end_batch(self, mask):
        for k, tb in self.tok_bad.items():
            self.kinds[k]["tokens_out"] += int((tb & mask).sum())
        for k, tb in self.tok_conv.items():
            self.kinds[k]["tokens_conv_out"] += int((tb & mask).sum())
        self.tok_bad, self.tok_conv = {}, {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--coeffs", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--tokens", type=int, default=65536)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--feed", choices=["poly", "exact"], default="poly")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    tok = RWKVTokenizer(a.vocab) if a.vocab else RWKVTokenizer()
    sites = ScanSites(a.feed)
    model = load_rwkv7(a.model, device=dev, sites=sites)
    sites.load_poly(json.load(open(a.coeffs)), clamp=False)
    sites.mode = "poly"
    wins = windows(tokenize_file(a.text, tok, a.tokens), a.ctx)
    n, t0 = 0, time.time()
    with torch.no_grad():
        for x, y in batches(wins, a.batch):
            x, y = x.to(dev), y.to(dev)
            model(x)
            m = y != -100
            sites.end_batch(m)
            n += int(m.sum())
    res = {"model": a.model, "coeffs": a.coeffs, "text": a.text, "tokens": n, "ctx": a.ctx, "feed": a.feed,
           "seconds": time.time() - t0, "kinds": sites.kinds}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{n} tokens, feed={a.feed}, unclamped")
    print(f"{'kind':18s} {'tok_out':>8s} {'frac':>9s} {'conv_out':>8s} {'nonfin':>7s} {'range':>24s} "
          f"{'seen':>24s} worst")
    for k, d in sorted(sites.kinds.items()):
        print(f"{k:18s} {d['tokens_out']:8d} {d['tokens_out'] / max(n, 1):9.2e} {d['tokens_conv_out']:8d} "
              f"{d['nonfinite']:7d} [{d['lo']:10.3g},{d['hi']:10.3g}] [{d['min']:10.3g},{d['max']:10.3g}] "
              f"{d['worst_excursion']:.3g}x width at {d['worst_site']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
