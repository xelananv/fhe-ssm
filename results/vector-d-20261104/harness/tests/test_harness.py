# SPDX-License-Identifier: MIT
import json
import os

import numpy as np
import torch

from conftest import RAW, make_reference_model
from rwkv7_model import RWKV7, RWKV7Config, Sites, drop_unused_ln0, load_rwkv7
from rwkv_tokenizer import RWKVTokenizer

EXPECTED_KINDS = {"ln0.rsqrt", "ln1.rsqrt", "ln2.rsqrt", "ln_out.rsqrt", "att.w_tanh", "att.decay", "att.v_sigmoid",
                  "att.a_sigmoid", "att.g_sigmoid", "att.kk_rsqrt", "att.gn_rsqrt", "ffn.relu2"}


def _mine_from_ref(ref):
    sd = drop_unused_ln0(ref.state_dict())
    m = RWKV7(RWKV7Config.from_state_dict(sd))
    m.load_state_dict(sd, strict=True)
    return m.eval()


def test_exact_equals_reference_fp32():
    ref, _ = make_reference_model(seed=1)
    mine = _mine_from_ref(ref)
    idx = torch.randint(0, 65536, (2, 37), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        a = ref(idx)
        b = mine(idx)
    err = (a - b).abs().max().item()
    scale = a.abs().max().item()
    print(f"max |ref - mine| = {err:.3e}  (max |logit| {scale:.3e})")
    assert err <= 1e-5 * max(1.0, scale)


def test_site_inventory():
    ref, cfg = make_reference_model(seed=2)
    mine = _mine_from_ref(ref)
    with torch.no_grad():
        mine(torch.zeros(1, 4, dtype=torch.long))
    kinds = {__import__("re").sub(r"^blocks\.\d+\.", "", n) for n in mine.sites.seen}
    assert kinds == EXPECTED_KINDS
    # value-residual gate only from layer 1 on (demo.py:273-276)
    assert "blocks.0.att.v_sigmoid" not in mine.sites.seen
    assert len(mine.sites.seen) == 2 + cfg["n_layer"] * 10 - 1


def test_tokenizer_roundtrip():
    tok = RWKVTokenizer(os.path.join(RAW, "rwkv_vocab_v20230424.txt"))
    s = "Hello world! The Eiffel tower is in the city of Paris. 東京 ünïcödé \n\t 12345"
    ids = tok.encode(s)
    assert tok.decode(ids) == s
    assert tok.encode(" A") == [300]   # vocab line 300: ' A'


def test_record_poly_and_scripts(ref_and_ckpt, text_files, tmp_path):
    import calibrate, eval as ev, fit
    _, ckpt = ref_and_ckpt
    txt, lam = text_files
    rng = str(tmp_path / "ranges.json")
    calibrate.main(["--model", ckpt, "--text", txt, "--tokens", "600", "--ctx", "64", "--batch", "4",
                    "--device", "cpu", "--out", rng])
    R = json.load(open(rng))
    assert set(R["meta"]["site_order"]) == set(R["sites"])
    for s in R["sites"].values():
        assert s["min"] <= s["p0.1"] <= s["p99.9"] <= s["max"]
    co = str(tmp_path / "coeffs.json")
    fit.main(["--ranges", rng, "--degree", "7", "--site-degree", "relu2=15", "--out", co])
    C = json.load(open(co))
    assert C["sites"]["ln0.rsqrt"].get("exact") and C["sites"]["ln_out.rsqrt"].get("exact")
    assert C["sites"]["blocks.1.ffn.relu2"]["degree"] == 15
    co2 = str(tmp_path / "coeffs_t.json")
    fit.main(["--ranges", rng, "--target-err", "1e-3", "--metric", "rel", "--max-degree", "40", "--out", co2])
    out = str(tmp_path / "ev.jsonl")
    common = ["--model", ckpt, "--wikitext", txt, "--lambada", lam, "--ctx", "64", "--max-tokens", "400",
              "--device", "cpu", "--out", out]
    ev.main(common + ["--mode", "exact"])
    ev.main(common + ["--mode", "poly", "--coeffs", co])
    ev.main(common + ["--mode", "poly", "--coeffs", co, "--clamp"])
    ev.main(common + ["--ablate", "--coeffs", co, "--also-full"])
    recs = [json.loads(l) for l in open(out)]
    assert recs[0]["config"] == "exact" and np.isfinite(recs[0]["wt_ppl"])
    # Unclamped polynomials explode outside their interval (a [p0.1, p99.9] interval leaves 0.2 % of inputs outside by
    # construction): non-finite there is the expected FHE behaviour, recorded, not a harness failure. Clamped and
    # one-site-at-a-time runs must stay finite.
    bad = [(r["config"], r.get("clamp"), r["wt_ppl"]) for r in recs
           if not np.isfinite(r["wt_ppl"]) and (r.get("clamp") or r["config"].startswith("ablate:"))]
    assert not bad, f"non-finite wt_ppl: {bad}"
    assert any(r.get("clamp") and np.isfinite(r["wt_ppl"]) for r in recs)
    assert 0.0 <= recs[1]["top1_agree"] <= 1.0
    abl = [r for r in recs if r["config"].startswith("ablate:")]
    assert len(abl) == len(EXPECTED_KINDS) - 2   # ln0 and ln_out are client-side, not ablated
    # exact line agrees with the ppl the fidelity pass recomputes in exact mode
    assert abs(recs[0]["wt_ppl"] - recs[1]["wt_ppl_exact_same_run"]) < 1e-3 * recs[0]["wt_ppl"]


def _ranges_from_model(model, idx):
    model.sites.mode = "record"
    with torch.no_grad():
        model(idx)
    R = model.sites.ranges()
    model.sites.mode = "exact"
    return R


def _coeffs(R, degree, margin):
    from fit import cheb_fit
    sites = {}
    for name, r in R.items():
        if name in ("ln0.rsqrt", "ln_out.rsqrt"):
            continue
        lo, hi = r["min"], r["max"]
        w = hi - lo
        lo, hi = lo - margin * w, hi + margin * w
        if name.endswith("rsqrt"):
            # the circuit's 1/sqrt: seed + Newton (fit.newton_fit), accurate at both degrees so the test isolates the rest
            from fit import newton_fit
            sites[name] = newton_fit(name, max(lo, 0.5 * r["min"]), hi, target_rel=1e-6, max_steps=40)
            continue
        sites[name] = {"lo": lo, "hi": hi, "coeffs": cheb_fit(name, lo, hi, degree).tolist()}
    return {"sites": sites}


def test_poly_high_degree_close_to_exact():
    ref, _ = make_reference_model(seed=4)
    mine = _mine_from_ref(ref)
    idx = torch.randint(0, 65536, (2, 48), generator=torch.Generator().manual_seed(5))
    R = _ranges_from_model(mine, idx)
    with torch.no_grad():
        ex = mine(idx)
    errs = {}
    for deg in (4, 64):
        mine.sites.load_poly(_coeffs(R, deg, 0.25), clamp=True)
        mine.sites.mode = "poly"
        with torch.no_grad():
            po = mine(idx)
        mine.sites.mode = "exact"
        errs[deg] = ((po - ex).abs().max() / ex.abs().max()).item()
        # a low-degree site moves the inputs of the sites after it, so they can leave the exact-mode range: only the
        # high-degree run must stay inside it
        if deg == 64:
            assert max(mine.sites.outside_fractions().values()) == 0.0
    print("relative max logit error by degree:", errs)
    assert errs[64] < 2e-3
    assert errs[64] < errs[4] / 10


def test_heal_lowers_loss(ref_and_ckpt, text_files, tmp_path):
    import heal
    _, ckpt = ref_and_ckpt
    txt, _ = text_files
    m = load_rwkv7(ckpt, sites=Sites("exact"))
    idx = torch.randint(0, 65536, (2, 32), generator=torch.Generator().manual_seed(6))
    R = _ranges_from_model(m, idx)
    co = tmp_path / "c3.json"
    co.write_text(json.dumps(_coeffs(R, 3, 0.1)))
    out = tmp_path / "heal"
    heal.main(["--model", ckpt, "--coeffs", str(co), "--clamp", "--text", txt, "--train", "blocks:1,2",
               "--steps", "40", "--ctx", "32", "--batch", "2", "--lr", "3e-3", "--warmup", "1",
               "--log-every", "1", "--save-every", "40", "--device", "cpu", "--seed", "0", "--out", str(out)])
    log = [json.loads(l) for l in open(out / "heal_log.jsonl")]
    steps = [r for r in log if r["event"] == "step"]
    first = np.mean([r["loss"] for r in steps[:5]])
    last = np.mean([r["loss"] for r in steps[-5:]])
    print(f"heal loss: first5 {first:.4f} last5 {last:.4f}")
    assert last < 0.8 * first
    assert any(r["event"] == "save" for r in log)
    # the polynomial coefficients are untouched (they are not parameters) and block 0 is frozen
    ck = torch.load(out / "healed_step40.pt", weights_only=True)["model"]
    base = torch.load(ckpt, weights_only=True)
    assert torch.equal(ck["blocks.0.att.key.weight"], base["blocks.0.att.key.weight"].float())
    assert not torch.equal(ck["blocks.2.att.key.weight"], base["blocks.2.att.key.weight"].float())
