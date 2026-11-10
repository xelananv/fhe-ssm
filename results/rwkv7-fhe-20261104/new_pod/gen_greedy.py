"""Greedy plaintext generation with the research harness (exact mode, or poly with --coeffs [--clamp]).
python gen_greedy.py --model M.pth --text T [--n 32] [--coeffs C.json --clamp] -> one JSON line on stdout."""
import argparse, json, sys, time
sys.path.insert(0, "/root/vd/harness")
import torch
from rwkv7_model import Sites, load_rwkv7, pick_device
from rwkv_tokenizer import RWKVTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--text", required=True); ap.add_argument("--tag", default="")
ap.add_argument("--n", type=int, default=32); ap.add_argument("--coeffs"); ap.add_argument("--clamp", action="store_true")
a = ap.parse_args()
text = a.text.encode().decode("unicode_escape") if "\\n" in a.text else a.text
tok, sites, dev = RWKVTokenizer(), Sites("exact"), pick_device("auto")
model = load_rwkv7(a.model, device=dev, sites=sites)
if a.coeffs:
    sites.load_poly(json.load(open(a.coeffs)), clamp=a.clamp); sites.mode = "poly"
ids = tok.encode(text); gen = []; t0 = time.time()
with torch.no_grad():
    for _ in range(a.n):
        logits = model(torch.tensor([ids + gen], device=dev))[0, -1]
        gen.append(int(torch.argmax(logits)))
print(json.dumps({"tag": a.tag, "model": a.model, "mode": "poly" + ("-clamp" if a.clamp else "") if a.coeffs else "exact",
                  "coeffs": a.coeffs, "prompt": text, "prompt_ids": ids, "generated_ids": gen,
                  "generated_text": tok.decode(gen), "seconds": round(time.time() - t0, 1)}, ensure_ascii=False))
