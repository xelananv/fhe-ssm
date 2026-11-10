"""Replay lanes of the 2026-09-19 148-tick run through the local serve_ollama server and compare the
continuation with the run's own picks (lanes where the encrypted pick equalled the plaintext pick on every tick)."""
import json, sys, urllib.request, collections
sys.path.insert(0, "ml-eval")
from transformers import AutoTokenizer
import train_fhe_native_ssm as tfns

FID = "results/dense-demo-s31/pod_longrun_20260919/pod_pull/s37/serve_LONG/fidelity.json"
tok = AutoTokenizer.from_pretrained(tfns.TOK_MODEL)
rows = json.load(open(FID))
lanes = collections.defaultdict(list)
for r in sorted(rows, key=lambda r: (r["lane"], r["tick"])):
    lanes[r["lane"]].append(r)

for lane in [int(x) for x in sys.argv[1:]]:
    rs = lanes[lane]
    prompt_ids = [r["fed"] for r in rs if r["fedSrc"] == "prompt"]
    gen = [r for r in rs if r["generating"]]
    assert all(r["pickAgree"] for r in gen), f"lane {lane}: not every pick agrees"
    pod_ids = [r["encPick"] for r in gen]
    prompt = tok.decode(prompt_ids)
    assert tok.encode(prompt) == prompt_ids, f"lane {lane}: prompt does not round-trip through the tokenizer"
    body = json.dumps({"model": "fhe-ssm-430m", "prompt": prompt, "raw": True, "stream": False,
                       "options": {"num_predict": len(pod_ids)}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11435/api/generate", body, {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))
    pod_text = tok.decode(pod_ids)
    srv = out["response"]
    same = srv == pod_text
    k = next((i for i, (a, b) in enumerate(zip(srv, pod_text)) if a != b), min(len(srv), len(pod_text)))
    print(json.dumps({"lane": lane, "prompt_tokens": len(prompt_ids), "server_prompt_eval": out.get("prompt_eval_count"),
                      "pod_generated": len(pod_ids), "server_generated": out.get("eval_count"),
                      "done_reason": out.get("done_reason"), "text_identical": same,
                      "first_diff_char": None if same else k}))
    print("  prompt:", repr(prompt[:80]))
    print("  reply :", repr(srv[:160]))
    if not same:
        print("  pod   :", repr(pod_text[max(0, k - 40):k + 80]))
        print("  server:", repr(srv[max(0, k - 40):k + 80]))
