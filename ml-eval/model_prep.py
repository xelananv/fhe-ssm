#!/usr/bin/env python3
"""
model_prep.py — download RWKV-4-169m-pile, export weights + config for the
evaluation pipeline, and validate our minimal reimplementation against the
HuggingFace reference on a short prompt.

Why RWKV-4: it is the standard public trained LM (~170M params) whose
sequence mixing is a *diagonal linear recurrence with per-token
plaintext-computable (stabilized) coefficients* — the exact circuit shape of
the generalized Level-Striding Lemma. Mamba-class selectivity is deliberately
out of scope.

Outputs (ml-eval/artifacts/):
  weights.npz   — float64 state dict (numpy)
  config.json   — the fields our runtime needs
  validation.json — max logit deltas vs the HF reference

Usage: .venv/bin/python ml-eval/model_prep.py
"""
import json
import pathlib
import sys

import numpy as np

MODEL_ID = "RWKV/rwkv-4-169m-pile"
HERE = pathlib.Path(__file__).resolve().parent
ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)


def main() -> None:
    import torch
    from transformers import AutoTokenizer, RwkvForCausalLM

    print(f"downloading {MODEL_ID} …", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = RwkvForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    cfg = model.config
    config = {
        "model_id": MODEL_ID,
        "n_layer": cfg.num_hidden_layers,
        "d_model": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "layer_norm_epsilon": cfg.layer_norm_epsilon,
    }
    (ART / "config.json").write_text(json.dumps(config, indent=2))

    sd = model.state_dict()
    np.savez(
        ART / "weights.npz",
        **{k: v.detach().numpy().astype(np.float64) for k, v in sd.items()},
    )
    n_params = sum(v.numel() for v in sd.values())
    print(f"exported {len(sd)} tensors, {n_params/1e6:.1f}M params -> {ART/'weights.npz'}")

    # ---- validate our reimplementation against the HF reference --------------
    sys.path.insert(0, str(HERE))
    from rwkv_numpy import RwkvRunner

    prompt = "The mathematics of lattice-based cryptography depends on the hardness of"
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        ref_logits = model(ids).logits[0].numpy()  # (T, vocab)

    runner = RwkvRunner(str(ART / "weights.npz"), str(ART / "config.json"))
    our_logits, _stats = runner.forward(ids[0].tolist(), backend="float")

    diff = np.abs(our_logits - ref_logits)
    rel = diff.max() / (np.abs(ref_logits).max() + 1e-12)
    argmax_agree = float((our_logits.argmax(-1) == ref_logits.argmax(-1)).mean())
    report = {
        "prompt_tokens": int(ids.shape[1]),
        "max_abs_logit_diff": float(diff.max()),
        "max_rel_logit_diff": float(rel),
        "next_token_argmax_agreement": argmax_agree,
    }
    (ART / "validation.json").write_text(json.dumps(report, indent=2))
    print("validation vs HF reference:", json.dumps(report))
    if argmax_agree < 1.0 or rel > 1e-3:
        print("WARNING: reimplementation disagrees with reference — do not proceed", file=sys.stderr)
        sys.exit(1)
    print("reimplementation validated OK")


if __name__ == "__main__":
    main()
