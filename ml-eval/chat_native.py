#!/usr/bin/env python3
"""
chat_native.py — interactive sampler over a trained lane-N checkpoint, for
talking to a trained model. Also usable non-interactively (--prompt) to generate a
handful of sample completions for the paper appendix, from both `native`
and `baseline` tags for a qualitative co-design comparison.

Generation is full-reforward-per-token (no incremental state cache): at
~138M params and short completions this is fast enough on an A100 and, more
importantly, is guaranteed to match the exact forward pass used everywhere
else in this project (train/eval/export) with zero risk of a second,
subtly-different incremental-decoding implementation drifting from it.

Usage:
  python ml-eval/chat_native.py --tag native --temp 0.8            # interactive REPL
  python ml-eval/chat_native.py --tag native --prompt "Once upon a time"  # one-shot
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import train_fhe_native_ssm as tfns  # noqa: E402


def load_model(tag, device, torch):
    tfns.TAG = tag
    if not tfns._cpath().exists():
        raise FileNotFoundError(f"{tfns._cpath()} missing -- train+export tag={tag} first")
    tfns.CFG.update(json.loads(tfns._cpath().read_text()))
    model = tfns.build_model(torch).to(device)
    sd = {k: torch.from_numpy(v) for k, v in dict(np.load(tfns._wpath())).items()}
    model.load_state_dict(sd)
    model.eval()
    return model


def generate(model, tok, prompt, max_new_tokens, temperature, device, torch):
    ids = tok.encode(prompt)
    ctx = tfns.CFG["ctx"]
    eos = tok.eos_token_id
    for _ in range(max_new_tokens):
        window = ids[-ctx:]
        inp = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(inp)[0, -1]
        if temperature <= 0:
            next_id = int(logits.argmax().item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
        ids.append(next_id)
        if eos is not None and next_id == eos:
            break
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="native")
    ap.add_argument("--temp", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--device", default=None)
    ap.add_argument("--prompt", default=None,
                     help="one-shot: generate for this prompt and exit (no REPL)")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(tfns.TOK_MODEL)
    model = load_model(args.tag, device, torch)
    print(f"[chat_native] tag={args.tag} nonlin={tfns.CFG['nonlin']} "
          f"d={tfns.CFG['d_model']} L={tfns.CFG['n_layer']} device={device}",
          file=sys.stderr)

    if args.prompt is not None:
        print(generate(model, tok, args.prompt, args.max_new_tokens, args.temp, device, torch))
        return

    print("Interactive mode (Ctrl-D to exit).", file=sys.stderr)
    while True:
        try:
            prompt = input("> ")
        except EOFError:
            print()
            break
        if not prompt.strip():
            continue
        print(generate(model, tok, prompt, args.max_new_tokens, args.temp, device, torch))


if __name__ == "__main__":
    main()
