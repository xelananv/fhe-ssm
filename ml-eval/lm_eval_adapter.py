#!/usr/bin/env python3
"""
lm_eval_adapter.py — exposes train_fhe_native_ssm.py's torch model (lane N,
poly-gate "native" AND the matched "baseline" tag) as a custom
lm-evaluation-harness LM, so both can be scored with the same PRIMARY/
SECONDARY task suite and stderr reporting as the Pythia anchors.

Written against lm-eval 0.4.9.1's api.model.TemplateLM (verified against the
installed package source, not assumed) -- record the pod's actual installed
version in the results file regardless, since task scoring has drifted
across releases.

Architecture note: NativeLM.forward(ids) has NO attention-mask input --
batches here are RIGHT-padded (never left-padded), which is safe because
every op in the model (token-shift, the causal parallel scan, RMSNorm) is
strictly backward-looking / per-token, so padding appended after a
sequence's real content cannot leak into that sequence's own logits. Do not
switch this to left-padding.

Usage (one tag per process -- see the CFG-is-a-module-global note below):
  python ml-eval/lm_eval_adapter.py --tag native \
      --tasks wikitext,lambada_openai,hellaswag,piqa,winogrande,arc_easy,arc_challenge,boolq \
      --device cuda:0 --batch-size 8 --output results/real-model/lm_eval_native.json
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import train_fhe_native_ssm as tfns  # noqa: E402

PRIMARY_TASKS = ["wikitext", "lambada_openai"]
SECONDARY_TASKS = ["hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge", "boolq"]


def _build(tag, device, torch):
    """Load a trained tag's shape+weights into a fresh NativeLM. NOTE:
    train_fhe_native_ssm.CFG is a module-level dict that TimeMix/ChannelMix
    read from AT FORWARD TIME (not frozen at construction) -- so this
    process must evaluate exactly ONE tag. Do not call _build twice in the
    same process and expect both models to behave independently."""
    tfns.TAG = tag
    if tfns._cpath().exists():
        tfns.CFG.update(json.loads(tfns._cpath().read_text()))
    else:
        raise FileNotFoundError(f"{tfns._cpath()} missing -- train+export tag={tag} first")
    model = tfns.build_model(torch).to(device)
    sd = {k: torch.from_numpy(v) for k, v in dict(np.load(tfns._wpath())).items()}
    model.load_state_dict(sd)
    model.eval()
    return model


class NativeSSMLM:
    """Subclasses lm_eval.api.model.TemplateLM lazily (import deferred to
    __init__) so this file imports cleanly even before lm-eval is pip
    installed (useful for local review without the pod's venv)."""

    def __new__(cls, *a, **kw):
        from lm_eval.api.model import TemplateLM

        # build the real subclass on first use, bound to the installed TemplateLM
        real_cls = globals().get("_NativeSSMLMImpl")
        if real_cls is None:
            real_cls = _make_impl(TemplateLM)
            globals()["_NativeSSMLMImpl"] = real_cls
        obj = real_cls.__new__(real_cls)
        obj.__init__(*a, **kw)
        return obj


def _make_impl(TemplateLM):
    import torch
    from tqdm import tqdm
    from lm_eval import utils

    class _Impl(TemplateLM):
        def __init__(self, tag="native", device=None, batch_size=8, max_length=None):
            super().__init__()
            self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = _build(tag, self._device, torch)
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tfns.TOK_MODEL)
            self._eot = self.tokenizer.eos_token_id
            if self._eot is None:
                self._eot = 0
            self._max_length = int(max_length or tfns.CFG["ctx"])
            self._batch_size = int(batch_size)
            self.tag = tag

        # ---- required TemplateLM surface ----
        @property
        def eot_token_id(self):
            return self._eot

        @property
        def max_length(self):
            return self._max_length

        @property
        def batch_size(self):
            return self._batch_size

        @property
        def device(self):
            return self._device

        def tok_encode(self, string, **kwargs):
            return self.tokenizer.encode(string)

        def tok_decode(self, tokens, **kwargs):
            return self.tokenizer.decode(tokens)

        def _model_call(self, inps):
            with torch.no_grad():
                if self._device.startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        return self.model(inps)
                return self.model(inps)

        def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
            res = [None] * len(requests)
            # longest-first: OOMs surface immediately, batching is efficient
            order = sorted(range(len(requests)),
                           key=lambda i: -(len(requests[i][1]) + len(requests[i][2])))
            pbar = tqdm(total=len(requests), disable=disable_tqdm, desc="loglikelihood")
            for start in range(0, len(order), self.batch_size):
                idxs = order[start:start + self.batch_size]
                seqs, meta = [], []
                for i in idxs:
                    _, context_enc, continuation_enc = requests[i]
                    full = (context_enc + continuation_enc)[-(self.max_length + 1):]
                    inp = full[:-1]
                    target = full[1:]
                    contlen = min(len(continuation_enc), len(inp))
                    seqs.append(inp)
                    meta.append((i, len(inp), contlen, target))
                maxlen = max(len(s) for s in seqs)
                padded = torch.full((len(seqs), maxlen), self.eot_token_id, dtype=torch.long)
                for j, s in enumerate(seqs):
                    padded[j, :len(s)] = torch.tensor(s, dtype=torch.long)
                padded = padded.to(self.device)
                logits = torch.log_softmax(self._model_call(padded).float(), dim=-1)
                for j, (i, inplen, contlen, target) in enumerate(meta):
                    cont_logits = logits[j, inplen - contlen:inplen, :]
                    cont_toks = torch.tensor(target[-contlen:], dtype=torch.long, device=self.device)
                    logprobs = cont_logits.gather(-1, cont_toks.unsqueeze(-1)).squeeze(-1)
                    greedy = cont_logits.argmax(-1)
                    is_greedy = bool((greedy == cont_toks).all().item())
                    res[i] = (float(logprobs.sum().item()), is_greedy)
                pbar.update(len(idxs))
            pbar.close()
            return res

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            out = []
            for req in tqdm(requests, disable=disable_tqdm, desc="loglikelihood_rolling"):
                (string,) = req.args
                windows = list(map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                ))
                reqs = [(None, ctx, cont) for ctx, cont in windows]
                nlls = self._loglikelihood_tokens(reqs, disable_tqdm=True)
                out.append(sum(x[0] for x in nlls))
            return out

        def generate_until(self, requests, disable_tqdm=False):
            raise NotImplementedError(
                "native_ssm adapter is loglikelihood-only (wikitext/lambada/"
                "hellaswag/piqa/winogrande/arc_*/boolq); no generate_until task "
                "in this evaluation needs it.")

    return _Impl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="native | baseline | native24 | ...")
    ap.add_argument("--tasks", default=",".join(PRIMARY_TASKS + SECONDARY_TASKS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=float, default=None,
                     help="cap examples/task -- use a small value to smoke-test the adapter first")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import lm_eval

    lm = NativeSSMLM(tag=args.tag, device=args.device, batch_size=args.batch_size)
    tasks = [t for t in args.tasks.split(",") if t]
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, limit=args.limit)

    out_path = pathlib.Path(args.output) if args.output else (
        HERE / "artifacts" / f"lm_eval_{args.tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in results.items() if k != "samples"}
    out_path.write_text(json.dumps(serializable, indent=2, default=str))
    print(json.dumps({"tag": args.tag, "tasks": tasks, "wrote": str(out_path)}))
    print(json.dumps(serializable.get("results", {}), indent=2, default=str))


if __name__ == "__main__":
    main()
