#!/usr/bin/env python3
"""
train_fhe_native_ssm.py — lane N of REAL_MODEL_PLAN.md, the HEADLINE
real-model result: train a real LM at real scale whose EVERY operation is
FHE-exact or a well-conditioned Newton iterate, so the exact circuit runs
it end-to-end with ONLY bootstrap noise (no approximation error at all on
the gates).

WHY this and not RWKV-4: the simulator proved RWKV-4's WKV ratio is
out-of-class (denominator spans 1e3–1e26 per channel; +568% ppl from range
control alone even with exact arithmetic — REAL_MODEL_PLAN.md §0). No
public pretrained LM is in the paper's non-selective bounded-diagonal
class. So we train one, at a scale (~110M params) where "you trained a toy"
does not apply.

ARCHITECTURE (in-class by construction — every choice is deliberate):
  token emb (tied to head)
  per layer, pre-norm residual:
    --- time mix (the SSM, Lemma 2) ---
    u  = RMSNorm(h) ⊙ γ1                    # rsqrt: Newton (well-conditioned:
                                            #   variance of a normed stream is
                                            #   O(1), tight range — NOT RWKV's
                                            #   pathological division)
    u  = shift_mix(u)                       # token-shift (RWKV trick, free/1lv)
    x  = W_in u                             # dense matvec
    s_t= a ⊙ s_{t-1} + b ⊙ x_t             # a = a_min+(a_max-a_min)σ(θ),
                                            #   b = 1-a  (LRU norm ⇒ ‖s‖≤‖x‖);
                                            #   a is a STATIC PARAMETER
                                            #   (NON-SELECTIVE) ⇒ plaintext
                                            #   decay ⇒ Lemma 2 applies exactly
    z  = c ⊙ s + d ⊙ x                     # readout
    g  = z ⊙ (p0 + p1 z + p2 z²)           # LEARNED degree-3 gate — EXACT in FHE
    h  = h + W_out g                        # residual
    --- channel mix (gated MLP) ---
    u2 = RMSNorm(h) ⊙ γ2
    k  = W_k u2 ; r = W_r u2
    a2 = q0 + q1 k + q2 k² + q3 k³          # LEARNED poly activation — EXACT
    gt = r0 + r1 r + r2 r²                  # LEARNED poly gate — EXACT
    h  = h + W_v (a2 ⊙ gt)
  RMSNorm ⊙ γ_out ; tied head
NO division of data, NO exp, NO unbounded gate, NO selectivity, bounded
state. The ONLY approximated op is RMSNorm rsqrt (Newton, 2–3 iters from a
per-layer calibrated seed) — the simulator showed LN-rsqrt was never the
problem (RWKV's LN passed; its division failed).

DATA: FineWeb-Edu sample (streamed, non-repeating — real quality) by
default; --data wt103 falls back to WikiText-103 (repeats, heavier
regularization). EVAL always on the WikiText-103 test slice (comparable to
the RWKV lanes and a standard small-LM benchmark).

Sizing default (~110M params, competitive small-LM territory; ~6–10 GPU-h
on one A100): d=768, L=12, ctx=1024. The QUALITY BAR is a real trained LM
with test ppl in small-transformer range (record whatever lands); the claim
is exact-circuit fidelity of a real trained model, not SOTA.

Outputs (ml-eval/artifacts/): native_weights.npz, native_config.json,
native_train_log.jsonl. `export` writes bundle_native.bin/.index.txt + a
per-layer calibration (RMSNorm variance ranges → Newton seeds) for the GPU
harness and the simulator's noise pre-check.

Usage (pod GPU3, Phase 0.5):
  python ml-eval/train_fhe_native_ssm.py train --data fineweb --steps 100000
  python ml-eval/train_fhe_native_ssm.py eval           # WT103 test ppl
  python ml-eval/train_fhe_native_ssm.py export --tokens 512

FLAGS (S2.8, 2026-08-22 — every one defaults to the pre-session behaviour, so
an unflagged invocation is bit-identical to this file before the edit):
  --tag S --layers N --d-model N --nonlin poly|standard --seed N
  --data fineweb|wt103|staged:<dir>[?offset=&on_exhaust=&chunk=]  --steps N
  --batch N          micro-batch. SHIPPED DEFAULT 16 DOES NOT FIT ON 48 GB at
                     d=1024/L=24 (62.9 GB) — size it with
                     `python training/vram_sizing.py --vram-gb X ...`
  --grad-accum N     keep effective batch = batch x grad_accum x world; both
                     are logged in the start event
  --ctx N            sequence length
  --scan hs|rfft     rfft = the causal-conv A/B (2.6x more rms error; NOT
                     adopted — training/TRAINER_FIXES_20260822.md)
  --gates expanded|horner   Horner form of the learned polynomials
  --ce-chunk N       chunked+checkpointed cross-entropy (N chunks); 0 = off
  --compile          torch.compile the model (measure determinism first)
  --skip-nonfinite   skip an optimizer update whose grad norm is non-finite.
                     The COUNTER is always on and costs nothing.
  --pin-memory       pinned + non_blocking host->device copies
  --window-carry     stop dropping each shard's ragged window tail
  --bf16             force bf16 autocast on ANY cuda device — the shipped
                     predicate is `dev == "cuda"`, which is FALSE under DDP
                     ("cuda:N"), so every torchrun run to date trained fp32
  --resume           continue {TAG}_resume.pt (now also restores the staged
                     data cursor, so a resumed run does not re-walk the corpus)
Unknown flags are now a hard error: they used to be silently ignored, which
trains the DEFAULTS under a command line that looks configured.
"""
import contextlib
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ART = HERE / "artifacts"
TOK_MODEL = "RWKV/rwkv-4-169m-pile"     # GPT-NeoX tokenizer (same as R lanes)

CFG = {
    "d_model": 768, "n_layer": 12, "ctx": 1024, "vocab": 50277,
    "d_ffn": 3072, "shift_mix": True,
    "a_min": 0.90, "a_max": 0.9995, "alpha_res": 1.0,
    "lr": 6e-4, "min_lr": 6e-5, "batch": 16, "grad_accum": 2,
    "steps": 100000, "warmup": 2000, "wd": 0.1, "dropout": 0.0,
    "grad_clip": 1.0, "seed": 1234,
    # --- S2.8 levers (2026-08-22). EVERY ONE DEFAULTS TO THE PRE-SESSION
    # BEHAVIOUR: an unflagged run is bit-identical to this file at
    # sha256 12069be324e573903a0e2f59ef08ece8ba98a421888dd15954b9c456bc6ec103.
    # Rationale + numerical checks: training/TRAINER_FIXES_20260822.md.
    "scan": "hs",             # hs | rfft         (--scan)
    "gates": "expanded",      # expanded | horner (--gates)
    "compile": False,         # --compile          (torch.compile the model)
    "ce_chunk": 0,            # 0 = one unchunked CE; N = N checkpointed chunks
    "skip_nonfinite": False,  # --skip-nonfinite   (guard; the COUNTER is always on)
    "pin_memory": False,      # --pin-memory       (pinned + non_blocking H2D)
    "window_carry": False,    # --window-carry     (carry the shard's ragged
                              #   window tail instead of dropping it)
    "bf16": False,            # --bf16: force bf16 autocast on ANY cuda device.
                              #   The shipped predicate is `dev == "cuda"`,
                              #   which is FALSE for the "cuda:N" that the DDP
                              #   branch sets -- so every torchrun run to date
                              #   trained fp32 (L40S_RUNBOOK.md §3.5). Default
                              #   off = shipped behaviour; turn it on for any
                              #   multi-GPU rental, and size VRAM accordingly.
    # "poly"     = FHE-native learned polynomial gates (the real lane N)
    # "standard" = MATCHED BASELINE: identical shape/params/data, but SiLU
    #   gates instead of learned polynomials. Training both and diffing the
    #   benchmark scores is the ONLY honest way to price the co-design --
    #   it is the first question any reviewer asks.
    "nonlin": "poly",
    # --- S3.3 Tier B (2026-09-03), results/theory/TIERB_SPEC_20260903.md. DEFAULT =
    # the shipped sequential block; --block parallel trains nothing here, it only
    # exists so the arm can be trained later. Replay [trace] under parent-first:
    # sequential 110.7 boots/tick, parallel 64.0 (trace_schedule_20260903.txt F).
    "block": "sequential",    # sequential | parallel (--block): parallel = one norm
                              #   feeds BOTH branches, h' = h + tm(n(h)) + cm(n(h))
    # d_ffn is now also a first-class flag (--d-ffn N); absent -> 4*d_model as before.
}


# ---------------------------------------------------------------------------
def build_model(torch):
    import torch.nn as nn
    import torch.nn.functional as F

    def rmsnorm(x, g, eps=1e-5):
        # matches the FHE circuit: 1/sqrt(mean(x^2)+eps) via Newton at eval
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g

    def rfft_scan(a, xb):
        """A/B ALTERNATIVE to parallel_scan, reached only by --scan rfft.
        DEFAULT OFF (CFG["scan"] == "hs").

        Same recurrence as ONE causal depthwise convolution with kernel a^j,
        evaluated by a single rFFT pair. Legal for exactly the reason
        parallel_scan's docstring gives -- the decay `a` is NON-SELECTIVE, so
        the kernel is a constant per channel and the recurrence IS a
        convolution. Same property that licenses Lemma 2 under FHE.

        NOT free, and NOT adopted on this evidence. Against an fp64
        Hillis-Steele reference (training/scan_cost_probe.py, reproduced
        2026-08-22): shipped fp32 HS relErrRms 1.498e-04, this path 3.862e-04
        -- 2.6x more rms error. Both an order of magnitude under bf16's ~4e-3
        resolution, but the tax is real. Traffic 213.0 -> 61.5 kB/token/layer,
        60 -> 17 aten ops; it stashes 12.3 kB/token/layer of activation that
        HS does not (HS's decay is constant, so autograd saves ~nothing) --
        and activation memory is what caps batch, which caps MFU."""
        Tl = xb.shape[1]
        n = 1
        while n < 2 * Tl:
            n *= 2
        j = torch.arange(Tl, device=a.device, dtype=torch.float32)
        kern = torch.exp(j[:, None] * torch.log(a)[None, :])
        K = torch.fft.rfft(kern, n=n, dim=0)
        X = torch.fft.rfft(xb.float(), n=n, dim=1)
        return torch.fft.irfft(X * K[None], n=n, dim=1)[:, :Tl]

    def parallel_scan(a, xb):
        """h_t = a (.) h_{t-1} + xb_t, computed in O(log T) parallel steps
        (Hillis-Steele doubling) instead of T sequential ones.

        Valid ONLY because the decay `a` is NON-SELECTIVE (a static
        per-channel parameter, not input-dependent) — the same property that
        makes Lemma 2 apply under FHE. After the k-th doubling step the
        partial result is h_t = sum_{j<2^{k+1}} a^j xb_{t-j}; after
        ceil(log2 T) steps it is the exact recurrence. Elementwise ops +
        shifts only, so it runs on any backend (CUDA/MPS/CPU) without an
        FFT kernel. Mathematically identical to the sequential scan (parity
        asserted in the smoke test); the FHE circuit still evaluates the
        blocked form of Lemma 2."""
        if CFG["scan"] == "rfft":
            return rfft_scan(a, xb)
        T = xb.shape[1]
        h = xb
        cur = a                                   # a^(2^k), broadcast over (B,T,d)
        step = 1
        while step < T:
            shifted = F.pad(h, (0, 0, step, 0))[:, :T]      # h[t-step], zero-filled
            h = h + cur * shifted
            cur = cur * cur
            step *= 2
        return h

    class TimeMix(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.g1 = nn.Parameter(torch.ones(d))
            self.mix = nn.Parameter(torch.ones(d) * 0.5) if CFG["shift_mix"] else None
            self.win = nn.Linear(d, d, bias=False)
            self.a_raw = nn.Parameter(torch.zeros(d))
            self.c = nn.Parameter(torch.randn(d) * 0.5)
            self.dd = nn.Parameter(torch.randn(d) * 0.5)
            self.p0 = nn.Parameter(torch.full((d,), 0.5))
            self.p1 = nn.Parameter(torch.full((d,), 0.25))
            self.p2 = nn.Parameter(torch.zeros(d))
            self.wout = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.wout.weight)      # residual-stable init

        def decay(self):
            lo, hi = CFG["a_min"], CFG["a_max"]
            return lo + (hi - lo) * torch.sigmoid(self.a_raw)

        def forward(self, h):
            return h + CFG["alpha_res"] * self.branch(rmsnorm(h, self.g1))

        def branch(self, u):
            """The time-mix branch from an ALREADY-normalised input (S3.3 Tier B:
            the parallel block feeds one norm output to both branches)."""
            if self.mix is not None:
                us = F.pad(u, (0, 0, 1, 0))[:, :-1]     # shift by one token
                u = u * self.mix + us * (1.0 - self.mix)
            x = self.win(u)
            a = self.decay().float()
            b = (1.0 - a)
            xb = (b * x.float())
            S = parallel_scan(a, xb).to(u.dtype)       # O(log T), fp32 (u: the branch input, S3.3)
            z = self.c * S + self.dd * x
            if CFG["nonlin"] == "standard":
                g = z * torch.sigmoid(z)                 # SiLU (baseline)
            elif CFG["gates"] == "horner":
                # --gates horner: same polynomial, fewer aten ops. NOT the
                # default; see the ChannelMix note below for the trap.
                g = z * (self.p0 + z * (self.p1 + z * self.p2))
            else:
                g = z * (self.p0 + self.p1 * z + self.p2 * z * z)
            return self.wout(g)

    class ChannelMix(nn.Module):
        def __init__(self, d, dff):
            super().__init__()
            self.g2 = nn.Parameter(torch.ones(d))
            self.wk = nn.Linear(d, dff, bias=False)
            self.wr = nn.Linear(d, dff, bias=False)
            self.q0 = nn.Parameter(torch.zeros(dff)); self.q1 = nn.Parameter(torch.ones(dff))
            self.q2 = nn.Parameter(torch.zeros(dff)); self.q3 = nn.Parameter(torch.zeros(dff))
            self.r0 = nn.Parameter(torch.full((dff,), 0.5)); self.r1 = nn.Parameter(torch.full((dff,), 0.25))
            self.r2 = nn.Parameter(torch.zeros(dff))
            self.wv = nn.Linear(dff, d, bias=False)
            nn.init.zeros_(self.wv.weight)

        def forward(self, h):
            return h + self.branch(rmsnorm(h, self.g2))

        def branch(self, u):
            k, r = self.wk(u), self.wr(u)
            if CFG["nonlin"] == "standard":
                act, gate = F.silu(k), torch.sigmoid(r)  # baseline GLU
            elif CFG["gates"] == "horner":
                # --gates horner (DEFAULT OFF). Identical polynomial, 15 -> 11
                # aten ops, 245.8 -> 180.2 kB/token/layer of write traffic
                # (training/scan_cost_probe.py). relErrRms 2.766e-14 vs the
                # expanded form -- measured with RANDOMISED coefficients,
                # which is the only check that means anything here: q2, q3 and
                # r2 (and tm.p2) ship initialised to ZERO, so at init both
                # forms collapse to q0 + q1*k and a naive A/B passes
                # trivially. torch.compile (--compile) subsumes this and goes
                # further (~16 kB/token/layer); Horner is the no-compile
                # fallback.
                act = self.q0 + k * (self.q1 + k * (self.q2 + k * self.q3))
                gate = self.r0 + r * (self.r1 + r * self.r2)
            else:
                act = self.q0 + self.q1 * k + self.q2 * k * k + self.q3 * k * k * k
                gate = self.r0 + self.r1 * r + self.r2 * r * r
            return self.wv(act * gate)

    def chunked_ce(h, targets, head, nchunk):
        """Cross-entropy WITHOUT ever materialising the full (B*T, V) logits
        tensor. Reached only by --ce-chunk N; DEFAULT OFF (CFG["ce_chunk"]==0).

        Why: at batch 16 / ctx 1024 / V=50,277 the logits are 3.3 GB in fp32
        plus the same again for log-softmax and grad
        (training/THROUGHPUT_AUDIT.md §6). Each chunk is wrapped in a
        non-reentrant checkpoint, so backward recomputes that chunk's logits
        instead of autograd stashing all of them -- peak CE memory drops by
        ~nchunk at the cost of one extra head matmul per chunk. Freed memory
        buys batch, and batch is what caps MFU.

        reduction="sum" then / N is the exact mean of the shipped path; the
        only difference is fp accumulation order. Verified against an fp64
        reference in training/verify_trainer_fixes.py."""
        import torch.utils.checkpoint as _ckpt
        hf = h.reshape(-1, h.shape[-1])
        tf = targets.reshape(-1)
        N = hf.shape[0]
        size = (N + nchunk - 1) // nchunk

        def seg(hs, ts):
            return F.cross_entropy(head(hs), ts, reduction="sum")

        tot = None
        for i in range(0, N, size):
            part = _ckpt.checkpoint(seg, hf[i:i + size], tf[i:i + size],
                                    use_reentrant=False)
            tot = part if tot is None else tot + part
        return tot / N

    class Block(nn.Module):
        def __init__(self, d, dff):
            super().__init__()
            self.tm = TimeMix(d); self.cm = ChannelMix(d, dff)

        def forward(self, h):
            if CFG["block"] == "parallel":
                # S3.3 Tier B: ONE norm (tm's g1) feeds both branches; cm's own norm
                # gain g2 is unused (kept in the parameter set so the export is
                # shape-stable; the exporter records block type in meta[6]).
                u = rmsnorm(h, self.tm.g1)
                return h + CFG["alpha_res"] * self.tm.branch(u) + self.cm.branch(u)
            return self.cm(self.tm(h))

    class NativeLM(nn.Module):
        def __init__(self):
            super().__init__()
            d, L, V, dff = (CFG["d_model"], CFG["n_layer"], CFG["vocab"], CFG["d_ffn"])
            self.emb = nn.Embedding(V, d)
            nn.init.normal_(self.emb.weight, std=0.02)
            self.blocks = nn.ModuleList(Block(d, dff) for _ in range(L))
            self.g_out = nn.Parameter(torch.ones(d))
            self.head = nn.Linear(d, V, bias=False)
            self.head.weight = self.emb.weight

        def forward(self, ids, targets=None):
            h = self.emb(ids)
            for b in self.blocks:
                h = b(h)
            h = rmsnorm(h, self.g_out)
            if targets is None:
                return self.head(h)          # the shipped path, unchanged
            # --ce-chunk N. The loss is computed INSIDE forward on purpose:
            # under DDP the head weight is tied to the embedding, and keeping
            # every use of it inside the wrapped forward keeps the reducer's
            # view of the graph the same as the unchunked path's.
            return chunked_ce(h, targets, self.head, CFG["ce_chunk"])

    return NativeLM()


# ---------------------------------------------------------------------------
def token_stream(torch, tok, data, split):
    """Yield 1-D LongTensors of token ids. FineWeb-Edu streamed (real
    quality) or WikiText-103 (fallback / eval).

    HOST-RAM TRAP, wt103 TRAIN split only (measured 2026-08-22: the S2.9 M3
    Kaggle cell died `rc=-9`, SIGKILL by the OOM killer, at 36 s). The wt103
    branch below is written for a big-RAM box and peaks several GB four times
    over: `to_pylist()` materialises the whole column as Python strings,
    `"".join()` builds one ~540 MB string, the tokenizer is handed ALL of it in
    a single call (fast tokenizers spike hard on one huge input), and the
    result is a ~130M-element Python int list before `torch.tensor()` sees it.
    On a ~13 GB box (Kaggle) this is fatal; on the pod (120 GB) it is merely
    wasteful. The `split == "test"` path shares this code but reads the 733 kB
    test parquet, so it is harmless.

    NOT FIXED ON PURPOSE. This is the WT103 anneal's data path -- the project's
    dominant quality lever (224 -> 54) -- and chunking the tokenisation would
    change the token stream at chunk boundaries, breaking comparability with
    every annealed record. If you need wt103 training tokens on a small-RAM
    box, stage them first (`ml-eval/stage_fineweb_tokens.py` +
    `staged_token_reader.py`, which is memory-safe and seekable) or use
    `--data fineweb`, whose branch below is streamed and safe by construction.
    """
    if data == "wt103" or split == "test":
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        fn = ("wikitext-103-raw-v1/test-00000-of-00001.parquet" if split == "test"
              else "wikitext-103-raw-v1/train-00000-of-00002.parquet")
        path = hf_hub_download(repo_id="Salesforce/wikitext",
                               repo_type="dataset", filename=fn)
        text = "".join(pq.read_table(path, columns=["text"]).column("text").to_pylist())
        yield torch.tensor(tok(text, return_tensors=None)["input_ids"], dtype=torch.long)
        return
    # FineWeb-Edu sample, streamed and tokenized in shards
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    buf = []
    for ex in ds:
        buf.extend(tok(ex["text"], return_tensors=None)["input_ids"])
        buf.append(tok.eos_token_id or 0)
        if len(buf) >= 1_000_000:
            yield torch.tensor(buf, dtype=torch.long); buf = []


def batch_iter(torch, tok, data, rank=0, world=1):
    """Yields (x,y) batches. With world>1 each rank takes a disjoint stride of
    the window space, so ranks never see the same tokens in a step.

    RAGGED-WINDOW-TAIL WART (identified and measured 2026-08-22, S2.8 --
    the trainer's flag tests and their record, not in this release).
    `carry` keeps only the sub-window remainder `ids[n*ctx:]`. The whole
    windows between the last emitted batch and `n` -- up to B*world-1 of them,
    i.e. up to (B*world-1)*ctx tokens per shard -- are DROPPED, not carried.
    Because the drop count depends on B, two runs at the same EFFECTIVE batch
    but different (batch, grad_accum) splits diverge in their token stream
    after the first shard boundary, even though the window ORDER is identical.

    Left as-is by default: changing it would alter what an unflagged run
    consumes, which would break comparability with every record in
    `results/a100-secure-20260730/`. --window-carry opts into the fixed
    behaviour (carry the tail, consume whole groups of B*world windows), which
    makes token consumption exactly batch-invariant."""
    ctx, B = CFG["ctx"], CFG["batch"]
    carry = torch.empty(0, dtype=torch.long)
    for shard in token_stream(torch, tok, data, "train"):
        ids = torch.cat([carry, shard])
        n = (len(ids) - 1) // ctx
        if CFG["window_carry"]:
            # consume whole groups of B*world windows; carry everything after
            # the last full group, so nothing is dropped at a shard boundary.
            groups = n // (B * world)
            for g in range(groups):
                base = g * B * world + rank * B
                xb = torch.stack([ids[(base + j) * ctx:(base + j) * ctx + ctx] for j in range(B)])
                yb = torch.stack([ids[(base + j) * ctx + 1:(base + j) * ctx + ctx + 1] for j in range(B)])
                yield xb, yb
            carry = ids[groups * B * world * ctx:]
            continue
        for base in range(rank * B, n - B + 1, B * world):
            xb = torch.stack([ids[(base + j) * ctx:(base + j) * ctx + ctx] for j in range(B)])
            yb = torch.stack([ids[(base + j) * ctx + 1:(base + j) * ctx + ctx + 1] for j in range(B)])
            yield xb, yb
        carry = ids[n * ctx:]


def cmd_train(data, steps, resume=False):
    """Single-GPU or multi-GPU DDP. Launch 4-GPU with:
         torchrun --nproc_per_node=4 ml-eval/train_fhe_native_ssm.py train ...
    Each rank streams a DISJOINT shard of the corpus (stride by world_size),
    so no sample is seen twice per epoch; gradients are all-reduced by DDP.
    Only rank 0 logs and checkpoints.

    resume=True continues an existing <tag>_resume.pt (model + optimizer +
    step) instead of starting fresh -- for pushing a run past a --steps
    target it already reached without re-spending the early steps. NOTE:
    only model/optimizer/step are persisted, not the data-stream position,
    so the resumed run's token_stream restarts from the beginning of the
    corpus; against a corpus far larger than the token budget this is a
    bounded, honest overlap, not a correctness issue, but report it as such
    rather than claim strictly-fresh tokens throughout."""
    import os
    import torch
    from transformers import AutoTokenizer
    torch.manual_seed(CFG["seed"])
    CFG["steps"] = steps
    # STAGED DATA (S2.8, 2026-08-22). `staged:<dir>[?...]` reads the uint16
    # corpus written by stage_fineweb_tokens.py instead of tokenizing
    # FineWeb-Edu inline. install() WRAPS token_stream, so batch_iter -- and
    # therefore the masked lane, which calls base.batch_iter -- picks it up
    # with no further edit. Anything not starting with "staged:" falls through
    # to the HF path byte-for-byte, so this is a no-op for a streaming run.
    data = _install_staged_reader(data, resume)
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank, world = 0, 1
    if ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dev = f"cuda:{os.environ['LOCAL_RANK']}"
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    # ONE definition of the autocast predicate, used by the loop AND by the run
    # stamp. Keeping them as two expressions let the stamp claim bf16 on a
    # device where the loop ran fp32 (caught 2026-08-22 in the S2.8 smoke).
    # `dev == "cuda"` is the SHIPPED predicate; it is False for the "cuda:N"
    # the DDP branch sets, which is why every torchrun run has trained fp32
    # (L40S_RUNBOOK.md §3.5). --bf16 opts out of that.
    amp_on = (dev == "cuda") or (CFG["bf16"] and dev.startswith("cuda"))
    tok = AutoTokenizer.from_pretrained(TOK_MODEL)
    model = build_model(torch).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    raw_model = model
    resume_ckpt = None
    if resume:
        # load before the DDP wrap so DDP's own rank0->all broadcast is the
        # thing that keeps ranks in sync, same as a fresh run.
        resume_ckpt = _load_resume_model(raw_model, torch, dev)
    if CFG["compile"]:
        # --compile (DEFAULT OFF). The largest single intensity lever: the
        # poly gates are 15 unfused elementwise ops over a dff-wide tensor,
        # 245.8 kB/token/layer, and a fused kernel reads k and r once
        # (training/THROUGHPUT_AUDIT.md §3b). Compiled BEFORE the DDP wrap so
        # `raw_model` stays the plain module the optimizer, the checkpoint
        # writer and `no_sync()` all expect; the parameters are the same
        # objects either way.
        # DETERMINISM: bitwise reproducibility across instances is a MEASURED
        # property of this stack (results/quality-ladder/
        # ladder_round3_rerun_console_20260808.log) and the thing that makes
        # every arm-to-arm comparison valid. Inductor re-associates and fuses,
        # so it is not guaranteed to preserve it -- measured verdict in
        # training/TRAINER_FIXES_20260822.md. Off until that verdict says on.
        _t_compile = time.time()
        model = torch.compile(model)
        if rank == 0:
            print(json.dumps({"event": "compile_requested",
                              "wrap_seconds": round(time.time() - _t_compile, 2),
                              "note": "graph capture happens on the first step"}),
                  flush=True)
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # nonlin="standard" (the matched baseline) leaves the learned polynomial
        # coefficients (tm.p0/p1/p2, cm.q0..q3, cm.r0..r2) allocated but UNUSED in
        # forward -- they stay allocated deliberately so the baseline keeps an
        # identical parameter shape/count to lane N. DDP aborts on params that
        # receive no gradient unless told to expect them, which is why the first
        # baseline launch died in 21s with "Expected to have finished reduction in
        # the prior iteration" (2026-07-18). Only enabled for the baseline: it
        # costs an extra autograd graph traversal per step, and lane N uses every
        # parameter.
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])],
                    find_unused_parameters=(CFG["nonlin"] == "standard"))
    if rank == 0:
        print(json.dumps({"event": "start", "params_M": round(nparams / 1e6, 1),
                          "device": dev, "data": data, "world_size": world,
                          "resume": resume,
                          # tokens per LOOP STEP. The `step` counter below
                          # increments once per micro-batch, so grad_accum must
                          # NOT appear here -- including it (as this line
                          # originally did) overstates tokens seen by exactly
                          # grad_accum x, which would have gone straight into
                          # the paper's "tokens seen" figure.
                          "tokens_per_step": CFG["batch"] * CFG["ctx"] * world,
                          "tokens_total": CFG["steps"] * CFG["batch"] * CFG["ctx"] * world,
                          # EFFECTIVE BATCH is the audit field: --batch and
                          # --grad-accum can trade off freely, and a silently
                          # different effective batch is a confound this
                          # project has already been bitten by
                          # (L40S_RUNBOOK.md §5 token-budget rule).
                          "effective_batch": CFG["batch"] * CFG["grad_accum"] * world,
                          "tokens_per_update": (CFG["batch"] * CFG["ctx"]
                                                * CFG["grad_accum"] * world),
                          "bf16_autocast": bool(amp_on),
                          **CFG}), flush=True)
    opt = torch.optim.AdamW(raw_model.parameters(), lr=CFG["lr"],
                            weight_decay=CFG["wd"], betas=(0.9, 0.95))
    start_step = 0
    if resume:
        start_step = _load_resume_optim(opt, resume_ckpt)
        cur = resume_ckpt.get("data_cursor") if isinstance(resume_ckpt, dict) else None
        restored = {}
        if _STAGED is not None and cur:
            # re-key onto the spec THIS run will use: --resume rewrote it to
            # on_exhaust=continue, and the cursor was saved under the old key.
            restored = _STAGED.rekey_cursors(cur, data)
            _STAGED.restore_cursors(restored)
        if rank == 0:
            print(json.dumps({"event": "resumed", "from_step": start_step,
                              "data_cursor": cur,
                              "data_cursor_restored": restored or False}),
                  flush=True)

    # LR SCHEDULE UNITS (bug fixed 2026-07-18). sched.step() is called only on
    # OPTIMIZER UPDATES -- i.e. once every `accum` iterations of the loop below --
    # but CFG["steps"]/CFG["warmup"] count LOOP iterations (micro-batches). The
    # original code compared the scheduler's update counter directly against
    # those loop-unit constants, so the schedule silently ran over a horizon
    # `accum` times shorter than intended. Measured consequence at steps=3000,
    # accum=2, warmup=2000: only 1500 scheduler ticks ever occur, so warmup
    # (2000) never completes, LR peaks at ~75% of target, and the cosine-decay
    # branch never executes at all -- the model never anneals. (Confirmed against
    # the live log: observed lr at loop steps 1000/1200/1400 matched lr_at(500/
    # 600/700) exactly.) Fix: convert the horizon into optimizer-update units and
    # scale warmup with it so the shape is preserved at any --steps value.
    _accum = CFG["grad_accum"]         # NB: local `accum` is bound further below
    updates_total = max(1, CFG["steps"] // _accum)
    warmup_updates = max(1, min(CFG["warmup"] // _accum, updates_total // 10))

    def lr_at(u):                      # u counts OPTIMIZER UPDATES, not loop steps
        if u < warmup_updates:
            return (u + 1) / warmup_updates
        prog = (u - warmup_updates) / max(1, updates_total - warmup_updates)
        cos = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        return (CFG["min_lr"] + (CFG["lr"] - CFG["min_lr"]) * cos) / CFG["lr"]

    # start_step is in loop units; the scheduler's counter is in update units.
    # LambdaLR with last_epoch != -1 requires 'initial_lr' on every param group,
    # which a fresh AdamW + load_state_dict() does NOT provide -- so --resume
    # would have died with KeyError('initial_lr') before ever training a step.
    if resume:
        for g in opt.param_groups:
            g.setdefault("initial_lr", CFG["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at,
                                              last_epoch=(start_step // _accum) - 1)
    if rank == 0:
        print(json.dumps({"event": "lr_schedule", "updates_total": updates_total,
                          "warmup_updates": warmup_updates,
                          "tokens_total": CFG["steps"] * CFG["batch"] * CFG["ctx"] * world}),
              flush=True)
    log = open(ART / f"{TAG}_train_log.jsonl", "a") if rank == 0 else None
    it = batch_iter(torch, tok, data, rank=rank, world=world)
    t0 = time.time(); step = start_step; accum = CFG["grad_accum"]
    # NON-FINITE UPDATE COUNTER (S2.8). Always on and behaviour-neutral: the
    # tally lives on-device and is only read at logging time, so the DEFAULT
    # path gains no host sync and no numerical change. training/DIAGNOSIS.md
    # §3 measured 1.2-1.85% of distill's updates skipped as non-finite -- over
    # the 1% stealth-undertraining threshold -- and this trainer had no way to
    # see the same thing happening. --skip-nonfinite additionally GATES the
    # update on the norm being finite (distill's guard); that one does change
    # behaviour, so it is opt-in.
    nf_count = torch.zeros((), dtype=torch.long, device=dev)
    nf_reported = 0
    opt.zero_grad(set_to_none=True)
    while step < CFG["steps"]:
        try:
            xb, yb = next(it)
        except StopIteration:
            # BUG (fixed 2026-07-18): this re-created the iterator WITHOUT
            # rank/world, so after the first corpus exhaustion every rank
            # silently fell back to the rank-0 shard and all four ranks trained
            # on IDENTICAL batches -- destroying the effective batch size and
            # quietly invalidating the token count. Must preserve the sharding.
            it = batch_iter(torch, tok, data, rank=rank, world=world); continue
        if CFG["pin_memory"] and dev != "cpu":
            # --pin-memory: pinned staging + async copy, so the H2D transfer
            # overlaps the previous step's tail instead of blocking on it.
            xb = xb.pin_memory().to(dev, non_blocking=True)
            yb = yb.pin_memory().to(dev, non_blocking=True)
        else:
            xb, yb = xb.to(dev), yb.to(dev)
        # DDP all-reduces gradients on EVERY backward() unless suppressed. With
        # grad_accum > 1 only the final micro-batch of an accumulation group needs
        # to sync, so without no_sync() this box was doing `accum`x the necessary
        # gradient communication -- ~552 MB per all-reduce over a PCIe-only
        # topology (nvidia-smi topo shows NODE, no NVLink), which profiling
        # implicated as the dominant cost (SM 100% / memory-bandwidth ~0%: the
        # signature of NCCL spin-wait, not compute).
        _is_sync_step = ((step + 1) % accum == 0)
        _nosync = (ddp and not _is_sync_step)
        _ctx = model.no_sync() if _nosync else contextlib.nullcontext()
        with _ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=amp_on):
                if CFG["ce_chunk"]:
                    # the head + CE run inside forward, chunk by chunk, so the
                    # (B*T, V) logits tensor is never materialised at once.
                    loss = model(xb, yb) / accum
                else:
                    logits = model(xb)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, CFG["vocab"]), yb.reshape(-1)) / accum
            loss.backward()
        if (step + 1) % accum == 0:
            gn = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), CFG["grad_clip"])
            # stays on-device: no .item(), no sync, no change to the default path
            nf_count += (~torch.isfinite(gn)).long()
            if CFG["skip_nonfinite"] and not bool(torch.isfinite(gn)):
                # distill's guard (distill_fhe_native_ssm.py:381-390). backward()
                # has already run on every rank and the norm is post-all-reduce,
                # so the skip decision is rank-identical and cannot deadlock.
                if rank == 0:
                    print(json.dumps({"event": "skip_nonfinite", "step": step,
                                      "total_skips": int(nf_count)}), flush=True)
            else:
                opt.step()
            sched.step(); opt.zero_grad(set_to_none=True)
        if step % 200 == 0 and rank == 0:
            rec = {"step": step, "loss": round(float(loss.item() * accum), 4),
                   "ppl": round(float(math.exp(min(loss.item() * accum, 20))), 2),
                   "lr": round(sched.get_last_lr()[0], 6),
                   "hrs": round((time.time() - t0) / 3600, 2)}
            # the ONE host sync the counter costs, once per logging interval
            nf_now = int(nf_count)
            updates = max(1, (step - start_step + 1) // accum)
            rec["nonfinite_updates"] = nf_now
            rec["nonfinite_pct"] = round(100.0 * nf_now / updates, 3)
            rec["skipped"] = nf_now if CFG["skip_nonfinite"] else 0
            if nf_now > nf_reported:
                # loud, not silent: >1% is the stealth-undertraining threshold
                # (kaggle_structmix.py:954-959 convention, DIAGNOSIS.md §3)
                print(json.dumps({"event": "nonfinite_grad_norm", "step": step,
                                  "total": nf_now, "pct_of_updates": rec["nonfinite_pct"],
                                  "guarded": bool(CFG["skip_nonfinite"]),
                                  "over_1pct_threshold": rec["nonfinite_pct"] > 1.0}),
                      flush=True)
                nf_reported = nf_now
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if (step % 5000 == 0 or step == CFG["steps"] - 1) and rank == 0:
            _save(raw_model, torch)
            _save_resume(raw_model, opt, step, torch,
                         extra={"data_cursor": _staged_cursor()})
        step += 1
    if rank == 0:
        _save(raw_model, torch)
        _save_resume(raw_model, opt, step - 1, torch,
                     extra={"data_cursor": _staged_cursor()})
    if rank == 0:
        print(json.dumps({"event": "done", "hrs": round((time.time() - t0) / 3600, 2)}))
    if ddp:
        import torch.distributed as dist
        # Rank 0's final _save()/_save_resume() writes ~2.4 GB (706 MB .npz +
        # 1.66 GB .pt). Without this barrier the other ranks fall straight through
        # to destroy_process_group() and EXIT while rank 0 is still writing, which
        # tears rank 0's NCCL communicator down mid-save and aborts it (SIGABRT).
        # Observed at the end of the first native run (2026-07-18): the checkpoint
        # itself survived intact, but torchrun returned nonzero and the campaign
        # driver treated a fully-completed 3000-step run as a failure, skipping
        # its export/eval/Gate-G0 phase.
        dist.barrier()
        dist.destroy_process_group()


TAG = "native"          # set by --tag; namespaces weights/config/bundle files


def _wpath(): return ART / f"{TAG}_weights.npz"
def _cpath(): return ART / f"{TAG}_config.json"
def _spath(): return ART / f"{TAG}_seeds.json"
def _rpath(): return ART / f"{TAG}_resume.pt"


_STAGED = None          # the staged_token_reader module, once installed


class _SelfModule:
    """Attribute get/set that lands in THIS module's globals.

    `staged_token_reader.install(base)` rebinds `base.token_stream`, so it needs
    a module-like target. `sys.modules[__name__]` is the obvious one and it is
    WRONG: a module loaded through `importlib.util.module_from_spec` without
    being registered — which is how the verification driver in
    `training/verify_trainer_fixes.py` and any comparison harness loads this
    file — raises KeyError there. globals() IS the module __dict__ in every
    case, so going through it is both equivalent and unconditional.
    """

    def __getattr__(self, k):
        try:
            return globals()[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        globals()[k] = v


def _install_staged_reader(data, resume):
    """Wire `ml-eval/staged_token_reader.py` in when --data is a staged spec.

    Returns the (possibly rewritten) data spec. A non-staged spec is returned
    untouched and nothing is installed, so a streaming run is unaffected.

    Under --resume the spec is rewritten to on_exhaust=continue unless the
    caller already said otherwise: `--resume` restores model+optimizer+step but
    NOT the data position, and training/DIAGNOSIS.md §3 traced ~48% of q2's
    tokens to seen-before data from exactly that. The cursor itself is carried
    in {TAG}_resume.pt (see _save_resume), so a kill/resume cycle picks up
    where the corpus walk stopped instead of re-walking it."""
    global _STAGED
    if not (isinstance(data, str) and data.startswith("staged:")):
        return data
    import importlib
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    _STAGED = importlib.import_module("staged_token_reader")
    _STAGED.install(_SelfModule())
    if resume:
        data = _STAGED.with_continue(data)
    print(json.dumps({"event": "staged_reader_installed", "data": data,
                      "resume": bool(resume)}), flush=True)
    return data


def _staged_cursor():
    return _STAGED.cursor_state() if _STAGED is not None else None


def _save(model, torch):
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(_wpath(), **sd)
    _cpath().write_text(json.dumps(CFG))


def _save_resume(model, opt, step, torch, extra=None):
    """Torch-native sibling of _save(): model + optimizer + step, so a run
    can genuinely continue past a --steps target it already reached instead
    of restarting from scratch. Does not replace _save()'s .npz output,
    which eval/export/simulate still read.

    `extra` (S2.8) carries the staged-corpus read cursor, so --resume does not
    re-walk the corpus. It is None for a streaming run, and an old checkpoint
    without the key resumes exactly as before."""
    ck = {"model": model.state_dict(), "optim": opt.state_dict(), "step": step}
    if extra:
        ck.update(extra)
    torch.save(ck, _rpath())


def _load_resume_model(model, torch, dev):
    if not _rpath().exists():
        raise FileNotFoundError(f"--resume requested but {_rpath()} does not exist")
    ckpt = torch.load(_rpath(), map_location=dev)
    model.load_state_dict(ckpt["model"])
    return ckpt


def _load_resume_optim(opt, ckpt):
    opt.load_state_dict(ckpt["optim"])
    return ckpt["step"] + 1


# ---------------------------------------------------------------------------
def _fwd_np(w, ids, capture=None):
    """float64 reference == exact circuit semantics. If capture is a dict,
    record per-layer RMSNorm mean-square ranges (for Newton seeds)."""
    d, L, dff = CFG["d_model"], CFG["n_layer"], CFG["d_ffn"]
    lo, hi = CFG["a_min"], CFG["a_max"]
    eps = 1e-5

    def rms(x, g, tag):
        ms = (x * x).mean(-1, keepdims=True) + eps
        if capture is not None:
            capture.setdefault(tag, []).append([float(ms.min()), float(ms.max())])
        return x / np.sqrt(ms) * g

    H = w["emb.weight"][ids].astype(np.float64)
    for i in range(L):
        p = f"blocks.{i}."
        # time mix
        u = rms(H, w[p + "tm.g1"], f"L{i}.tm")
        u_norm = u                                         # pre-shift norm output (parallel block reads it)
        if CFG["shift_mix"]:
            us = np.vstack([np.zeros((1, d)), u[:-1]])
            u = u * w[p + "tm.mix"] + us * (1.0 - w[p + "tm.mix"])
        x = u @ w[p + "tm.win.weight"].T
        a = lo + (hi - lo) / (1.0 + np.exp(-w[p + "tm.a_raw"]))
        b = 1.0 - a
        s = np.zeros(d); S = np.empty_like(x)
        for t in range(x.shape[0]):
            s = a * s + b * x[t]; S[t] = s
        z = w[p + "tm.c"] * S + w[p + "tm.dd"] * x
        # BUG FIXED 2026-07-19: this reference forward ALWAYS applied the
        # polynomial gate, ignoring CFG["nonlin"] -- while build_model()
        # branches on it. So evaluating the matched baseline (trained with
        # SiLU/sigmoid) ran its weights through poly gates whose coefficients
        # were never trained, silently producing garbage: the first baseline
        # eval returned ppl 1.44e9. This invalidated the baseline's eval,
        # export AND exported bundle -- i.e. the entire matched ablation, which
        # is the paper's credibility evidence. The numpy path must mirror the
        # torch path exactly, since it is the float64 reference the whole FHE
        # circuit is checked against.
        if CFG.get("nonlin") == "standard":
            g = z / (1.0 + np.exp(-z))                     # SiLU
        else:
            g = z * (w[p + "tm.p0"] + w[p + "tm.p1"] * z + w[p + "tm.p2"] * z * z)
        attn = CFG["alpha_res"] * (g @ w[p + "tm.wout.weight"].T)
        if CFG.get("block", "sequential") == "parallel":
            u2 = u_norm                                    # S3.3 Tier B: shared norm output
        else:
            H = H + attn
            # channel mix
            u2 = rms(H, w[p + "cm.g2"], f"L{i}.cm")
        k = u2 @ w[p + "cm.wk.weight"].T
        r = u2 @ w[p + "cm.wr.weight"].T
        if CFG.get("nonlin") == "standard":
            act = k / (1.0 + np.exp(-k))                   # SiLU
            gate = 1.0 / (1.0 + np.exp(-r))                # sigmoid  (baseline GLU)
        else:
            act = w[p + "cm.q0"] + w[p + "cm.q1"] * k + w[p + "cm.q2"] * k * k + w[p + "cm.q3"] * k * k * k
            gate = w[p + "cm.r0"] + w[p + "cm.r1"] * r + w[p + "cm.r2"] * r * r
        ffn = (act * gate) @ w[p + "cm.wv.weight"].T
        H = H + attn + ffn if CFG.get("block", "sequential") == "parallel" else H + ffn
    Hf = rms(H, w["g_out"], "ln_out")
    return Hf @ w["emb.weight"].T


def _fwd_circuit(w, ids, sigma_bs, seeds, rng):
    """Level-tracked FHE-circuit forward: identical to _fwd_np but with
    (a) RMSNorm rsqrt via finite Newton from the calibrated linear seed,
    (b) bootstrap noise N(0,sigma_bs) injected whenever a stream's tracked
    level would exhaust. Shares the harness level model (depth 29,
    levelsAfter 10, boot when remaining<5). This is lane N's go/no-go: an
    in-class model has NO division/exp, so the ONLY error is Newton
    truncation + bootstrap noise — it must stay BOUNDED (the RWKV contrast).
    Returns (logits, stats)."""
    import fhe_circuit_sim as S
    if CFG.get("block", "sequential") != "sequential":
        sys.exit("simulate: the level model here is the SEQUENTIAL block's; the parallel "
                 "block's trace is results/theory/tick_level_trace.py --parallel-block")
    d, L, dff = CFG["d_model"], CFG["n_layer"], CFG["d_ffn"]
    lo, hi = CFG["a_min"], CFG["a_max"]
    eps = 1e-5
    # Faithful single-stream level model (mirrors gpu_full_layer.cu's
    # TrackedCt threading the residual stream): ONE accumulating `used`
    # counter carried through the whole forward; a level-consuming op that
    # would drop remaining below BOOT_FLOOR bootstraps FIRST (noise injected,
    # used reset), then consumes. The residual stream and its derived
    # activation path share this counter, so the boot COUNT matches the
    # measured ~2-3/layer (~§7e) rather than the buggy per-tag reset.
    state = {"used": S.DEPTH - S.LEVELS_AFTER, "boots": 0, "worst_abs": 0.0}

    def noisy(x, tag, lv=1):
        if S.DEPTH - state["used"] < lv + S.BOOT_FLOOR:
            x = x + rng.normal(0.0, sigma_bs, x.shape)   # bootstrap: inject noise
            state["used"] = S.DEPTH - S.LEVELS_AFTER
            state["boots"] += 1
        state["used"] += lv
        state["worst_abs"] = max(state["worst_abs"], float(np.abs(x).max()))
        return x

    def rms(x, g, tag):
        ms = (x * x).mean(-1, keepdims=True) + eps
        sd = seeds[tag]
        y = sd["a"] + sd["b"] * ms                     # linear seed
        for _ in range(sd["iters"]):
            y = y * (1.5 - 0.5 * ms * y * y)           # Newton rsqrt
        y = noisy(y, tag + ".rsqrt", 3)
        return noisy(x * y * g, tag + ".out")

    H = w["emb.weight"][ids].astype(np.float64)
    for i in range(L):
        p = f"blocks.{i}."
        u = rms(H, w[p + "tm.g1"], f"L{i}.tm")
        if CFG["shift_mix"]:
            us = np.vstack([np.zeros((1, d)), u[:-1]])
            u = noisy(u * w[p + "tm.mix"] + us * (1.0 - w[p + "tm.mix"]), f"L{i}.tm.shift")
        x = noisy(u @ w[p + "tm.win.weight"].T, f"L{i}.tm.win")
        a = lo + (hi - lo) / (1.0 + np.exp(-w[p + "tm.a_raw"]))
        b = 1.0 - a
        s = np.zeros(d); S_ = np.empty_like(x)
        for t in range(x.shape[0]):
            s = a * s + b * x[t]; S_[t] = s
        S_ = noisy(S_, f"L{i}.tm.scan")                # 1 level per block (amortized: per-token here)
        z = noisy(w[p + "tm.c"] * S_ + w[p + "tm.dd"] * x, f"L{i}.tm.readout")
        g = noisy(z * (w[p + "tm.p0"] + w[p + "tm.p1"] * z + w[p + "tm.p2"] * z * z), f"L{i}.tm.gate", 2)
        H = H + CFG["alpha_res"] * noisy(g @ w[p + "tm.wout.weight"].T, f"L{i}.tm.wout")
        u2 = rms(H, w[p + "cm.g2"], f"L{i}.cm")
        k = noisy(u2 @ w[p + "cm.wk.weight"].T, f"L{i}.cm.wk")
        r = noisy(u2 @ w[p + "cm.wr.weight"].T, f"L{i}.cm.wr")
        act = noisy(w[p + "cm.q0"] + w[p + "cm.q1"] * k + w[p + "cm.q2"] * k * k + w[p + "cm.q3"] * k * k * k, f"L{i}.cm.act", 2)
        gate = noisy(w[p + "cm.r0"] + w[p + "cm.r1"] * r + w[p + "cm.r2"] * r * r, f"L{i}.cm.gate", 2)
        H = H + noisy((act * gate) @ w[p + "cm.wv.weight"].T, f"L{i}.cm.wv")
    Hf = rms(H, w["g_out"], "ln_out")
    return Hf @ w["emb.weight"].T, state


def cmd_simulate(tokens, sigma_bs):
    """Go/no-go: run the trained (or init) model through the noise circuit,
    report ppl/top1/KL vs the float reference. Proves BOUNDEDNESS (contrast
    with RWKV) and predicts FHE quality before any GPU spend."""
    sys.path.insert(0, str(HERE))
    from run_eval import wikitext_tokens, nll_stats
    import fhe_circuit_sim as S
    w = dict(np.load(_wpath()))
    seeds = (json.loads(_spath().read_text()) if _spath().exists()
             else _calib_seeds(w, wikitext_tokens(512)))
    ids = wikitext_tokens(tokens)
    inputs, targets = ids[:-1], np.array(ids[1:])
    base = _fwd_np(w, inputs)
    sim, st = _fwd_circuit(w, inputs, sigma_bs, seeds, np.random.default_rng(0))
    nb, ns = nll_stats(base, targets), nll_stats(sim, targets)
    ppl_b, ppl_s = float(np.exp(nb.mean())), float(np.exp(ns.mean()))
    top1 = float((base.argmax(-1) == sim.argmax(-1)).mean())
    finite = bool(np.isfinite(sim).all())
    rep = {"mode": "simulate_native", "tokens": len(inputs), "sigma_bs": sigma_bs,
           "ppl_float": ppl_b, "ppl_circuit": ppl_s if finite else None,
           "ppl_delta_pct": (100 * (ppl_s - ppl_b) / ppl_b) if finite else None,
           "top1": top1, "finite": finite, "boots": st["boots"],
           "worst_abs_value": st["worst_abs"],
           "gate_G0": bool(finite and abs(ppl_s - ppl_b) / ppl_b < 0.05 and top1 > 0.97)}
    (ART / f"{TAG}_sim_{sigma_bs:.0e}.json").write_text(json.dumps(rep))
    print(json.dumps(rep))


def _calib_seeds(w, ids):
    import fhe_circuit_sim as S
    cap = {}
    _fwd_np(w, ids[:-1], capture=cap)
    seeds = {}
    for tag, rows in cap.items():
        arr = np.array(rows); m, M = float(arr[:, 0].min()), float(arr[:, 1].max())
        # FIXED 2026-07-19: the seed used to be fitted on [m/2, 2M] -- 2x
        # headroom on EACH side of the calibrated range, which quadruples the
        # effective spread (worst tag: M/m 46 -> 186) and therefore the Newton
        # iteration count. Measured consequence: worst tag needed iters=7, and
        # at ~4 levels per iteration that is 1 + 4*7 = 29 levels for RMSNorm
        # alone -- the ENTIRE depth-29 budget, so the op could not fit no
        # matter how bootstraps were scheduled. It showed up on-device as
        # rmsnorm thrashing (~16 bootstraps in one call) and still exhausting
        # the ciphertext to used=28/29, where the remaining modulus cannot
        # represent the message -> "Decode(): approximation error too high".
        # The calibration slice already records the observed min/max, so a
        # modest 20%/25% margin is the honest amount of headroom. Measured:
        # worst-tag levels 29 -> 21, max iters 7 -> 5.
        a, b = S.linear_rsqrt_seed(0.8 * m, 1.25 * M)
        n, ok = S.rsqrt_iters_for(a, b, m, M)
        seeds[tag] = {"a": a, "b": b, "iters": int(n), "ms_range": [m, M], "ok": ok}
    return seeds


def cmd_eval(tokens=2048):
    sys.path.insert(0, str(HERE))
    from run_eval import wikitext_tokens, nll_stats
    w = dict(np.load(_wpath()))
    ids = wikitext_tokens(tokens)
    logits = _fwd_np(w, ids[:-1])
    nll = nll_stats(logits, np.array(ids[1:]))
    print(json.dumps({"native_test_ppl": float(np.exp(nll.mean())),
                      "tokens": tokens}))


def cmd_export(tokens, lanes=1, lane_stride=0, suffix=""):
    # --lanes NL (T1c full-pass driver, 2026-08-29): NL DISJOINT WikiText
    # windows, one fresh float64 forward per lane (fresh state == the
    # harness's cold start), stacked LANE-MAJOR into inputs/logits_ref with a
    # `lanes` tensor [NL, T]. Newton-seed calibration merges every lane's
    # captures (the shared `cap` dict accumulates across _fwd_np calls). A
    # broadcast run of this bundle with --tokens T reads rows 0..T-1 = lane 0
    # exactly; lane r's broadcast reference uses --bundle-row-offset r*T.
    # --bundle-suffix keeps the lanes bundle (and its seeds sidecar) from
    # clobbering the base tag's artifacts.
    sys.path.insert(0, str(HERE))
    from run_eval import wikitext_tokens, nll_stats
    w = dict(np.load(_wpath()))
    if lanes < 1:
        sys.exit(f"--lanes must be >= 1, got {lanes}")
    if lanes == 1:
        ids = wikitext_tokens(tokens)
        inputs, targets = ids[:-1], np.array(ids[1:])
        cap = {}
        logits = _fwd_np(w, inputs, capture=cap)
        nll = nll_stats(logits, targets)
        print(json.dumps({"native_test_ppl": float(np.exp(nll.mean())),
                          "tokens": len(inputs)}))
    else:
        stride = lane_stride if lane_stride > 0 else max(tokens + 1, 512)
        if stride < tokens + 1:
            sys.exit(f"--lane-stride {stride} overlaps: need >= tokens+1 = {tokens + 1}")
        ids = wikitext_tokens((lanes - 1) * stride + tokens + 1)
        cap = {}
        lane_in, lane_lg, lane_ppl = [], [], []
        for r in range(lanes):
            win = ids[r * stride: r * stride + tokens + 1]
            li, lt = win[:-1], np.array(win[1:])
            lg = _fwd_np(w, li, capture=cap)        # fresh state per lane
            lane_in.append(np.asarray(li)); lane_lg.append(lg)
            lane_ppl.append(float(np.exp(nll_stats(lg, lt).mean())))
        inputs = np.concatenate(lane_in)             # (NL*T,) lane-major ids
        logits = np.concatenate(lane_lg, axis=0)     # (NL*T, V) lane-major
        print(json.dumps({"lanes": lanes, "tokens_per_lane": tokens,
                          "lane_stride": stride, "native_test_ppl_per_lane": lane_ppl}))
    # per-tag RMSNorm mean-square range -> Newton rsqrt seed (linear)
    sys.path.insert(0, str(HERE))
    import fhe_circuit_sim as S
    seeds = {}
    for tag, rows in cap.items():
        arr = np.array(rows); m, M = float(arr[:, 0].min()), float(arr[:, 1].max())
        # FIXED 2026-07-19: the seed used to be fitted on [m/2, 2M] -- 2x
        # headroom on EACH side of the calibrated range, which quadruples the
        # effective spread (worst tag: M/m 46 -> 186) and therefore the Newton
        # iteration count. Measured consequence: worst tag needed iters=7, and
        # at ~4 levels per iteration that is 1 + 4*7 = 29 levels for RMSNorm
        # alone -- the ENTIRE depth-29 budget, so the op could not fit no
        # matter how bootstraps were scheduled. It showed up on-device as
        # rmsnorm thrashing (~16 bootstraps in one call) and still exhausting
        # the ciphertext to used=28/29, where the remaining modulus cannot
        # represent the message -> "Decode(): approximation error too high".
        # The calibration slice already records the observed min/max, so a
        # modest 20%/25% margin is the honest amount of headroom. Measured:
        # worst-tag levels 29 -> 21, max iters 7 -> 5.
        a, b = S.linear_rsqrt_seed(0.8 * m, 1.25 * M)
        n, ok = S.rsqrt_iters_for(a, b, m, M)
        seeds[tag] = {"a": a, "b": b, "iters": int(n), "ms_range": [m, M], "ok": ok}

    tensors, blobs, off = {}, [], 0

    def put(name, arr):
        nonlocal off
        a = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
        tensors[name] = {"offset": off, "shape": list(a.shape)}
        blobs.append(a.tobytes()); off += a.nbytes

    lo, hi = CFG["a_min"], CFG["a_max"]
    meta = [CFG["d_model"], CFG["n_layer"], CFG["vocab"], CFG["d_ffn"],
            CFG["alpha_res"], 1.0 if CFG["shift_mix"] else 0.0]
    if CFG.get("block", "sequential") == "parallel":
        meta.append(1.0)          # meta[6] = block type: 1 parallel (absent/0 = sequential; S3.3 Tier B)
    put("meta", meta)
    for i in range(CFG["n_layer"]):
        p = f"blocks.{i}."
        put(f"L{i}.decay", lo + (hi - lo) / (1.0 + np.exp(-w[p + "tm.a_raw"])))
        put(f"L{i}.b", 1.0 - (lo + (hi - lo) / (1.0 + np.exp(-w[p + "tm.a_raw"]))))
        put(f"L{i}.tm.g1", w[p + "tm.g1"])
        if CFG["shift_mix"]:
            put(f"L{i}.tm.mix", w[p + "tm.mix"])
        put(f"L{i}.tm.win", w[p + "tm.win.weight"])
        for nm in ("c", "dd", "p0", "p1", "p2"):
            put(f"L{i}.tm.{nm}", w[p + f"tm.{nm}"])
        put(f"L{i}.tm.wout", w[p + "tm.wout.weight"])
        put(f"L{i}.cm.g2", w[p + "cm.g2"])
        put(f"L{i}.cm.wk", w[p + "cm.wk.weight"]); put(f"L{i}.cm.wr", w[p + "cm.wr.weight"])
        for nm in ("q0", "q1", "q2", "q3", "r0", "r1", "r2"):
            put(f"L{i}.cm.{nm}", w[p + f"cm.{nm}"])
        put(f"L{i}.cm.wv", w[p + "cm.wv.weight"])
        put(f"L{i}.tm.rsqrt", [seeds[f"L{i}.tm"]["a"], seeds[f"L{i}.tm"]["b"], seeds[f"L{i}.tm"]["iters"]])
        put(f"L{i}.cm.rsqrt", [seeds[f"L{i}.cm"]["a"], seeds[f"L{i}.cm"]["b"], seeds[f"L{i}.cm"]["iters"]])
    put("g_out", w["g_out"])
    put("ln_out.rsqrt", [seeds["ln_out"]["a"], seeds["ln_out"]["b"], seeds["ln_out"]["iters"]])
    put("head", w["emb.weight"])
    put("inputs", w["emb.weight"][inputs])
    put("logits_ref", logits.astype(np.float64))
    if lanes > 1:
        put("lanes", [lanes, tokens])
    name = f"{TAG}{suffix}"
    (ART / f"bundle_{name}.bin").write_bytes(b"".join(blobs))
    lines = [f"{k} {v['offset']} {int(np.prod(v['shape'])) if v['shape'] else 1} "
             f"{' '.join(str(x) for x in v['shape'])}" for k, v in tensors.items()]
    (ART / f"bundle_{name}.index.txt").write_text("\n".join(lines) + "\n")
    spath = ART / f"{name}_seeds.json" if suffix else _spath()
    spath.write_text(json.dumps(seeds, indent=2))
    print(json.dumps({"exported": f"bundle_{name}.bin", "bytes": off,
                      "rsqrt_basin_ok": all(s["ok"] for s in seeds.values()),
                      "worst_rsqrt_iters": max(s["iters"] for s in seeds.values())}))


# ---------------------------------------------------------------------------
# CLI. Factored out of __main__ (S2.8, 2026-08-22) so there is exactly ONE
# place CFG-mutating flags are parsed. `train_fhe_native_ssm_masked.py` and any
# other import-as-module lane call apply_cli_overrides() instead of
# re-implementing the parse -- which is what "one edit reaches both lanes"
# actually requires; before this, the masked lane had its own copy and would
# have silently ignored --batch.
CLI_FLAGS_WITH_VALUE = ("--tag", "--layers", "--d-model", "--nonlin", "--seed",
                        "--data", "--steps", "--tokens", "--sigma-bs",
                        "--batch", "--grad-accum", "--ctx", "--scan", "--gates",
                        "--ce-chunk", "--lanes", "--lane-stride", "--bundle-suffix",
                        "--block", "--d-ffn")
CLI_FLAGS_BOOL = ("--resume", "--compile", "--skip-nonfinite", "--pin-memory",
                  "--window-carry", "--bf16")
CLI_ENUMS = {"scan": ("hs", "rfft"), "gates": ("expanded", "horner"),
             "nonlin": ("poly", "standard"), "block": ("sequential", "parallel")}


def cli_opt(args, flag, default, cast=str):
    if flag not in args:
        return default
    raw = args[args.index(flag) + 1]
    try:
        return cast(raw)
    except (TypeError, ValueError):
        # queue files carry deliberate argv placeholders (the FLAGS_* /
        # *_FROM_* convention in campaign/queues/*.json) that must REFUSE to
        # run rather than silently fall back to a default.
        sys.exit(f"{flag} got {raw!r}, which is not a valid "
                 f"{getattr(cast, '__name__', cast)}. If this is a queue "
                 f"placeholder (e.g. BATCH_FROM_CARD_INTAKE), the card-intake "
                 f"step has not filled it in — see training/vram_sizing.py.")


def reject_unknown_flags(args, extra_value_flags=(), extra_bool_flags=()):
    """Hard-fail on a flag this parser does not know.

    L40S_RUNBOOK.md §3.7: "the parser ignores unknown flags", so a queue cmd
    with a typo or a flag from another lane SILENTLY TRAINS THE DEFAULTS.
    That is a wrong measurement, not a failure, and it is exactly the class
    the one-shot run cannot afford. Valid invocations are unaffected."""
    val = set(CLI_FLAGS_WITH_VALUE) | set(extra_value_flags)
    boo = set(CLI_FLAGS_BOOL) | set(extra_bool_flags)
    i, unknown = 0, []
    while i < len(args):
        a = args[i]
        if a in val:
            i += 2; continue
        if a in boo:
            i += 1; continue
        if a.startswith("--"):
            unknown.append(a)
        i += 1
    if unknown:
        sys.exit(f"unknown flag(s) {unknown} — this parser silently ignored "
                 f"unknown flags before 2026-08-22, which trains the DEFAULTS "
                 f"under a command line that looks configured. Known: "
                 f"{sorted(val)} (value) {sorted(boo)} (bool)")


def apply_cli_overrides(args, cfg=None):
    """Parse every CFG-mutating flag. Returns cfg (CFG by default)."""
    cfg = CFG if cfg is None else cfg
    cfg["n_layer"] = cli_opt(args, "--layers", cfg["n_layer"], int)
    if "--d-model" in args:
        cfg["d_model"] = cli_opt(args, "--d-model", cfg["d_model"], int)
        cfg["d_ffn"] = 4 * cfg["d_model"]
    # S3.3 Tier B: d_ffn as a first-class flag (applied AFTER --d-model's 4x default)
    if "--d-ffn" in args:
        cfg["d_ffn"] = cli_opt(args, "--d-ffn", cfg["d_ffn"], int)
    cfg["block"] = cli_opt(args, "--block", cfg.get("block", "sequential"))
    cfg["nonlin"] = cli_opt(args, "--nonlin", cfg["nonlin"])
    cfg["seed"] = cli_opt(args, "--seed", cfg["seed"], int)
    # --- S2.8 levers. Absent flag => the CFG default => pre-session behaviour.
    cfg["batch"] = cli_opt(args, "--batch", cfg["batch"], int)
    cfg["grad_accum"] = cli_opt(args, "--grad-accum", cfg["grad_accum"], int)
    cfg["ctx"] = cli_opt(args, "--ctx", cfg["ctx"], int)
    cfg["scan"] = cli_opt(args, "--scan", cfg["scan"])
    cfg["gates"] = cli_opt(args, "--gates", cfg["gates"])
    cfg["ce_chunk"] = cli_opt(args, "--ce-chunk", cfg["ce_chunk"], int)
    cfg["compile"] = cfg["compile"] or ("--compile" in args)
    cfg["skip_nonfinite"] = cfg["skip_nonfinite"] or ("--skip-nonfinite" in args)
    cfg["pin_memory"] = cfg["pin_memory"] or ("--pin-memory" in args)
    cfg["window_carry"] = cfg["window_carry"] or ("--window-carry" in args)
    cfg["bf16"] = cfg["bf16"] or ("--bf16" in args)
    for key, allowed in CLI_ENUMS.items():
        if cfg[key] not in allowed:
            sys.exit(f"--{key.replace('_', '-')} must be one of {allowed}, "
                     f"got {cfg[key]!r}")
    for key in ("batch", "grad_accum", "ctx", "n_layer", "d_model", "d_ffn"):
        if int(cfg[key]) <= 0:
            sys.exit(f"--{key.replace('_', '-')} must be positive, got {cfg[key]}")
    if cfg["ce_chunk"] < 0:
        sys.exit(f"--ce-chunk must be >= 0 (0 = off), got {cfg['ce_chunk']}")
    return cfg


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "eval", "export", "simulate"):
        sys.exit(__doc__)
    cmd = sys.argv[1]
    args = sys.argv[2:]

    def opt(flag, default, cast=str):
        return cast(args[args.index(flag) + 1]) if flag in args else default

    reject_unknown_flags(args)
    # --tag namespaces every artifact (weights/config/seeds/bundle), so the
    # depth-stress variant trains+exports alongside the base model:
    #   train --tag native24 --layers 24   ->  bundle_native24.*
    TAG = opt("--tag", TAG)
    apply_cli_overrides(args)
    if _cpath().exists() and cmd != "train":
        CFG.update(json.loads(_cpath().read_text()))   # match the trained shape

    if cmd == "train":
        # --resume continues an existing <tag>_resume.pt (model + optimizer +
        # step) instead of training from scratch -- used to push a run past
        # a --steps target it already hit (see cmd_train docstring).
        cmd_train(opt("--data", "fineweb"), opt("--steps", CFG["steps"], int),
                  resume="--resume" in args)
    elif cmd == "eval":
        cmd_eval(opt("--tokens", 2048, int))
    elif cmd == "simulate":
        cmd_simulate(opt("--tokens", 512, int), opt("--sigma-bs", 4.9e-4, float))
    else:
        cmd_export(opt("--tokens", 512, int), lanes=opt("--lanes", 1, int),
                   lane_stride=opt("--lane-stride", 0, int),
                   suffix=opt("--bundle-suffix", ""))
