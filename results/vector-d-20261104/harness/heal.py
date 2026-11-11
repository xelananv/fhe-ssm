# SPDX-License-Identifier: MIT
"""Block-wise distillation ("healing") of the polynomial model against the exact model.

Teacher: the checkpoint in exact mode, frozen. Student: a copy of it in poly mode with the coefficients JSON fixed
(the polynomials are the circuit; only weights adapt). Loss per step:

  loss = --mse-weight * mean_i MSE(h_student_i, h_teacher_i) / var(h_teacher_i)  +  --kl-weight * KL(teacher || student)

h_i is the residual stream after block i. With --blockwise global (default) the student runs on its own trajectory;
with --blockwise local each student block is fed the TEACHER's input to that block (and the teacher's v_first), so
block i is trained to undo only its own polynomial error; the KL term always comes from a full student pass.
The variance normalisation makes blocks with large residual norms comparable to early ones.

Trainable parameters: --train all | blocks:0,3,7 | regex:<pattern on parameter names>. Everything else is frozen.
Data: random windows of --ctx tokens from a raw text file. bf16 autocast for the student (CUDA or CPU); the
teacher runs in fp32 unless --no-teacher-fp32; the recurrence and every site run in fp32 regardless. Fixed seed; checkpoints every --save-every steps; one JSONL line per --log-every steps.

  python heal.py --model M.pth --coeffs C.json --text wiki.train.raw --train blocks:5,6 --steps 2000 --out heal_run/
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import BOS, append_jsonl, sha256_file, tokenize_file
from rwkv7_model import Sites, _layer_norm, load_rwkv7, pick_device
from rwkv_tokenizer import RWKVTokenizer


def select_params(model, spec: str):
    if spec == "all":
        pred = lambda n: True
    elif spec.startswith("blocks:"):
        ids = {int(i) for i in spec[7:].split(",") if i}
        pred = lambda n: (m := re.match(r"blocks\.(\d+)\.", n)) is not None and int(m.group(1)) in ids
    elif spec.startswith("regex:"):
        pat = re.compile(spec[6:])
        pred = lambda n: pat.search(n) is not None
    else:
        raise ValueError(spec)
    names = []
    for n, p in model.named_parameters():
        p.requires_grad_(pred(n))
        if p.requires_grad:
            names.append(n)
    if not names:
        raise ValueError(f"--train {spec!r} selects no parameters")
    return names


def teacher_states(teacher, x):
    """Teacher residual stream before and after every block, its v_first, and its logits."""
    s = teacher.sites
    h = teacher.emb(x)
    v_first = torch.zeros_like(h)
    ins, outs, vfs = [], [], []
    for blk in teacher.blocks:
        ins.append(h)
        vfs.append(v_first)
        h, v_first = blk(s, h, v_first)
        outs.append(h)
    logits = teacher.head(_layer_norm(s, "ln_out.rsqrt", h, teacher.ln_out))
    return ins, outs, vfs, logits


def heal_loss(student, teacher, x, a):
    with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=not a.teacher_fp32):
        t_in, t_out, t_vf, t_logits = teacher_states(teacher, x)
    logits, hidden = student(x, return_hidden=True)
    terms = []
    for i, blk in enumerate(student.blocks):
        if a.blockwise == "local":
            h_s, _ = blk(student.sites, t_in[i], t_vf[i] if i > 0 else torch.zeros_like(t_in[i]))
        else:
            h_s = hidden[i]
        tv = t_out[i].float().var().clamp(min=1e-8)
        terms.append(F.mse_loss(h_s.float(), t_out[i].float()) / tv)
    mse = torch.stack(terms).mean()
    lt = F.log_softmax(t_logits.float(), -1)
    ls = F.log_softmax(logits.float(), -1)
    kl = (lt.exp() * (lt - ls)).sum(-1).mean()
    loss = a.mse_weight * mse + a.kl_weight * kl
    return loss, mse.detach(), kl.detach(), [t.item() for t in terms]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--coeffs", required=True)
    ap.add_argument("--clamp", action="store_true")
    ap.add_argument("--text", required=True)
    ap.add_argument("--max-text-tokens", type=int, default=None)
    ap.add_argument("--train", default="all")
    ap.add_argument("--blockwise", choices=["global", "local"], default="global")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--tokens", type=int, default=None, help="stop after this many training tokens (overrides --steps)")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--mse-weight", type=float, default=1.0)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--teacher-fp32", action=argparse.BooleanOptionalAction, default=True,
                    help="run the teacher outside autocast: bf16 rounding (~4e-3 relative) is the size of the "
                         "polynomial errors being healed, so bf16 targets would blur them")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "heal_log.jsonl")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = pick_device(a.device)

    teacher = load_rwkv7(a.model, device=dev, sites=Sites("exact"))
    for p in teacher.parameters():
        p.requires_grad_(False)
    s_sites = Sites("poly")
    t_sites, teacher.sites = teacher.sites, None   # Sites holds a torch.Generator: keep it out of deepcopy
    student = copy.deepcopy(teacher)
    teacher.sites = t_sites
    student.sites = s_sites
    cfg = json.load(open(a.coeffs))
    s_sites.load_poly(cfg, clamp=a.clamp)
    student.train()
    student.grad_ckpt = a.grad_ckpt
    names = select_params(student, a.train)
    params = [p for p in student.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.99))

    tok = RWKVTokenizer(a.vocab) if a.vocab else RWKVTokenizer()
    ids = tokenize_file(a.text, tok, a.max_text_tokens)
    if len(ids) < a.ctx + 1:
        raise ValueError("text shorter than one window")
    steps = a.steps if a.tokens is None else math.ceil(a.tokens / (a.batch * a.ctx))
    meta = {"event": "start", "model": a.model, "weights_sha256": sha256_file(a.model), "coeffs": a.coeffs,
            "n_poly_sites": len(s_sites.polys), "train": a.train, "n_train_params": n_train,
            "n_train_tensors": len(names), "args": vars(a), "steps": steps, "text_tokens": int(len(ids)),
            "device": str(dev)}
    append_jsonl(log, meta)
    print(json.dumps({k: meta[k] for k in ("n_poly_sites", "train", "n_train_params", "steps", "device")}))

    use_ac = a.bf16 and dev.type in ("cuda", "cpu")
    t0 = time.time()
    tokens_seen = 0
    for step in range(1, steps + 1):
        starts = rng.integers(0, len(ids) - a.ctx, size=a.batch)
        x = torch.stack([torch.as_tensor(np.concatenate([[BOS], ids[s: s + a.ctx - 1]])) for s in starts]).to(dev)
        lr = a.lr * min(1.0, step / max(1, a.warmup))
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_ac):
            loss, mse, kl, per_block = heal_loss(student, teacher, x, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip).item()
        opt.step()
        tokens_seen += x.numel()
        if step % a.log_every == 0 or step == 1 or step == steps:
            rec = append_jsonl(log, {"event": "step", "step": step, "tokens": tokens_seen, "loss": loss.item(),
                                     "mse_norm": mse.item(), "kl": kl.item(), "grad_norm": gn, "lr": lr,
                                     "per_block_mse": per_block, "seconds": round(time.time() - t0, 1)})
            print(f"step {step} loss {rec['loss']:.5f} mse {rec['mse_norm']:.5f} kl {rec['kl']:.5f} "
                  f"gn {gn:.3f} {rec['seconds']} s", flush=True)
        if step % a.save_every == 0 or step == steps:
            path = os.path.join(a.out, f"healed_step{step}.pt")
            torch.save({"model": student.state_dict(), "step": step, "tokens": tokens_seen,
                        "coeffs": a.coeffs, "base_weights_sha256": meta["weights_sha256"]}, path)
            append_jsonl(log, {"event": "save", "step": step, "path": path, "sha256": sha256_file(path)})
    append_jsonl(log, {"event": "end", "steps": steps, "tokens": tokens_seen, "seconds": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
