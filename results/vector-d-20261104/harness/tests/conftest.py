# SPDX-License-Identifier: MIT
import os
import re
import sys

try:
    import pytest
except ImportError:  # tests/run_without_pytest.py: fixtures become plain functions
    class pytest:  # noqa: N801
        @staticmethod
        def fixture(*a, **k):
            return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(HERE)
RAW = os.path.join(HARNESS, "..", "raw")
sys.path.insert(0, HARNESS)

TINY = dict(n_layer=3, n_embd=128, head_size=64, d_decay=16, d_aaa=16, d_mv=8, d_gate=32, vocab_size=65536)


def load_reference_namespace():
    """Execute the model part of the reference GPT-mode demo (raw/rwkv_v7_demo.py:1-372: tokenizer class, the
    non-CUDA RWKV7_OP, TimeMix, ChannelMix, Block, RWKV) with five textual patches and nothing else changed:
    the CUDA kernel off (line 43), fp32 instead of fp16 (line 38), plain nn.Module instead of TorchScript (45-46), and
    no tokenizer construction at import (129). Lines 374+ (checkpoint load, inference) are not executed."""
    src = open(os.path.join(RAW, "rwkv_v7_demo.py"), encoding="utf-8").read().splitlines()[:372]
    src = "\n".join(src)
    patches = [
        ("USE_CUDA_KERNEL = True", "USE_CUDA_KERNEL = False"),
        ("DTYPE = torch.half # better", "DTYPE = torch.float"),
        ("MyModule = torch.jit.ScriptModule", "MyModule = nn.Module"),
        ("MyFunction = torch.jit.script_method", "MyFunction = lambda f: f"),
        ('tokenizer = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")', "pass"),
    ]
    for old, new in patches:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    ns = {"__name__": "rwkv_v7_demo_ref"}
    exec(compile(src, "rwkv_v7_demo.py", "exec"), ns)
    return ns


def make_reference_model(seed=0, **over):
    cfg = {**TINY, **over}
    ns = load_reference_namespace()
    ns["D_DECAY_LORA"], ns["D_AAA_LORA"], ns["D_MV_LORA"], ns["D_GATE_LORA"] = (
        cfg["d_decay"], cfg["d_aaa"], cfg["d_mv"], cfg["d_gate"])
    ns["HEAD_SIZE"] = cfg["head_size"]
    args = ns["args"]
    args.n_layer, args.n_embd, args.vocab_size, args.head_size_a = (
        cfg["n_layer"], cfg["n_embd"], cfg["vocab_size"], cfg["head_size"])
    torch.manual_seed(seed)
    ref = ns["RWKV"](args)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if name.endswith(("ln0.weight", "ln1.weight", "ln2.weight", "ln_out.weight", "ln_x.weight")):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif re.search(r"\.x_[rwkvag]$|\.x_k$", name):
                p.copy_(torch.rand(p.shape, generator=g))
            elif name.endswith(("w0",)):
                p.copy_(-1.0 + 1.5 * torch.randn(p.shape, generator=g))
            elif name.endswith("k_a"):
                p.copy_(torch.rand(p.shape, generator=g))
            elif name.endswith("k_k"):
                p.copy_(0.8 + 0.2 * torch.rand(p.shape, generator=g))
            elif p.dim() >= 2 and "emb" not in name:
                p.copy_(torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5)
            else:
                p.copy_(0.5 * torch.randn(p.shape, generator=g))
    return ref.float().eval(), cfg



@pytest.fixture(scope="session")
def ref_and_ckpt(tmp_path_factory):
    ref, cfg = make_reference_model()
    d = tmp_path_factory.mktemp("ckpt")
    path = str(d / "tiny.pth")
    torch.save({k: v.detach().to(torch.bfloat16).float() for k, v in ref.state_dict().items()}, path)
    return ref, path


@pytest.fixture(scope="session")
def text_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    para = ("The quick brown fox jumps over the lazy dog. In 1854 the river flooded the valley, and the town "
            "council voted to build a stone bridge across it. Numbers like 3.14159 and 2,718 appear too. ")
    txt = d / "text.raw"
    txt.write_text(para * 40, encoding="utf-8")
    lam = d / "lambada.jsonl"
    import json
    lines = [{"text": "the cat sat on the mat and looked at the dog"},
             {"text": "she opened the door and saw her brother"},
             {"text": "the river flooded the valley and the town council voted"}]
    lam.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return str(txt), str(lam)
