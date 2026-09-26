#!/usr/bin/env python3
"""
serve_ollama.py — chat with the dense demo checkpoint from any Ollama client.

WHY A SERVER AND NOT A MODELFILE
--------------------------------
Ollama cannot load this model. It runs GGUF through llama.cpp (and its own Go
engine); both dispatch on an architecture string that must already be
implemented in the binary — a Modelfile only selects and configures an
architecture, it cannot describe a new one. This checkpoint's architecture
(`ml-eval/train_fhe_native_ssm.py::build_model`: token-shift + a NON-selective
per-channel decay scan + learned polynomial gates, no attention, no KV cache)
is not one of the implemented ones, and the nearest neighbour in llama.cpp
(rwkv6/7) computes different math, so `ollama create` cannot be made to work
without patching llama.cpp itself.

So this serves the SAME forward pass behind the Ollama HTTP API on its own
port. `OLLAMA_HOST=127.0.0.1:11435 ollama run fhe-ssm-430m` talks to it, as do
Open WebUI and anything that speaks /v1/chat/completions.

PLAINTEXT. This is the plaintext model that the FHE harness evaluates
homomorphically. Nothing here is encrypted; no timing printed here says
anything whatsoever about the FHE side (that is `hpc_gpu_port/`, seconds per
token, not tokens per second).

EXACTNESS
---------
`ml-eval/chat_native.py` re-forwards the whole window per token, on purpose, to
avoid a second decoding implementation drifting from the trained one. That
costs O(T) per token and is too slow to chat with. This file instead carries
the recurrence's own state, which is exact here for the same reason Lemma 2
applies under FHE: the decay `a` is non-selective, so

    S_t = a·S_{t-1} + (1-a)·x_t

is a plain linear recurrence and a chunk of T tokens can be resumed from
(prev_u, S) with a closed-form carry. ONE code path serves prefill (T=P) and
decode (T=1); `--selftest` checks it against the model's own batched forward
and writes the record to results/. Run it after touching anything here.

Usage
  # one-time check, writes results/serve-ollama-<date>/SELFTEST.json
  .venv/bin/python ml-eval/serve_ollama.py --selftest

  # terminal chat, no ollama involved
  .venv/bin/python ml-eval/serve_ollama.py --repl

  # serve, then:  OLLAMA_HOST=127.0.0.1:11435 ollama run fhe-ssm-430m
  .venv/bin/python ml-eval/serve_ollama.py --port 11435
"""
import argparse
import datetime
import json
import os
import pathlib
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import train_fhe_native_ssm as tfns  # noqa: E402

DEFAULT_TAG = "pbd430a"
DEFAULT_NAME = "fhe-ssm-430m"
# the canonical demo checkpoint (PUBLICATION_PLAN.md step 2; chmod 444,
# sha256 3ec3f1d8b9b71e7eee38055a7621213991c6417a89576f1c02590963293e784b,
# recorded in results/dense-demo-s31/*/pod_pull/stage_verify.txt)
DEMO_WEIGHTS = pathlib.Path.home() / "Documents/fhe-ssm-backup/pbd430a_weights_DEMO_MODEL_20260901.npz"
MIRROR_WEIGHTS = REPO / "results/dense-demo-s31/pod/checkpoints/ckpt_mirror/pbd430a_weights.npz"

# A base LM trained on FineWeb-Edu and annealed on WikiText-103. It has never
# seen an instruction, a chat turn or a ChatML marker. Handed raw chat text it
# autocompletes; handed literal <|im_start|> it treats the markers as prose.
# The few-shot scaffold below is what makes a chat client usable at all, and is
# the same shape that was verified to work for the pythia:410m-qa smoke model.
QA_PREAMBLE = ("The following are questions with short, factual answers.\n\n"
               "Q: What is the capital of France?\nA: Paris.\n\n"
               "Q: Who wrote Hamlet?\nA: William Shakespeare.\n\n")
QA_STOPS = ["\nQ:", "\n\n", "<|endoftext|>"]
DIALOGUE_STOPS = ["\nUser:", "\nuser:", "<|endoftext|>"]


def now_rfc3339():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
def pick_weights(explicit):
    if explicit:
        return pathlib.Path(explicit).expanduser()
    for cand in (DEMO_WEIGHTS, MIRROR_WEIGHTS, tfns.ART / f"{DEFAULT_TAG}_weights.npz"):
        if cand.exists():
            return cand
    raise SystemExit("no weights found; pass --weights")


def load_model(tag, weights, config, device, dtype, torch):
    cfg_path = pathlib.Path(config).expanduser() if config else tfns.ART / f"{tag}_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    tfns.TAG = tag
    tfns.CFG.update(json.loads(cfg_path.read_text()))
    cfg = dict(tfns.CFG)

    model = tfns.build_model(torch)
    sd = model.state_dict()
    wpath = pick_weights(weights)
    t0 = time.time()
    with np.load(wpath) as z:
        have = set(z.files)
        missing = [k for k in sd if k not in have]
        if missing:
            raise SystemExit(f"{wpath.name}: missing {len(missing)} keys, first {missing[:3]}")
        # emb.weight and head.weight are TIED in build_model() (one tensor, two
        # state_dict keys). If the export ever wrote two different arrays the
        # second copy_ would silently win, so check instead of assuming.
        if "emb.weight" in have and "head.weight" in have:
            if not np.array_equal(z["emb.weight"], z["head.weight"]):
                raise SystemExit(f"{wpath.name}: emb.weight != head.weight but the model ties them")
        with torch.no_grad():
            for name, p in sd.items():
                arr = z[name]
                if tuple(arr.shape) != tuple(p.shape):
                    raise SystemExit(f"shape mismatch {name}: npz {arr.shape} vs model {tuple(p.shape)}")
                p.copy_(torch.from_numpy(arr))
    model.eval().to(device=device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, cfg_path, wpath, time.time() - t0


def load_tokenizer():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(tfns.TOK_MODEL)
    except Exception:
        os.environ.pop("HF_HUB_OFFLINE", None)
        return AutoTokenizer.from_pretrained(tfns.TOK_MODEL)


# ---------------------------------------------------------------------------
# the one forward path: a chunk of T tokens, resumable
# ---------------------------------------------------------------------------
def make_state(cfg, device, dtype, torch):
    """(prev_u, S) per layer — the whole carried state. No KV cache exists in
    this architecture, so this is O(n_layer * d_model) whatever the context
    length: 24 * 1024 * 2 floats here, ~200 kB, constant per conversation."""
    d, L = cfg["d_model"], cfg["n_layer"]
    return [{"prev_u": torch.zeros(1, 1, d, device=device, dtype=dtype),
             "S": torch.zeros(1, 1, d, device=device, dtype=torch.float32)}
            for _ in range(L)]


def clone_state(st):
    return [{"prev_u": s["prev_u"].clone(), "S": s["S"].clone()} for s in st]


def _rmsnorm(x, g, torch, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g


def _scan(a, xb, s0, torch):
    """h_t = a·h_{t-1} + xb_t over a chunk, resuming from s0.

    First term: the trainer's Hillis-Steele doubling, verbatim (valid because
    `a` is a static per-channel parameter, not input-dependent). Second term:
    the carry a^{t+1}·s0, exact because the recurrence is linear. At T=1 the
    loop does not run and this reduces to xb + a·s0 with no extra error."""
    T = xb.shape[1]
    h = xb
    cur = a
    step = 1
    while step < T:
        shifted = torch.nn.functional.pad(h, (0, 0, step, 0))[:, :T]
        h = h + cur * shifted
        cur = cur * cur
        step *= 2
    pw = torch.cumprod(a.unsqueeze(0).expand(T, -1), dim=0)     # a^1 .. a^T
    return h + pw.unsqueeze(0) * s0


def _tm_branch(tm, u, st_i, cfg, torch):
    """TimeMix from an ALREADY-NORMALISED input, mirroring TimeMix.branch()."""
    if tm.mix is not None:
        us = torch.cat([st_i["prev_u"], u[:, :-1]], dim=1)      # u[t-1], carried
        st_i["prev_u"] = u[:, -1:].clone()
        u = u * tm.mix + us * (1.0 - tm.mix)
    x = tm.win(u)
    a = tm.decay().float()
    xb = (1.0 - a) * x.float()
    s_full = _scan(a, xb, st_i["S"], torch)
    st_i["S"] = s_full[:, -1:].clone()
    S = s_full.to(u.dtype)
    z = tm.c * S + tm.dd * x
    if cfg["nonlin"] == "standard":
        g = z * torch.sigmoid(z)
    elif cfg["gates"] == "horner":
        g = z * (tm.p0 + z * (tm.p1 + z * tm.p2))
    else:
        g = z * (tm.p0 + tm.p1 * z + tm.p2 * z * z)
    return tm.wout(g)


def _cm_branch(cm, u, cfg, torch):
    """ChannelMix from an ALREADY-NORMALISED input, mirroring ChannelMix.branch().
    Stateless: the channel mix has no token shift."""
    import torch.nn.functional as F
    k, r = cm.wk(u), cm.wr(u)
    if cfg["nonlin"] == "standard":
        act, gate = F.silu(k), torch.sigmoid(r)
    elif cfg["gates"] == "horner":
        act = cm.q0 + k * (cm.q1 + k * (cm.q2 + k * cm.q3))
        gate = cm.r0 + r * (cm.r1 + r * cm.r2)
    else:
        act = cm.q0 + cm.q1 * k + cm.q2 * k * k + cm.q3 * k * k * k
        gate = cm.r0 + cm.r1 * r + cm.r2 * r * r
    return cm.wv(act * gate)


def forward_chunk(model, ids, st, cfg, torch, last_only=True):
    """ids: (1,T) int64 -> logits (1,T,V) or (1,1,V). Mutates st in place."""
    alpha = cfg["alpha_res"]
    h = model.emb(ids)
    for i, blk in enumerate(model.blocks):
        if cfg["block"] == "parallel":
            u = _rmsnorm(h, blk.tm.g1, torch)
            h = h + alpha * _tm_branch(blk.tm, u, st[i], cfg, torch) + _cm_branch(blk.cm, u, cfg, torch)
        else:
            h = h + alpha * _tm_branch(blk.tm, _rmsnorm(h, blk.tm.g1, torch), st[i], cfg, torch)
            h = h + _cm_branch(blk.cm, _rmsnorm(h, blk.cm.g2, torch), cfg, torch)
    if last_only:
        h = h[:, -1:]
    return model.head(_rmsnorm(h, model.g_out, torch))


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
class Sampler(object):
    """The pod's decoding policy, plus the sampling the Ollama API exposes.

    The logit surgery is a port of `spec_decode/decode_policy.py::modified`
    ("norepeat"), in that file's order and with its constants — repetition
    penalty 1.3 over the last 64 FED tokens (prompt included, the CTRL/HF rule:
    a positive logit is divided, a negative one multiplied), then a no-repeat
    4-gram ban, then argmax. That policy exists because greedy alone repeats
    itself: 64 of 64 lanes of the 2026-09-19 selection repeat a 6-word sequence,
    median first repeat at word 21 (decode_policy.py's own docstring).

    Kept in float64 like the original so the two agree exactly;
    `ml-eval/test_serve_ollama_policy.py` asserts they do over random cases.
    Temperature/top-k/top-p run AFTER, and are a no-op at temperature 0, which
    is the pod's setting."""

    def __init__(self, temperature=0.0, top_p=0.9, top_k=40,
                 repeat_penalty=1.3, repeat_last_n=64, no_repeat_ngram=4,
                 seed=None, torch=None):
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repeat_penalty = float(repeat_penalty)
        self.repeat_last_n = int(repeat_last_n)
        self.no_repeat_ngram = int(no_repeat_ngram)
        self.gen = None
        if seed is not None:
            self.gen = torch.Generator().manual_seed(int(seed))

    def pick(self, logits, history, torch, vocab_limit=None):
        """history = prompt + generated so far, INCLUDING the token that
        produced these logits — the same convention as decode_policy.pick."""
        # .cpu() FIRST, then float64: MPS has no float64, so a combined
        # .to("cpu", torch.float64) tries to build the f64 tensor on-device and
        # raises. .clone() is not optional either: .to()/.cpu() are no-ops when
        # the tensor already matches, and every line below writes in place.
        # decode_policy.py copies for the same reason (np.array(..., copy=True));
        # without it the caller's logits come back penalised — caught by
        # test_serve_ollama_policy.py at 1 flipped argmax in 400 cases.
        lg = logits.detach().cpu().to(torch.float64).reshape(-1).clone()
        if vocab_limit is not None and vocab_limit < lg.shape[0]:
            lg[vocab_limit:] = float("-inf")
        # 1. repetition penalty over the last `window` fed tokens
        if self.repeat_penalty and self.repeat_penalty != 1.0 and history:
            idx = torch.tensor(sorted(set(history[-self.repeat_last_n:])), dtype=torch.long)
            v = lg[idx]
            lg[idx] = torch.where(v > 0, v / self.repeat_penalty, v * self.repeat_penalty)
        # 2. no-repeat n-gram: ban any token that would complete an n-gram the
        #    history already contains
        n = self.no_repeat_ngram
        if n and n > 1 and len(history) >= n - 1:
            tail = tuple(history[-(n - 1):])
            for i in range(len(history) - n + 1):
                if tuple(history[i:i + n - 1]) == tail:
                    lg[history[i + n - 1]] = float("-inf")
        if self.temperature <= 0:
            return int(lg.argmax().item())
        lg = lg / self.temperature
        if self.top_k > 0 and self.top_k < lg.shape[0]:
            kth = torch.topk(lg, self.top_k).values[-1]
            lg[lg < kth] = float("-inf")
        probs = torch.softmax(lg, dim=-1)
        if 0 < self.top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sp, dim=-1)
            cut = int(torch.searchsorted(cdf, torch.tensor([self.top_p])).item()) + 1
            keep = si[:cut]
            mask = torch.zeros_like(probs)
            mask[keep] = 1.0
            probs = probs * mask
            probs = probs / probs.sum()
        return int(torch.multinomial(probs, 1, generator=self.gen).item())


class StopWatcher(object):
    """Streams text but never emits a fragment that could still turn out to be
    the head of a stop string."""

    def __init__(self, stops):
        self.stops = [s for s in (stops or []) if s]
        self.hold = max([len(s) for s in self.stops] or [0]) - 1
        self.emitted = 0

    def feed(self, text, final=False):
        for s in self.stops:
            at = text.find(s)
            if at >= 0:
                out = text[self.emitted:at]
                self.emitted = at
                return out, True
        safe = len(text) if final else max(0, len(text) - self.hold)
        if safe <= self.emitted:
            return "", False
        out = text[self.emitted:safe]
        self.emitted = safe
        return out, False


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class Engine(object):
    def __init__(self, args, torch):
        self.torch = torch
        self.lock = threading.Lock()
        dev = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = dev
        self.dtype = {"float32": torch.float32, "float16": torch.float16,
                      "bfloat16": torch.bfloat16}[args.dtype]
        self.tok = load_tokenizer()
        self.model, self.cfg, self.cfg_path, self.wpath, self.load_sec = load_model(
            args.tag, args.weights, args.config, dev, self.dtype, torch)
        self.name = args.name
        self.tag = args.tag
        self.vocab_limit = min(len(self.tok), self.cfg["vocab"])
        self.eos = self.tok.eos_token_id
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def banner(self):
        return (f"[serve_ollama] tag={self.tag} d={self.cfg['d_model']} L={self.cfg['n_layer']} "
                f"d_ffn={self.cfg['d_ffn']} nonlin={self.cfg['nonlin']} block={self.cfg['block']}\n"
                f"[serve_ollama] params={self.n_params/1e6:.1f}M weights={self.wpath} "
                f"({self.load_sec:.1f}s) device={self.device} dtype={str(self.dtype).split('.')[-1]}\n"
                f"[serve_ollama] PLAINTEXT model — this path says nothing about FHE timings")

    def generate(self, prompt, sampler, max_new, stops, on_delta=None, chunk=256):
        """Yields nothing; calls on_delta(text). Returns a dict of counters."""
        torch = self.torch
        ids = self.tok.encode(prompt) if prompt else []
        st = make_state(self.cfg, self.device, self.dtype, torch)
        watcher = StopWatcher(stops)
        t0 = time.time()
        logits = None
        with torch.no_grad():
            for i in range(0, len(ids), chunk):
                part = torch.tensor([ids[i:i + chunk]], dtype=torch.long, device=self.device)
                logits = forward_chunk(self.model, part, st, self.cfg, torch, last_only=True)
            t_prefill = time.time() - t0
            if logits is None:                       # empty prompt: seed with eos
                part = torch.tensor([[self.eos if self.eos is not None else 0]],
                                    dtype=torch.long, device=self.device)
                logits = forward_chunk(self.model, part, st, self.cfg, torch, last_only=True)
            out_ids, text, done_reason, truncated = [], "", "length", False
            t1 = time.time()
            for _ in range(max_new):
                nxt = sampler.pick(logits[0, -1], ids + out_ids, torch, self.vocab_limit)
                if self.eos is not None and nxt == self.eos:
                    done_reason = "stop"
                    break
                out_ids.append(nxt)
                text = self.tok.decode(out_ids)
                delta, hit = watcher.feed(text)
                if delta and on_delta:
                    on_delta(delta)
                if hit:
                    done_reason, truncated = "stop", True
                    break
                part = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
                logits = forward_chunk(self.model, part, st, self.cfg, torch, last_only=True)
            if not truncated:
                delta, _ = watcher.feed(text, final=True)
                if delta and on_delta:
                    on_delta(delta)
        t_eval = time.time() - t1
        return {"prompt_tokens": len(ids), "eval_tokens": len(out_ids),
                "prefill_sec": t_prefill, "eval_sec": t_eval,
                "text": text[:watcher.emitted], "done_reason": done_reason}


# ---------------------------------------------------------------------------
# prompt shaping — a base model has no chat format; do not invent one silently
# ---------------------------------------------------------------------------
def build_prompt(messages, style, system):
    sys_txt = "".join(m["content"] for m in messages if m.get("role") == "system") or (system or "")
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if style == "plain":
        # Ollama's own default template would inject literal ChatML markers,
        # which this model has never seen. Plain concatenation is the fix.
        return (sys_txt + "".join(m["content"] for m in turns)), ["<|endoftext|>"]
    if style == "dialogue":
        head = (sys_txt + "\n\n") if sys_txt else ""
        body = "".join(("User: " if m["role"] == "user" else "Assistant: ") + m["content"].strip() + "\n"
                       for m in turns)
        return head + body + "Assistant:", list(DIALOGUE_STOPS)
    head = (sys_txt.strip() + "\n\n") if sys_txt else QA_PREAMBLE
    body = ""
    for m in turns:
        if m["role"] == "user":
            body += "Q: " + m["content"].strip() + "\n"
        else:
            body += "A: " + m["content"].strip() + "\n\n"
    return head + body + "A:", list(QA_STOPS)


def options_to_sampler(opts, defaults, torch):
    o = dict(defaults)
    o.update({k: v for k, v in (opts or {}).items() if v is not None})
    return Sampler(temperature=o.get("temperature", 0.0), top_p=o.get("top_p", 0.9),
                   top_k=o.get("top_k", 40), repeat_penalty=o.get("repeat_penalty", 1.3),
                   repeat_last_n=o.get("repeat_last_n", 64),
                   no_repeat_ngram=o.get("no_repeat_ngram", 4),
                   seed=o.get("seed"), torch=torch), o


# ---------------------------------------------------------------------------
# HTTP: the Ollama API surface `ollama run` needs, plus the OpenAI one
# ---------------------------------------------------------------------------
def make_handler(eng, args):
    torch = eng.torch
    details = {"parent_model": "", "format": "npz", "family": "fhe-ssm",
               "families": ["fhe-ssm"], "parameter_size": "430M",
               "quantization_level": str(eng.dtype).split(".")[-1].upper()}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "fhe-ssm-shim"

        def log_message(self, fmt, *a):
            if args.verbose:
                sys.stderr.write("[http] " + fmt % a + "\n")

        # -- plumbing ---------------------------------------------------
        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _open_stream(self, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Transfer-Encoding", "chunked")
            self._cors()
            self.end_headers()

        def _chunk(self, data):
            b = data if isinstance(data, bytes) else data.encode()
            self.wfile.write(("%x\r\n" % len(b)).encode())
            self.wfile.write(b)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        def _end_stream(self):
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        # -- routes -----------------------------------------------------
        def do_HEAD(self):
            # `ollama` heartbeats with HEAD / before every command; a 501 here
            # is what makes the CLI report "something went wrong".
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "0")
            self._cors()
            self.end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                body = b"Ollama is running"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif p == "/api/version":
                self._json({"version": "0.0.0-fhe-ssm-shim"})
            elif p in ("/api/tags", "/v1/models"):
                self._json(self._tags(p == "/v1/models"))
            elif p == "/api/ps":
                self._json({"models": [self._model_row()]})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            p = self.path.split("?")[0]
            try:
                req = self._body()
            except Exception as e:
                return self._json({"error": str(e)}, 400)
            try:
                if p == "/api/show":
                    return self._json(self._show())
                if p == "/api/generate":
                    return self._generate(req)
                if p == "/api/chat":
                    return self._chat(req)
                if p == "/v1/chat/completions":
                    return self._chat(req, openai=True)
                if p == "/v1/completions":
                    return self._generate(req, openai=True)
                if p in ("/api/pull", "/api/create", "/api/copy"):
                    return self._json({"status": "success"})
                if p in ("/api/embed", "/api/embeddings"):
                    return self._json({"error": "this shim serves generation only"}, 400)
                return self._json({"error": "not found"}, 404)
            except BrokenPipeError:
                pass
            except Exception as e:
                import traceback
                traceback.print_exc()
                try:
                    self._json({"error": repr(e)}, 500)
                except Exception:
                    pass

        # -- payloads ---------------------------------------------------
        def _model_row(self):
            return {"name": eng.name + ":latest", "model": eng.name + ":latest",
                    "modified_at": now_rfc3339(), "size": eng.wpath.stat().st_size,
                    "digest": "sha256:3ec3f1d8b9b71e7eee38055a7621213991c6417a89576f1c02590963293e784b",
                    "details": details, "expires_at": "0001-01-01T00:00:00Z"}

        def _tags(self, openai):
            if openai:
                return {"object": "list", "data": [{"id": eng.name, "object": "model",
                                                    "created": int(time.time()), "owned_by": "local"}]}
            return {"models": [self._model_row()]}

        def _show(self):
            return {
                "license": "MIT",
                "modelfile": f"# served by ml-eval/serve_ollama.py from {eng.wpath}",
                "parameters": (f"temperature {args.temperature}\nrepeat_penalty {args.repeat_penalty}\n"
                               f"repeat_last_n {args.repeat_last_n}\n"
                               f"no_repeat_ngram {args.no_repeat_ngram}"),
                "template": "{{ .Prompt }}",
                "details": details,
                "model_info": {
                    "general.architecture": "fhe-ssm",
                    "general.parameter_count": eng.n_params,
                    "fhe-ssm.context_length": eng.cfg["ctx"],
                    "fhe-ssm.embedding_length": eng.cfg["d_model"],
                    "fhe-ssm.block_count": eng.cfg["n_layer"],
                    "fhe-ssm.feed_forward_length": eng.cfg["d_ffn"],
                    "tokenizer.ggml.model": tfns.TOK_MODEL,
                },
                "capabilities": ["completion"],
                "modified_at": now_rfc3339(),
            }

        def _defaults(self):
            return {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                    "repeat_penalty": args.repeat_penalty, "repeat_last_n": args.repeat_last_n,
                    "no_repeat_ngram": args.no_repeat_ngram, "num_predict": args.num_predict}

        def _run(self, prompt, stops, opts, stream, emit, finish):
            sampler, o = options_to_sampler(opts, self._defaults(), torch)
            if opts and opts.get("stop"):
                stops = opts["stop"] if isinstance(opts["stop"], list) else [opts["stop"]]
            n_pred = int(o.get("num_predict", args.num_predict))
            if n_pred < 0:
                n_pred = 4096
            if args.verbose:
                sys.stderr.write("[prompt] " + json.dumps(prompt)[:600] + "\n")
            with eng.lock:
                return eng.generate(prompt, sampler, n_pred, stops,
                                    on_delta=emit if stream else None)

        # -- /api/generate, /v1/completions -----------------------------
        def _generate(self, req, openai=False):
            prompt = req.get("prompt")
            if isinstance(prompt, list):
                prompt = "".join(prompt)
            stream = bool(req.get("stream", not openai))
            if not prompt and not openai:            # ollama's warm-up ping
                return self._json({"model": eng.name, "created_at": now_rfc3339(),
                                   "response": "", "done": True, "done_reason": "load"})
            sysmsg = req.get("system") or args.system
            if args.generate_style == "raw":
                prompt = (sysmsg + "\n\n" if sysmsg else "") + (prompt or "")
                stops = ["<|endoftext|>"]
            else:
                prompt, stops = build_prompt([{"role": "user", "content": prompt or ""}],
                                             args.generate_style, sysmsg)
            stops = list(req.get("options", {}).get("stop") or stops)
            opts = dict(req.get("options") or {})
            for k_openai, k_ours in (("max_tokens", "num_predict"), ("temperature", "temperature"),
                                     ("top_p", "top_p"), ("seed", "seed")):
                if openai and req.get(k_openai) is not None:
                    opts[k_ours] = req[k_openai]
            if openai and req.get("stop"):
                opts["stop"] = req["stop"]
            cid = "cmpl-" + uuid.uuid4().hex[:24]

            if not stream:
                r = self._run(prompt, stops, opts, False, None, None)
                if openai:
                    return self._json(self._openai_done(cid, "text_completion", r, text_field=True))
                return self._json(self._ollama_done(r, {"response": r["text"]}))

            ctype = "text/event-stream" if openai else "application/x-ndjson"
            self._open_stream(ctype)

            def emit(txt):
                if openai:
                    self._chunk("data: " + json.dumps({
                        "id": cid, "object": "text_completion", "created": int(time.time()),
                        "model": eng.name, "choices": [{"index": 0, "text": txt,
                                                        "finish_reason": None}]}) + "\n\n")
                else:
                    self._chunk(json.dumps({"model": eng.name, "created_at": now_rfc3339(),
                                            "response": txt, "done": False}) + "\n")

            r = self._run(prompt, stops, opts, True, emit, None)
            if openai:
                self._chunk("data: " + json.dumps({
                    "id": cid, "object": "text_completion", "created": int(time.time()),
                    "model": eng.name,
                    "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]}) + "\n\n")
                self._chunk("data: [DONE]\n\n")
            else:
                self._chunk(json.dumps(self._ollama_done(r, {"response": ""})) + "\n")
            self._end_stream()

        # -- /api/chat, /v1/chat/completions ----------------------------
        def _chat(self, req, openai=False):
            messages = req.get("messages") or []
            stream = bool(req.get("stream", not openai))
            if not messages:
                return self._json({"model": eng.name, "created_at": now_rfc3339(),
                                   "message": {"role": "assistant", "content": ""},
                                   "done": True, "done_reason": "load"})
            prompt, stops = build_prompt(messages, args.chat_style, args.system)
            opts = dict(req.get("options") or {})
            for k_openai, k_ours in (("max_tokens", "num_predict"), ("temperature", "temperature"),
                                     ("top_p", "top_p"), ("seed", "seed")):
                if openai and req.get(k_openai) is not None:
                    opts[k_ours] = req[k_openai]
            if openai and req.get("stop"):
                opts["stop"] = req["stop"]
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]

            if not stream:
                r = self._run(prompt, stops, opts, False, None, None)
                txt = r["text"].strip()
                if openai:
                    return self._json(self._openai_done(cid, "chat.completion", r, text=txt))
                return self._json(self._ollama_done(r, {"message": {"role": "assistant", "content": txt}}))

            ctype = "text/event-stream" if openai else "application/x-ndjson"
            self._open_stream(ctype)
            state = {"first": True}

            def emit(txt):
                if state["first"]:
                    txt = txt.lstrip()
                    if not txt:
                        return
                    state["first"] = False
                if openai:
                    self._chunk("data: " + json.dumps({
                        "id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": eng.name, "choices": [{"index": 0, "delta": {"content": txt},
                                                        "finish_reason": None}]}) + "\n\n")
                else:
                    self._chunk(json.dumps({"model": eng.name, "created_at": now_rfc3339(),
                                            "message": {"role": "assistant", "content": txt},
                                            "done": False}) + "\n")

            r = self._run(prompt, stops, opts, True, emit, None)
            if openai:
                self._chunk("data: " + json.dumps({
                    "id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": eng.name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n")
                self._chunk("data: [DONE]\n\n")
            else:
                self._chunk(json.dumps(self._ollama_done(
                    r, {"message": {"role": "assistant", "content": ""}})) + "\n")
            self._end_stream()

        # -- shared tails -----------------------------------------------
        def _ollama_done(self, r, extra):
            out = {"model": eng.name, "created_at": now_rfc3339(), "done": True,
                   "done_reason": r["done_reason"],
                   "total_duration": int((r["prefill_sec"] + r["eval_sec"]) * 1e9),
                   "load_duration": 0,
                   "prompt_eval_count": r["prompt_tokens"],
                   "prompt_eval_duration": int(r["prefill_sec"] * 1e9),
                   "eval_count": r["eval_tokens"],
                   "eval_duration": int(r["eval_sec"] * 1e9)}
            out.update(extra)
            return out

        def _openai_done(self, cid, obj, r, text=None, text_field=False):
            usage = {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["eval_tokens"],
                     "total_tokens": r["prompt_tokens"] + r["eval_tokens"]}
            if text_field:
                choices = [{"index": 0, "text": r["text"], "finish_reason": "stop"}]
            else:
                choices = [{"index": 0, "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop"}]
            return {"id": cid, "object": obj, "created": int(time.time()), "model": eng.name,
                    "choices": choices, "usage": usage}

    return H


# ---------------------------------------------------------------------------
# selftest: this file's decode path vs the model's own batched forward
# ---------------------------------------------------------------------------
def selftest(eng, args, torch):
    text = args.selftest_text
    ids = eng.tok.encode(text)[:args.selftest_tokens]
    T = len(ids)
    inp = torch.tensor([ids], dtype=torch.long, device=eng.device)
    with torch.no_grad():
        ref = eng.model(inp)[0].to("cpu", torch.float32)                 # trained path
        st = make_state(eng.cfg, eng.device, eng.dtype, torch)
        one = forward_chunk(eng.model, inp, st, eng.cfg, torch, last_only=False)[0].to("cpu", torch.float32)
        st = make_state(eng.cfg, eng.device, eng.dtype, torch)
        rows = []
        for t in range(T):
            step = torch.tensor([[ids[t]]], dtype=torch.long, device=eng.device)
            rows.append(forward_chunk(eng.model, step, st, eng.cfg, torch, last_only=True)[0, 0])
        seq = torch.stack(rows).to("cpu", torch.float32)
        st = make_state(eng.cfg, eng.device, eng.dtype, torch)
        half = T // 2
        forward_chunk(eng.model, inp[:, :half], st, eng.cfg, torch, last_only=True)
        resumed = forward_chunk(eng.model, inp[:, half:], st, eng.cfg, torch,
                                last_only=False)[0].to("cpu", torch.float32)

    def cmp(name, a, b):
        d = (a - b).abs()
        scale = b.abs().max().item()
        agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
        return {"case": name, "maxAbs": d.max().item(), "meanAbs": d.mean().item(),
                "relMax": d.max().item() / scale, "top1Agree": agree,
                "rows": int(a.shape[0])}

    rec = {"selftest": "serve_ollama", "when": now_rfc3339(), "tag": eng.tag,
           "weights": str(eng.wpath), "device": eng.device,
           "dtype": str(eng.dtype).split(".")[-1], "tokens": T,
           "cases": [cmp("chunk_T_vs_batched", one, ref),
                     cmp("token_by_token_vs_batched", seq, ref),
                     cmp("resumed_half_vs_batched", resumed, ref[half:])]}
    tol = args.selftest_tol
    rec["pass"] = all(c["relMax"] < tol and c["top1Agree"] == 1.0 for c in rec["cases"])
    rec["tol"] = tol
    out = REPO / f"results/serve-ollama-{datetime.date.today().strftime('%Y%m%d')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "SELFTEST.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    print(f"[selftest] record -> {out / 'SELFTEST.json'}")
    return 0 if rec["pass"] else 1


def bench(eng, args, torch):
    sampler = Sampler(temperature=0, repeat_penalty=1.0, no_repeat_ngram=0, torch=torch)
    prompt = " ".join(["the quick brown fox jumps over the lazy dog"] * 12)
    r = eng.generate(prompt, sampler, args.num_predict, ["<|endoftext|>"])
    print(json.dumps({"bench": "serve_ollama", "device": eng.device,
                      "dtype": str(eng.dtype).split(".")[-1],
                      "promptTokens": r["prompt_tokens"], "prefillSec": round(r["prefill_sec"], 3),
                      "prefillTokPerSec": round(r["prompt_tokens"] / max(r["prefill_sec"], 1e-9), 2),
                      "evalTokens": r["eval_tokens"], "evalSec": round(r["eval_sec"], 3),
                      "decodeTokPerSec": round(r["eval_tokens"] / max(r["eval_sec"], 1e-9), 2)},
                     indent=2))
    return 0


def cli_sampler(args, torch):
    return Sampler(args.temperature, args.top_p, args.top_k, args.repeat_penalty,
                   args.repeat_last_n, args.no_repeat_ngram, torch=torch)


def repl(eng, args, torch):
    print(eng.banner(), file=sys.stderr)
    print(f"[repl] style={args.chat_style} — Ctrl-D to exit", file=sys.stderr)
    history = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            return 0
        if not line.strip():
            continue
        history.append({"role": "user", "content": line})
        prompt, stops = build_prompt(history, args.chat_style, args.system)
        sampler = cli_sampler(args, torch)
        r = eng.generate(prompt, sampler, args.num_predict, stops,
                         on_delta=lambda t: (sys.stdout.write(t), sys.stdout.flush()))
        print()
        history.append({"role": "assistant", "content": r["text"].strip()})
        if args.verbose:
            print(f"[{r['eval_tokens']} tok, {r['eval_tokens']/max(r['eval_sec'],1e-9):.1f} tok/s]",
                  file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--weights", default=None, help="default: the chmod-444 demo checkpoint")
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default=DEFAULT_NAME, help="model name reported to clients")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--chat-style", default="qa", choices=["qa", "plain", "dialogue"],
                    help="how /api/chat turns messages into text for a BASE model")
    ap.add_argument("--generate-style", default="raw", choices=["raw", "qa", "dialogue"],
                    help="/api/generate is a raw completion by API contract (raw); "
                         "qa applies the same few-shot scaffold as chat, which is what "
                         "makes one-shot `ollama run MODEL \"question\"` answer instead of ramble")
    ap.add_argument("--system", default=None)
    # defaults ARE the pod's free-running policy (spec_decode/decode_policy.py
    # "norepeat"): greedy, penalty 1.3 over 64 fed tokens, no-repeat 4-gram.
    # A client's per-request `options` still override any of them.
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (pod)")
    ap.add_argument("--top-p", type=float, default=0.9, help="ignored at temperature 0")
    ap.add_argument("--top-k", type=int, default=40, help="ignored at temperature 0")
    ap.add_argument("--repeat-penalty", type=float, default=1.3)
    ap.add_argument("--repeat-last-n", type=int, default=64)
    ap.add_argument("--no-repeat-ngram", type=int, default=4, help="0 disables the ban")
    ap.add_argument("--num-predict", type=int, default=128)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--repl", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--prompt", default=None, help="one-shot raw completion, then exit")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-tokens", type=int, default=48)
    ap.add_argument("--selftest-tol", type=float, default=2e-3)
    ap.add_argument("--selftest-text", default=(
        "The city of Cambridge is home to a university founded in 1209. "
        "Its libraries hold several million volumes, and the river Cam runs past the colleges."))
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    eng = Engine(args, torch)
    print(eng.banner(), file=sys.stderr)

    if args.selftest:
        return selftest(eng, args, torch)
    if args.bench:
        return bench(eng, args, torch)
    if args.prompt is not None:
        sampler = cli_sampler(args, torch)
        r = eng.generate(args.prompt, sampler, args.num_predict, ["<|endoftext|>"],
                         on_delta=lambda t: (sys.stdout.write(t), sys.stdout.flush()))
        print()
        return 0
    if args.repl:
        return repl(eng, args, torch)

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), make_handler(eng, args))
    except OSError as e:
        if e.errno != 48:
            raise
        # almost always an earlier copy of this script still holding the port,
        # often one running pre-edit code. A traceback here says none of that.
        sys.stderr.write(
            f"\n[serve_ollama] port {args.port} is already in use.\n"
            f"[serve_ollama] if that is an older copy of this server, stop it with:\n"
            f"[serve_ollama]     pkill -f 'serve_ollama.py --port {args.port}'\n"
            f"[serve_ollama] or serve on another port with --port {args.port + 1}\n")
        return 1
    print(f"[serve_ollama] listening on http://{args.host}:{args.port}  model={args.name}\n"
          f"[serve_ollama]   OLLAMA_HOST={args.host}:{args.port} ollama run {args.name}",
          file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve_ollama] bye", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
