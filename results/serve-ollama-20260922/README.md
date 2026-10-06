# Chatting with the dense demo checkpoint from Ollama — 2026-09-22

`ml-eval/serve_ollama.py`. Plaintext only: this is the model the FHE harness
evaluates homomorphically, run natively on the Mac. **No number on this page
says anything about the FHE side** (that is `hpc_gpu_port/`, and it is seconds
per token, not tokens per second).

## Why Ollama cannot load the checkpoint directly

Ollama runs GGUF through llama.cpp and through its own Go engine. Both dispatch
on an architecture string that must already be implemented **in the binary**; a
Modelfile selects and configures an architecture, it cannot describe one. The
demo checkpoint's architecture (`ml-eval/train_fhe_native_ssm.py::build_model`)
is token-shift + a non-selective per-channel decay scan + learned polynomial
gates, with no attention and no KV cache. That is not an implemented
architecture, and the nearest neighbour in llama.cpp (`rwkv6`/`rwkv7`) computes
different math — RWKV's WKV is a normalised exp-weighted ratio, ours is
`S_t = a·S_{t-1} + (1-a)·x_t` followed by `z·(p0 + p1·z + p2·z²)`. So there is
no Modelfile, no conversion flag and no quantisation setting that makes
`ollama create` work. A real GGUF port means writing the graph in ggml plus a
converter, i.e. patching llama.cpp and pointing Ollama at the patched build.

`serve_ollama.py` serves the same forward pass behind the Ollama HTTP API
instead, on its own port, so every Ollama-shaped client works unchanged.

## Decode path and its check

`chat_native.py` re-forwards the whole window per token on purpose (its
docstring: one forward pass, no second implementation to drift). That is O(T)
per token and too slow to chat with. `serve_ollama.py` carries the recurrence
state instead — `(prev_u, S)` per layer, ~200 kB total, **constant in context
length** because there is no KV cache — which is exact here for the same reason
Lemma 2 applies under FHE: `a` is non-selective, so the recurrence is linear and
a chunk resumes from its carry in closed form. One code path serves prefill
(T=P) and decode (T=1).

`--selftest` checks it against the model's own batched forward
(`results/serve-ollama-20260922/SELFTEST.json`, 30 tokens, MPS, fp32):

| case | maxAbs | relMax | top-1 agreement |
|---|---|---|---|
| chunk of T vs batched forward | **0.0** (bit-identical) | 0.0 | 1.0 |
| token-by-token vs batched forward | 6.96e-05 | 4.55e-06 | 1.0 |
| prefill 15 + resume 15 vs batched | 3.91e-05 | 2.55e-06 | 1.0 |

The token-by-token residual is the scan's accumulation order (sequential vs the
trainer's Hillis-Steele doubling), not a different function. Re-run the selftest
after touching anything in that file.

## Weights

`~/Documents/fhe-ssm-backup/pbd430a_weights_DEMO_MODEL_20260901.npz`,
sha256 `3ec3f1d8b9b71e7eee38055a7621213991c6417a89576f1c02590963293e784b`,
**re-verified 2026-09-22** against the staged record in
`results/dense-demo-s31/pod_2xbw_20260917/pod_pull/stage_verify.txt:5`.
The loader also asserts `emb.weight == head.weight` in the npz, because
`build_model` ties them and a silent second `copy_` would otherwise decide
which one wins. 404.7M parameters counted from the built model (the "430M"
in the name is the nominal size).

## Speed (M5, 16 GB, MPS, fp32, `--bench`)

| | |
|---|---|
| prefill | 108 tokens in 0.806 s — **134.0 tok/s** |
| decode | 40 tokens in 0.750 s — **53.3 tok/s** |
| weights load | ~1.3 s |

Same-binary, same-machine, uncontended. Per the repo's timing rule these are
not comparable to any other build or load state.

## Quality — what to expect before you type

A base LM: FineWeb-Edu pretrain, WikiText-103 anneal, never instruction-tuned.
From `results/dense-demo-s31/pod/state/runs/s31_eval_pbd430a_anchors/anchor.jsonl`:

| axis | value |
|---|---|
| WikiText-103 token ppl | 28.55 (word ppl 53.08, 280 windows, 285,830 tokens) |
| held-out token ppl | 27.92 (2,000,000 tokens) |
| LAMBADA accuracy | **9.02 %** (5,153 items; last-word ppl 135.82) |

wt103 is *in-domain after the anneal*; LAMBADA 9 % is the honest out-of-domain
signal. Observed here: greedy, few-shot scaffold — "What is the capital of
France?" → "Paris." ✓; "Who wrote Pride and Prejudice?" → "Louis XIV." ✗;
"What colour is the sky?" → "Green." ✗. It autocompletes fluent English and
gets facts wrong. Fidelity ≠ quality applies to the FHE circuit; this table is
the quality column.

## Decoding policy — the pod's, ported (2026-09-23)

The shim's sampler now runs the pod client's free-running policy by default:
greedy, repetition penalty **1.3** over the last **64 fed tokens** (prompt
included, the CTRL/HF rule), then a no-repeat **4-gram** ban, then argmax.
`Sampler.pick` is a port of `spec_decode/decode_policy.py::modified`
("norepeat"), in that file's order and with its constants, kept in float64 like
the original. A client's per-request `options` still override any of it
(`temperature`, `repeat_penalty`, `repeat_last_n`, `no_repeat_ngram`).

Why it is the default: greedy alone repeats itself. decode_policy.py's own
docstring records 64 of 64 lanes of the 2026-09-19 selection repeating a 6-word
sequence, median first repeat at word 21. Reproduced here on
"The Emancipation Proclamation stands as the", greedy, 70 tokens:

- `--repeat-penalty 1.0 --no-repeat-ngram 0` → "most important document in **the
  history of the United States**. It is **the most important document in the
  history of the United States**. It is the document that established … and it
  is the document that has been the most influential document in **the history
  of the United States**."
- defaults (1.3 / 4) → "most important document in American history. It is a
  declaration of freedom for all slaves, and it was signed by President Abraham
  Lincoln on January 1, 1863. …"

`ml-eval/test_serve_ollama_policy.py` asserts the two implementations pick the
same token: **400/400 exact**, with the n-gram ban actually firing in **117** of
them (the test plants repeats on purpose and fails if the ban never triggers —
a random history almost never contains a repeated 3-gram, so the naive version
of this test proves nothing). It caught two real bugs in the port:

1. **In-place mutation.** `decode_policy` copies (`np.array(..., copy=True)`);
   the first port did not, so `pick()` handed the caller back penalised logits.
   One flipped argmax in 400 cases — a rate low enough to survive casual
   testing.
2. **MPS has no float64.** A combined `.to("cpu", torch.float64)` builds the f64
   tensor on-device and raises; `.cpu()` must come first. Invisible to the
   parity test, whose tensors are already on CPU — only the real MPS run finds it.

End-to-end check: with these defaults the shim reproduces the pod-side
continuation of that prompt **byte-for-byte, 790/790 characters**.

## The base-model chat trap (carried over from the pythia smoke model)

Ollama's default template injects literal ChatML (`<|im_start|>…`) for models
whose template lacks `.Messages`. This model has never seen those markers and
tokenises them as prose. `serve_ollama.py` never emits ChatML. `/api/chat`
applies `--chat-style`:

- `qa` (default) — few-shot `Q:`/`A:` scaffold, stops on `\nQ:`, `\n\n`,
  `<|endoftext|>`. The only style that gives turn-taking on a base model.
- `plain` — messages concatenated with no separators (the verified Pythia fix).
- `dialogue` — `User:`/`Assistant:` transcript, stops on `\nUser:`.

`/api/generate` is a raw completion by API contract (`--generate-style raw`,
default). One-shot `ollama run MODEL "question"` goes to `/api/generate`, so
start the server with `--generate-style qa` if you want it to answer rather
than ramble.

## Verified against the real client (ollama 0.33.3)

| call | result |
|---|---|
| `ollama list` | shows `fhe-ssm-430m:latest`, 1.8 GB |
| `ollama show fhe-ssm-430m` | architecture `fhe-ssm`, 430M, ctx 1024 |
| `ollama run fhe-ssm-430m "…"` | streams; `--generate-style qa` → "Paris. The capital of France is Paris." |
| `POST /api/chat` | NDJSON stream, `done_reason: stop` |
| `POST /v1/chat/completions` | OpenAI shape, streaming and non-streaming |

Trap found: the CLI heartbeats with `HEAD /` before every command. A 501 there
makes it print "something went wrong, please see the ollama server logs" with
no further detail — `do_HEAD` is required, not optional. Second trap: `ollama
run MODEL "prompt"` also reads stdin when stdin is not a TTY, so it hangs in a
script unless you redirect `</dev/null`.

## Run it

```
.venv/bin/python ml-eval/serve_ollama.py --selftest          # after any edit
.venv/bin/python ml-eval/test_serve_ollama_policy.py         # decode policy parity
.venv/bin/python ml-eval/serve_ollama.py --repl              # no ollama at all
.venv/bin/python ml-eval/serve_ollama.py --port 11435 --generate-style qa
OLLAMA_HOST=127.0.0.1:11435 ollama run fhe-ssm-430m
```

Port 11435 because the real ollama holds 11434; the two do not interfere, and
`ollama list` against 11434 will not show this model — it lives only in this
process.
