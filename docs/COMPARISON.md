# Comparison with published systems

Numbers for other systems are transcribed from their papers or repositories (linked); where a paper does not state a
quantity, the cell says so. Ours are from [RESULTS.md](RESULTS.md). Read the caveats before the table: the systems differ
in model size by 3× to 60× (130 M to 8 B parameters; this one is 405 M), in what one "token" costs (a decode step of a KV-cache transformer vs a tick of a recurrence), in
hardware generation, and in what was actually measured.

## Non-interactive FHE generation of text (per generated token)

| system | model | security | hardware | per generated token | what was measured | fidelity measured under encryption? |
|---|---|---|---|---|---|---|
| **this repository** | 405M recurrent LM (trained for the circuit) | ring 2^17, 128-bit classical (uniform ternary secret) | 2× RTX PRO 6000 96 GB | **2.7–2.9 s per lane-token** (175.6 s layer loop / 188 s request→reply per 64-lane tick) | **148 consecutive ticks, 9,472 tokens, free-running**; a separate 18-tick client-keyed session | yes: top-1 99.7 % vs the model's own plaintext on the same tokens, every tick — on a prompt set pre-screened in plaintext against near-ties, so conditional on that selection ([RESULTS.md](RESULTS.md)) |
| Sylph ([arXiv 2601.18511](https://arxiv.org/abs/2601.18511)) | Llama-3-8B | not stated explicitly in the text we could access; ring degrees up to 2^16 | 8× RTX PRO 6000 (also 1–4× B200) | **18 s** (8× RP6000), 31 s (1× RP6000), 24 s (1× B200), 12 s (4× B200), per output token | prefill of 128 encrypted tokens (20 s) + the cost of one decode step at T = 1; no multi-token run reported | quality from a CKKS-noise-injection simulator (12-bit precision matches FP16 perplexity), not from an encrypted run |
| Cachemir ([arXiv 2602.11470](https://arxiv.org/abs/2602.11470)) | Llama-3-8B (also GPT-2, TinyLlama-1.1B) | ring 2^16, 1763-bit modulus, 128-bit claimed, sparse secret h = 192 | 1× A100 80 GB | **96.6 s** (1.61 min per generated token, their §7.3) | one decode step at context 512, single sequence; encrypted KV cache carried (≈ 17 GB at 1,024 tokens) | not reported; the token-selection / re-embedding step is not described |
| NEXUS ([NDSS 2025](https://www.ndss-symposium.org/wp-content/uploads/2025-868-paper.pdf)) | Llama-3-8B | 128-bit | 4× A100 40 GB | 51.84 s (batch-32 amortised; Sylph reads the same figure as per input token, 414.72 s for the 8-token query) | one generated token from an 8-token input | — |
| Cerium ([arXiv 2512.11269](https://arxiv.org/abs/2512.11269)) | Llama-3-8B | not extracted | 8× H100 / 8× B200 | no decode stage: 134 s is a 128-token **prefill** to the first token | prefill only | — |
| fhe-mamba ([Hosi121/fhe-mamba](https://github.com/Hosi121/fhe-mamba), its README at commit `c5dd1fe`, 2026-09-26, and its 2026-09-22 client-generation note) | Mamba-2-130M (pretrained, 24 layers, one lane); a Mamba-3 SISO 187M variant (12 layers) is also run | ring 2^16, depth 44, `security=not-set` (its own notes: full-chain 128-bit security and a secret-key-free server are open work; its opt-in dual-ring mode moves ordinary arithmetic to 2^15 and "does not establish equal security for both ring sizes") | DGX Spark (GB10) | **32.4 min for 4 generated tokens** in 5 encrypted evaluations (1,942.99 s; ≈ 8 min per token); the Mamba-3 187M variant 7.2 min (430.51 s; 107.6 s per token by its own figure, "not steady-state token latency") | 4 tokens; the client decrypts the final hidden vector, picks the greedy token and re-encrypts its embedding; the recurrent state stays encrypted between the evaluations; a single-process client loop with public weights | its error gate compares CKKS to the matching polynomial circuit (max 0.0092 in the first complete run, 0.0037 in its latest study, gate 0.05); the exact-model hidden-output error is reported separately (up to 0.073, tokens still matching); no paper |

**Reading the table.** The transformer systems' numbers are the cost of one decode step at T = 1 (Sylph, Cachemir: one
new token appended to a context of 128 or 512 with a KV cache; NEXUS: one forward over an 8-token input, batch-amortised),
priced per hardware, and none of them reports a fidelity against the original model measured under encryption; fhe-mamba
reports four tokens on one lane, the state carried encrypted between its five evaluations, checked against its own polynomial
circuit. None reports a long chain. Ours is a sustained
run: 148 consecutive ticks, every lane fed its own encrypted output, the fidelity against the model itself at every tick.

GPU-seconds per generated token: **5.5–5.9 here** (two cards × 175.6–188 s ÷ 64 lanes), against 24 (one B200) to
31 (one RTX PRO 6000) and 48–144 (four B200 to eight RTX PRO 6000) for Sylph's single step and 96.6 for Cachemir's — and that at ring 2^17, where every operation costs
about twice what it does at 2^16 (ring 2^16 was not available to this circuit: its 128-bit modulus ceiling of 1,747 bits does
not hold the depth at a 59-bit scaling factor, and smaller scaling factors were numerically broken in the GPU bootstrap of
the library build used), and with two cards instead of one. Per parameter the picture is different: the 8B systems' single
step costs ≈ 3–4 (Sylph) and ≈ 12 (Cachemir) GPU-ns per parameter-token against ≈ 14–15 here (5.5–5.9 GPU-s ÷ 404.7 M). Which of the two normalisations
matters depends on what is being bought — tokens or parameters — and the two are stated side by side here for that reason.

## What this repository has that the others do not report

- A **measured chain**: 148 consecutive decode steps with the state carried under encryption, and 9,472 exact comparisons
  against the model's plaintext. The other systems price one step.
- **Constant cost and memory in context length**, measured over the chain (a first-order recurrence, no KV cache). Cachemir's
  KV cache grows with context (17 GB at 1,024 tokens); Sylph's decode attends over a public/private KV cache.
- **The model is the circuit**: no approximation error on top of CKKS noise. Cachemir and Sylph approximate a pretrained
  transformer's nonlinearities and report the approximation's accuracy separately.
- **A public verification kit**: the keys and every ciphertext of a recorded session ([VERIFICATION_KIT.md](VERIFICATION_KIT.md)).
- **Uniform ternary secret at ring 2^17** with the modulus chain 204 bits under OpenFHE's 128-bit ceiling ([SECURITY.md](SECURITY.md));
  NEXUS and Cachemir use sparse secrets (h = 192) under key encapsulation; Sylph's secret distribution is not stated in the text we could access.

## What the others have that this repository does not

- **Model quality.** Llama-3-8B is a real language model; `pbd430a` is a 405M-parameter model with WikiText-103 token
  perplexity 28.55 and LAMBADA 9 % ([RESULTS.md](RESULTS.md) section 4).
- **Single-stream latency.** Cachemir generates a token for an 8B model in 96.6 s on one 2020-era A100; a single tick here
  takes 176–188 s on two 2025 cards, whatever the lane count — the 64 lanes are throughput, not latency.
- **Reading throughput.** Transformers process a prompt in one prefill (Sylph: 128 tokens in 20 s on eight cards); here a
  64-token prompt costs 64 ticks per lane.
- **A selective recurrence.** fhe-mamba (above) runs a pretrained Mamba-2 model under CKKS with a selective (input-dependent)
  decay; the model here has a fixed, trained decay. Its run is four tokens on one lane at parameters without a set security
  level, released the same week as the long run here (32.4 min for the four tokens in its README of 2026-09-26, 38.5 at release).

## Encoder-only and interactive systems (cited, not raced)

THOR ([ePrint 2024/1881](https://eprint.iacr.org/2024/1881), BERT-base, one 128-token pass 626 s), MOAI
([ePrint 2025/991](https://eprint.iacr.org/2025/991), BERT-base, 141.6 s per input amortised), Powerformer
([ACL 2025](https://aclanthology.org/2025.acl-long.543.pdf)), Rho et al. ([ICLR 2025](https://arxiv.org/abs/2410.02486),
a 2-layer softmax-free GPT-2-style model, 26.5 s per 128-token pass) evaluate one encoder pass and do not generate;
BumbleBee ([NDSS 2025](https://www.ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf)) and EncFormer
([arXiv 2604.09975](https://arxiv.org/abs/2604.09975)) are interactive two-party protocols, a different threat model.
