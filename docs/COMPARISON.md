# Comparison with published systems

Numbers for other systems are transcribed from their papers or repositories (linked); where a paper does not state a
quantity, the cell says so. Ours are from [RESULTS.md](RESULTS.md). Read the caveats before the table: the systems differ
in model size by 3× to 65× (124 M to 8 B parameters; this one is 405 M), in what one "token" costs (a decode step of a KV-cache transformer vs a tick of a recurrence), in
hardware generation, and in what was actually measured.

## Non-interactive FHE generation of text (per generated token)

| system | model | security | hardware | per generated token | what was measured | fidelity measured under encryption? |
|---|---|---|---|---|---|---|
| **this repository** | 405M recurrent LM (trained for the circuit) | ring 2^17, 128-bit classical (uniform ternary secret) | 2× RTX PRO 6000 96 GB | **2.7–2.9 s per lane-token** (175.6 s layer loop / 188 s request→reply per 64-lane tick) | **148 consecutive ticks, 9,472 tokens, free-running**; a separate 18-tick client-keyed session | yes: top-1 99.7 % vs the model's own plaintext on the same tokens, every tick — on a prompt set pre-screened in plaintext against near-ties, so conditional on that selection ([RESULTS.md](RESULTS.md)) |
| Sylph ([arXiv 2601.18511](https://arxiv.org/abs/2601.18511)) | Llama-3-8B | not stated explicitly in the text we could access; ring degrees up to 2^16 | 8× RTX PRO 6000 (also 1–4× B200) | **18 s** (8× RP6000), 31 s (1× RP6000), 24 s (1× B200), 12 s (4× B200), per output token | prefill of 128 encrypted tokens (20 s) + the cost of one decode step at T = 1 (the decode figures equal its per-layer times, 0.98 s and 0.56 s, × 32 layers); no multi-token run reported | quality from a CKKS-noise-injection simulator (12-bit precision matches FP16 perplexity), not from an encrypted run |
| Cachemir ([arXiv 2602.11470](https://arxiv.org/abs/2602.11470)) | Llama-3-8B (also GPT-2, TinyLlama-1.1B) | ring 2^16, 1763-bit modulus, 128-bit claimed, sparse secret h = 192 under sparse-secret encapsulation | 1× A100 80 GB | **96.6 s** (1.61 min per generated token, their §7.3; one decoder timed module by module, × 32 layers) | one decode step at context 512, single sequence; encrypted KV cache carried (≈ 17 GB at 1,024 tokens) | not reported; the token-selection / re-embedding step is not described |
| NEXUS ([NDSS 2025](https://www.ndss-symposium.org/wp-content/uploads/2025-868-paper.pdf)) | Llama-3-8B | 128-bit; sparse secret h = 192 (stated in its error appendix); no key encapsulation described | 4× A100 40 GB | 51.84 s per 8-token input, batch-32 amortised, "roughly the aggregation of the microbenchmarks" (Sylph reads the same figure as per input token, 414.72 s for the 8-token query) | one generated token from an 8-token input, selected under encryption (secure argmax over the vocabulary) | task accuracy under encryption only (e.g. SST-2 94.94 → 94.46 %) |
| Cerium ([arXiv 2512.11269](https://arxiv.org/abs/2512.11269)) | Llama-3-8B | 128-bit, ring 2^16, 1,782-bit maximum modulus, ternary secret of Hamming weight 32K (dense); no encapsulation described | 8× H100 / 8× B200 | no decode stage: 134 s is a 128-token **prefill** to the first token | prefill only, decoder blocks | not for Llama-3-8B (encrypted BERT-base RTE accuracy equals plaintext, one pass) |
| fhe-mamba ([Hosi121/fhe-mamba](https://github.com/Hosi121/fhe-mamba), README and research notes at commit `5aa03cce`, 2026-10-02; no paper) | Mamba-2-130M (pretrained, 24 layers) at release; since 2026-09-24 a trained Mamba-3 SISO 187M (12 layers); one lane | at release (2026-09-22) ring 2^16, `security=not-set`; **since 2026-09-27** a classical-128 profile: ring 2^17, uniform ternary secret, no sparse secret, depth 44, scale 59 | DGX Spark (GB10) at release; one B300 since 2026-09-27 | at release 32.4 min for 4 tokens (≈ 8 min per token, not-set); at 128-bit: **67.41 s per token** for a 16-token request (1,078.48 s, public-state reuse; 72.55 s evaluation-only) | at release 4 tokens; at 128-bit (2026-09-28) **16 free-running tokens**, the client decrypting the final hidden vector and picking each token; a 64-token attempt failed its own error gate at token 24 | yes, at every selection: hidden-vector error against the exact model and its polynomial circuit (max 6.63e-5 vs the exact model on the 16-token run, gate 0.001) |
| HEAT ([arXiv 2609.01730](https://arxiv.org/abs/2609.01730), 2026-09-01; the Perseus group) | GPT-2 small (124M, pretrained, fine-tuned so its approximations need fewer iterations) | ring 2^16, 128-bit enforced by OpenFHE, dense uniform ternary key with a transient Hamming-weight-32 key for the bootstrap | 1× A100 64 GB | **37.6 s** per token (54.3 s for the uncalibrated baseline), incl. 9.3 s of encrypted argmax | **128 teacher-forced chains of 128 tokens**, one sequence each, KV cache carried | yes: per-position top-1 and KL vs its plaintext model; top-1 falls along the chain (baseline 94 % → 50 %, HEAT 90 % → 76 %), pooled 74.0 % / 82.9 % |
| Perseus ([gladia-research-group/perseus](https://github.com/gladia-research-group/perseus), README at commit `b47fd430`, 2026-10-06; no paper published) | GPT-2 small (124M, pretrained; one sequence) | ring 2^16, OpenFHE `HEStd_128_classic`, uniform ternary persistent key; the bootstrap mod-raises under an ephemeral Hamming-weight-32 key through switching keys | 1× RTX PRO 6000 96 GB | **11.78 s** per token (32-bit composite chain), 16.38 s (64-bit) | a 16-token teacher-forced decode with the KV cache carried under encryption and an encrypted argmax at every token; encrypted-feedback generation (8 tokens after a 4-token prompt in its README) implemented, not timed | yes: top-1 15/16 (93.8 %) and KL 0.026 vs its plaintext model (100 % / 0.002 on the 64-bit chain) — a fitted polynomial approximation of a pretrained transformer |
| AR-HE ([arXiv 2610.04912](https://arxiv.org/abs/2610.04912), 2026-10-04) | GPT-2 small and a nanoGPT (65-token vocabulary), both on **random weights** ("no claim about language quality") | ring 2^16, sparse ternary h = 192, 1,465-bit modulus, 128-bit classical by the lattice estimator; the persistent key, no encapsulation | 1× H100 | **544 s** per generated step for GPT-2 small at context 33 (the prompt step 4,630 s); nanoGPT 187 s | the whole loop on the server: token selected under encryption, re-embedded under encryption, encrypted KV cache, client offline; chains of 24, 16 and 8 generated tokens — by their bootstrap counts on the nanoGPT (the 16-token run's 71,006 bootstraps are nanoGPT's per-step law exactly); the token count is fixed before the run | token agreement with the plaintext model on the same random weights: 24/24 (best of 20 prompts), 16/16, 5/8; for GPT-2 small, agreement at the first step |
| HE-Guardrail ([arXiv 2609.21484](https://arxiv.org/abs/2609.21484), 2026-09-18) | Llama-3-8B-Instruct (pretrained), with jailbreak guardrails evaluated beside it | not stated (2^15 slots, 17 levels after bootstrapping; desilofhe) | 2× H200 | not reported | states that every target response of ≈ 4,000 prompts was generated under encryption: a 200-token budget, each selected token fed back as an encrypted one-hot vector times the embedding table, 16 requests interleaved per ciphertext at a padded length of 2,048; how the token is selected under encryption, the run times and the total compute are not described | not reported against the plaintext model (guardrail decisions and harmfulness of the responses only) |

**Reading the table.** The transformer systems' numbers are the cost of one decode step at T = 1 (Sylph, Cachemir: one
new token appended to a context of 128 or 512 with a KV cache; NEXUS: one forward over an 8-token input, batch-amortised),
priced per hardware (Sylph's and Cachemir's composed from per-layer or per-module timings, NEXUS's from microbenchmarks),
and none of the three reports a fidelity against the original model measured under encryption. Multi-step chains: HEAT
(2026-09-01) reports 128 teacher-forced chains of 128 tokens on one sequence each, with fidelity that degrades along the
chain; HE-Guardrail (2026-09-18) states 200-token encrypted generation with 16 requests per ciphertext but reports no timing,
security level or fidelity; Perseus a 16-token teacher-forced decode; fhe-mamba 16 free-running tokens on one lane at
128-bit (2026-09-28); AR-HE 24-token server-side loops on a random-weight nanoGPT (2026-10-04). Ours (run 2026-09-19, public
2026-09-23): 148 consecutive ticks × 64 lanes, free-running, every lane fed the token chosen from its own encrypted output,
the fidelity against the model itself at every tick, at set 128-bit parameters, with constant cost per tick. HEAT, Perseus,
NEXUS, AR-HE and (by its description) HE-Guardrail select the token under encryption on the server; here the client decrypts
and picks it every tick.

GPU-seconds per generated token: **5.5–5.9 here** (two cards × 175.6–188 s ÷ 64 lanes), against 24 (one B200) to
31 (one RTX PRO 6000) and 48–144 (four B200 to eight RTX PRO 6000) for Sylph's single step and 96.6 for Cachemir's — and that at ring 2^17, where every operation costs
about twice what it does at 2^16 (ring 2^16 was not available to this circuit: its 128-bit modulus ceiling of 1,747 bits does
not hold the depth at a 59-bit scaling factor, and smaller scaling factors were numerically broken in the GPU bootstrap of
the library build used), and with two cards instead of one. Per parameter the picture is different: the 8B systems' single
step costs ≈ 3–4 (Sylph) and ≈ 12 (Cachemir) GPU-ns per parameter-token against ≈ 14–15 here (5.5–5.9 GPU-s ÷ 404.7 M). Which of the two normalisations
matters depends on what is being bought — tokens or parameters — and the two are stated side by side here for that reason.

## What this repository has that the others do not report

- A **measured free-running chain at 128-bit**: 148 consecutive decode steps with the state carried under encryption, and 9,472
  exact comparisons against the model's plaintext, flat along the chain. HEAT and Perseus report teacher-forced encrypted
  decoding chains (128 and 16 tokens, one sequence, KV cache carried); HE-Guardrail states 200-token encrypted generation
  without timing, a security level or a fidelity measurement; fhe-mamba (16 tokens, 2026-09-28) and AR-HE (24 tokens on
  random weights, 2026-10-04) followed; Sylph, Cachemir and NEXUS price one step.
- **Constant cost and memory in context length**, measured over the chain (a first-order recurrence, no KV cache). Cachemir's
  KV cache grows with context (17 GB at 1,024 tokens); Sylph's decode attends over a public/private KV cache.
- **The model is the circuit**: no approximation error on top of CKKS noise. Cachemir and Sylph approximate a pretrained
  transformer's nonlinearities and report the approximation's accuracy separately.
- **A public verification kit**: the keys and every ciphertext of a recorded session ([VERIFICATION_KIT.md](VERIFICATION_KIT.md)).
  Several of the others publish their code (Cachemir, NEXUS, THOR, fhe-mamba, Perseus); the kit adds a recorded session to it.
- **Uniform ternary secret at ring 2^17** with the modulus chain 204 bits under OpenFHE's 128-bit ceiling ([SECURITY.md](SECURITY.md));
  Cachemir uses a sparse secret (h = 192) under sparse-secret encapsulation, HEAT and Perseus a transient h = 32 key for the
  bootstrap; NEXUS and AR-HE state h = 192 and describe no encapsulation; Sylph's secret distribution is not stated. Cerium
  (dense ternary at 2^16) and fhe-mamba (uniform ternary at 2^17, since 2026-09-27) also do without a sparse secret.

## What the others have that this repository does not

- **Model quality.** Llama-3-8B is a real language model; `pbd430a` is a 405M-parameter model with WikiText-103 token
  perplexity 28.55 and LAMBADA 9 % ([RESULTS.md](RESULTS.md) section 4).
- **Single-stream latency.** Cachemir generates a token for an 8B model in 96.6 s on one 2020-era A100; a single tick here
  takes 176–188 s on two 2025 cards, whatever the lane count — the 64 lanes are throughput, not latency.
- **Reading throughput.** Transformers process a prompt in one prefill (Sylph: 128 tokens in 20 s on eight cards); here a
  64-token prompt costs 64 ticks per lane.
- **A selective recurrence.** fhe-mamba (above) runs Mamba models under CKKS with a selective (input-dependent) decay; the
  model here has a fixed, trained decay. Its first release (2026-09-22) was four tokens at parameters without a set security
  level; since 2026-09-27 it runs at a classical-128 profile (16 tokens on 2026-09-28, 67.41 s per token on one B300).
- **Token selection on the server.** HEAT, Perseus, NEXUS and AR-HE pick the next token under encryption; here the client
  decrypts the final hidden rows, applies the output head and picks, every tick.

## Encoder-only and interactive systems (cited, not raced)

THOR ([ePrint 2024/1881](https://eprint.iacr.org/2024/1881), BERT-base, one 128-token pass 602.26 s on one A100 in its
current version; 626 s is the figure other papers quote), MOAI ([ePrint 2025/991](https://eprint.iacr.org/2025/991),
BERT-base, 141.3 s per input on one H200, amortised over 256 inputs per ciphertext), Powerformer
([ACL 2025](https://aclanthology.org/2025.acl-long.543.pdf)), Rho et al. ([ICLR 2025](https://arxiv.org/abs/2410.02486),
a 2-layer softmax-free encoder-only BERT-style classifier, 26.5 s per 128-token pass) evaluate one encoder pass and do not
generate; MOAI and Powerformer do report agreement with the plaintext model under encryption for that single pass.
BumbleBee ([NDSS 2025](https://www.ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf)), EncFormer
([arXiv 2604.09975](https://arxiv.org/abs/2604.09975)) and CryptoGen ([arXiv 2602.08798](https://arxiv.org/abs/2602.08798),
GPT-2 base generating 64–512 tokens with HE for the linear layers and two-party computation for the rest, encrypted perplexity
reported against plaintext) are interactive two-party protocols, a different threat model.
