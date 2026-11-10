# Comparison with published systems

Numbers for other systems are transcribed from their preprints, publications or repositories (linked); where a source does
not state a quantity, the cell says so. Ours are from [RESULTS.md](RESULTS.md). Read the caveats before the table: the systems
differ in model size by up to 65× (124 M to 8 B parameters; ours are 405 M, and RWKV-7 at 191 M and 0.4 B), in what one
"token" costs (a decode step of a KV-cache transformer vs a step of a recurrence), in hardware generation, and in what was
actually measured. **A batched aggregate rate and a single-stream rate are different quantities**
([arXiv 2610.19607](https://arxiv.org/abs/2610.19607) makes the same warning for private transformer inference): our batched
rows below give both the aggregate and the time each conversation waits per token.

## Non-interactive FHE generation of text (per generated token)

| system | model | security | hardware | per generated token | what was measured | fidelity measured under encryption? |
|---|---|---|---|---|---|---|
| **this repository, November 2026** | 405M recurrent LM (trained for the circuit) | ring 2^16, key set 2^131.28 by the lattice estimator (classical); client role on the GPU machine | 1× RTX PRO 6000 Blackwell Workstation | **8.857 s per 64-conversation pass**: 7.23 tokens/s aggregate, 8.86 s per token per conversation | 58 passes of a three-document scenario with no secret in the server session; 148 free-running steps (in-process keys) at 8.929 s per step | yes: top-1 9,462 / 9,472 over the 148 steps on the pre-screened prompt set; logit relErrRms 0.00059–0.00069 per 10-step block ([RESULTS.md](RESULTS.md) §5–6) |
| **this repository, November 2026** | RWKV-7 World 0.1B and 0.4B (public checkpoints, fitted polynomials) | as above; a client object in the same process encrypts inputs and decrypts replies | 1× RTX PRO 6000 Blackwell Workstation | 0.1B: **9.74–9.76 s** single-stream; 32 conversations 19.36 s per pass = 1.65 tokens/s aggregate. 0.4B: 19.74–19.77 s; 39.08 s per pass = 0.82 tokens/s | decode, prefill, 32-conversation batches of 61 passes, 148 free-running steps (0.1B), a continuous-batching loop | yes, against the polynomial model: 141 / 141 (0.1B decode), 1,300 / 1,306 and 1,302 / 1,306 (batched), 155 / 155 (free run) ([RESULTS.md](RESULTS.md) §7) |
| **this repository, September 2026** | 405M recurrent LM (trained for the circuit) | ring 2^17, 128-bit classical (uniform ternary secret) | 2× RTX PRO 6000 96 GB | **2.7–2.9 s per lane-token** (175.6 s layer loop / 188 s request→reply per 64-lane tick) | **148 consecutive ticks, 9,472 tokens, free-running**; a separate 18-tick client-keyed session | yes: top-1 99.7 % vs the model's own plaintext on the same tokens, every tick — on a prompt set pre-screened in plaintext against near-ties, so conditional on that selection ([RESULTS.md](RESULTS.md)) |
| Perseus ([arXiv 2611.04488](https://arxiv.org/abs/2611.04488), 2026-11-03) | Llama-3-8B; GPT-2 small (124M) | ring 2^16, 128-bit stated, a 32-bit RNS chain with composite scaling of degree 2, 27 levels over 51 primes; key distribution not stated | 1× RTX PRO 6000 | **7.8 s** (Llama-3-8B) and **1.9 s** (GPT-2 small) per token, batch 1, context 512, end to end; prefill of 128 tokens 54.8 s and 6.8 s | one decode step with a complex-packed KV cache and an encrypted argmax (CutMax); no multi-token run and no multi-sequence decode reported | perplexity at context 512: 26.46 vs 26.47 in plaintext (GPT-2 small) |
| Sylph ([arXiv 2601.18511](https://arxiv.org/abs/2601.18511)) | Llama-3-8B | not stated explicitly in the text we could access; ring degrees up to 2^16 | 8× RTX PRO 6000 (also 1–4× B200) | **18 s** (8× RP6000), 31 s (1× RP6000), 24 s (1× B200), 12 s (4× B200), per output token | prefill of 128 encrypted tokens (20 s) + the cost of one decode step at T = 1 (the decode figures equal its per-layer times, 0.98 s and 0.56 s, × 32 layers); no multi-token run reported | quality from a CKKS-noise-injection simulator (12-bit precision matches FP16 perplexity), not from an encrypted run |
| Cachemir ([arXiv 2602.11470](https://arxiv.org/abs/2602.11470)) | Llama-3-8B (also GPT-2, TinyLlama-1.1B) | ring 2^16, 1763-bit modulus, 128-bit claimed, sparse secret h = 192 under sparse-secret encapsulation | 1× A100 80 GB | **96.6 s** (1.61 min per generated token, their §7.3; one decoder timed module by module, × 32 layers) | one decode step at context 512, single sequence; encrypted KV cache carried (≈ 17 GB at 1,024 tokens) | not reported; the token-selection / re-embedding step is not described |
| NEXUS ([NDSS 2025](https://www.ndss-symposium.org/wp-content/uploads/2025-868-paper.pdf)) | Llama-3-8B | 128-bit; sparse secret h = 192 (stated in its error appendix); no key encapsulation described | 4× A100 40 GB | 51.84 s per 8-token input, batch-32 amortised, "roughly the aggregation of the microbenchmarks" (Sylph reads the same figure as per input token, 414.72 s for the 8-token query) | one generated token from an 8-token input, selected under encryption (secure argmax over the vocabulary) | task accuracy under encryption only (e.g. SST-2 94.94 → 94.46 %) |
| Cerium ([arXiv 2512.11269](https://arxiv.org/abs/2512.11269)) | Llama-3-8B | 128-bit, ring 2^16, 1,782-bit maximum modulus, ternary secret of Hamming weight 32K (dense); no encapsulation described | 8× H100 / 8× B200 | no decode stage: 134 s is a 128-token **prefill** to the first token | prefill only, decoder blocks | not for Llama-3-8B (encrypted BERT-base RTE accuracy equals plaintext, one pass) |
| fhe-mamba ([Hosi121/fhe-mamba](https://github.com/Hosi121/fhe-mamba), README and research notes at commit `5aa03cce`, 2026-10-02; no preprint) | Mamba-2-130M (pretrained, 24 layers) at release; since 2026-09-24 a trained Mamba-3 SISO 187M (12 layers); one lane | at release (2026-09-22) ring 2^16, `security=not-set`; **since 2026-09-27** a classical-128 profile: ring 2^17, uniform ternary secret, no sparse secret, depth 44, scale 59 | DGX Spark (GB10) at release; one B300 since 2026-09-27 | at release 32.4 min for 4 tokens (≈ 8 min per token, not-set); at 128-bit: **67.41 s per token** for a 16-token request (1,078.48 s, public-state reuse; 72.55 s evaluation-only) | at release 4 tokens; at 128-bit (2026-09-28) **16 free-running tokens**, the client decrypting the final hidden vector and picking each token; a 64-token attempt failed its own error gate at token 24 | yes, at every selection: hidden-vector error against the exact model and its polynomial circuit (max 6.63e-5 vs the exact model on the 16-token run, gate 0.001) |
| HEAT ([arXiv 2609.01730](https://arxiv.org/abs/2609.01730), 2026-09-01; the Perseus group) | GPT-2 small (124M, pretrained, fine-tuned so its approximations need fewer iterations) | ring 2^16, 128-bit enforced by OpenFHE, dense uniform ternary key with a transient Hamming-weight-32 key for the bootstrap | 1× A100 64 GB | **37.6 s** per token (54.3 s for the uncalibrated baseline), incl. 9.3 s of encrypted argmax | **128 teacher-forced chains of 128 tokens**, one sequence each, KV cache carried | yes: per-position top-1 and KL vs its plaintext model; top-1 falls along the chain (baseline 94 % → 50 %, HEAT 90 % → 76 %), pooled 74.0 % / 82.9 % |
| Perseus ([gladia-research-group/perseus](https://github.com/gladia-research-group/perseus), README at commit `b47fd430`, 2026-10-06, before its preprint) | GPT-2 small (124M, pretrained; one sequence) | ring 2^16, OpenFHE `HEStd_128_classic`, uniform ternary persistent key; the bootstrap mod-raises under an ephemeral Hamming-weight-32 key through switching keys | 1× RTX PRO 6000 96 GB | **11.78 s** per token (32-bit composite chain), 16.38 s (64-bit) | a 16-token teacher-forced decode with the KV cache carried under encryption and an encrypted argmax at every token; encrypted-feedback generation (8 tokens after a 4-token prompt in its README) implemented, not timed | yes: top-1 15/16 (93.8 %) and KL 0.026 vs its plaintext model (100 % / 0.002 on the 64-bit chain) — a fitted polynomial approximation of a pretrained transformer |
| AR-HE ([arXiv 2610.04912](https://arxiv.org/abs/2610.04912), 2026-10-04) | GPT-2 small and a nanoGPT (65-token vocabulary), both on **random weights** ("no claim about language quality") | ring 2^16, sparse ternary h = 192, 1,465-bit modulus, 128-bit classical by the lattice estimator; the persistent key, no encapsulation | 1× H100 | **544 s** per generated step for GPT-2 small at context 33 (the prompt step 4,630 s); nanoGPT 187 s | the whole loop on the server: token selected under encryption, re-embedded under encryption, encrypted KV cache, client offline; chains of 24, 16 and 8 generated tokens — by their bootstrap counts on the nanoGPT (the 16-token run's 71,006 bootstraps are nanoGPT's per-step law exactly); the token count is fixed before the run | token agreement with the plaintext model on the same random weights: 24/24 (best of 20 prompts), 16/16, 5/8; for GPT-2 small, agreement at the first step |
| HE-Guardrail ([arXiv 2609.21484](https://arxiv.org/abs/2609.21484), 2026-09-18) | Llama-3-8B-Instruct (pretrained), with jailbreak guardrails evaluated beside it | not stated (2^15 slots, 17 levels after bootstrapping; desilofhe) | 2× H200 | not reported | states that every target response of ≈ 4,000 prompts was generated under encryption: a 200-token budget, each selected token fed back as an encrypted one-hot vector times the embedding table, 16 requests interleaved per ciphertext at a padded length of 2,048; how the token is selected under encryption, the run times and the total compute are not described | not reported against the plaintext model (guardrail decisions and harmfulness of the responses only) |

**Reading the table.** The transformer systems' numbers are the cost of one decode step at T = 1 (Sylph, Cachemir, Perseus:
one new token appended to a context of 128 or 512 with a KV cache; NEXUS: one forward over an 8-token input,
batch-amortised), priced per hardware (Sylph's and Cachemir's composed from per-layer or per-module timings, NEXUS's from
microbenchmarks, Perseus's end to end), and none of them reports a fidelity against the original model measured under
encryption over a generated chain. Multi-step chains: HEAT (2026-09-01) reports 128 teacher-forced chains of 128 tokens on
one sequence each, with fidelity that degrades along the chain; HE-Guardrail (2026-09-18) states 200-token encrypted
generation with 16 requests per ciphertext but reports no timing, security level or fidelity; Perseus's repository a 16-token
teacher-forced decode; fhe-mamba 16 free-running tokens on one lane at 128-bit (2026-09-28); AR-HE 24-token server-side loops
on a random-weight nanoGPT (2026-10-04). Ours: 148 consecutive steps × 64 conversations, free-running, every conversation fed
the token chosen from its own encrypted output, the fidelity against the model itself at every step, with constant cost per
step, on two runtimes (2026-09-19, ring 2^17; 2026-11-02, ring 2^16), and 148 steps of RWKV-7 0.1B on one conversation
(2026-11-05). HEAT, Perseus, NEXUS, AR-HE and (by its description) HE-Guardrail select the token under encryption on the
server; here the client decrypts and picks it every step.

**GPU-seconds per generated token, two normalisations side by side.** GPU-seconds per token = cards × seconds per pass ÷
tokens per pass; per parameter, divide by the parameter count (ours: 404.7 M total, 353.2 M in the encrypted blocks; Llama-3-8B
8.03 B; GPT-2 small 124 M). All figures below are arithmetic on the rows above.

| system, row | GPU-s per token | GPU-ns per parameter-token | per conversation, s per token |
|---|---|---|---|
| this repository, September (2 cards × 175.6–188 s ÷ 64) | 5.49–5.88 | 13.6–14.5 | 175.6–188 |
| this repository, November, 405M (8.857 s ÷ 64) | **0.138** | 0.342 (total) / 0.392 (encrypted blocks) | 8.86 |
| this repository, November, RWKV-7 0.1B batched (19.36 s ÷ 32) | 0.605 | 3.17 (191.0 M total) / 6.70 (90.4 M server-side) | 19.36 |
| this repository, November, RWKV-7 0.1B single stream | 9.74–9.76 | 51 (total) | 9.74–9.76 |
| this repository, November, RWKV-7 0.4B batched (39.08 s ÷ 32) | 1.22 | parameter count not on record | 39.08 |
| Perseus, GPT-2 small, batch 1 | 1.9 | 15.3 | 1.9 |
| Perseus, Llama-3-8B, batch 1 | 7.8 | 0.97 | 7.8 |
| Sylph, Llama-3-8B, one step | 24 (one B200) to 144 (eight RTX PRO 6000) | ≈ 3–18 | 12–31 |
| Cachemir, Llama-3-8B, one step | 96.6 | ≈ 12 | 96.6 |

- **The September figure is beaten.** Perseus's GPT-2 small spends 1.9 GPU-s per token on one card, about a third of the
  September run's 5.5–5.9, and is level with it per parameter (15.3 against 13.6–14.5 GPU-ns). That run paid for ring 2^17
  (every operation about twice its 2^16 cost; ring 2^16 did not hold the circuit's depth at a 59-bit scaling factor under the
  128-bit modulus ceiling of 1,747 bits on that library build) and for two cards.
- **The November 405M figure** is 0.138 GPU-s per token: about 2.5× (encrypted blocks) to 2.8× (total) Perseus's Llama-3-8B row
  per parameter. Per conversation it is slower than both Perseus rows: 8.86 s against 7.8 s and 1.9 s.
- **RWKV-7** is below Perseus's Llama-3-8B row per parameter even batched (3.17 against 0.97 GPU-ns at 0.1B), and its 0.1B
  single stream is about 5× slower per conversation than Perseus's GPT-2 small.
- **HE-Guardrail** packs 16 requests per ciphertext but reports no time, so it has no row here.
- **Batched and single-stream figures are different quantities.** Our 0.138 and 0.605 are aggregates over 64 and 32
  conversations of one key holder; every transformer row above is batch 1. Which normalisation matters depends on what is being
  bought (tokens, parameters, or one user's latency), and the columns are stated side by side for that reason.

## What this repository has that the others do not report

- **Measured free-running chains on two runtimes**: 148 consecutive decode steps of the 405M model with the state carried under
  encryption, each with 9,472 exact comparisons against the model's plaintext. September (ring 2^17, set 128-bit parameters):
  top-1 between 98.6 and 100 % in every 10-tick block while the logit error grows about as tick^0.24
  ([FIDELITY_TREND.md](../results/dense-demo-s31/pod_longrun_20260919/FIDELITY_TREND.md)). November (ring 2^16, in-process keys):
  top-1 9,462 / 9,472 with no trend in the logit error ([RESULTS.md](RESULTS.md) §6). Plus 148 steps of RWKV-7 0.1B, 155 / 155.
  The mechanism is old: encrypted control has carried a bootstrapped or bounded recurrent state indefinitely since Kim, Shim
  and Han ([arXiv 1912.07362](https://arxiv.org/abs/1912.07362)), with the same loop shape (only the output decrypted and
  re-encrypted each step) in Lee, Lee, Kim and Shim ([arXiv 2307.03451](https://arxiv.org/abs/2307.03451)) and proven bounds in
  Schlor and Allgöwer ([arXiv 2403.18571](https://arxiv.org/abs/2403.18571)); what is added here is the measurement at
  language-model scale. HEAT and Perseus's repository report teacher-forced encrypted decoding chains (128 and 16 tokens, one
  sequence, KV cache carried); HE-Guardrail states 200-token encrypted generation without timing, a security level or a
  fidelity measurement; fhe-mamba (16 tokens, 2026-09-28) and AR-HE (24 tokens on random weights, 2026-10-04) followed; Sylph,
  Cachemir, NEXUS and Perseus's preprint price one step.
- **A reported batched decode throughput**: 64 conversations per pass at 7.23 tokens/s aggregate (405M) and 32 at 1.65 and
  0.82 tokens/s (RWKV-7), each with the time a conversation waits per token beside it. HE-Guardrail states 16 requests per
  ciphertext without a time; every other row above is one sequence.
- **Constant cost and memory in context length**, measured over the chain (a first-order recurrence, no KV cache). Cachemir's
  KV cache grows with context (17 GB at 1,024 tokens); Sylph's and Perseus's decodes attend over a KV cache.
- **The model is the circuit (405M only)**: no approximation error on top of CKKS noise. Cachemir, Sylph and Perseus
  approximate a pretrained transformer's nonlinearities, as the RWKV-7 port here does; for RWKV-7 this repository reports a
  whole-test-set gate of the polynomial model, unclamped ([RESULTS.md](RESULTS.md) §7.5).
- **A public verification kit**: the keys and every ciphertext of a recorded session ([VERIFICATION_KIT.md](VERIFICATION_KIT.md)).
  Several of the others publish their code (Cachemir, NEXUS, THOR, fhe-mamba, Perseus); the kit adds a recorded session to it.
- **Uniform ternary main secrets**: at ring 2^17 in September, with the modulus chain 204 bits under OpenFHE's 128-bit ceiling;
  at ring 2^16 in November, with a Hamming-weight-32 key used only at the bottom of the chain for encapsulation
  ([SECURITY.md](SECURITY.md)). Cachemir uses a sparse secret (h = 192) under sparse-secret encapsulation, HEAT and Perseus's
  repository a transient h = 32 key for the bootstrap; NEXUS and AR-HE state h = 192 and describe no encapsulation; Sylph's and
  Perseus's preprints do not state the distribution. Cerium (dense ternary at 2^16) and fhe-mamba (uniform ternary at 2^17,
  since 2026-09-27) also do without a sparse secret.

## What the others have that this repository does not

- **Model quality.** Llama-3-8B is a real language model; `pbd430a` is a 405M-parameter model with WikiText-103 token
  perplexity 28.55 and LAMBADA 9 % ([RESULTS.md](RESULTS.md) section 4). The exact RWKV-7 World 0.1B and 0.4B score 48.98 % and 58.49 % on LAMBADA (5,153 items;
  their WikiText-103 perplexities use another tokenizer and do not compare).
- **Single-stream latency.** Perseus generates a Llama-3-8B token in 7.8 s and a GPT-2 small token in 1.9 s on one card; here a
  conversation waits 8.86 s per token in the 405M batch and 9.74–9.76 s alone on RWKV-7 0.1B. The lanes are throughput, not
  latency.
- **Reading throughput.** Perseus reads a 128-token prompt in 54.8 s (Llama-3-8B, 0.43 s per prompt token) and 6.8 s (GPT-2
  small, 0.053 s), 18–36× its own decode per token. Here RWKV-7 0.1B reads 0.76 s per prompt token, 12.8× its own decode, and
  the 405M model reads 64 positions of one document per 8.8 s pass.
- **A selective recurrence.** fhe-mamba (above) runs Mamba models under CKKS with a selective (input-dependent) decay; the
  405M model here has a fixed, trained decay (RWKV-7's decay is input-dependent). Its first release (2026-09-22) was four
  tokens at parameters without a set security level; since 2026-09-27 it runs at a classical-128 profile (16 tokens on
  2026-09-28, 67.41 s per token on one B300).
- **Token selection on the server.** HEAT, Perseus, NEXUS and AR-HE pick the next token under encryption; here the client
  decrypts the reply, applies the output head (and for RWKV-7 the final LayerNorm) and picks, every step.
- **Secret off the server machine, in November.** Every November run had the client role on the GPU machine; the only session
  with the secret on another machine is the 2026-09-18 session ([SECURITY.md](SECURITY.md)).

## Encoder-only and interactive systems (cited, not raced)

THOR ([ePrint 2024/1881](https://eprint.iacr.org/2024/1881), BERT-base, one 128-token pass 602.26 s on one A100 in its
current version; 626 s is the figure other works quote), MOAI ([ePrint 2025/991](https://eprint.iacr.org/2025/991),
BERT-base, 141.3 s per input on one H200, amortised over 256 inputs per ciphertext), Powerformer
([ACL 2025](https://aclanthology.org/2025.acl-long.543.pdf)), Rho et al. ([ICLR 2025](https://arxiv.org/abs/2410.02486),
a 2-layer softmax-free encoder-only BERT-style classifier, 26.5 s per 128-token pass) evaluate one encoder pass and do not
generate; MOAI and Powerformer do report agreement with the plaintext model under encryption for that single pass.
BumbleBee ([NDSS 2025](https://www.ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf)), EncFormer
([arXiv 2604.09975](https://arxiv.org/abs/2604.09975)) and CryptoGen ([arXiv 2602.08798](https://arxiv.org/abs/2602.08798),
GPT-2 base generating 64–512 tokens with HE for the linear layers and two-party computation for the rest, encrypted perplexity
reported against plaintext) are interactive two-party protocols, a different threat model.
