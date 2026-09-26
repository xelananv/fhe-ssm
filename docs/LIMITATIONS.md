# Limits of what was measured, and open questions

## What the records do not cover

- **One model, one size, one ring.** Everything was measured on the 405M-parameter model at ring 2^17 on one kind of box.
  How the cost moves with model size was not measured.
- **The logit error was still rising slowly at the end of the long run.** The per-tick median relErrRms went from 0.0010
  (ticks 0–9) to 0.0025 (ticks 130–139), roughly as tick^0.24 over ticks 5–135, without visible effect on the text or on the
  top-1 agreement (every disagreement sat at a near-tie of the plaintext logits). The state carried between ticks cannot
  accumulate error without bound — each channel's state decays geometrically with a trained factor a ≤ 0.9918 (a time
  constant of at most 121 tokens), so a perturbation washes out within a few hundred ticks — but that alone does not
  reproduce the recorded rise; its cause is not established
  ([results/dense-demo-s31/pod_longrun_20260919/ERROR_HEADROOM.md](../results/dense-demo-s31/pod_longrun_20260919/ERROR_HEADROOM.md)).
- **Whether the disagreement rate changes over a long session.** 27 top-1 disagreements in 9,472 comparisons; 7 of them
  fell in the last 8 ticks (top-1 505/512 there) against 0–4 per block of ten ticks earlier. Every one is a near-tie of the
  plaintext logits (top-two margin ≤ 0.03 logits — the encrypted pick was one of the model's own top choices), so a rising
  rate would mean a rising logit error, not wrong text; one run cannot tell a cluster from a trend.
- **Quality is separate from fidelity.** Top-1 99.7 % says the circuit computes what the model computes; the model itself
  scores WikiText-103 token perplexity 28.55 (after an anneal whose mix includes that corpus's train split) and LAMBADA 9 %
  ([RESULTS.md](RESULTS.md) section 4).
- **Larger models.** Not measured: no larger model of this architecture was trained, and no size ladder was run on the GPU.
  What is known: the weight store holds 688,128 plaintexts = 2 × 24 × (3 × 4,096 + 2 × 1,024), i.e. the number of weight
  products per tick is proportional to L·(3·d_ffn + 2·d), and a ring-2^17 ciphertext holds 65,536 ÷ d lanes. For an 8B-class
  shape (32 blocks, d = 4,096, d_ffn = 16,384) that is 5.3× the products and 16 lanes instead of 64, with a third more
  bootstraps (32 blocks). If the tick time followed those counts — an assumption, not a measurement — a 16-lane tick would take
  roughly 10–15 minutes on the same two cards, i.e. 40–60 s per lane-token, with a weight store of about 350 GB (5× the 68 GB),
  which no longer fits the 256 GB host RAM of the box used. The ring, the security parameters and the constant-in-context
  property would be unchanged.
- **Energy** was sampled during the stock-range control arm only, GPU board power only.

## Claims not made here, and why

- **"Fastest encrypted inference."** Per parameter and in a single stream, the Llama-3-8B systems are ahead at their one
  measured decode step ([COMPARISON.md](COMPARISON.md)). What none of them reports is a run of many generated tokens, or a
  fidelity measured under encryption. A chain would not change their per-step cost much for a few hundred tokens (the
  context-dependent part of Cachemir's decode step is 3 % of the layer time at 512 tokens, by their own table); it would grow
  their memory (a KV cache of ciphertexts, 17 GB at 1,024 tokens for Cachemir), which here stays constant.
- **"First recurrent language model under FHE with generation."** Hosi121/fhe-mamba (release 2026-09-22, no paper) generated
  four tokens from a pretrained Mamba-2-130M under CKKS: five encrypted evaluations in 38.5 minutes on one lane (about 10
  minutes per token), at parameters without a set security level, the client decrypting to pick each token. What this
  repository adds is the chain — 148 ticks, 9,472 tokens — at set 128-bit parameters, with the fidelity measured under
  encryption.

## Open questions

- Why the extended bootstrap range (K = 768, "split") is *more* precise than the stock range on the GPU evaluator
  (0.64× the logit error) while the CPU probe on OpenFHE's evaluator showed the opposite ordering (1.78× worse).
- What an overflow beyond 512 does under the stock range on the GPU: the control arm's 8 ticks contained none (max 483);
  the CPU probe's thresholds (degraded from 519, exceptions from 534–539) are the only evidence.
- The recorder's mid-tail: bootstraps with maximum overflow between 460 and 512 are 13–19 % fewer than the Gaussian
  model predicts (−3.4 σ at > 460) while the body and the far tail match it.
- The one-off host-memory step of +512 kB / +1,024 kB at the second request (reproduced in three arms): which mapping.

## Reading the records

- `reqMsPerToken` is a sum of timers, not a wall clock; the tick time is `reqLayerLoopMs` or the driver's `serveSec`.
- A `relErr` of exactly 0 in a summary line is a NaN masked by `std::max`, not a perfect result.
- Timings are a property of the build (LTO on/off, library commit, CUDA version) and of the machine's load state; the
  reports never compare milliseconds across binaries or beside a CPU-heavy neighbour. Correctness fields do cross binaries.
- Under greedy decoding this model looped on all 64 prompts of the greedy arm (a repeated 6-word sequence, median first repeat at
  word 21) and on 719 of the 734 candidates screened; the recorded long run uses the client-side anti-repetition rule, and its
  prompt set was selected under that rule. The 2026-09-18 session (up to 8 tokens per lane) is greedy.
- **The prompt set was pre-screened, and the fidelity number is conditional on it.** The 64 prompts of the long run were chosen by a
  plaintext emulation that rejected candidates which looped, stressed the norm or had many near-ties (top-two margin under 0.05 logits),
  and near-ties are where the encrypted argmax can flip. The 99.7 % top-1 agreement is the fidelity on that set; on unscreened prompts the
  flip rate would be higher and was not measured. The per-step logit error is the number that carries across prompt sets
  ([RESULTS.md](RESULTS.md), the selection script `spec_decode/prompt_select_longrun.py` and its reports).
