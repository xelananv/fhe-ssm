# Limits of what was measured, and open questions

## What the November runs do not cover

- **The secret on another machine.** In every November run the client role (key generation, decryption, the token pick) ran
  on the GPU machine. The best of them opened a server session without the secret (`has_secret: false`); for the 405M model
  the session itself encrypted the inputs with the public key from plaintext rows in the same process. The only session that
  kept the secret on another machine is the 2026-09-18 one. Per-run labels: [SECURITY.md](SECURITY.md).
- **Per-conversation latency.** A conversation waits 8.86 s per token in the 405M batch, 9.74–9.76 s alone on RWKV-7 0.1B and
  39.08 s in a 32-conversation 0.4B batch; Perseus reports 1.9 s (GPT-2 small) and 7.8 s (Llama-3-8B) single-stream on the
  same card class ([COMPARISON.md](COMPARISON.md)).
- **Error trends from one run per build.** The 405M model's logit error is flat over the November 148 steps and rose 2.4×
  over the September 148 steps; the cause of the rise is not established, and no controlled pair separates runtime, ring,
  linears and keys. RWKV-7's hidden-state error rose 14 % over the last four 20-position windows of its 148-step run.
- **RWKV-7 robustness is empirical.** No site left its interval over 700,659 tokens of WikiText-103 test and LAMBADA,
  unclamped, nor over 61 passes of 32 conversations with the final fit. Corpus scans do not cover a model's own generated
  text, and an earlier, narrower fit broke all 32 conversations of a batch from one conversation's generated answer
  ([RESULTS.md](RESULTS.md) §7.5). An adversarial prompt is not excluded.
- **The gate's margins are thin and do not scale yet.** Top-1 against exact is 98.17 % (0.1B) and 98.08 % (0.4B) against a
  98 % bar; the 1.5B fails (WikiText-103 +1.9 %, top-1 93.07 %) and has no encrypted run. The gate's evaluation and the full
  scans count the inverse-square-root inputs below their interval differently; the two are not yet reconciled.
- **Unexplained errors.** The RWKV-7 prefill passes' error exceeds the decode-only runs' (prompt length and position are an
  untested confound), and the filled layout's error grows token by token in single-conversation decode while it does not in
  the batched runs.
- **PCMM fidelity.** No PCMM configuration of the 405M model reaches a relErrRms under 1e-3 (0.002–0.005 against 1–2e-4 for
  diagonals); top-1 holds because the differences sit at near-ties.
- **No 405M prefill at more than one position per lane, and no 405M run with client-side input encryption** on the Perseus
  runtime.
- **The serve loop** ran with in-process keys only, at K = 4 and K = 8; its steps are slower than the standalone figures, and
  the cause is not established. At 0.4B and K = 32 it ran out of device memory at CUDA graph instantiation, a term the
  runtime's memory model does not count.
- **"The model is the circuit"** holds for the 405M model only; RWKV-7 is a fitted polynomial approximation of a pretrained
  model, and its fidelity is measured against that polynomial model.
- **Parameter counts.** The RWKV-7 0.4B and 1.5B parameter counts are not on record, so no per-parameter figure is given for
  them.

## What the first run does not cover

- **One model, one size, one ring.** The September run measured the 405M-parameter model at ring 2^17 on one kind of box.
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
  measured decode step ([COMPARISON.md](COMPARISON.md)); Perseus (arXiv 2611.04488) decodes Llama-3-8B at 7.8 s and GPT-2 small
  at 1.9 s per token on one card, faster per conversation than every configuration here. What this repository adds is a
  batched aggregate (7.23 tokens/s over 64 conversations) and a measured chain. Sylph, Cachemir and NEXUS do not report a run
  of many generated tokens or a fidelity measured under encryption (HE-Guardrail, also on Llama-3-8B, states 200-token
  encrypted generation but reports no timing, security level or fidelity). A chain would not change their per-step cost much
  for a few hundred tokens (the context-dependent part of Cachemir's decode step is 3 % of the layer time at 512 tokens, by
  their own table); it would grow their memory (a KV cache of ciphertexts, 17 GB at 1,024 tokens for Cachemir), which here
  stays constant.
- **"First recurrent language model under FHE with generation."** Hosi121/fhe-mamba (release 2026-09-22, no preprint) generated
  four tokens from a pretrained Mamba-2-130M under CKKS: five encrypted evaluations in 38.5 minutes on one lane (about 10
  minutes per token), at parameters without a set security level, the client decrypting to pick each token. Since 2026-09-27
  it runs at a classical-128 profile (ring 2^17, uniform ternary secret) and on 2026-09-28 generated 16 tokens of a Mamba-3
  187M with a per-step error gate. Encrypted recurrent networks are older still (an encrypted GRU under CKKS in MatHEGRU,
  DOI 10.1145/3488932.3523253, 2022). What this repository adds is the length and the lanes (148 ticks × 64 conversations, 9,472 tokens, run
  2026-09-19 and published 2026-09-23) at set 128-bit parameters, with the fidelity measured under encryption.
- **"First encrypted generation chain."** HEAT (arXiv 2609.01730, 2026-09-01) ran 128 teacher-forced encrypted chains of 128
  tokens at 128-bit with per-position fidelity, and HE-Guardrail (arXiv 2609.21484, 2026-09-18) states 200-token encrypted
  generation (no timing, security level or fidelity reported). What is particular here is the combination: free-running, at
  set 128-bit parameters, fidelity at every step, 64 conversations per ciphertext, constant cost per step, public records.
- **New mechanisms.** None of the following is new, and none is claimed: a recurrent state carried under encryption at
  constant cost per step (encrypted control since arXiv 1912.07362, the same client-in-the-loop shape in arXiv 2307.03451);
  two reals per complex slot (nGraph-HE2, 2019) or a complex-packed carried decode state (Perseus's complex KV cache,
  2026-11-03); token-parallel prefill (Sylph, Perseus and others); several tensors per bootstrap (fhe-mamba's batched refresh);
  join-and-leave scheduling of encrypted batched serving (described in patents); plaintext weight delivery as the bottleneck
  (Cerium). What is reported here are measurements of them at language-model scale.
- **"32 users."** All lanes of a batch decrypt under one secret key: a batch serves one key holder's conversations.

## Open questions

- Why the extended bootstrap range (K = 768, "split") is *more* precise than the stock range on the GPU evaluator
  (0.64× the logit error) while the CPU probe on OpenFHE's evaluator showed the opposite ordering (1.78× worse).
- What an overflow beyond 512 does under the stock range on the GPU: the control arm's 8 ticks contained none (max 483);
  the CPU probe's thresholds (degraded from 519, exceptions from 534–539) are the only evidence.
- The recorder's mid-tail: bootstraps with maximum overflow between 460 and 512 are 13–19 % fewer than the Gaussian
  model predicts (−3.4 σ at > 460) while the body and the far tail match it.
- The one-off host-memory step of +512 kB / +1,024 kB at the second request (reproduced in three arms): which mapping.

## Reading the records

- On the Perseus runtime a pass's `seconds` includes the host's mask staging and the encryption of block 0's inputs; an
  **unfenced** pass is the production wall clock, a **fenced** one synchronises after every stage to time it. Quote unfenced
  passes. RWKV-7's `seconds` covers client encryption, the device programs and client decryption, not the client's head and
  pick or the scoring reference.
- RWKV-7 record headers carry no build stamp; the Python checkout of each run comes from the runsheet log. All of them share
  one `_pc` binary, but runs on different checkouts are not paired as controls.
- Host memory in the RWKV-7 records is the whole machine's `free -m`, not the session's alone.
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
  ([RESULTS.md](RESULTS.md), the selection script `spec_decode/prompt_select_longrun.py` and its reports). The
  November 148-step run of the 405M model used the same set; the prompt of the RWKV-7 free run was not screened.
