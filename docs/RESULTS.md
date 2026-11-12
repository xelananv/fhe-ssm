# Results — every number with the file it comes from

Conventions: errors are rms (a max does not reproduce across repeats); times are wall clocks (the harness's `reqMsPerToken`
is a sum of per-stage timers that omits an untimed part of the layer loop — it is in the records but it is not a tick time);
no timing is compared across different binaries or machine load states; fidelity (the circuit against the model's own
plaintext) is stated beside quality (whether the model is any good).

Sections 1–4 are the first run (FIDESlib runtime, ring 2^17, two cards, September 2026). Sections 5–8 are the November runs
on the Perseus runtime at ring 2^16 on one card: the 405M model (5, 6), RWKV-7 (7) and the key set both used (8). A
timing from one section is never divided by a timing from the other.

## 1. The long free-running session (2026-09-19/20)

Setup: 2× RTX PRO 6000 Blackwell 96 GB, ring 2^17, `HEStd_128_classic`, 64 lanes, the served binary
`gpu_real_model_x` (sha256 `df8ada73…`, [build identity](../results/dense-demo-s31/pod_longrun_20260919/pod_pull/build/BUILD_IDENTITY.txt)),
flags `--canonical-carry --periodic-encode --drop-wv 7 --drop-wkr 4 --drop-wout 7 --drop-win 4 --newton-robust 0.1 12
--pt-cache`, extended bootstrap range (`FIDESLIB_BOOT_UNIFORM_EXT=split`), the bootstrap flight recorder on, **server-generated
test keys** (the flight recorder reads the secret key on the server, so this session is a measurement run, not the
privacy demonstration — that is the 2026-09-18 session in section 2; the test keys were deleted at the end). Prompts: 64 = 16 long reading (64–80 tokens) + 16 medium (33–47) + 16 short reading (9–14) + 16 short openers
(3–11); the reading classes half from FineWeb-Edu, half written for the run
([prompt set](../results/dense-demo-s31/pod_longrun_20260919/prompt_select/longrun_prompts64_v4.txt), [selection report](../results/dense-demo-s31/pod_longrun_20260919/prompt_select/longrun_prompts64_v4.report.md)).
Decoding: the client-side `norepeat` rule. Report: [LONG_REPORT.md](../results/dense-demo-s31/pod_longrun_20260919/LONG_REPORT.md) (includes all
64 lanes' generated text); tables for plotting: [plots_data/](../results/dense-demo-s31/pod_longrun_20260919/plots_data/).

| quantity | value | where |
|---|---|---|
| ticks served | 148 in 8 h 04 min (15:18:48Z → 23:22:33Z), stopped by a timer, 0 decode failures, 0 fallback feeds | LONG_REPORT §1, §3 |
| layer-loop wall clock per tick, warm ticks 1..147 | median **175.6 s** (174.7–176.4); first 8 vs last 8 ticks −0.14 %; slope −1.09 ms/tick | §1 |
| request → reply as the driver saw it | 188.0 s on every warm tick | §1 |
| bootstraps per tick | 577 on the cold tick, **529** on every warm tick | §1 |
| weight encodes on warm ticks | 0 (`reqEncPtMs` 0); plaintext-cache misses 1,140 / 4 / 0 / 0 … | §1 |
| device pool after each request | device 0: 73.0244 GB, device 1: 72.0244 GB — a single value each over requests 1..147 | §2 |
| host RSS of the server | +512 kB at request 2, +1,024 kB at request 20, then unchanged in 415 one-minute samples (127 requests) | §2; [mem_sampler.log](../results/dense-demo-s31/pod_longrun_20260919/pod_pull/mem_sampler.log) |
| raw top-1 agreement with plaintext | **9,445 / 9,472 = 99.715 %** (reading 1,986/1,991; generating 7,459/7,481) | §3 |
| decode-rule pick agreement | 9,452 / 9,472 = 99.789 % | §3 |
| the 27 disagreements | all near-ties: plaintext top-two margin ≤ 0.0298 logits (this model's median margin: 0.925); the encrypted pick was one of the model's own top choices every time | [ERROR_HEADROOM.md](../results/dense-demo-s31/pod_longrun_20260919/ERROR_HEADROOM.md) |
| relErrRms per lane-tick | median 0.001849; per-tick medians by block of ten ticks 0.001045 → 0.002540 (still rising at the end; fit 0.000704·tick^0.242 over ticks 5–135) | [FIDELITY_TREND.md](../results/dense-demo-s31/pod_longrun_20260919/FIDELITY_TREND.md) |
| GPU power (control arm, 3 warm ticks, request-to-request) | 339.7 + 317.1 = **656.7 W** ⇒ 34.9 Wh per 64-token tick ≈ 1,963 J per token; 82 % utilisation; ≤ 68 °C | [gpu_power.csv](../results/dense-demo-s31/pod_longrun_20260919/pod_pull/gpu_power.csv) |

**Bootstrap-input overflow** ([RECORDER_EVENTS.md](../results/dense-demo-s31/pod_longrun_20260919/RECORDER_EVENTS.md)): 78,340 bootstraps recorded,
71,236 with non-trivial inputs; the coefficient overflow I has rms 85.360 against the predicted sqrt((h+1)/12) = 85.36 for
the key's Hamming weight h; bootstraps whose maximum |I| exceeds 512 (the stock table's bound): **17 observed, 17.97 expected**
under the Gaussian model; maximum 552; every tick containing such an event decoded (raw top-1 63–64 of 64). Counts in the
460–512 range run 13–19 % under the model (−3.4 σ at > 460) — unexplained.

**The control arm** ([CTRL512_VS_LONG.md](../results/dense-demo-s31/pod_longrun_20260919/CTRL512_VS_LONG.md)): 8 ticks on the stock bootstrap range
(K = 512), same binary, box, store, prompts and rule. On 512 paired lane-ticks with identical fed history the stock table's
relErrRms is **1.572× the extended table's** (2 s.e. 1.461–1.691; larger in 371 of 512 pairs); both decode 512/512 on top-1;
the extended table costs 176.0 vs 172.5 s per tick (**+2.0 %**).


**How the 64 prompts were chosen, and what that does to the fidelity number.** The prompts were not sampled at random.
[spec_decode/prompt_select_longrun.py](../spec_decode/prompt_select_longrun.py) rolled every candidate through the exact plaintext model for
172 steps (the feeding the encrypted run would do if it agreed with the plaintext at every step), kept only the candidates that passed three
hard filters — the Newton inverse-square-root converged at every norm site at every step, the norm inputs stayed under a bound, and the
continuation did not fall into a repetition loop — and among those preferred the ones with the fewest near-ties (steps where the plaintext
top-two margin is under 0.05 logits) and then the most diverse text. The first round screened 640 FineWeb-Edu candidates (160 per length
class) and a later one 94 self-written; the final set is 24 corpus prompts, 24 written for the run and 16 short openers from the demo
prompt pool ([reports and candidate lists](../results/dense-demo-s31/pod_longrun_20260919/prompt_select/)). A near-tie is exactly where the
encrypted argmax can differ from the plaintext one, so selecting against near-ties lowers the number of disagreements: **the 99.7 % is the
fidelity on this pre-screened set, not an estimate for arbitrary prompts.** The 27 disagreements all happened at near-ties the screen left
in (its plaintext emulation of the selected set still had 282 near-tie steps); on unscreened prompts, or under a decoding rule that visits
near-ties more often, the flip rate would be higher, and it was not measured. The relative rms logit error (0.0010 → 0.0025) is a property
of the circuit on the same tokens and is the number to compare across prompt sets.

## 2. The recorded client-keyed demo session (2026-09-18)

Setup: same box and ring; keys generated on the client (a Mac), the secret key never on the server; binary `a6778d2b…`;
`--newton-robust 0.3 8` (480 bootstraps per warm tick); 64 lanes × 8 generated tokens, greedy, stop on newline; the
client encrypts/decrypts every tick over the internet. Records: [results/dense-demo-s31/pod_2xbw_20260917/](../results/dense-demo-s31/pod_2xbw_20260917/)
(client transcript, tokens, per-lane text, pod logs, the kit manifests).

| quantity | value |
|---|---|
| ticks | 18; **491 / 492 generated tokens identical to plaintext greedy decoding; 63 / 64 lanes token-exact** (lane 33: a near-tie newline vs " It") |
| layer-loop wall clock per tick | median 181.8 s (180.9–182.8); 480 bootstraps; 0 weight encodes |
| client-side tick (request up 88.1 MB, reply down 10.5 MB, over the client's link) | median 264.7 s (243.5–286.8) |
| what left the client | every request and reply ciphertext + the keys: the verification kit ([docs/VERIFICATION_KIT.md](VERIFICATION_KIT.md)) |

The same box's earlier arms are in [a11_final/](../results/dense-demo-s31/pod_2xbw_20260917/a11_final/) and
[x2_final/](../results/dense-demo-s31/pod_2xbw_20260917/x2_final/): **A11** (45 ticks, teacher-forced, pod test keys, the stock
bootstrap range, `--newton-robust 0.3 8`): ticks 0–41 decoded with top-1 2,675 / 2,688 = 99.5 %, and **ticks 42, 43 and 44
failed OpenFHE's decode check on every lane** — the failure that led to the bootstrap-input overflow investigation
([a11_tick42/](../results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/)), to the flight recorder and to the extended bootstrap
range used by the long run (which then ran 148 ticks without a failure); **X2**: the plaintext cache, −24 s per tick by the wall
clock (layer loop 182.2 → 158.0 s), kept.

## 3. Bootstrap range on the CPU twin (2026-09-18)

[results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.md](../results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.md):
planted-overflow sweeps of OpenFHE's bootstrap at rings 2^12–2^13. The stock K = 512 table degrades from |I| = 519 and
throws decode exceptions from 534–539; the extended table's edge is at 775–798. The straight swap to K = 768 loses
2.43× precision because the scalar 1/(768·N) is rounded to an integer at the encoding scale; folding the factor 3 into the
CoeffsToSlots precomputation ("split") recovers most of it (1.78× on the CPU, and *better* than stock on the GPU — the
ordering differs between the two evaluators; not explained).

## 4. Model quality (fidelity is not quality)

The fidelity numbers above say the encrypted circuit computes what the model computes. Whether the model is good is a
separate measurement ([results/dense-demo-s31/pod/state/runs/](../results/dense-demo-s31/pod/state/runs/), `anchor.jsonl` of each run; the evaluation harness
was validated against Pythia-410M's published numbers first, [QUALITY_ANCHORS.md](../results/dense-demo-s31/QUALITY_ANCHORS_20260828.md)):

| model | WikiText-103 test, token ppl (285,830 tokens, ctx 1024) | pretraining-corpus holdout, token ppl (2,000,000 tokens) | LAMBADA accuracy (5,153 items) |
|---|---|---|---|
| `pbd430a` — the served checkpoint, after a 410 M-token anneal on WikiText-103 train + Gutenberg fiction + a FineWeb-Edu replay ([provenance](../results/dense-demo-s31/pod_tools/anneal_staged_provenance.json)) | **28.55** | 27.91 | 9.0 % |
| `pbd430` — the same model before the anneal | 60.74 | — | 10.1 % |

This is a 405M-parameter model pretrained on 12.1 B FineWeb-Edu tokens in 61 h on one GPU ($94 of rental for the two-card box;
records: [s31_train_pbd430/](../results/dense-demo-s31/pod/state/runs/s31_train_pbd430/), [s31_anneal_pbd430a_a2/](../results/dense-demo-s31/pod/state/runs/s31_anneal_pbd430a_a2/));
it writes fluent English and is not a reasoning model (Pythia-410M, for scale: LAMBADA 51.6 % on the same harness).

## 5. The 405M model on the Perseus runtime (2026-11-02)

**Box.** One NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB, driver 595.84, CUDA 13.2, compute capability 12.0),
20 host cores. Ring 2^16, the `n32` chain, planned refreshes. Weights `pbd430a_weights.npz`, sha256 `3ec3f1d8…784b`.

**Three builds.** The Python commit alone does not fix a build: the `_pc` Rust extension's hash does, and a new `.so` is a new
binary. Timings are compared only inside one row of this table.

| build | commit | `_pc` .so | what ran on it |
|---|---|---|---|
| first | `c174f0ae` | `009427e7463beeca` | setup, key generation and contact tests; no whole-model pass |
| phase B | `3afbfdf3` | `009427e7463beeca` | the diagonal and PCMM controls, the pass-kernel configurations, a bundle-keyed PCMM run |
| add-on | `50666f6f` | `5bbae5e8d2c91ef7` | the stack (`cfg-best`, `cfg-best-em`), the three-document scenario, the publication run, the 148-step run |

A **pass** pushes one token-value through all 24 blocks and the tail for each of the 64 lane-slots. Aggregate tokens/s is
64 ÷ (seconds per pass); each conversation advances one token per pass. An **unfenced** pass does not wait for the device
between stages (the production wall clock); a **fenced** pass synchronises after every stage to time it (`--profile-every 2`
fences every second pass). `relErrRms` is the decrypted reply's error against the exact float64 forward on the same inputs.

**Phase B build** (in-process keys unless noted; records: [numbers.md](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_r2/numbers.md),
[runs](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_r2/gpu/)):

| run | s per pass (unfenced) | tok/s aggregate | relErrRms | top-1 |
|---|---|---|---|---|
| diagonal linears, two controls | mean **31.879** (31.77–31.96) | 2.01 | 1.0e-4–1.9e-4 | 64/64 |
| PCMM-24 linears, two controls | mean **14.035** (13.956–14.098) | 4.56 | 0.0043–0.0051 | 63–64/64 |
| PCMM-24, client bundle (`KEYS=bundle`) | 14.028 / 14.11 / 14.165 | 4.52–4.56 | 0.0040–0.0049 | 64/64, 62/64 on pass 5 |

On this build the PCMM linears run a pass 2.27× faster than the diagonals (31.879 ÷ 14.035) and pay for it in fidelity.
The diagonal method multiplies the input by every rotated diagonal of the weight matrix, one rotation and one product per
diagonal. PCMM packs the weights into 24-bit plaintext matrices and finishes with a correction-factor rescale; that finish
carries the extra error. Its relErrRms on one block against the correction factor (`--pcmm-cf`) is U-shaped: 0.318 at 4, 0.0252
at 6, **0.0020 at 8**, 0.0047 at 10, 0.019 at 12, 0.075 at 14 ([pod.log](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_r2/pod.log)).
Every whole-model run uses 8. No PCMM configuration reaches a relErrRms under 1e-3.

**Add-on build** (in-process keys unless noted; `--pcmm-cf 8` with a per-site table and `--stc-cache 160` on every run):

| run | s per pass (unfenced) | tok/s aggregate | relErrRms | top-1 | record |
|---|---|---|---|---|---|
| PCMM-24, two controls | mean **14.043** (13.965–14.11) | 4.56 | 0.0022–0.0039 | 63–64/64 | [ctlC](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_best/gpu/run_pcmm24_ctlC/record.json) |
| `cfg-best`, the stack | 8.817 / 8.848 / 8.875 | 7.21–7.26 | 0.0031–0.0037 | 64/64, 63/64 on pass 5 | [record](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_best/gpu/run_cfg-best_tick/record.json) |
| `cfg-best-em`, the stack + fused rescale | 8.67 / 8.703 / 8.741 | 7.32–7.38 | 0.0030–0.0038 | 64/64, 61/64 on pass 5 | [record](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_best/gpu/run_cfg-best-em_tick/record.json) |
| three documents on `cfg-best-em`, 58 passes | read A **8.772** (7 steady passes); 58 timed passes in 510.9 s | 7.30 (reading) | per pass 0.0016–0.0024 | 63–64/64 on full passes; 17/18 on one reading pass (pass 23) | [record](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_best/gpu/run_cfg-best-em_own/record.json) |
| the same 58 passes on PCMM-24 | 58 timed passes in 823.7 s | 4.5 | per pass 0.0015–0.0023 | 63–64/64 on full passes; 17/18 on one reading pass (pass 23) | [record](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_best/gpu/run_cfg-pcmm24-own_own/record.json) |

- **The stack** is Newton depth-2 at every norm site, the norm statistics packed in halves, PCMM-24 linears, a sibling schedule
  and paired refreshes. On this build it runs the three-document scenario **1.61×** faster than PCMM-24 alone (823.7 ÷ 510.9
  s) at indistinguishable fidelity.
- **Where a pass goes.** The last fenced pass of `cfg-best-em` is 25 stages summing to 8.667 s: 24 blocks at 0.358 s mean
  (0.341–0.359) and a tail of 0.067 s.
- **The fused rescale** (`cfg-best-em` against `cfg-best`) is −0.134 to −0.147 s per pass, one run per arm, about the size of
  the controls' own spread (0.145 s): not a measured gain.
- **The three-document scenario.** Three FineWeb-Edu documents: A is read 64 positions per pass for passes 0–15, B and C
  for passes 16–25 while A's conversations generate, then all 64 lanes generate for passes 26–57 (the `config` field of
  each pass in the record).

**The publication run** ([record](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_pub/run_cfg-best-em_pub/record.json),
[log](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_pub/pub/pub.log)). The same
scenario on `cfg-best-em` with a client key bundle: the server session recorded `has_secret: false`. The client role (key
generation, decryption of the replies) ran on the same machine, and the session encrypted block 0's inputs with the bundle's
public key from plaintext embedding rows held in the same process; this port has no client-side input encryption.

| quantity | value |
|---|---|
| keys | 89 switching keys + public key + PCMM key, 7,680,224,518 bytes, generated in 11 s; key request 113 s; deleted after the run |
| passes | 58 timed + 1 untimed warm-up; 29 unfenced |
| unfenced mean per pass | **8.840 s** over all 29 (8.717–8.872): reading A **8.793** (8 passes), reading B and C 8.863 (5), generating **8.857** (16) |
| aggregate | 64 ÷ 8.857 = **7.23 tokens/s** generated; 8.86 s per token per conversation |
| top-1 against the exact model | 1.0 on 56 timed passes, 0.984375 on two (passes 26 and 34) |

**The summarisation attempt** (quality only, no encrypted run). The model does not summarise a synthetic terms-of-service document (1,815 prompt tokens with its cue), before or
after a 63,873,024-token anneal on summarisation data ([evaluation](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/runs_d/anneal_out/eval/SUMM_EVAL_pbd430s.md),
[plaintext outputs](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/runs_d/D_PLAIN_SUMMARIES.md)).
The reason is the memory of the recurrence: the trained decays are at most 0.99175, so the best channel keeps 0.000252 of a
token 1,000 positions back, and 2.79 % of channels keep more than 1 % of a token 300 positions back
([decay_stats.txt](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/runs_d/anneal_out/decay_stats.txt)).
A short form of the document failed too ([short form](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/runs_d/anneal_out/short/SHORT_D.md)), so the limit is also the summarising itself.

## 6. The two 148-step runs of the 405M model, side by side

Both runs fed the same 64 pre-screened prompts (section 1) under the same `norepeat` rule, so their fidelity fields
compare. The time row describes each run and is not a ratio: the builds differ in runtime, ring, linears, evaluator,
cards and keys.

| | 2026-09-19, FIDESlib | 2026-11-02, Perseus |
|---|---|---|
| build, ring, cards | `gpu_real_model_x` `df8ada73…`, 2^17, 2× RTX PRO 6000 | add-on `50666f6f` / `5bbae5e8d2c91ef7`, 2^16, 1× RTX PRO 6000 Blackwell Workstation |
| keys | server-generated test keys | in-process (the session held the secret) |
| raw top-1 | 9,445 / 9,472 (27 differences, margins ≤ 0.0298) | **9,462 / 9,472** (10 differences, margins ≤ 0.0067, median 0.0021) |
| decode-rule picks | 9,452 / 9,472 | 9,466 / 9,472 |
| logit relErrRms over the run | block mean of per-step medians 0.001045 → 0.002540, still rising (fit 0.000704·step^0.242) | median per 10-step block **0.00059–0.00069**, no trend |
| reply relErrRms over the run | — | median per block 0.00164 → 0.00181; mean of the first ten steps 0.00163, of the last ten 0.00182 |
| memory | device pool one value per card over 147 requests | `mem_free_bytes` one value (30,000,349,184) over 148 steps |
| wall clock per step (descriptor) | 175.6 s layer loop (median of steps 1–147) | 8.929 s unfenced mean (74 steps, 8.793–8.951) |
| record | [LONG_REPORT.md](../results/dense-demo-s31/pod_longrun_20260919/LONG_REPORT.md), [FIDELITY_TREND.md](../results/dense-demo-s31/pod_longrun_20260919/FIDELITY_TREND.md) | [record.json](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_148/run_best_free148/record.json), [per-block table](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_148/run_best_free148/record.json) |

The error grew 2.4× over 148 steps on the first build and shows no trend in the logits on the second; the cause of the rise
is not established. On the second build the reply error rises about 12 % over the first 60–90 steps and then stays flat.
That shape has a simple bound: a per-step error e on a channel that decays by a per step accumulates to at most e/(1 − a),
approached as 1 − a^k, which takes about 3/(1 − a) ≈ 360 steps for the slowest channel (a = 0.99175). The first build's
power-law rise did not follow that bound's shape. Each trend is one run per build.

## 7. RWKV-7 World 0.1B and 0.4B (2026-11-04/05)

**Box.** One NVIDIA RTX PRO 6000 Blackwell Workstation (600 W, 97,887 MiB), 32 cores, 186,252 MiB of host RAM, CUDA 13.2. Ring
2^16, the `n32` chain, PCMM linears, planned refreshes, the levers `PAIR=1`, fused EvalMod and the sibling schedule.
**Build:** branch `fhe-ssm-rwkv7-any`, one `_pc` .so (`1ef01cc1f9604226`) and three Python checkouts: `071cfa2c` (0.1B decode,
prefill, batched, free run), `197cb0ba` (the host-memory flag, the 0.1B serve loop), `b4d825c1` (the device-memory flag,
every 0.4B run). The record headers carry no build stamp; the checkout of each run comes from the runsheet log
([sheet.log](../results/rwkv7-fhe-20261104/new_pod/sheet.log)). **Keys:** every run below except the serve loops opened the
session from a client bundle (`keys: bundle`, `has_secret: false`, `input_encryption: client`, `params_hash d364d0968ad178ca`):
a client object in the same process on the GPU machine encrypted every input and decrypted every reply.

**What is scored.** Top-1 and `logits_relErrRms` compare the encrypted run with the plaintext *polynomial* model on the same
tokens (fidelity). Quality is the gate below. `seconds` covers client encryption, the device programs and client decryption;
the client's `ln_out`, head and pick, and the plaintext reference used for scoring, are outside it.

### 7.1 Decode, one conversation ([f2_01b/run_best_P1.jsonl](../results/rwkv7-fhe-20261104/new_pod/f2_01b/run_best_P1.jsonl), `_P2`, `_P3`)

| prompt | positions (prompt + generated) | top-1 | logits relErrRms median | s per token, median |
|---|---|---|---|---|
| P1 | 39 (7 + 32) | 39 / 39 | 1.81e-3 | **9.740** |
| P2 | 41 (9 + 32) | 41 / 41 | 1.73e-3 | **9.755** |
| P3 | 61 (29 + 32) | 61 / 61 | 2.38e-3 | **9.760** |

330 compiled refresh sites per token, 22 of them paired, plus 167 PCMM finishes. The 0.4B decode figure, 19.74–19.77 s per
token, comes from the decode tails of its two prefill sessions (7.2); no standalone client-keyed 0.4B decode record exists.

### 7.2 Prefill through the decode lanes ([f2_01b/run_pf32_best_L1.jsonl](../results/rwkv7-fhe-20261104/new_pod/f2_01b/run_pf32_best_L1.jsonl), `_L5`, `_L9`, `_L25`; [f2_04b/run_pf32_best_L1.jsonl](../results/rwkv7-fhe-20261104/new_pod/f2_04b/run_pf32_best_L1.jsonl), `_L5`)

A pass reads T = 32 prompt tokens of one conversation, one per lane, then the session decodes the rest.

| model | s per 32-token pass | **s per prompt token** | decode in the same sessions | ratio | top-1 prefilled + decoded |
|---|---|---|---|---|---|
| 0.1B (4 prompts) | 24.2–24.6 | **0.76** | 9.750–9.766 s | 12.9× | 409 / 416 + 198 / 201 |
| 0.4B (2 prompts) | 48.8–49.9 | **1.53–1.54** | 19.74–19.77 s | 12.9× | 223 / 224 + 90 / 90 |

The prefill passes' error (0.1B medians 3.2–6.6e-3) exceeds the decode-only runs' (1.7–2.4e-3), and so do the decode tails
of the same sessions (4.3–9.6e-3), so prompt length or position is an untested confound; the cause is not established.

### 7.3 Batched decode, 32 conversations ([f2_01b/batch_k32_filled_best_pob2.jsonl](../results/rwkv7-fhe-20261104/new_pod/f2_01b/batch_k32_filled_best_pob2.jsonl), [f2_04b/batch_k32_filled_best_pob2.jsonl](../results/rwkv7-fhe-20261104/new_pod/f2_04b/batch_k32_filled_best_pob2.jsonl))

32 conversations, 32 generated tokens each, 61 passes. Each pass advances every active conversation by one token.

| | 0.1B | 0.4B |
|---|---|---|
| s per pass, median (min–max) | **19.36** (18.89–19.45) | **39.08** (38.49–39.25) |
| tokens/s at 32 active lanes | **1.65** (16× its own decode) | **0.82** (16× its own decode) |
| tokens/s realised over the run (lanes idle as conversations finish) | 1.11 | 0.55 |
| s per token per conversation | 19.36 | 39.08 |
| compiled refresh sites per pass | 684 | 1,358 |
| top-1 | 1,300 / 1,306 | 1,302 / 1,306 |
| logits relErrRms median; per-pass medians first → last | 9.91e-4; 9.6e-4 → 1.3e-3 | 2.14e-3; 2.25e-3 → 2.67e-3 |

Both runs use the **filled layout**: each conversation's carried state has two value groups, packed into one complex
ciphertext as S = S0 + i·S1, so every state product, mask and carry refresh acts on both for one price. On the 0.1B, same build
and checkout, the current layout ran 27.99 s per pass with 1,068 refresh sites
([batch_k32_best_pob2.jsonl](../results/rwkv7-fhe-20261104/new_pod/f2_01b/batch_k32_best_pob2.jsonl)); filled is 1.446× faster.
The first → last trends compare a pass with 32 active lanes to one with a single active lane, so they are not error trends.
The filled layout in single-conversation decode was 3.0 % faster but its error grew token by token (logits median 8.19e-3,
two flips in 39, [run_filled_best_P1.jsonl](../results/rwkv7-fhe-20261104/new_pod/f2_01b/run_filled_best_P1.jsonl)); the cause
is not established, and decode stays in the current layout.

### 7.4 148 free-running steps, 0.1B ([f2_01b/run_best_free148.jsonl](../results/rwkv7-fhe-20261104/new_pod/f2_01b/run_best_free148.jsonl))

Prompt P1 (7 positions), then 148 greedy steps, the client decrypting each pick and feeding it back. The prompt was not
pre-screened.

| per 20-position window | values |
|---|---|
| s per token, median | 9.680, 9.765, 9.769, 9.773, 9.768, 9.772, 9.769, 9.774 |
| logits relErrRms, median | 1.3e-3, 2.1e-3, 2.7e-3, 4.5e-3, 3.7e-3, 2.8e-3, 2.4e-3, 2.5e-3 |
| hidden state against the circuit's mirror, median | 4.5e-3, 8.0e-3, 1.1e-2, 1.4e-2, 1.4e-2, 1.5e-2, 1.6e-2, 1.6e-2 |

Top-1 155 / 155; no site left its interval. The hidden-state error still rose 17 % over the last four windows (medians 0.0137 → 0.0159).

### 7.5 The polynomial models and the gate ([gate/](../results/rwkv7-fhe-20261104/new_pod/gate/))

A ciphertext cannot be clamped, so the gate scores the polynomial model **unclamped** in plaintext over whole test sets.
Bars: WikiText-103 perplexity within 0.5 % of the exact model, LAMBADA within 0.5 points, top-1 agreement with exact ≥ 98 %,
every value finite unclamped.

| model | fit | WikiText-103 token ppl | LAMBADA | top-1 vs exact | unclamped |
|---|---|---|---|---|---|
| 0.1B | exact | 19.4044 | 48.98 % | — | — |
| 0.1B | final | **19.4144** | **48.96 %** | **98.17 %** | equals clamped |
| 0.4B | exact | 13.7530 | 58.49 % | — | — |
| 0.4B | final | **13.7484** | **58.33 %** | **98.08 %** | equals clamped |
| 1.5B | exact | 9.618 | 69.34 % | — | — |
| 1.5B | final | 9.800 (**+1.9 %, fails**) | 69.07 % | 93.07 % (**fails**) | equals clamped |

Sources: [0.1B](../results/rwkv7-fhe-20261104/new_pod/gate/0.1b/eval_final.jsonl), [0.4B](../results/rwkv7-fhe-20261104/new_pod/gate/0.4b/eval_final.jsonl),
[0.4B exact](../results/rwkv7-fhe-20261104/new_pod/gate/0.4b/eval.jsonl), [1.5B](../results/rwkv7-fhe-20261104/new_pod/vd_runs/1.5b_pipe/verdict_final.json).
The 0.1B exact LAMBADA figure comes from the earlier evaluation of the same checkpoint (sha256 `60c98129…` in both).

- **Full scans.** The final fits were run unclamped over the whole WikiText-103 test (286,849 tokens) and LAMBADA's texts
  (413,810 tokens): zero non-finite values on both models. The only out-of-interval inputs are to the GroupNorm inverse
  square root, below its accuracy range: 0.1B 1 token (WikiText) and 0 (LAMBADA), 0.4B 2 and 1, all finite
  ([0.1B wiki](../results/rwkv7-fhe-20261104/new_pod/gate/0.1b/scanfull_final_wiki.json), [lambada](../results/rwkv7-fhe-20261104/new_pod/gate/0.1b/scanfull_final_lambada.json)).
- **How the fit was reached.** Fitting each site on its observed range plus a 10 % margin was quality-neutral clamped but went
  non-finite unclamped: a degree-53–127 Chebyshev series explodes just past its interval, and every failing window started at
  a sigmoid or tanh site. Widening every interval by 50 % at degree ≤ 255 removed every non-finite value but cost top-1
  (93.66 % at 0.1B, 91.30 % at 0.4B), almost all of it at the ReLU² sites. The final fit keeps the wide intervals and fits ReLU²
  on a 20 % margin; the inverse square roots are a constant-seed Newton iteration.
- **The 1.5B** (24 blocks, width 2,048) fails on perplexity and top-1 with nothing non-finite (696 GroupNorm and 1 ReLU² inputs
  out of interval, all finite); five tanh and sigmoid sites missed their fit target. No 1.5B encrypted run exists.

**A negative result: one conversation broke a batch.** With the earlier, narrower coefficients a 32-conversation batched run
(0.1B, an earlier build, in-process keys, [r01b_batch/batch32.jsonl](../results/rwkv7-fhe-20261104/new_pod/r01b_batch/batch32.jsonl))
held 575 / 576 top-1 over passes 0–17. At pass 18 one conversation's own generated answer to "What is 7 times 8?" drove
`blocks.9.att.a_sigmoid` out of its interval, and the median relErrRms of **all 32** lanes jumped to 0.86. Each conversation's
state is its own ciphertext set, but the token-side ciphertexts are shared: their refresh mixes all slots, and the masked lane
rotations carry the blow-up to every lane. With the final coefficients the same 32 prompts ran 61 passes at 1,302 / 1,306 and no
excursion. Corpus scans do not cover a model's own generated text; an adversarial prompt is not excluded.

### 7.6 Continuous batching ([f2_01b/serve_k4_pf32_best_pob2.jsonl](../results/rwkv7-fhe-20261104/new_pod/f2_01b/serve_k4_pf32_best_pob2.jsonl), [f2_04b/serve_k8_pf32_filled_best_pob2.jsonl](../results/rwkv7-fhe-20261104/new_pod/f2_04b/serve_k8_pf32_filled_best_pob2.jsonl))

A serve session holds four program sets (decode, prefill T = 32, batched K, join). Eight conversations arrive at passes
0, 0, 0, 1, 2, 4, 6 and 9 with prompts of 5–45 tokens; each is prefilled or decoded alone, joins a free lane (a register
hand-over and one small program), generates 8 tokens and leaves. In-process keys: a functional result, not a throughput row.

| | 0.1B, K = 4 | 0.4B, K = 8 |
|---|---|---|
| top-1 (all positions) | 245 / 248 | 247 / 248 |
| join, median | 3 ms | 5 ms |
| batched pass / prefill pass / decode step, medians | 14.18 / 29.77 / 11.91 s | 30.88 / 57.32 / 24.34 s |
| share of the loop in decode steps | 74 % of 1,412 s | 74 % of 2,895 s |
| VRAM max | 85,900 MiB | 89,356 MiB |

The 0.4B serve session at K = 32 opened at a modelled 88,018 MiB and failed in `cudaGraphInstantiate` at 97,174 MiB: the
instantiated CUDA graphs of its 76 programs are not in the runtime's own memory accounting.

### 7.7 Memory ([shed](../results/rwkv7-fhe-20261104/new_pod/f2_01b/batch_k32_best_pob2_shed.jsonl), [lean](../results/rwkv7-fhe-20261104/new_pod/f2_01b/batch_k32_best_pob2_lean.jsonl))

Two flags change what the host and the device hold without changing a compiled program (the programs are byte-identical).
Host figures are the whole machine's `free -m` maximum over the session; device figures are the runtime's memory model.

| 0.1B, 32 conversations, current layout | without | with | pass time |
|---|---|---|---|
| host, `RWKV7_SHED_TWIN=1` (the CPU twin keeps one slot of vectors only the record references, and shares equal plaintexts) | 149,520 MiB | **26,425 MiB** | 27.99 → 27.96 s |
| device, `RWKV7_LEAN_PREFIX=1` (one shared encryption of the public prefix state instead of one per conversation) | 90,905 MiB | **63,689 MiB** | 27.99 s (52 of 61 passes; the run was stopped there) |

The 0.4B 32-conversation filled session, with both flags: 70,687 MiB on the device (VRAM max 75,904 MiB), 39,462 MiB host-wide
([record](../results/rwkv7-fhe-20261104/new_pod/f2_04b/batch_k32_filled_best_pob2.jsonl)). Graph instantiation adds 9,156 MiB
(the failed K = 32 serve) and 11,469 MiB (the K = 8 serve: 89,356 against a modelled 77,887) outside the model.

## 8. The security of the November key set

The 405M publication run and every bundle-keyed RWKV-7 run carry `params_hash d364d0968ad178ca`: one parameter set. Priced with
the lattice estimator `3e48ef4` under its default cost model, classical, IND-CPA
([battery.txt](../results/estimator-speedrun-port-20261103/battery.txt), [battery.jsonl](../results/estimator-speedrun-port-20261103/battery.jsonl)):

| key | secret | modulus (log2) | weakest attack | bits |
|---|---|---|---|---|
| main key (rotations, relinearisation, conjugation, sparse → dense, public key) | uniform ternary, ring 2^16 | Q·P 1,705.34 (54 + 9 primes) | uSVP / BDD, β 333 | **131.28** |
| dense → sparse encapsulation key | Hamming weight 32 | 307.64 (bottom shape) | primal MITM hybrid | **142.97** |
| PCMM ring-switch key | uniform ternary, ring 2^12 | 86.98 | BDD, β 450 | **159.71** |

The key set is 2^131.28, capped by the main key. Not covered: quantum cost models, attacks that use decrypted values
(the server never sees a decryption), and the circular-security assumption every relinearisation and rotation key makes.
