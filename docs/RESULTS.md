# Results — every number with the file it comes from

Conventions: errors are rms (a max does not reproduce across repeats); times are wall clocks (the harness's `reqMsPerToken`
is a sum of per-stage timers that omits an untimed part of the layer loop — it is in the records but it is not a tick time);
no timing is compared across different binaries or machine load states; fidelity (the circuit against the model's own
plaintext) is stated beside quality (whether the model is any good).

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
