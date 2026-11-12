# Figures

`make_figures.py` draws six figures from the run records and nothing else: every plotted number is read from a
named record file, and the script prints each one beside its record path. `sources.json` holds the sha256 of every
record read on the last run, so a reader can check that the figures were made from the files in this tree.

    python3 figures/make_figures.py --out figures

Output: `<name>.png` (200 dpi) and `<name>.svg` per figure, 7.5 in wide. Matplotlib and numpy only.

## Rules the figures follow

- **Error is reported as rms, never max.** Every error series is a `relErrRms` field (rms of the encrypted-minus-plaintext
  difference over rms of the plaintext): over repeated cells the rms reproduces to a few percent, the max does not.
- **No timing ratio crosses a build or a machine.** Wall clock belongs to the build and the host (link options, runtime
  version, host-CPU load all move it). Timings from different builds are drawn side by side as descriptors with the build
  named on each, never divided. The only ratios drawn are inside one build: prompt read vs generation inside one session
  (figure 4), and 32 conversations vs 1 conversation of one model on one build (figure 3).
- **Fidelity is not quality.** The error figures compare each encrypted run with that same model's own plaintext; they
  say the circuit is faithful, not that the model is good. Quality is figure 5 (plaintext evaluation of the polynomial
  model against the exact one).

## The builds and machines named in the figures

| short name | runtime | ring | host / card | runs |
|---|---|---|---|---|
| FIDESlib 2026-09-19 | FIDESlib harness | 2^17 | 2× RTX PRO 6000 96 GB (rented box) | 148-tick run, 64 conversations, server-side test keys |
| Perseus 2026-11-02 | Perseus speedrun runtime, fhe_ssm build, commit `50666f6f` (stamped in each `record.json` `build_id`) | 2^16 | 1× RTX PRO 6000 Blackwell Workstation, 20-core host | pbd430a 148-tick run, best-arm scenario, publication run |
| perseus-rwkv7 2026-11-05 | perseus-rwkv7 build, one `_pc` library `1ef01cc1` with three Python checkouts (the RWKV record headers carry no commit stamp; the checkout per run is in `sheet.log`) | 2^16 | 1× RTX PRO 6000 Blackwell Workstation 600 W, 32-core host | every RWKV-7 timing |

The three are different builds on three different hosts. The two 1-card hosts carry the same card model.

## 1. `cost_per_step` — per-step wall clock over 148 free-running steps

**Shows.** Seconds per step against the step index, log y, for three free runs; the median of each is written at the
right edge. Each curve is flat in the step index after the first step: the step's graph is the same at every step and
the carried state has a fixed size, so the cost does not grow with the context length.

**Records and fields.**
- (a) `results/dense-demo-s31/pod_longrun_20260919/plots_data/LONG_per_tick.csv`, column `wall_layer_loop_s` (the wall
  clock of the layer loop, already in seconds; the column `timer_sum_ms_NOT_A_WALL_CLOCK` is a sum of timers and is not
  used). All 148 ticks are drawn, tick 0 (cold) included and labelled.
- (b) `results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_148/run_best_free148/record.json`,
  `ticks[].seconds` where `fenced == false` (74 of 148 ticks; a fenced tick syncs the device after every stage for
  profiling and is not the production wall).
- (c) `results/rwkv7-fhe-20261104/new_pod/f2_01b/run_best_free148.jsonl`, rows `kind == "token"`, field `seconds`
  (155 positions: 7 prompt + 148 generated, one conversation).

**Numbers printed (last run).** (a) median 175.642 s over all 148 ticks, 175.638 s over ticks 1–147, range 174.748–176.418 s,
cold tick 0 201.911 s. (b) 74 unfenced ticks, median 8.935 s, mean 8.929 s, 8.793–8.951 s. (c) median 9.7686 s,
9.4331–9.7898 s; 20-position window medians 9.680, 9.765, 9.769, 9.773, 9.768, 9.772, 9.769, 9.774 s.

**Caveats.** Three builds on three hosts, two model families, 64 vs 1 conversation, ring 2^17 vs 2^16: the vertical
gaps between the curves are not speedups and the figure carries no ratio. The 09-19 run and the 11-02 148-tick run held
the secret key in the serving process (test keys); the RWKV-7 run used a secret-free session with a co-located client.

## 2. `error_per_step` — per-step error over the same free runs

**Shows.** Relative rms error per step, log y.
- (a) 09-19: per-tick median over 64 conversations of the logits `relErrRms` (column `relerr_rms_median` of
  `LONG_per_tick.csv`), with the power law `amp · tick^b` drawn dashed over ticks 5–135. The law is **refitted by the
  script** from the same column, the way the run's own trend script fits it: mean of the per-tick medians in each full
  10-tick block (14 blocks, 0–139), least squares of ln(level) on ln(block centre b0 + 5). Printed: `0.000704 · tick^0.242`
  (reproduces the published fit). Block means 0.001045 (first) → 0.002540 (last).
- (b) 11-02: two quantities per tick from `ticks[]` of the 148-tick `record.json`: `relErrRms_vs_exact` (the reply, i.e.
  the decrypted state the server returns; solid) and `logits_relErrRms_vs_exact` (the logits; dotted). All 148 ticks
  (fencing does not change the arithmetic). A saturating curve `c − d·exp(−k/60)` (time constant fixed at 60, `k` = the
  record's `pos`) is fitted to the reply series by linear least squares in (c, d): printed `c = 0.001831` (plateau),
  `d = 0.000253`. Reply first-10 mean 0.001626, last-10 mean 0.001822; logits first-10 0.000642, last-10 0.000681.
- (c) RWKV-7 0.1B: `logits_relErrRms` per token row of `run_best_free148.jsonl` (one conversation, so no median over
  conversations), drawn thinner and at alpha 0.6 so the pbd430a series read first. Median 0.00238, max 0.01722; 20-position window medians 1.30e-3, 2.11e-3, 2.68e-3, 4.52e-3, 3.70e-3,
  2.75e-3, 2.36e-3, 2.46e-3.
- Points: 10-step block medians for (a) and for both (b) series. Blocks are [0,10), [10,20), …; the last block,
  140–147, has 8 ticks and is plotted at the mean of its tick indices. (a): 0.001085, 0.001381, 0.001533, 0.001493,
  0.001635, 0.001959, 0.001910, 0.002035, 0.001969, 0.002099, 0.002270, 0.002071, 0.002071, 0.002477, 0.002403.
  (b) logits: 0.00059–0.00069 with no trend; (b) reply: 0.00164 → 0.00181 (last block partial).
- Inset (upper left, its own y-range): the shape of the geometric-series bound (dotted) against the 11-02 reply series
  (solid). A per-step error e entering a state channel that decays by a
  factor a per step accumulates to e·(1 − a^k)/(1 − a) after k steps, approaching e/(1 − a). Drawn with a = the slowest
  trained decay of pbd430a (0.99175) and e = the first reply value of (b), so it equals that value at k = 1 and reaches
  0.1186 at k = 148. It is a worst-case envelope shape (every step's error adding in phase on the slowest channel), not a
  prediction; it lies far above every measured series, which is why it sits in an inset with its own y-range rather
  than on the main panel.

**Records.** `LONG_per_tick.csv` and the two run files above;
`results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/runs_d/anneal_out/decay_stats.txt`, the line
starting `pbd430a` ("channels 24576; decay a min 0.90882 median 0.94811 p99 0.98905 max 0.99175"): the script reads
a = the `max` field, 0.99175.

**Caveats.** The three series measure different things: (a) and (c) are logits errors, (b) has both the reply/state
error and the logits error. The builds differ in runtime, ring, linear layers, evaluator and keys; fidelity fields may be
set side by side across builds, but the cause of the 09-19 rise is not established and the figure does not attribute it.
The fits describe these records only; neither is a law.

## 3. `throughput_latency` — aggregate throughput and per-conversation latency

**Shows.** Left: tokens per second summed over all conversations of a pass at full lanes. Right: seconds per token as
seen by one conversation (= seconds per pass for a batched run, since each conversation gets one token per pass).

| category | record | statistic | aggregate tok/s | s per token, one conversation |
|---|---|---|---|---|
| pbd430a 405M, 64 conversations | `…/broadcast_port_20261030/gpu_20261102_pub/run_cfg-best-em_pub/record.json` | mean `seconds` of the 16 unfenced passes whose `config` starts with `gen` (8.843–8.866) | 64 / 8.8567 = 7.226 | 8.857 |
| RWKV-7 0.1B, 1 conversation | `new_pod/f2_01b/run_best_P{1,2,3}.jsonl` | median `seconds` over all token rows pooled (per file 9.7403 / 9.7553 / 9.7599) | 0.1025 | 9.7554 |
| RWKV-7 0.1B, 32 conversations | `new_pod/f2_01b/batch_k32_filled_best_pob2.jsonl` | median `seconds` of the 61 `pass` rows (18.895–19.448) | 32 / 19.362 = 1.6527 | 19.362 |
| RWKV-7 0.4B, 1 conversation | `new_pod/f2_04b/run_pf32_best_L{1,5}.jsonl` | median `seconds` of token rows with `mode == "decode"` (the decode tail after the prefill passes), pooled (per file 19.7709 / 19.7413) | 0.0506 | 19.7595 |
| RWKV-7 0.4B, 32 conversations | `new_pod/f2_04b/batch_k32_filled_best_pob2.jsonl` | median `seconds` of the 61 `pass` rows (38.495–39.255) | 32 / 39.0825 = 0.8188 | 39.0825 |

(`new_pod` = `results/rwkv7-fhe-20261104/new_pod`.) Annotated same-build ratios: 0.1B 32 vs 1 conversation **16.1×**;
0.4B **16.2×**.

**Caveats.** pbd430a ran on the Perseus speedrun build `50666f6f` on the 2026-11-02 host; the RWKV-7 runs on the
perseus-rwkv7 build on another host (both on the RTX PRO 6000 Blackwell Workstation card, ring 2^16). No ratio is drawn
between pbd430a and RWKV-7. The 0.4B single-conversation figure comes from the decode tails of two prefill sessions
because no standalone secret-free 0.4B decode record completed. The publication run's session held no secret; its
client role (key generation, reply decryption) ran in the same process on the same host, and the server encrypted the
inputs with the public key. The aggregate figure assumes all lanes are busy; over the whole 61-pass batched runs, where
conversations finish at different passes, the realised rates were lower (0.1B 1.1065, 0.4B 0.5476 tok/s, the records'
summaries).

## 4. `prefill` — reading a prompt vs generating, inside one session per model

**Shows.** For each model, seconds per prompt token read (pale) and seconds per generated token for one conversation
(solid), log x, with the ratio inside the same session.
- RWKV-7 0.1B: `new_pod/f2_01b/run_pf32_best_L{1,5,9,25}.jsonl`. Note: the prefill rows are
  `kind == "prefill"` (fields `seconds`, `T`, `seconds_per_token`), not `kind == "pass"`; the decode tail is the token
  rows with `mode == "decode"` (token rows with `mode == "prefill"` are the per-position fidelity rows of the prefill
  passes and carry the pass time). Per prompt token = `seconds / T` (T = 32 in every row), median over 13 passes = 0.7583 s;
  decode median over the four tails = 9.7624 s; ratio **12.9×**.
- RWKV-7 0.4B: `new_pod/f2_04b/run_pf32_best_L{1,5}.jsonl`: 7 passes, 1.5272 s per prompt token; decode 19.7595 s;
  ratio **12.9×**.
- pbd430a: `…/broadcast_port_20261030/gpu_20261102_best/gpu/run_cfg-best-em_own/record.json`. The reading of document A
  is the passes with `config` `readA.*` (`describe` "seq A [0,64) …"; `token_values` = 64 positions of one document per
  pass); 8 unfenced passes, mean 8.7621 s → 0.1369 s per prompt token. Generation: the 16 unfenced `gen.*` passes, mean
  8.8094 s per generated token per conversation. Ratio **64.3×**.

**Caveats.** Every ratio is inside one session of one build. The ratio printed for RWKV-7 is from pooled medians
(12.9×); computed per session from the records' ranges it is 12.8× (0.76 s vs 9.75–9.77 s). The pbd430a ratio
compares unlike passes: a read pass carries 64 positions of one document, a generation pass one token for each of 64
conversations, so per pass the two cost the same (8.76 vs 8.81 s) and the 64× is the lane count. The figure says so
next to the number. The pbd430a scenario
run held the secret in the serving process; the RWKV-7 sessions were secret-free with a co-located client.

## 5. `rwkv7_gate` — the polynomial RWKV-7 models against the exact models

**Shows.** Three panels (WikiText-103 token perplexity, LAMBADA accuracy, top-1 agreement with the exact model) for
0.1B, 0.4B and 1.5B, exact vs polynomial **unclamped** (no clamping of polynomial inputs). Plaintext evaluation of the
polynomial model; no encryption involved. The gate bars are read from the 1.5B verdict record's `bars`: ppl within
0.5 %, LAMBADA within 0.5 points, top-1 ≥ 98 %. The 1.5B fails (ppl +1.9 %, top-1 93.07 %).

| model | exact WT103 ppl | poly ppl | exact LAMBADA | poly LAMBADA | top-1 |
|---|---|---|---|---|---|
| 0.1B | 19.4044 | 19.4144 | 48.98 % | 48.96 % | 98.17 % |
| 0.4B | 13.7530 | 13.7484 | 58.49 % | 58.33 % | 98.08 % |
| 1.5B | 9.6180 | 9.7997 | 69.34 % | 69.07 % | 93.07 % (FAIL) |

**Records.** `new_pod/gate/{0.1b,0.4b}/eval_final.jsonl` holds two lines per model,
`tag` `final_clamp` (`clamp: true`) and `final_unclamped` (`clamp: false`); there is **no separate exact row** in these
files. The exact perplexity is the field `wt_ppl_exact_same_run` of the unclamped row. The exact LAMBADA accuracy is not
in `eval_final.jsonl`: for 0.4B it is read from `new_pod/gate/0.4b/eval.jsonl` (row `tag == "exact"`, whose `wt_ppl`
13.75298 matches `wt_ppl_exact_same_run`, checked by the script); for 0.1B it exists only in
`results/rwkv7-fhe-20261104/old_pod_mirror/out_d_run_0.1b.log`, the line `exact rc=0` (its `wt_ppl` 19.40443 also
matches, checked). 1.5B: `new_pod/vd_runs/1.5b_pipe/verdict_final.json` keys `exact.*`, `unclamped.*`, `verdict`.
Clamped and unclamped are identical on every plotted field for all three models.

**Caveats.** Perplexity bars start at zero. The 1.5B has no encrypted run. The bars were set during the project, not by
an external standard.

## 6. `memory` — device and host memory of a 32-conversation RWKV-7 session

**Shows.** Two panels for four sessions. Left: device memory as the runtime's own memory report totals it (header
`mem_report_text`, the last item `total … MB`; the runtime's "MB" are MiB, 2^20 bytes). Right: the maximum host RAM in
use over the session's window, from the 5-second sampler (column 9 of `new_pod/monitor/usage_20261104.csv`, `free -m`
used, MiB: **host-wide, not the session's own process**).

| session | record | flags (header) | device total MiB | host RAM max MiB (median) | sampler window (sheet.log end − duration) |
|---|---|---|---|---|---|
| 0.1B, 32 conv., no flag | `f2_01b/batch_k32_best_pob2.jsonl` | none | 90,904.8 | 149,520 (148,954) | 14:29:01–15:07:18Z, line "01b batched K32 b2 rc=0 2297 s" |
| 0.1B, host flag `RWKV7_SHED_TWIN=1` | `f2_01b/batch_k32_best_pob2_shed.jsonl` | `shed_twin` | 90,904.8 | 26,425 (26,062) | 16:32:20–17:10:57Z, line "01b batched K32 b2 SHED … 2317 s" |
| 0.1B, both flags (+ `RWKV7_LEAN_PREFIX=1`) | `f2_01b/batch_k32_best_pob2_lean.jsonl` | `shed_twin`, `lean_prefix` | 63,688.8 | 26,272 (25,682) | 19:01:12–19:34:38Z, line "01b batched K32 b2 LEAN … rc=1 2006 s" |
| 0.4B, 32 conv., both flags | `f2_04b/batch_k32_filled_best_pob2.jsonl` | `shed_twin`, `lean_prefix` | 70,686.6 | 39,462 (38,801) | 21:20:10–22:16:33Z, line "04b-filled batched K32 b2 LEAN … 3383 s" |

All windows are 2026-11-05. Every window was found in `sheet.log` (no value was transcribed from a results table); the
window is [end − duration, end] of the sheet line, which includes the session's setup. The sampler's GPU-memory column
(column 4) gives maxima of 94,664 / 94,664 / 67,448 / 75,904 MiB over the same windows (printed, not plotted). All four
reproduce the values in `new_pod/RESULTS.md` (149,520; 26,425; 26,272; 39,462 MiB host; 90,905 / 63,689 / 70,687 MiB
device). The host flag drops what the instrument twin holds on the host; the device flag shares the prefix encryptions
instead of copying them per conversation (`ct_pool` 32,256 → 5,040 MiB in the report).

**Caveats.** The both-flags 0.1B run stopped at pass 52 of 61 (`rc=1`, no summary row); its memory figures stand, its
run is partial. The three 0.1B sessions use the current (non-filled) layout and the 0.4B session the filled layout, on
two different Python checkouts of the same build (the no-flag run on one, the flagged runs on later ones, per
`sheet.log`); the host figure is the whole host, so other processes on the pod are included.

## What did not reproduce or needed a correction

- Figure 3: the 0.4B batched-vs-decode multiple prints as 16.2× (0.8188 / 0.0506); the README rounds it to 16×.
- Figure 4: pooled medians give 12.9× for both RWKV-7 sizes; dividing the rounded per-session ranges gives 12.8× (0.76 s vs
  9.750–9.766 s; 1.53–1.54 s vs 19.74–19.77 s). Same records, different pooling.
- Figure 5: no exact row in `eval_final.jsonl`; exact values come from `wt_ppl_exact_same_run` and two other files (above).
