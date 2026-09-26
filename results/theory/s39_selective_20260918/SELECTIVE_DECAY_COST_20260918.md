# What would a non-fixed-coefficient (selective) model cost per tick? (2026-09-18)

*The question: "can we run a non-fixed param model under our architecture? is mamba possible? … whether its higher time per tick?"
This file answers the TIME half, offline. It is a **replay, not a measurement**: no selective model exists in this repository, none was
trained, nothing here ran on a GPU. Labels: **[record]** transcribed from a named file; **[replay]** counted by the level-trace replayer;
**[derived]** arithmetic shown; **[projection]** a rule applied to a named constant; **[estimate]** arithmetic without a replay behind it.
The quality half (would such a model be better?) is untouched: R4, and §6.*

## 0. Answer
1. **Input-dependent decay is cheap in the T = 1 decode tick.** On the demo's frozen configuration (64 lanes, 480 bootstraps per tick):
   a per-channel ("diagonal") selective decay costs **+0 bootstraps and +48 ciphertext multiplications** per tick; a fully projected one
   (one new d→d matrix per layer, the RG-LRU / Griffin shape) **+24 bootstraps and +24 matvecs ≈ +3.6–4.9 % of the tick**; the RG-LRU pair
   (recurrence gate + input gate) **≈ +7–10 %**. [replay + projection]
2. **What is expensive is the WIDTH of the carried state, not its selectivity.** Every extra state ciphertext per layer is exactly +24
   bootstraps per tick (+2.5 % of the tick) on every baseline tried. Mamba's N = 16 inside our block: **+360 bootstraps ≈ +38 %**; with
   selective Δ, B, C and a 4-tap convolution as well: **+432 bootstraps ≈ +50 %** for the same 64 lanes. [replay + projection]
3. **Mamba proper** (expansion 2, 48 blocks, N = 16, no FFN) is a different block; by arithmetic on the per-unit findings it carries
   1,824 ciphertexts between ticks instead of 48 and needs roughly 2,400–2,550 bootstraps per tick: **about 3× today's tick** for the same 64
   lanes (or a third of the lanes at the same time). [estimate, §5]
4. A **150M-class shape** of our own family (L = 12, d = 768, d_ff = 3,072) replays at 216 bootstraps per tick ≈ 77 s per 64-lane tick
   [projection]; the selective increments are the same in relative terms (+0 / +12 / +24 boots).
5. The replayer reproduces **three recorded boot counts exactly (380 / 404 / 480)** and the multiplication counts that the memory-leak
   trace derived from device memory (1,040 / 1,706), so its baseline is the demo's. It also makes two predictions for the next pod run (§7).

## 1. Instrument and its calibration
`selective_decay_trace.py` = `results/theory/tick_level_trace.py` (the S3.3 replay; imported, untouched) + the canonical-carry hand-off of
`results/theory/canonical_carry_trace.py` (imported, untouched) + three additions of this file: the S3.7 X8 exact drops
(`--drop-wv/-wkr/-wout/-win`, modelled from `gpu_real_model_x.cu` `x8DropTo`: the operand is dropped in place to depth − R before a matvec of
its class), `--robust-iters K` (every norm runs max(iters, K) Newton steps: the harness's `--newton-robust FRAC K`), and ONE replaced section
of the layer (the scan). `--selfcheck`: with every lever off it reproduces the imported replay exactly (boots, rotations, Λ; 3 configurations × 3 ticks).

| configuration | replay: boots/tick | record | replay: ct×ct per tick | independent count |
|---|---|---|---|---|
| canonical carry, no drops | 380 | 380: `serve_D1_real_canon`, `serve_L1r`, `serve_L9`, `serve_L9e` (table of `../../dense-demo-s31/pod_2xbw_20260917/MEMORY_RESIDUE_TRACE_20260918.md` §1b); `../TICK_COMPLEXITY_20260904.md` ("332 + 48 = 380 … exact") | 1,040 | M = 1,040 (stock seed), same trace: pool-used growth per request = 144 B × (48 × boots + M), reproduced to the byte |
| + X8 drops 7/4/7/4 | **404** | 404: `serve_L7` and seven `server.log.*.prev` (same table) | 1,040 | 1,040 |
| + `--newton-robust 0.3 8` (take 9, A11, X2) | **480** | 480: `serve_A11`, take 9's `server.log`, `serve_X2` (same table); `a11_final/A11_REPORT.md` `reqBoots` | **1,706** | M = 1,706 (robust seed), same trace |

Without the X8 drops the robust configurations replay chaotically (K = 6: 264, K = 8: 552, K = 9: 456 boots): the stock refresh rule is
chaotic in level placement (the repository conventions "Instruments that lie"). With the drops every matvec output is bootstrapped anyway and the trajectory
is regular; all deltas below are therefore read on the DEMO configuration and cross-checked on a family of five other baselines (`SWEEP.md`).

## 2. The variants (what replaces `S ← a ⊙ S + b ⊙ x`, a and b plaintext)
- **diag**: `a_t = q(w ⊙ u)`, q of degree 2 (or 4), no matrix: each channel's decay depends on its own normalised input. Update
  `S ← x' + a_t ⊙ (S − x')` with `x' = b ⊙ x`: ONE ciphertext multiplication; `a_t` is born at u + 2, in parallel with `x = W_in u`.
  The gate reads the norm output BEFORE `W_in`'s X8 drop (elementwise work does not need the drop).
- **proj**: `a_t = q(W_a u)`, one new d→d matvec per layer (same class and drop as `W_in`). With `--input-gate` (the S3.8 hybrid gate,
  `../s38_gating_20260908/`) this is the RG-LRU pair.
- **lowrank**: `a_t = q(A(Bu))`, d→64→d (Mamba's Δ shape) as two rectangular BSGS matvecs — a primitive the harness does NOT have [projection].
- **state width N**: N state ciphertexts per layer per 64-lane block (a d × N state under a packing whose ciphertext is lanes × channels);
  each is updated, read out and carried. Equivalent reading: 64/N lanes per block at the same cost.
- **Mamba-shaped time-mix inside our block**: lowrank Δ with a degree-4 softplus stand-in, a degree-4 exp ladder per state component,
  selective B_t, C_t (2N per-lane inner products per layer: a ct×pt row, the norm's up-tree, the lanes-block mask and down-tree, then a
  ciphertext multiplication each at the update and at the readout), 4-tap convolution (2 more carried ciphertexts per layer). Our FFN stays.
Every new chain has the harness's guard discipline (`refresh(v, need)` before it consumes levels), so no variant can overflow the depth silently.

## 3. Results on the demo's configuration [replay] (`SWEEP.md`, first table; all variants repeat identically every tick)
`--lanes-block --robust-iters 8 --drop-wv 7 --drop-wkr 4 --drop-wout 7 --drop-win 4`, L = 24, d = 1,024, d_ff = 4,096, depth 41, λ = 22

| variant | boots/tick | Δ boots | carried cts | Δ rotations | Δ ct×ct | Δ matvecs | ΔΛ (limb-diagonals) | Δ boots over the six baselines |
|---|---|---|---|---|---|---|---|---|
| fixed coefficients (today) | 480 | — | 48 | — | — | — | — | — |
| diagonal selective decay, degree 2 | 480 | **+0** | 48 | 0 | +48 | 0 | 0 | 0 on all six |
| diagonal selective decay, degree 4 | 480 | +0 | 48 | 0 | +72 | 0 | 0 | −16 … +23 |
| projected selective decay | 504 | **+24** | 48 | +2,256 | +48 | +24 | +245,640 | 0 … +24 |
| low-rank selective decay (r = 64) | 504 | +24 | 48 | +1,440 | +48 | (2 rectangular) | +27,648 | −16 … +24 |
| input gate only (S3.8) | 504 | +24 | 48 | +2,256 | +48 | +24 | +245,640 | −16 … +24 |
| RG-LRU pair (projected decay + input gate) | 528 | **+48** | 48 | +4,512 | +96 | +48 | +491,280 | −16 … +48 |
| state width 2, fixed coefficients | 504 | +24 | 72 | 0 | 0 | 0 | 0 | +24 on all six |
| state width 4 | 552 | +72 | 120 | 0 | 0 | 0 | 0 | +72 on all six |
| state width 16 | 840 | **+360** | 408 | 0 | 0 | 0 | 0 | +360 on all six |
| Mamba-shaped time-mix inside our block | 912 | **+432** | 456 | +16,800 | +1,968 | (2 rectangular) | +27,648 | +419 … +716 |

The negative deltas on the un-dropped stock baseline are the placement chaos, not savings. The state-width rows are exact on every
baseline because they are carry bootstraps: 24 layers × (N − 1) more ciphertexts re-booted at every tick end.

## 4. In seconds [projection]
Constants, all from the 2 × RTX PRO 6000 box of 2026-09-17/18:
- bootstrap **190.8 ms** = `gpuBootstrapMs` 144,461 ÷ `boots` 757, synced timers, `../../dense-demo-s31/pod_2xbw_20260917/pod_pull/s37/V2_timers/summary.json`
  [record ÷ record; the REAL binary on the plain shape — applied here to the copy binary's tick: same box, same FIDESlib build, R6 caveat];
- the demo tick: layer-loop wall **182.2 s**, 480 boots (A11) ⇒ bootstraps ≈ 91.6 s ≈ half of the tick [derived];
- one limb-diagonal of matvec work, all-in: **18.1 µs** = (27.0 s + 24 × 0.1908 s) ÷ 1,746,944, from the X8 contrast L1r → L7 (wall 161.6 → 134.6 s,
  `SESSION_REPORT.md` lever table; replay: Λ 3,835,904 → 2,088,960, boots 380 → 404) [derived]; lower bracket 8.21 µs (`../HYBRID_GATING_COST_20260908.md` §1, another box);
- ct×ct 0.46–0.93 ms (same file, [projection]).

| variant | bootstraps | matvec work | other | total | of the 182.2 s tick |
|---|---|---|---|---|---|
| diagonal selective decay | 0 | 0 | +48 ct×ct: 0.02–0.05 s | **< 0.1 s** | **< 0.1 %** |
| projected selective decay | +24: 4.6 s | +245,640 × 8.21–18.1 µs: 2.0–4.4 s | 0.05 s | **6.6–9.0 s** | **+3.6–4.9 %** |
| RG-LRU pair | +48: 9.2 s | 4.0–8.9 s | 0.1 s | **13–18 s** | **+7–10 %** |
| state width N (fixed coefficients) | +24 (N − 1): 4.6 s each | 0 | elementwise ct×pt, ~1–3 s at N = 16 | N = 16: **≈ 70 s** | **+2.5 % per unit of N; ≈ +38 % at N = 16** |
| Mamba-shaped time-mix inside our block | +432: 82.4 s | 0.2–0.5 s | rotations +16,800 (≈ 8 s at 0.47 ms), ct×ct 1–2 s, elementwise 1–3 s | **≈ 92–96 s** | **≈ +50 %** |

## 5. Mamba proper [estimate — not replayed: a different block]
Mamba-370M-like: 48 blocks, d = 1,024, expansion 2 (d_inner = 2,048 = two ciphertexts per tensor at 1,024 channel slots per lane), N = 16,
conv 4, no FFN. Carried per block: 2 × 16 state + 2 × 3 convolution = 38 ciphertexts ⇒ **1,824 carried ciphertexts** (today: 48) ⇒ 1,824
carry bootstraps per tick. Chain bootstraps: one norm at 8 Newton steps (2–3), in_proj 4 chunks and out_proj 2 chunks with X8-style drops
(one boot per matvec output: 5–6), conv + SiLU polynomial (1–2), Δ / B / C (2–3), update and readout on two ciphertexts (1–2) ≈ 12–15 per
block ⇒ 576–720. Total ≈ **2,400–2,550 boots ≈ 460–490 s**, plus about today's matvec work (48 × 7 = 336 matvec-equivalents, as today) ⇒
≈ 540–580 s per 64-lane tick ≈ **3×** today's. The driver is N × expansion × blocks, i.e. how much state is carried, not the selectivity.

## 6. What this does NOT show
- **Quality.** No selective variant of this architecture has been trained; whether a diagonal or projected selective decay closes any of
  the gap to Mamba is unmeasured (the pre-registered bar of `PROJECT_BACKLOG.md` Vector A is a ppl delta at ~130M). A training run reverses
  the 2026-09-08 "no retraining" decision: the author's call.
- **Numerics of an encrypted decay.** Today `a` is an exact plaintext; a ciphertext `a_t` carries bootstrap / approximation error ε_a, and a
  slow channel integrates it: the state error from the gate alone is of order ε_a |S| / (1 − a). Additive bootstrap noise is amplified the
  same way already today, so this roughly adds a second term of the same kind — it must be measured (plaintext emulation with injected
  noise on a trained gate: no GPU needed), and the gate polynomial must keep a_t inside (0, 1) over the inputs free-running text produces
  (the Newton-window lesson of `../../dense-demo-s31/pod_longrun_prep/freerun_check/`).
- **Packed prefill.** With a ciphertext decay the plaintext block coefficients of Lemma 2 (k-step blocking when T > 1 tokens share a
  ciphertext) no longer exist. The demo does not use them (T = 1 decode ticks, prompt fed token by token), so nothing above changes; a
  packed-prefill design would.
- **Wall clock.** §4 prices counts with constants from one box; the rectangular low-rank matvec and the per-lane inner products are not
  harness primitives. Treat §4 as ±30 %, §5 as a factor.

## 7. Two predictions for the next pod run (made before it; both checkable in its first minutes)
- The long-run configuration (`--newton-robust 0.1 12`, X8 drops) replays at **529 boots per tick** and **2,294 ct×ct per tick**
  (`SWEEP.md`, second table, first row). `POD_RUN_PLAN.md` had estimated ≈ 547 (+67) by linear scaling; the replay says +49 ⇒ ≈ +9.3 s per
  tick instead of +16 s. The store build's boots per pass will say which.
- If the leak fix L1 did NOT take, pool-used would grow by 144 B × (48 × 529 + 2,294) = **3,986,784 B per request per card**; with it, 0 B.

## 8. Files
`selective_decay_trace.py` (the instrument; `--selfcheck`), `run_sweep.py` → `SWEEP.md`, `SWEEP.json` (6 baselines × 12 variants, 5 ticks each).
