# Probe B: does OpenFHE's CKKS bootstrap fail when the ModRaise overflow max|I| leaves K = 512, and at what overshoot? (2026-09-18)

**Scope.** CPU, vendored OpenFHE v1.5.1 (`vendor/install`, single-threaded), rings 2^13 and 2^12 (`HEStd_NotSet`), `UNIFORM_TERNARY`
as the library's key-distribution parameter, `FLEXIBLEAUTO`, scaling 59 bits, first modulus 60 bits, level budget {3,3}, full packing
(slots = N/2), depth 23 (3 levels after a bootstrap depth of 20). This is the library whose constants FIDESlib imports; it is NOT the
GPU bootstrap of the demo, and nothing here was run at ring 2^17. Every number below is transcribed from a file in this directory;
the tables between the `T` markers are pasted by `assemble_reports.py` from `probe_b_tables.md`, which `analyze_probe_b.py` generates from
the JSONL records.

## 1. What was built

- `harness/boot_overflow_probe.cpp` (sha256 `da7c0fd6127e…`), binary `harness/build/boot_overflow_probe` (sha256 `bcaaea2a47d6…`), built by
  `PATH=$PWD/.venv/bin:$PATH tools/memguard.sh 6 ninja -C harness/build boot_overflow_probe` (target added to the `foreach` list of
  `harness/CMakeLists.txt`); log: `build_boot_overflow_probe.log` (three builds; the third is the binary of record, both hashes are printed at
  the top of `planted_loadstate.txt` and `dose_loadstate.txt`). Every record line carries the run stamp (`harnessSha256 da7c0fd6127e0ba7`, commit
  `426a7ca5e03a`, dirty tree, exact argv). `dev_smoke/` holds the development runs of the two earlier builds; no table uses them.
- For EVERY trial the probe computes the **exact** overflow vector of the ciphertext that enters ModRaise,
  `I_j = round(([c0]_q0 + [c1]_q0 * s)_j / q0)`, by a negacyclic convolution over the integers in `__int128` (no floating point), with the
  library's own centering rule (`v > q0>>1` is negative). **Route and why it is exact:** `FLEXIBLEAUTO`. `EvalBootstrap` goes from its argument to
  ModRaise through `Clone -> ModReduceInternalInPlace(noiseScaleDeg-1) -> AdjustCiphertext(2^-corr, 0)` (`ckksrns-fhe.cpp:558-562`), all
  deterministic. The probe calls those SAME library functions on a clone of the SAME ciphertext (the private `AdjustCiphertext` is reached through
  an explicit template instantiation, which the language exempts from access checking; no header is touched), and also re-derives the step from
  the public API line for line. The two agree bit for bit on every trial of every run (`adjustMatchesLibrary`, counts in the tables).
  `FIXEDMANUAL` would not have been simpler: in a 64-bit build that branch also multiplies by the correction and mod-reduces before ModRaise
  (`ckksrns-fhe.cpp:2264-2273`).
- Two readouts per trial: the client's view (`cc->Decrypt`; its decode throws "approximation error is too high", and it ADDS Gaussian noise of
  the estimated size to every slot) and a decode-free view (b + a*s over the output towers, CRT-interpolated, compared coefficient by
  coefficient with the encoded reference), which localises the damage.
- `bootRestored` (output towers 5 > input towers 2) is true on every trial, so no row measures the `EvalBootstrap` no-op.
- Library facts with file:line: `LIBRARY_FACTS.md`.

Three experiments:

1. **Soundness of the hand-built key set** (`sound_*.jsonl`): library `KeyGen` vs a secret built by hand with coefficients uniform in {-1,0,1},
   public key built by hand as `PKEBase::KeyGenInternal` does, every evaluation key from the library's own `EvalMultKeyGen` /
   `EvalBootstrapKeyGen` on that secret.
2. **Planted overflow** (`planted_*.jsonl`, an instrument added to the brief): stock library keys, uniform ternary secret. The ModRaise input is
   rebuilt so that ONE coefficient j* carries a chosen overflow I* while the ciphertext stays a valid encryption of the same message with the
   same noise under the same key (`c1'' = U + A*sgn-pattern of s`, `c0'' = c0' + (c1' - c1'')*s mod q0`), then pulled back through the library's
   adjustment step (linear on the q0 tower when the dropped tower is zero; its multiplier is measured from the library, `adjustMultiplier`
   records). Before bootstrapping, the probe runs the library's adjustment on the crafted ciphertext and requires bit-equality with the intended
   `(c0'', c1'')` (`plantVerified`), and the recorded overflow is the exact recomputation, not the target (`plantAchieved`). This gives the error
   as a function of I in steps of ONE, with a ternary secret, without waiting for a 6-sigma event.
3. **Dose-response** (`dose_r13_F10.jsonl`, the brief's design): hand-built secrets with coefficients uniform in {-a..a}, a = 5..10, fresh
   random encryptions, exact max|I| per trial.

## 2. Results

### 2.1 The hand-built key set is sound

| file | mode | N | F | sum s_i^2 | sigma_I pred | sigma_I measured (mean over trials) | trials | decode fails | adjust==library | relErrRms median | relErrRms max | coefErrRms median | bootRestored |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sound_r13_hand_a1.jsonl | hand | 8192 | 10 | 5555 | 21.517 | 21.472 | 40 | 0 | 40/40 | 2.828e-05 | 2.882e-05 | 1.789e-07 | 40/40 |
| sound_r13_stock.jsonl | stock | 8192 | 10 | 5435 | 21.284 | 21.289 | 40 | 0 | 40/40 | 2.854e-05 | 2.930e-05 | 1.814e-07 | 40/40 |

Same precision (median relErrRms 2.83e-05 hand-built vs 2.85e-05 library keys), no decode failure, and the measured sigma_I equals
sqrt((sum s_i^2 + 1)/12) for both.

### 2.2 Planted overflow: the error is a function of the overflow, it starts just above K, and it is the library's own polynomial

| sweep | N | F | I* grid | baseline relErrRms (median over |I*| <= 510) | median relErrRms > 2x baseline | median >= 1e-3 | median >= 1e-2 | median >= 1e-1 | first decode exception | all trials throw from | whole ciphertext destroyed (median coefficient error > 1) | measured / model, median over 518 <= |I*| <= 540 (min..max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| planted_r12_F11_pos.jsonl | 4096 | 11 | 500..560 step 1 | 1.403e-05 | 518 | 522 | 525 | 528 | 534 | 534 | 553 | 1.001 (0.758..1.160), n = 69 |
| planted_r13_F07_pos.jsonl | 8192 | 7 | 500..560 step 1 | 3.540e-06 | 519 | 526 | 529 | 532 | 539 | 546 | 553 | 0.995 (0.303..1.666), n = 69 |
| planted_r13_F10_imag.jsonl | 8192 | 10 | 510..545 step 5 | 2.876e-05 | 520 | 525 | 525 | 530 | 535 | 540 | None | 1.349 (0.521..2.139), n = 15 |
| planted_r13_F10_neg.jsonl | 8192 | 10 | 500..560 step 2 | 2.835e-05 | 520 | 522 | 526 | 528 | 534 | 536 | 552 | 1.523 (0.003..3.926), n = 24 |
| planted_r13_F10_pos.jsonl | 8192 | 10 | 500..560 step 1 | 2.914e-05 | 519 | 522 | 526 | 529 | 535 | 548 | 553 | 1.116 (0.218..2.674), n = 69 |

Reading of the per-I tables (`probe_b_tables.md`, section T2; 635 bootstraps in all, every one `plantVerified`, `achieved == target`,
`adjust==library`):

- **|I*| <= 512: nothing; 513-515: nothing a client could see.** The decoded precision is the baseline; the error AT the planted coefficient
  starts to lift off the floor (F = 10 sweep, median: 2.7e-07 at 513, 4.6e-07 at 514, 9.3e-07 at 515, against a floor of 1.2e-07), as the model
  column predicts, but it is one coefficient in 8,192 at the noise level.
- **From |I*| = 516-519 the damage appears and then roughly doubles per unit of I** (N = 2^12 sweep, median |coef err at j*|: 1.676e-04 at 520,
  3.716e-04 at 521, 9.857e-04 at 522, 2.325e-03 at 523, ... 2.840e-01 at 530, 5.901e-01 at 531). It sits on exactly TWO plaintext coefficients,
  the overflowing one j* and the other half of its slot, j* + N/2: their share of the squared coefficient error goes 0.0014 (I* = 515) -> 0.1364 (517)
  -> 0.9680 (520) -> 1.0000 (524) in the F = 7 sweep while the median error of the remaining coefficients (`floor`) does not move. The partner's part
  varies from trial to trial (the error in that slot is complex-valued; why was not investigated). A single-coefficient error of size d is an
  error of size about d in EVERY slot after decoding (|zeta^j| = 1), which is what the slot-domain relErrRms shows.
- **The size of the error is the library's Chebyshev + double-angle polynomial evaluated outside its fitted range, times 2^F**
  (F = the correction factor): measured |coef err at j*| / (|g(I/K)| * 2^F) has median 1.00 (N = 2^12, F = 11), 0.995 (N = 2^13, F = 7),
  1.12 (N = 2^13, F = 10) over 518 <= I* <= 540, 69 bootstraps each. The float64 re-evaluation in `../boot_overflow_estimate.py` is therefore
  a faithful model of what the real bootstrap does to an overflowing coefficient.
- Thresholds at N = 2^13: relErrRms >= 1e-3 from I = 522 (F = 10) / 526 (F = 7); >= 1e-2 from 526 / 529; >= 0.1 from 529 / 532; the library's
  decode check first throws at 535 / 539; from **I = +553 the whole ciphertext is destroyed** in all three positive sweeps (F = 7, 10, 11, both
  rings): the median error over ALL coefficients jumps from 1e-7 at 552 to 3e8-2e14 at 553 and 1e27 at 554, i.e. the damage is no longer
  confined to one slot. On the negative side (step 2) that floor already rises at -548 (2.2e-02) and -550 (1.5e-01) and is 2.4e16 at -552.
- Sign and position do not matter: -I* behaves as +I* (`planted_r13_F10_neg`), a coefficient of the second half (j* = 6000) as one of the
  first (`planted_r13_F10_imag`).
- F moves the thresholds by about one unit of I per factor 2.4 in 2^F, because the same sine-output error is multiplied back by 2^F. The demo's F on
  the GPU is NOT known from this machine (the serialized client context carries `corFactor` 0, see probe C); the library's default formula
  gives 7 at N = 2^17 / 2^16 slots (`LIBRARY_FACTS.md`).

### 2.3 Dose-response with wide secrets: failures happen when, and only when, max|I| is above the planted thresholds, at the Gaussian rate

### dose_r13_F10.jsonl -- N = 8192, F = 10, levelBudget [3, 3], depth 23

Outcome classes per trial (absolute thresholds, the same for every width): `exception` = the library's decode threw; `garbage` = relErrRms >= 0.1; `degraded` = 1e-3 <= relErrRms < 0.1; `normal` = relErrRms < 1e-3. `baseline` = median relErrRms of that width's trials with max|I| <= 500 (`-` when there is no such trial).

The three `predicted` columns use the thresholds the PLANTED sweep of the same ring and F measured (T2s, planted_r13_F10_pos: median relErrRms >= 1e-3 from |I*| = 522, >= 0.1 from 529, first decode exception at 535), i.e. P(max|I| > 521), P(max|I| > 528), P(max|I| > 534) under the Gaussian model.

| a | sum s_i^2 | sigma_I pred | sigma_I measured | K/sigma | trials | adjust==library | baseline relErrRms | normal | degraded | garbage | exception | observed P(not normal) | predicted P(max|I|>521) | observed P(garbage or exception) | predicted P(max|I|>528) | observed P(exception) | predicted P(max|I|>534) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 82276 | 82.80 | 82.94 | 6.183 | 100 | 100/100 | 1.014e-04 | 100 | 0 | 0 | 0 | 0.0000 | 2.469e-06 | 0.0000 | 1.426e-06 | 0.0000 | 8.863e-07 |
| 6 | 114387 | 97.63 | 97.59 | 5.244 | 200 | 200/200 | 1.195e-04 | 200 | 0 | 0 | 0 | 0.0000 | 0.0007553 | 0.0000 | 0.0005073 | 0.0000 | 0.0003593 |
| 7 | 152564 | 112.76 | 112.75 | 4.541 | 300 | 300/300 | 1.380e-04 | 287 | 2 | 2 | 9 | 0.0433 | 0.03021 | 0.0367 | 0.02244 | 0.0300 | 0.01733 |
| 8 | 195520 | 127.65 | 127.63 | 4.011 | 300 | 300/300 | 1.567e-04 | 210 | 19 | 12 | 59 | 0.3000 | 0.3025 | 0.2367 | 0.2473 | 0.1967 | 0.2064 |
| 9 | 244435 | 142.72 | 142.68 | 3.587 | 300 | 300/300 | 1.754e-04 | 29 | 18 | 18 | 235 | 0.9033 | 0.8795 | 0.8433 | 0.8255 | 0.7833 | 0.7718 |
| 10 | 306507 | 159.82 | 159.62 | 3.204 | 150 | 150/150 | - | 0 | 1 | 0 | 149 | 1.0000 | 0.9999 | 0.9933 | 0.9996 | 0.9933 | 0.9988 |

Predicted P(max_j |I_j| > B) = 1 - (1 - 2Q((B + 0.5)/sigma_I))^N with sigma_I = sqrt((sum s_i^2 + 1)/12), next to the OBSERVED fraction of trials whose exact max|I| exceeded B (this checks the Gaussian model of I itself, independently of any error threshold):

| a | trials | B=512: predicted P | expected count | observed count | B=521: predicted P | expected count | observed count | B=523: predicted P | expected count | observed count | B=528: predicted P | expected count | observed count | B=532: predicted P | expected count | observed count | B=534: predicted P | expected count | observed count | B=538: predicted P | expected count | observed count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 100 | 4.949e-06 | 0.00 | 0 | 2.469e-06 | 0.00 | 0 | 2.112e-06 | 0.00 | 0 | 1.426e-06 | 0.00 | 0 | 1.039e-06 | 0.00 | 0 | 8.863e-07 | 0.00 | 0 | 6.435e-07 | 0.00 | 0 |
| 6 | 200 | 0.001251 | 0.25 | 0 | 0.0007553 | 0.15 | 0 | 0.0006745 | 0.13 | 0 | 0.0005073 | 0.10 | 0 | 0.0004032 | 0.08 | 0 | 0.0003593 | 0.07 | 0 | 0.0002848 | 0.06 | 0 |
| 7 | 300 | 0.04396 | 13.19 | 15 | 0.03021 | 9.06 | 12 | 0.02776 | 8.33 | 11 | 0.02244 | 6.73 | 10 | 0.01889 | 5.67 | 10 | 0.01733 | 5.20 | 9 | 0.01455 | 4.37 | 9 |
| 8 | 300 | 0.3855 | 115.65 | 101 | 0.3025 | 90.75 | 85 | 0.2859 | 85.77 | 77 | 0.2473 | 74.18 | 69 | 0.2194 | 65.82 | 59 | 0.2064 | 61.92 | 58 | 0.1823 | 54.70 | 50 |
| 9 | 300 | 0.9328 | 279.84 | 281 | 0.8795 | 263.84 | 266 | 0.8651 | 259.53 | 260 | 0.8255 | 247.64 | 246 | 0.7904 | 237.11 | 233 | 0.7718 | 231.55 | 230 | 0.7332 | 219.95 | 222 |
| 10 | 150 | 1 | 150.00 | 150 | 0.9999 | 149.98 | 150 | 0.9998 | 149.97 | 149 | 0.9996 | 149.93 | 149 | 0.9991 | 149.87 | 149 | 0.9988 | 149.83 | 149 | 0.9979 | 149.69 | 149 |
| all | 1350 | - | 558.93 (binomial sd 10.14) | 547 | - | 513.79 (binomial sd 10.20) | 513 | - | 503.73 (binomial sd 10.22) | 497 | - | 478.59 (binomial sd 10.29) | 474 | - | 458.55 (binomial sd 10.34) | 451 | - | 448.57 (binomial sd 10.36) | 446 | - | 428.76 (binomial sd 10.40) | 430 |

Outcome vs the exact max|I| of the trial, pooled over all widths in this file (each row is one value range of max|I|):

| max|I| | trials | normal | degraded | garbage | exception | relErrRms: median (min..max) over non-throwing | worst coefficient is argmax|I| or its slot partner |
|---|---|---|---|---|---|---|---|
| 1..480 | 627 | 627 | 0 | 0 | 0 | 0.000135 (9.76e-05..0.000164) | 0/627 |
| 481..500 | 113 | 113 | 0 | 0 | 0 | 0.000156 (0.000119..0.000179) | 0/113 |
| 501..505 | 30 | 30 | 0 | 0 | 0 | 0.000156 (0.000139..0.000178) | 0/30 |
| 506..510 | 23 | 23 | 0 | 0 | 0 | 0.000157 (0.000137..0.000177) | 0/23 |
| 511..512 | 10 | 10 | 0 | 0 | 0 | 0.000157 (0.000138..0.000175) | 0/10 |
| 513..514 | 6 | 6 | 0 | 0 | 0 | 0.000156 (0.000138..0.000173) | 0/6 |
| 515..516 | 6 | 6 | 0 | 0 | 0 | 0.000168 (0.00016..0.000181) | 5/6 |
| 517..518 | 7 | 7 | 0 | 0 | 0 | 0.00025 (0.00015..0.000625) | 7/7 |
| 519..520 | 8 | 2 | 6 | 0 | 0 | 0.00188 (0.000463..0.00249) | 8/8 |
| 521..522 | 15 | 2 | 13 | 0 | 0 | 0.00236 (0.000798..0.0082) | 15/15 |
| 523..524 | 12 | 0 | 12 | 0 | 0 | 0.0176 (0.0032..0.087) | 11/12 |
| 525..526 | 12 | 0 | 8 | 4 | 0 | 0.0615 (0.00371..0.398) | 12/12 |
| 527..528 | 7 | 0 | 0 | 7 | 0 | 0.323 (0.138..1.59) | 7/7 |
| 529..530 | 14 | 0 | 1 | 13 | 0 | 1.31 (0.0332..2.33) | 13/14 |
| 531..532 | 9 | 0 | 0 | 7 | 2 | 3.19 (0.709..6.07) | 9/9 |
| 533..534 | 5 | 0 | 0 | 1 | 4 | 6.41 (6.41..6.41) | 5/5 |
| 535..536 | 7 | 0 | 0 | 0 | 7 | - | 7/7 |
| 537..538 | 9 | 0 | 0 | 0 | 9 | - | 9/9 |
| 539..540 | 5 | 0 | 0 | 0 | 5 | - | 5/5 |
| 541..545 | 15 | 0 | 0 | 0 | 15 | - | 15/15 |
| 546..550 | 26 | 0 | 0 | 0 | 26 | - | 25/26 |
| 551..560 | 41 | 0 | 0 | 0 | 41 | - | 12/41 |
| 561..580 | 86 | 0 | 0 | 0 | 86 | - | 0/86 |
| 581..inf | 257 | 0 | 0 | 0 | 257 | - | 0/257 |

- trials: 1350; largest max|I| with a NORMAL outcome: 522; smallest max|I| with a non-normal outcome: 519; smallest max|I| with garbage-or-exception: 525; smallest max|I| with an exception: 532; largest max|I| WITHOUT an exception: 534
- non-normal trials with max|I| <= 512: 0; normal trials with max|I| > 512: 23

Reading:

- 1,350 bootstraps, every one with `adjust==library` and `bootRestored`. **All 803 trials with max|I| <= 512 are normal; all 505 trials with
  max|I| >= 523 are not** (the rows from 523 down in the last table); the 42 trials in 513..522 are the transition (23 normal, 19 degraded; first
  non-normal at 519, last normal at 522). First garbage at 525, first decode exception at 532, no trial above 534 decodes. These are the planted
  thresholds of the same ring and F (522 / 529 / 535 for the medians) reproduced by RANDOM ciphertexts.
- In the damaged trials the worst plaintext coefficient is the argmax of |I| or its slot partner (e.g. 15/15 at 521..522, 12/12 at 525..526,
  15/15 at 541..545); once max|I| is in the 550s and beyond the whole ciphertext is destroyed and the worst coefficient is anywhere (12/41 at
  551..560, 0/86, 0/257).
- **Rates.** The observed fraction of abnormal bootstraps follows the Gaussian tail with sigma_I = sqrt((sum s_i^2 + 1)/12): a = 8 (K/sigma 4.01):
  0.3000 observed vs 0.3025 predicted; a = 9 (3.59): 0.9033 vs 0.8795; a = 7 (4.54): 13/300 = 0.0433 vs 0.0302 (9.06 expected, binomial sd 3.0);
  a = 6 (5.24): 0/200 vs 7.6e-04; a = 5 (K/sigma 6.18, the demo-like margin): 0/100 vs 2.5e-06. The model of I itself, independent of any error
  threshold: over all widths 547 trials had max|I| > 512 against 558.93 +- 10.14 expected, 513 vs 513.79 for B = 521, 474 vs 478.59 for B = 528,
  430 vs 428.76 for B = 538.

## 3. What the data support, and what they do not

Supported (measured, this library, CPU):

- The mechanism exists as hypothesised: the approximate modular reduction is harmless for |I| <= 515, degrades from 516-519, gives
  message-sized garbage in the high 520s / low 530s, throws the decode check in the mid/high 530s and destroys the whole ciphertext from 553.
  The README's "harmless to ~523, message-sized at ~532, explodes by 563" agrees with the F = 7 column of these measurements (relErrRms 2.4e-04 at 523, 1.2e-01 at 532), except that the
  whole-ciphertext destruction starts at 553, not 563: the estimate script's grid jumps from 552 to 563; evaluated at every integer the same float64
  model also explodes at 553 (`model` column of T2: 1.3e+02 at 552, 4.6e+25 at 553 for F = 10), and so does the library.
- One overflowing coefficient damages one slot pair in the coefficient domain and therefore every decoded slot; nothing else in the ciphertext moves.
- With random ciphertexts the outcome of a bootstrap is decided by max|I| (section 2.3: 0 abnormal trials of 803 at or below 512, 0 normal
  trials of 505 at or above 523), and the frequency of max|I| > B follows 1-(1-2Q((B+0.5)/sigma_I))^N with sigma_I = sqrt((sum s_i^2+1)/12)
  for K/sigma between 3.2 and 6.2.

Not shown:

- That the A11 tick-42 failure WAS such an event: the ciphertext that entered the failing bootstrap was never saved, so its I cannot be computed.
- That FIDESlib's GPU bootstrap has the same error-versus-I curve. It imports the same K, R and table (README), and a polynomial is a polynomial,
  but its evaluation order, its correction factor and its behaviour once intermediate values stop fitting the modulus were not run here.
- The rate at K/sigma = 6.0: the widest margin at which a few hundred trials can fail at all is K/sigma = 4.5 (13 of 300); at the demo-like
  K/sigma = 6.2 the probe saw what the model predicts for 100 trials, zero events, which by itself excludes only rates above a few percent. The
  rate at 2^17 remains an extrapolation of a tail law that this probe confirms at 3.2-4.5 sigma and that probe C checks to about 5 sigma on
  real ciphertexts (with one caveat, see `PROBE_C_REPORT.md`).
- Wide secrets are an instrument, not a recommendation: they raise the noise floor of the bootstrap (baseline relErrRms 1.0e-04 at a = 5 to
  1.8e-04 at a = 9, against 2.8e-05 for ternary keys at the same ring), which is why the outcome classes use absolute thresholds.

## 4. Commands, binary, load state

- Builds: `PATH=$PWD/.venv/bin:$PATH tools/memguard.sh 6 ninja -C harness/build boot_overflow_probe` (log `build_boot_overflow_probe.log`).
- Runs: `run_planted.sh` (5 sweeps, 13:19:53Z-13:33:39Z) and `run_dose.sh` (soundness + dose-response, 13:37:16Z-14:18:38Z); every process is
  `caffeinate -i tools/memguard.sh 6 harness/build/boot_overflow_probe ...`, one at a time; the exact argv of each is in `planted_runlog.txt`,
  `dose_runlog.txt` and in the header record of each JSONL. All exits 0, all stderr files empty, no guard trip.
- Analysis: `.venv/bin/python analyze_probe_b.py` -> `probe_b_tables.md`; `.venv/bin/python demo_rate_from_thresholds.py` ->
  `demo_rate_from_thresholds.md` (ARITHMETIC: Gaussian tail at N = 2^17 for the measured thresholds; not a measurement).
- Load state: `planted_loadstate.txt`, `dose_loadstate.txt` (`tools/mem_probe.sh` before and after: `heavyProcs 0`, i.e. nothing else that
  links or compiles OpenFHE; swap 1,600 -> 1,993 MB over the dose run). During the dose-response run light numpy / JSON analysis scripts also ran
  on the machine, so `bootMs` in the records is NOT a clean timing; no timing is quoted anywhere in this report (R6). Correctness fields do
  not depend on load.
