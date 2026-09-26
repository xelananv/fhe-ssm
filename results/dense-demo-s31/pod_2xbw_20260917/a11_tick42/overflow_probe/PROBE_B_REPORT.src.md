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

<<T1>>

Same precision (median relErrRms 2.83e-05 hand-built vs 2.85e-05 library keys), no decode failure, and the measured sigma_I equals
sqrt((sum s_i^2 + 1)/12) for both.

### 2.2 Planted overflow: the error is a function of the overflow, it starts just above K, and it is the library's own polynomial

<<T2s>>

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

<<T3>>

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
