# A11 ticks 42-44 undecryptable: what was checked, what the library's constants predict (2026-09-18)

**Observation (records: `../pod_pull/s37/serve_A11/fidelity.json`, `../a11_final/A11_REPORT.md`).** A11 (stateful, 64 lanes,
teacher-forced, frozen flags + `--newton-robust 0.3 8`, binary `a6778d2b...`, pod test keys): ticks 0-41 decrypt, top-1
2675/2688, per-tick median relErrRms 0.0010-0.0031; ticks 42, 43, 44: every lane's reply fails OpenFHE's decode check
("approximation error is too high"); the server's own timing/boot fields on those ticks are normal (137-139 s, 480 boots).
Abrupt (tick 41 is clean: median 0.0031, worst 0.017), total (all 64 lanes = the whole block ciphertext), persistent (the
carried state stays corrupt).

**Excluded by plaintext emulation of the same 64 prompts over 50 ticks** (`plain_amplitudes.txt`, `plain_amplitudes2.txt`,
and `spec_decode/plain_newton_probe.py --robust-frac 0.3 --robust-iters 8`):
- the Newton rsqrt seed (tonight's fixed defect): no divergent lane-site at any tick 0-49, worst relErr 9.3e-6 (tm/cm sites;
  the final `ln_out` norm is NOT hooked by the probe);
- magnitudes at tick 42 are ordinary: max|S| 2.3 (peak 2.6 at tick 36), |h| 0.4, |z| 1.0, |k| 3.7, |act| 2.0, |gate| 1.0,
  ms/hi 1.09 (tick 1 had 1.82 and decrypted). Nothing in the plaintext model singles out tick 42.

**A candidate that needs NO data dependence: the bootstrap's own failure probability** (`boot_overflow_estimate.py`, `.txt`;
arithmetic on library constants, not a pod measurement):
- The harness sets `UNIFORM_TERNARY` (`gpu_real_model_x.cu:1411`, the stock secure default). OpenFHE v1.5.1 then uses
  `K_UNIFORM = 512`, `R_UNIFORM = 6` and the degree-88 table `g_coefficientsUniform` (`ckksrns-fhe.h`); FIDESlib's GPU
  bootstrap imports exactly these (`FIDESlib/src/CKKS/openfhe-interface/RawCiphertext.cu:578-579`: `coefficientsCheby =
  g_coefficientsUniform; bootK = K_UNIFORM`). The library's own comment: "For larger composite degrees, larger K used to
  achieve a reasonable probability of failure" (it switches to K = 768 only for composite scaling at N >= 2^17).
- After ModRaise every coefficient is m + q0*I_j with I_j ~ a sum of h uniforms on (-1/2, 1/2): at N = 2^17, h ~ 87,381,
  sigma_I = 85.33, so **K = 512 is a 6.00-sigma bound on each of 131,072 coefficients, per bootstrap**.
- The library's approximation evaluated OUTSIDE its fitted range (the script reproduces it to 1.7e-12 inside): harmless up
  to I ~ 523 (error 1e-6 of the sine output, the message being <= ~1e-3 there), 5.5e-4 at I = 532 (message-sized: garbage),
  2.8e-2 at I = 538, 1e150 at I = 563. One bad coefficient becomes an error of the same size in EVERY slot after
  SlotToCoeff/decoding: the whole block dies, and a corrupted carry never recovers — the observed signature.
- P(some |I_j| > 532) = 5.7e-5 per bootstrap; at 480 bootstraps per tick = 2.7 % per tick: **mean 37 ticks to the first
  failure** (58 ticks if the threshold is I > 538). At N = 2^16 the same K is 8.5 sigma: never (which is why shorter-ring
  work never saw it).
- Tonight's tally on the fixed configuration: take 9 18 ticks clean (P(survive 18) ~ 0.6-0.7 under this rate), A11 42 clean
  then failure, X2 4 clean. Take 6's unexplained tick-1 refusal (pre-fix, 404 boots/tick) is the one other candidate event.

**Status: NOT proven.** The estimate matches the observation; a match is not a demonstration. Deciding tests:
1. repeat A11 on the same prompts: the failure tick must MOVE (or vanish) if it is this; it must stay at 42 if it is data;
2. a bootstrap soak at N = 2^17 / uniform ternary with a decrypt check: expect ~1 failure per 17,000-27,000 bootstraps;
3. offline, on take 9's saved ciphertexts + the Mac secret: measure I = round((c0 + c1*s)/q0) directly, confirm sigma_I.
**If it is this, the fixes stay inside the library's own security envelope:** the library's K = 768 table (9.0 sigma:
3e-14 per bootstrap; costs approximation depth), sparse-secret encapsulation (shipped in OpenFHE 1.5.1 as
`SPARSE_ENCAPSULATED`; FIDESlib support unknown), or fewer bootstraps per tick (exposure is linear in the boot count).

**Exposure tally (added 12:31Z; sum of `reqBoots` over every served request in the mirrored ring-2^17 server logs of the
night, `../pod_pull/**/server.jsonl` + `../pod_pull/server.log*`): 63,769 bootstraps** (store builds, selftests and the
fidelity gates add a few thousand more). At 3.65e-5 to 5.7e-5 per bootstrap the estimate expects 2.3 to 3.6 whole-block
failures in that exposure. Whole-block decode failures of the night NOT explained by the Newton-seed emulation: A11 tick 42
(this one), take 6 tick 1 (pre-fix configuration, the emulation did not reproduce it), and possibly take 5b tick 2
(attributed to `--carry-prescale 3` without a reproduction). The tally does not contradict the estimate; it cannot prove it.

**Offline additions, 2026-09-18 afternoon (after the pod was released).**
- `newton_probe_lnout_robust.txt`: the Newton emulation now hooks the final norm too (`ln_out`): with the robust seed 0.3/8 there is no
  divergent lane-site at any of the 49 sites x 64 lanes x ticks 0-49 (worst 9.3e-6). (`newton_probe_lnout_stockseed.txt`: the STOCK seed would have
  diverged on these prompts at tick 1, L3.tm lane 26, ms/hi 1.84 — the defect fixed during the night.) The rsqrt is excluded at every site.
- `overflow_tail_montecarlo.py/.txt`: 3,000 synthetic (uniform c0, c1; uniform-ternary s) ciphertexts at N = 2^17: sigma_I measured 85.336 vs
  predicted 85.337; P(max|I| > B) observed vs the formula: 380 0.662/0.661, 400 0.311/0.297, 420 0.110/0.103, 440 0.034/0.032, 460 0.0087/0.0089,
  480 0.0040/0.0024, 500 0.0007/0.0006; ONE of the 3,000 had max|I| = 517 > K = 512 (expected 0.75). The tail formula holds where it can be counted.
- `boot_overflow_ext_table.py/.txt`: the library's own alternative for a dense secret, `K_UNIFORMEXT = 768` + the degree-118 table: same depth
  bucket (8) and the same R = 6 as the degree-88 table, accurate to 8.6e-12 inside |I| <= 768, breaks at ~784-790; at N = 2^17 it is a 9.00-sigma
  bound: 2.8e-14 per bootstrap. `SPARSE_ENCAPSULATED` in OpenFHE 1.5.1 makes the MAIN secret sparse (h = 192): not a drop-in under the security rule.
- Delegated (results land in `overflow_probe/`): a CPU probe of the mechanism in the real library (wide hand-built secret at a small ring so that
  K/sigma_I is 3-6; failures vs the exact max|I| of each bootstrapped ciphertext) and the I distribution on take 9's saved ciphertexts.
- The experiments that need a GPU are specified in `../NEXT_SESSION_BRIEF.md` (harvest-and-bootstrap; the K = 768 A/B; A11 to >= 64 ticks).

**CPU probes delivered, 2026-09-18 evening (`overflow_probe/`; ring <= 2^13 for the bootstraps, load-only at 2^17; memguard; nothing on a GPU).**
- **The mechanism is CONFIRMED in the library itself (OpenFHE 1.5.1 CPU `EvalBootstrap`, 2,065 bootstraps, `PROBE_B_REPORT.md`).** For every
  trial the exact overflow vector of the ciphertext that enters ModRaise was computed (the library's own `Clone -> ModReduceInternalInPlace ->
  AdjustCiphertext` on a clone, verified bit for bit against a public-API replica on all trials; negacyclic product in `__int128`). Outcome is
  decided by max|I|: **all 803 trials with max|I| <= 512 were normal, all 505 with max|I| >= 523 were not**; first non-normal at 519, first
  decode exception at 532, nothing above 534 decoded. A planted-overflow sweep (one coefficient driven to a chosen I* in steps of 1, 635
  bootstraps) gives the thresholds directly: relErrRms > 2x baseline from 518-519, >= 1e-3 from 522-526, >= 0.1 from 528-532, first decode
  exception ("approximation error is too high") at 534-539, the whole ciphertext destroyed from 553; the error at the overflowing
  coefficient equals this directory's float64 re-evaluation of the library's polynomial x 2^F (F = the bootstrap correction factor;
  median measured/model 0.995-1.116). The wide-secret dose-response (K/sigma_I from 6.2 down to 3.2) matches the Gaussian-tail prediction
  row by row (observed vs predicted P(not normal): 0 / 2.5e-6, 0 / 7.6e-4, 0.043 / 0.030, 0.300 / 0.3025, 0.903 / 0.880, 1.000 / 0.9999).
- **The model's input holds on the session's REAL ciphertexts (`PROBE_C_REPORT.md`, 36 ciphertexts of take 9's kit, 4.72 M coefficients, Mac
  secret; its coefficients never entered the repo and the scratch copies were deleted):** Hamming weight 87,520, predicted sigma_I 85.40,
  pooled 85.44, excess kurtosis -0.0009, tails at expectation to 4 sigma; K/sigma = 5.995.
- **With the measured thresholds the arithmetic at the demo's parameters (`overflow_probe/demo_rate_from_thresholds.md`; an extrapolation, the
  probe could not reach K/sigma = 6 rates directly) is 3.8e-5 (decode exception, I >= 539) to 6.4e-5 (relErr >= 0.1, I >= 532) per bootstrap
  = 1.8-3.0 % per 480-boot tick = a mean of 33-56 ticks to the first event.** A11's first failure was tick 42.
- **Unresolved, and it cuts toward a HIGHER rate:** evaluated ciphertexts (the GPU replies) run high in the far tail — one reply has
  |I| = 503 (5.89 sigma; chance 0.019 under the model), 36 vs 24.7 expected beyond 4.5 sigma over 56 ciphertexts, per-ciphertext sigma
  over-dispersed for replies (p = 0.011) and not for fresh requests; a CPU check at ring 2^12 found one 7.12-sigma value in evaluated
  ciphertexts in one of two runs. The Gaussian model holds to 5.5 sigma; whether a rare extreme component exists beyond is open.
- **Still NOT shown:** that this event is what hit A11 at tick 42 (the failing ciphertext was not kept), and FIDESlib's GPU curve (it imports
  the same table and K; not run). The flight recorder of `../../pod_longrun_prep/` measures exactly this on the GPU, per bootstrap.


**K = 768 confirmed on CPU, with a precision cost the arithmetic had NOT predicted, 2026-09-18 night (`overflow_probe/k768/K768_CPU_REPORT.md`;
a patched COPY of the vendored OpenFHE 1.5.1 with a run-time switch, rings 2^12 / 2^13, 1,735 trial records, memguard, nothing on a GPU; the
the tables were recomputed: the baseline medians, the edge thresholds, the old-zone counts and both slopes from the JSONL records and the constant
rounding in exact rationals: identical).**
- **The failure side behaves as `boot_overflow_ext_table.txt` predicted.** With the switch off the patched build reproduces the stock thresholds
  (519 / 522 / 529 / 534 / 553). With `g_coefficientsUniformExt` / K = 768: 275 planted bootstraps with 500 <= |I*| <= 766, 0 decode exceptions,
  relErrRms 6.97e-05..8.44e-05 against that arm's un-planted median 7.07e-05 (the control in the same zone: 79 decode exceptions in 186
  bootstraps); new edge at N = 2^13 / F = 10: first degraded 775, relErrRms >= 1e-3 from 777, >= 0.1 from 782, first decode exception 788,
  ciphertext destroyed from 798 (predicted 778 / 784 / 790 for an injected error of 1e-6 / 1e-4 / 1e-2); measured / float model at the
  overflowing coefficient 1.22 (69 bootstraps). Same bootstrap depth (20), towers (24), output level (19). At F = 7 the edge is 1-4 units later.
- **The precision side does NOT behave as predicted ("at most 1.5x").** Un-planted median relErrRms, 40 trials per arm, same binary: 2.914e-05
  (stock table) -> 7.066e-05 (Ext): **2.43x = 1.28 bits** at N = 2^13 / F = 10; 2.45x at F = 7; 3.35x = 1.75 bits at N = 2^12.
  - Part of it has a found and measured cause: the library divides the raised ciphertext by K with ONE scalar, `EvalMult(ct, pre/(K N))`, which
    is encoded as the integer round(c x SF). For K = 512 that constant is a power of two; for 768 = 3 x 2^8 it is not, and the rounding is a
    relative scale error eta of the sine's input, i.e. an error I x eta x 2^F on every coefficient, proportional to its own overflow. Measured
    eta +7.31e-12 +- 0.04e-12 at N = 2^13 (335 of 335 planted trials with the predicted sign; arithmetic +7.2475e-12) and -3.75e-12 at N = 2^12
    (42 of 42; arithmetic -3.6664e-12). **The same arithmetic at N = 2^17: +9.55e-11 for 1/(768 N) against -2.09e-11 for the stock 1/(512 N)**
    (FirstPrime(59, 2^18) = 2^59 + 12,058,625, so the stock constant is not exact there either). FIDESlib builds the same rounded constant
    (`Bootstrap.cu` `constantEvalMult`, `Context.cu` `ElemForEvalMult`): source reading, not run.
  - The fix that was run on CPU (variant B): fold the 3 into the CoeffsToSlots precomputation (`scaleEnc = pre / 3`, where the library folds
    the whole K of its sparse cases) and divide by the power of two 256 N. Slope -0.6e-13 +- 0.9e-13 (gone), relErrRms 5.179e-05 = **1.78x =
    0.83 bits**, same depth and the same edge.
  - What remains is random-sign noise on the coefficients whose own overflow is 0, +-1, +-2 (7.4x the control's rms at I = 0, 1.3x from
    |I| >= 9). Its 0 : 1 : 2 profile matches the amplification of the double-angle iterations; why the degree-118 evaluation injects more of it
    than the degree-88 one was NOT established. Coefficients with I = 0 are 1.9 % of all at sigma_I = 21 and about 0.5 % at the demo's 85.
- **Consequence for the next pod run** (`../../pod_longrun_prep/reports/K768_SPLIT_VARIANT_NOTE.md`): the GPU switch now has two forms,
  `FIDESLIB_BOOT_UNIFORM_EXT=1` (straight swap) and `=split` (variant B: OpenFHE patch O1 + `bootK = 256` in FIDESlib), and the run's first
  cell measures the stock table and both forms on the two cards before the long run starts. Neither form has run on a GPU.
- **Still NOT shown:** FIDESlib's GPU edge and GPU precision with the Ext table, anything at ring 2^17, the failure RATE with K = 768 (it is
  the Gaussian tail at 9.0 sigma with the measured edge put in: below 1e-14 per bootstrap, subject to the far-tail question above).
