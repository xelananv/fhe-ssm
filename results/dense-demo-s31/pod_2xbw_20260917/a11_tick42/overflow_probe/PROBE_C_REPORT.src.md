# Probe C: the distribution of the ModRaise overflow I on the REAL ciphertexts of the recorded ring-2^17 session (2026-09-18)

**What was measured.** For each saved ciphertext of the verification kit `~/Documents/fhe-ssm-backup/demo_verification_demo_20260918T074952Z/`
(18 ticks: `req.N.0`, the fresh encryption that left the Mac, and `resp.N.0`, the reply of the GPU server; 36 ciphertexts, ring 2^17), the q0
tower of `c0` and `c1` was read as centered residues in coefficient format and
`x_j = c0_j/q0 + sum_k s-hat(j,k) c1_k/q0`, `I_j = round(x_j)` was computed with the Mac secret key (negacyclic convolution by a twisted FFT in
float64, checked against exact Python-integer arithmetic). This is the quantity ModRaise would produce if that ciphertext's q0 tower were
raised; its distribution is the input of the failure estimate in `../README.md`.

**What was built and run.**

- `harness/overflow_dist_dump.cpp` (sha256 `ded7526b4f2b…`), binary `harness/build/overflow_dist_dump` (sha256 `d5d78ed9667c…`, both printed in
  `probe_c_loadstate.txt`), built by `PATH=$PWD/.venv/bin:$PATH tools/memguard.sh 6 ninja -C harness/build overflow_dist_dump`
  (`build_overflow_dist_dump.log`). LOAD-ONLY: it deserialises the context, the secret key and the ciphertexts; it loads no evaluation key and
  performs no homomorphic operation. It refuses any output directory outside `/tmp/`.
- `run_probe_c.sh dump` ran it ALONE under `tools/memguard.sh 8` (the one ring-2^17 process; `heavyProcs 0` before it, exit 0, a few seconds, no guard trip:
  `probe_c_loadstate.txt`, public metadata of the dump in `probe_c_dump.jsonl`). The kit was opened in place, read-only; nothing in it was
  modified, moved, copied or symlinked.
- `overflow_dist_stats.py` (numpy) computed the statistics and wrote AGGREGATES ONLY into this directory (`probe_c_stats.json`,
  `probe_c_tables.md`): the Hamming weight, one autocorrelation sum of the secret, histograms and order statistics of I pooled over ciphertexts,
  per-ciphertext moments. No coefficient of the secret and no per-coefficient value derived from it was written inside the repository.
- **Extension beyond the brief (same rules):** the two earlier kits `demo_verification_demo_20260904T183832Z` (2 ticks) and
  `demo_verification_demo_20260918T025113Z` (8 ticks) carry the SAME `secret.key`, `public.key` and `cryptocontext.bin` sha256 in their
  `MANIFEST.sha256`, so their 20 ciphertexts were dumped the same way (`run_probe_c.sh dump-ext`, `probe_c_dump_ext.jsonl`) and pooled
  (`probe_c_stats_ext.json`, `probe_c_tables_ext.md`): 56 ciphertexts, 7.34 M samples.
- **The scratch directories** `$SCRATCH/overflow_dist` and `$SCRATCH/overflow_dist_ext` (mode 700, outside the repo; the secret's
  coefficient vector and the tower-0 dumps) **were deleted** by `run_probe_c.sh clean`; `probe_c_loadstate.txt` records the deletion and the
  failing `ls` after it. They were created twice (the statistics script was corrected once, see "dispersion" below) and deleted twice.

## 1. The brief's kit (36 ciphertexts)

<<C_PRIMARY>>

## 2. All three kits under the same key (56 ciphertexts)

<<C_EXT>>

## 3. Reading

- **The secret** is ternary (towers 0 and 1 agree), Hamming weight **h = 87,520** (2N/3 = 87,381), so the predicted
  **sigma_I = sqrt((h+1)/12) = 85.40** and **K = 512 is 5.995 sigma**.
- **The bulk is Gaussian with that sigma.** Pooled sigma 85.44 (36 ciphertexts) / 85.46 (56), i.e. 1.0004 / 1.0007 of the prediction; mean ~0;
  excess kurtosis -0.0009 / -0.0012 (standard error ~0.002); the normal-quantile table agrees to better than 1 % up to p = 0.9999 and the
  0.5-sigma histogram gives chi^2 = 11.4 on 17 degrees of freedom (36 ciphertexts), 17.6 on 19 (56). Tail counts match the Gaussian expectation within
  counting noise out to 4 sigma (e.g. beyond 3 sigma: 12,735 observed vs 12,595 expected; beyond 4 sigma: 286 vs 300).
- The float computation is exact where it was checked: 252 (392 with the extension) coefficients recomputed with Python integers, including
  every ciphertext's argmax, 0 mismatches in I, worst |x_float - x_exact| below 3e-13.
- `c1 mod q0` is uniform to the precision this sample allows (std of c1/q0 per ciphertext 0.2878-0.2897 against 0.28868), for requests and for
  responses alike; request ciphertexts have a tiny fractional part (rms 2.5e-05: the fresh message), responses a larger one (rms 5.9e-03), as expected.
- **As the brief anticipated, the 6-sigma tail itself is not observable**: 0 of 4.7 M (7.3 M) samples exceed 512; the Gaussian model expects 0.01.

### Surprising, and not resolved

- **One coefficient of one RESPONSE (`tick_012_req12/resp.12.0`) has |I| = 503 = 5.89 sigma**, nine short of K. It was verified with exact
  integer arithmetic (it is that ciphertext's argmax). Under the Gaussian model the probability that the largest of 4.72 M samples reaches 503
  is 0.019 (0.029 for the 7.34 M of the extension; 0.009 / 0.015 for the responses alone).
- Split by kind, the **fresh requests follow the model everywhere** (56-ciphertext sample: 222 vs 233.7 expected beyond 4 sigma, 20 vs 24.7
  beyond 4.5 sigma, 1 vs 2.0 beyond 5 sigma, max 442), while the **responses run high in the far tail**: 10,013 vs 9,796 beyond 3 sigma,
  249 vs 233.7 beyond 4 sigma, 36 vs 24.7 beyond 4.5 sigma, 4 vs 2.0 beyond 5 sigma, 1 vs 0.14 beyond 5.5 sigma, and their pooled sigma is
  85.50 against 85.40 predicted.
- **Dispersion of the per-ciphertext sigma.** The N overflows of one ciphertext are not independent (they are N shifted signed sums over the same
  c1), which inflates the sampling variance of a per-ciphertext variance estimate by `1 + R2 - 0.6(h-1)/h`, R2 = the sum over nonzero lags of the
  squared normalised autocorrelation of the secret (1.0087 here, so a factor 1.41; derivation in `overflow_dist_stats.py`; checked on a toy ring by
  `sim_sigma_dispersion.py`: predicted 1.34, simulated 1.27 for uniform c1 and 1.35 for public-key encryptions; and by the 50,000-ciphertext
  classes below, chi^2 = 50,245 and 50,077 on 50,000). With that standard error the requests scatter as expected (chi^2 16.9 on 28) and the
  responses scatter more than expected (48.0 on 28, p = 0.011).
- Each of these is a 1-3 % observation on its own, they were found by looking at about ten statistics, and they all point the same way. They do
  NOT establish a heavier tail for evaluated ciphertexts. Theory says there should be none: if `c1` is uniform given the secret, I_j is a sum
  of h independent uniforms whatever produced the ciphertext. If the excess were real, the per-bootstrap failure probability in `../README.md`
  (a Gaussian tail at 6.0-6.3 sigma) would be an UNDERESTIMATE for the ciphertexts that actually get bootstrapped, which are evaluated ones.

### CPU check of exactly that question: fresh vs evaluated ciphertexts at ring 2^12 (addendum, not in the brief)

`harness/overflow_tail_probe.cpp` (sha256 `cf1a32c91398…`, binary `ac4ca2fb63f4…`, `build_overflow_tail_probe.log`, `tail_probe.jsonl`,
`tail_probe_loadstate.txt`): one stock uniform-ternary key, no bootstrapping, exact integer I of the q0 tower of 50,000 fresh public-key
encryptions and of 50,000 outputs of a short layer-like pipeline (ct*ct with relinearisation, rotation, plaintext product, mixed-level
addition, square, subtraction, scalar product, rescales; 5 towers left, noiseScaleDeg 1, like a served reply): 204.8 M overflows per class.

<<TAIL>>

Run 1 agrees with the model for both kinds out to 5 sigma (and the excess kurtosis of the two histograms is -0.000443 / -0.000462 against the
-1.2/h = -0.00044 of a sum of h uniforms, skewness 0: moments listed under the pooled table below), but the `evaluated` class has 14 values beyond 5.5 sigma against 8.1 expected, 2 beyond 6 sigma against
0.35, and **one value of 7.12 sigma**. So the run was repeated with a diagnostic build on a new key, `evaluated` only, 60,000 ciphertexts
(`harness/overflow_tail_probe2.cpp`, sha256 `6c9c5f231b34…`, binary `b7cefa9e4942…`, `tail_probe2.jsonl`, `tail_probe2_loadstate.txt`):

<<TAILC>>

Run 2 shows **no excess** (6 beyond 5.5 sigma against 8.5 expected, none beyond 6 sigma, maximum 5.68 sigma); pooled, the two `evaluated`
runs give 20 vs 16.6 beyond 5.5 sigma and 2 vs 0.9 beyond 6 sigma over 450 M samples. Run 2 logged every ciphertext whose maximum reached 5.3 sigma
(22 of them, 25.3 expected; table in `tail_probe2_tables.md`): each extreme is an isolated coefficient (that ciphertext has 1, in one case 2, values beyond 4 sigma) in a ciphertext whose sigma-hat is ordinary (15.12-15.72 against 15.32),
the ciphertext decrypts to the right message (relErrRms 5.9e-14 to 1.1e-13 against the clear computation), and the same coefficient had an
ordinary overflow (-31..+24) one operation earlier: every operation re-draws c1, as the model assumes. Signs: 15 positive, 7 negative
(p = 0.13).

**Where this leaves the question.** On 450 M evaluated and 205 M fresh CPU overflows the model holds to 5.5 sigma, and the far-tail excess of
run 1 did not reproduce. What stays unexplained is one value at 7.12 sigma (expected 6e-04 such values in all the evaluated samples) on
the CPU and one at 5.89 sigma on the GPU replies (expected 0.015-0.03). Two isolated extremes do not make a law, and they are not nothing
either: both occurred in EVALUATED ciphertexts, none in the 204.8 M fresh CPU samples or the 3.67 M request samples. A soak of the same probe (about 10 M overflows per minute on this Mac at
ring 2^12) to 10^10 samples would settle whether evaluated ciphertexts have a rare extreme component; nothing here measures it.

## 4. What this does and does not show

- It shows that the INPUT of the failure model is right on real session ciphertexts: I is centred, Gaussian in the bulk, with
  sigma = sqrt((h+1)/12) to 0.1 %, on a key whose K/sigma is 5.995.
- The tower-0 residues of ANY ciphertext have the distribution of a ModRaise input provided `c1 mod q0` is uniform given the secret. These 56
  ciphertexts are requests and replies, not the ciphertexts that entered the 480 bootstraps per tick inside the server; none of those was saved.
- It does not observe the tail that matters (beyond 6 sigma) and cannot: that needs ~10^9 samples. The rate in the README stays a Gaussian
  extrapolation from a bulk that is confirmed (to 5 sigma here, to 5.5 sigma on 450 M CPU samples), with the two unexplained extremes above as the caveat.
- The serialized client context carries `corFactor` = 0 (`serializedCorrectionFactor` in `probe_c_dump.jsonl`): the context was written before
  `EvalBootstrapSetup`, so the correction factor the GPU bootstrap used cannot be read from the kit.

## 5. Gaussian-tail rates at N = 2^17 for the thresholds probe B measured (ARITHMETIC, not a measurement)

<<RATES>>
