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

ring N = 131072, q0 = 1152921504606584833; secret: ternary = True, towers 0/1 agree = True, Hamming weight h = 87520 (h/N = 0.66772; uniform ternary expects 2/3), predicted sigma_I = sqrt((h+1)/12) = 85.4015, K/sigma = 5.9952

pooled over 36 ciphertexts: n = 4718592, mean = -0.0228, sigma = 85.4360 (measured/predicted = 1.00040), max|I| = 503 = 5.890 sigma_pred, excess kurtosis = -0.00085; requests only sigma = 85.4119, responses only sigma = 85.4601

under the Gaussian model with sigma_pred, P(max of n samples >= 503) = 0.0187; R2 = sum over nonzero lags of the squared normalised negacyclic autocorrelation of the secret = 1.0087, so a sigma estimate has standard error sigma/sqrt(2n) * sqrt(1 + R2 - 0.6(h-1)/h) (factor 1.1869; derivation in the script, checked by sim_sigma_dispersion.py): 0.0330 pooled, 0.1980 for one ciphertext (it would be 0.0278 and 0.1668 for independent samples); standard error of the excess kurtosis (independent-sample formula) = 0.00226

requests: 18 ciphertexts, n = 2359296, sigma = 85.4119 (+- 0.0467), max|I| = 442 (P(max >= that) = 0.4249); chi^2 of the per-ciphertext sigma estimates against sigma_pred = 10.7 on 18 (upper-tail p = 0.9056) (it would read 15.1 with the independent-sample standard error); tail counts observed/expected: |I|>256: 6292/6297.64, |I|>341: 130/150.23, |I|>384: 13/15.86, |I|>427: 1/1.31, |I|>469: 0/0.09

responses: 18 ciphertexts, n = 2359296, sigma = 85.4601 (+- 0.0467), max|I| = 503 (P(max >= that) = 0.0094); chi^2 of the per-ciphertext sigma estimates against sigma_pred = 29.8 on 18 (upper-tail p = 0.0392) (it would read 42.0 with the independent-sample standard error); tail counts observed/expected: |I|>256: 6443/6297.64, |I|>341: 156/150.23, |I|>384: 23/15.86, |I|>427: 4/1.31, |I|>469: 1/0.09

ten largest |I| of the pooled sample: [503, 442, 435, 432, 431, 427, 424, 416, 414, 411] (in units of sigma_pred: [5.89, 5.176, 5.094, 5.058, 5.047, 5.0, 4.965, 4.871, 4.848, 4.813])

float-vs-exact validation: 252 coefficients recomputed with Python integers (each ciphertext's argmax|I| plus 6 random), I mismatches = 0, worst |x_float - x_exact| = 1.705e-13

per ciphertext (Gaussian expectation per ciphertext: #(|I| > 3 sigma) = 349.9, #(|I| > 4 sigma) = 8.35):

| k | file | towers | noiseScaleDeg | log2 SF | mean I | sigma_I | max|I| | max|I|/sigma_pred | #>3 sigma | #>4 sigma | rms frac | max|frac| | std(c1/q0) (uniform: 0.28868) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | demo_verification_demo_20260918T074952Z/tick_000_req0/req.0.0 | 42 | 1 | 59.00 | +0.048 | 85.123 | 397 | 4.649 | 333 | 6 | 2.638e-05 | 6.763e-04 | 0.28810 |
| 1 | demo_verification_demo_20260918T074952Z/tick_000_req0/resp.0.0 | 5 | 1 | 59.00 | +0.064 | 85.112 | 403 | 4.719 | 360 | 7 | 5.947e-03 | 1.043e-01 | 0.28839 |
| 2 | demo_verification_demo_20260918T074952Z/tick_001_req1/req.1.0 | 42 | 1 | 59.00 | -0.038 | 85.583 | 386 | 4.520 | 350 | 11 | 2.891e-05 | 6.682e-04 | 0.28898 |
| 3 | demo_verification_demo_20260918T074952Z/tick_001_req1/resp.1.0 | 5 | 1 | 59.00 | +0.184 | 85.861 | 364 | 4.262 | 351 | 2 | 6.000e-03 | 8.817e-02 | 0.28902 |
| 4 | demo_verification_demo_20260918T074952Z/tick_002_req2/req.2.0 | 42 | 1 | 59.00 | +0.002 | 85.377 | 401 | 4.695 | 344 | 7 | 2.613e-05 | 4.379e-04 | 0.28906 |
| 5 | demo_verification_demo_20260918T074952Z/tick_002_req2/resp.2.0 | 5 | 1 | 59.00 | +0.080 | 85.486 | 408 | 4.777 | 351 | 14 | 5.954e-03 | 5.206e-02 | 0.28933 |
| 6 | demo_verification_demo_20260918T074952Z/tick_003_req3/req.3.0 | 42 | 1 | 59.00 | +0.138 | 85.359 | 386 | 4.520 | 330 | 8 | 2.311e-05 | 2.536e-04 | 0.28916 |
| 7 | demo_verification_demo_20260918T074952Z/tick_003_req3/resp.3.0 | 5 | 1 | 59.00 | -0.160 | 85.236 | 399 | 4.672 | 337 | 5 | 5.888e-03 | 5.260e-02 | 0.28784 |
| 8 | demo_verification_demo_20260918T074952Z/tick_004_req4/req.4.0 | 42 | 1 | 59.00 | -0.270 | 85.524 | 384 | 4.496 | 376 | 8 | 2.528e-05 | 3.962e-04 | 0.28857 |
| 9 | demo_verification_demo_20260918T074952Z/tick_004_req4/resp.4.0 | 5 | 1 | 59.00 | -0.078 | 85.834 | 374 | 4.379 | 363 | 6 | 5.963e-03 | 5.072e-02 | 0.28906 |
| 10 | demo_verification_demo_20260918T074952Z/tick_005_req5/req.5.0 | 42 | 1 | 59.00 | -0.120 | 85.395 | 378 | 4.426 | 347 | 6 | 2.374e-05 | 3.092e-04 | 0.28840 |
| 11 | demo_verification_demo_20260918T074952Z/tick_005_req5/resp.5.0 | 5 | 1 | 59.00 | -0.183 | 85.467 | 401 | 4.695 | 368 | 13 | 5.933e-03 | 4.990e-02 | 0.28909 |
| 12 | demo_verification_demo_20260918T074952Z/tick_006_req6/req.6.0 | 42 | 1 | 59.00 | +0.267 | 85.536 | 386 | 4.520 | 359 | 7 | 2.424e-05 | 2.930e-04 | 0.28844 |
| 13 | demo_verification_demo_20260918T074952Z/tick_006_req6/resp.6.0 | 5 | 1 | 59.00 | -0.160 | 85.675 | 431 | 5.047 | 374 | 10 | 5.907e-03 | 4.740e-02 | 0.28893 |
| 14 | demo_verification_demo_20260918T074952Z/tick_007_req7/req.7.0 | 42 | 1 | 59.00 | +0.249 | 85.515 | 375 | 4.391 | 384 | 4 | 2.411e-05 | 2.608e-04 | 0.28851 |
| 15 | demo_verification_demo_20260918T074952Z/tick_007_req7/resp.7.0 | 5 | 1 | 59.00 | -0.157 | 85.552 | 414 | 4.848 | 372 | 15 | 5.891e-03 | 4.808e-02 | 0.28905 |
| 16 | demo_verification_demo_20260918T074952Z/tick_008_req8/req.8.0 | 42 | 1 | 59.00 | +0.127 | 85.765 | 361 | 4.227 | 330 | 4 | 2.613e-05 | 2.363e-04 | 0.28967 |
| 17 | demo_verification_demo_20260918T074952Z/tick_008_req8/resp.8.0 | 5 | 1 | 59.00 | -0.103 | 85.792 | 389 | 4.555 | 392 | 10 | 5.913e-03 | 5.034e-02 | 0.28935 |
| 18 | demo_verification_demo_20260918T074952Z/tick_009_req9/req.9.0 | 42 | 1 | 59.00 | -0.106 | 85.257 | 395 | 4.625 | 338 | 12 | 2.591e-05 | 2.121e-04 | 0.28900 |
| 19 | demo_verification_demo_20260918T074952Z/tick_009_req9/resp.9.0 | 5 | 1 | 59.00 | -0.466 | 85.839 | 406 | 4.754 | 374 | 13 | 5.859e-03 | 4.130e-02 | 0.28847 |
| 20 | demo_verification_demo_20260918T074952Z/tick_010_req10/req.10.0 | 42 | 1 | 59.00 | -0.019 | 85.469 | 343 | 4.016 | 344 | 1 | 2.546e-05 | 2.239e-04 | 0.28822 |
| 21 | demo_verification_demo_20260918T074952Z/tick_010_req10/resp.10.0 | 5 | 1 | 59.00 | -0.021 | 85.276 | 361 | 4.227 | 328 | 4 | 5.876e-03 | 4.594e-02 | 0.28868 |
| 22 | demo_verification_demo_20260918T074952Z/tick_011_req11/req.11.0 | 42 | 1 | 59.00 | +0.012 | 85.147 | 408 | 4.777 | 336 | 9 | 2.455e-05 | 2.196e-04 | 0.28808 |
| 23 | demo_verification_demo_20260918T074952Z/tick_011_req11/resp.11.0 | 5 | 1 | 59.00 | +0.171 | 85.154 | 435 | 5.094 | 335 | 7 | 5.871e-03 | 3.788e-02 | 0.28824 |
| 24 | demo_verification_demo_20260918T074952Z/tick_012_req12/req.12.0 | 42 | 1 | 59.00 | -0.244 | 85.401 | 442 | 5.176 | 369 | 13 | 2.457e-05 | 2.383e-04 | 0.28863 |
| 25 | demo_verification_demo_20260918T074952Z/tick_012_req12/resp.12.0 | 5 | 1 | 59.00 | -0.231 | 85.161 | 503 | 5.890 | 332 | 8 | 5.905e-03 | 4.349e-02 | 0.28788 |
| 26 | demo_verification_demo_20260918T074952Z/tick_013_req13/req.13.0 | 42 | 1 | 59.00 | +0.107 | 85.376 | 381 | 4.461 | 354 | 6 | 2.439e-05 | 2.486e-04 | 0.28884 |
| 27 | demo_verification_demo_20260918T074952Z/tick_013_req13/resp.13.0 | 5 | 1 | 59.00 | -0.093 | 85.305 | 389 | 4.555 | 353 | 7 | 5.897e-03 | 3.907e-02 | 0.28862 |
| 28 | demo_verification_demo_20260918T074952Z/tick_014_req14/req.14.0 | 42 | 1 | 59.00 | +0.011 | 85.427 | 399 | 4.672 | 353 | 9 | 2.486e-05 | 2.318e-04 | 0.28836 |
| 29 | demo_verification_demo_20260918T074952Z/tick_014_req14/resp.14.0 | 5 | 1 | 59.00 | -0.005 | 85.241 | 432 | 5.058 | 348 | 7 | 5.888e-03 | 3.340e-02 | 0.28867 |
| 30 | demo_verification_demo_20260918T074952Z/tick_015_req15/req.15.0 | 42 | 1 | 59.00 | +0.144 | 85.239 | 386 | 4.520 | 316 | 6 | 2.437e-05 | 2.174e-04 | 0.28872 |
| 31 | demo_verification_demo_20260918T074952Z/tick_015_req15/resp.15.0 | 5 | 1 | 59.00 | -0.035 | 85.552 | 402 | 4.707 | 382 | 7 | 5.880e-03 | 4.035e-02 | 0.28920 |
| 32 | demo_verification_demo_20260918T074952Z/tick_016_req16/req.16.0 | 42 | 1 | 59.00 | -0.075 | 85.408 | 380 | 4.450 | 358 | 7 | 2.486e-05 | 2.317e-04 | 0.28834 |
| 33 | demo_verification_demo_20260918T074952Z/tick_016_req16/resp.16.0 | 5 | 1 | 59.00 | -0.090 | 85.439 | 424 | 4.965 | 360 | 6 | 5.880e-03 | 4.363e-02 | 0.28849 |
| 34 | demo_verification_demo_20260918T074952Z/tick_017_req17/req.17.0 | 42 | 1 | 59.00 | +0.210 | 85.508 | 374 | 4.379 | 371 | 6 | 2.470e-05 | 2.165e-04 | 0.28813 |
| 35 | demo_verification_demo_20260918T074952Z/tick_017_req17/resp.17.0 | 5 | 1 | 59.00 | +0.020 | 85.289 | 400 | 4.684 | 363 | 15 | 5.866e-03 | 4.591e-02 | 0.28833 |

Tail counts, pooled (Gaussian expectation uses sigma_pred and the integer continuity correction P(|I| > b) = 2Q((b+0.5)/sigma)):

| z | bound b = floor(z*sigma_pred) | observed #(|I| > b) | Gaussian expectation | observed/expected | sqrt(expected) |
|---|---|---|---|---|---|
| 1 | 85 | 1496537 | 1494626.66 | 1.0013 | 1222.55 |
| 2 | 170 | 216930 | 216511.40 | 1.0019 | 465.31 |
| 3 | 256 | 12735 | 12595.27 | 1.0111 | 112.23 |
| 3.5 | 298 | 2239 | 2234.77 | 1.0019 | 47.27 |
| 4 | 341 | 286 | 300.46 | 0.9519 | 17.33 |
| 4.5 | 384 | 36 | 31.72 | 1.1348 | 5.63 |
| 5 | 427 | 5 | 2.63 | 1.9045 | 1.62 |
| 5.5 | 469 | 1 | 0.18 | 5.5035 | 0.43 |
| 6 | 512 | 0 | 0.01 | 0.0000 | 0.10 |

Normal-quantile comparison, pooled:

| p | empirical quantile of I | sigma_pred * z_p | ratio |
|---|---|---|---|
| 0.5 | 0.0 | 0.0 | - |
| 0.75 | 58.0 | 57.60 | 1.0069 |
| 0.9 | 109.0 | 109.45 | 0.9959 |
| 0.99 | 199.0 | 198.67 | 1.0016 |
| 0.999 | 264.0 | 263.91 | 1.0003 |
| 0.9999 | 318.0 | 317.61 | 1.0012 |
| 0.99999 | 367.0 | 364.23 | 1.0076 |
| 0.999999 | 431.0 | 405.95 | 1.0617 |

histogram in 0.5-sigma_pred bins over [-6, 6] sigma: chi^2 = 11.4 on 17 degrees of freedom (bins with expectation >= 20)

## 2. All three kits under the same key (56 ciphertexts)

ring N = 131072, q0 = 1152921504606584833; secret: ternary = True, towers 0/1 agree = True, Hamming weight h = 87520 (h/N = 0.66772; uniform ternary expects 2/3), predicted sigma_I = sqrt((h+1)/12) = 85.4015, K/sigma = 5.9952

pooled over 56 ciphertexts: n = 7340032, mean = 0.0037, sigma = 85.4625 (measured/predicted = 1.00071), max|I| = 503 = 5.890 sigma_pred, excess kurtosis = -0.00123; requests only sigma = 85.4290, responses only sigma = 85.4960

under the Gaussian model with sigma_pred, P(max of n samples >= 503) = 0.0290; R2 = sum over nonzero lags of the squared normalised negacyclic autocorrelation of the secret = 1.0087, so a sigma estimate has standard error sigma/sqrt(2n) * sqrt(1 + R2 - 0.6(h-1)/h) (factor 1.1869; derivation in the script, checked by sim_sigma_dispersion.py): 0.0265 pooled, 0.1980 for one ciphertext (it would be 0.0223 and 0.1668 for independent samples); standard error of the excess kurtosis (independent-sample formula) = 0.00181

requests: 28 ciphertexts, n = 3670016, sigma = 85.4290 (+- 0.0374), max|I| = 442 (P(max >= that) = 0.5771); chi^2 of the per-ciphertext sigma estimates against sigma_pred = 16.9 on 28 (upper-tail p = 0.9504) (it would read 23.8 with the independent-sample standard error); tail counts observed/expected: |I|>256: 9805/9796.32, |I|>341: 222/233.69, |I|>384: 20/24.67, |I|>427: 1/2.04, |I|>469: 0/0.14

responses: 28 ciphertexts, n = 3670016, sigma = 85.4960 (+- 0.0374), max|I| = 503 (P(max >= that) = 0.0146); chi^2 of the per-ciphertext sigma estimates against sigma_pred = 48.0 on 28 (upper-tail p = 0.0108) (it would read 67.6 with the independent-sample standard error); tail counts observed/expected: |I|>256: 10013/9796.32, |I|>341: 249/233.69, |I|>384: 36/24.67, |I|>427: 4/2.04, |I|>469: 1/0.14

ten largest |I| of the pooled sample: [503, 442, 435, 432, 431, 427, 424, 416, 414, 411] (in units of sigma_pred: [5.89, 5.176, 5.094, 5.058, 5.047, 5.0, 4.965, 4.871, 4.848, 4.813])

float-vs-exact validation: 392 coefficients recomputed with Python integers (each ciphertext's argmax|I| plus 6 random), I mismatches = 0, worst |x_float - x_exact| = 2.842e-13

per ciphertext (Gaussian expectation per ciphertext: #(|I| > 3 sigma) = 349.9, #(|I| > 4 sigma) = 8.35):

| k | file | towers | noiseScaleDeg | log2 SF | mean I | sigma_I | max|I| | max|I|/sigma_pred | #>3 sigma | #>4 sigma | rms frac | max|frac| | std(c1/q0) (uniform: 0.28868) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | demo_verification_demo_20260918T074952Z/tick_000_req0/req.0.0 | 42 | 1 | 59.00 | +0.048 | 85.123 | 397 | 4.649 | 333 | 6 | 2.638e-05 | 6.763e-04 | 0.28810 |
| 1 | demo_verification_demo_20260918T074952Z/tick_000_req0/resp.0.0 | 5 | 1 | 59.00 | +0.064 | 85.112 | 403 | 4.719 | 360 | 7 | 5.947e-03 | 1.043e-01 | 0.28839 |
| 2 | demo_verification_demo_20260918T074952Z/tick_001_req1/req.1.0 | 42 | 1 | 59.00 | -0.038 | 85.583 | 386 | 4.520 | 350 | 11 | 2.891e-05 | 6.682e-04 | 0.28898 |
| 3 | demo_verification_demo_20260918T074952Z/tick_001_req1/resp.1.0 | 5 | 1 | 59.00 | +0.184 | 85.861 | 364 | 4.262 | 351 | 2 | 6.000e-03 | 8.817e-02 | 0.28902 |
| 4 | demo_verification_demo_20260918T074952Z/tick_002_req2/req.2.0 | 42 | 1 | 59.00 | +0.002 | 85.377 | 401 | 4.695 | 344 | 7 | 2.613e-05 | 4.379e-04 | 0.28906 |
| 5 | demo_verification_demo_20260918T074952Z/tick_002_req2/resp.2.0 | 5 | 1 | 59.00 | +0.080 | 85.486 | 408 | 4.777 | 351 | 14 | 5.954e-03 | 5.206e-02 | 0.28933 |
| 6 | demo_verification_demo_20260918T074952Z/tick_003_req3/req.3.0 | 42 | 1 | 59.00 | +0.138 | 85.359 | 386 | 4.520 | 330 | 8 | 2.311e-05 | 2.536e-04 | 0.28916 |
| 7 | demo_verification_demo_20260918T074952Z/tick_003_req3/resp.3.0 | 5 | 1 | 59.00 | -0.160 | 85.236 | 399 | 4.672 | 337 | 5 | 5.888e-03 | 5.260e-02 | 0.28784 |
| 8 | demo_verification_demo_20260918T074952Z/tick_004_req4/req.4.0 | 42 | 1 | 59.00 | -0.270 | 85.524 | 384 | 4.496 | 376 | 8 | 2.528e-05 | 3.962e-04 | 0.28857 |
| 9 | demo_verification_demo_20260918T074952Z/tick_004_req4/resp.4.0 | 5 | 1 | 59.00 | -0.078 | 85.834 | 374 | 4.379 | 363 | 6 | 5.963e-03 | 5.072e-02 | 0.28906 |
| 10 | demo_verification_demo_20260918T074952Z/tick_005_req5/req.5.0 | 42 | 1 | 59.00 | -0.120 | 85.395 | 378 | 4.426 | 347 | 6 | 2.374e-05 | 3.092e-04 | 0.28840 |
| 11 | demo_verification_demo_20260918T074952Z/tick_005_req5/resp.5.0 | 5 | 1 | 59.00 | -0.183 | 85.467 | 401 | 4.695 | 368 | 13 | 5.933e-03 | 4.990e-02 | 0.28909 |
| 12 | demo_verification_demo_20260918T074952Z/tick_006_req6/req.6.0 | 42 | 1 | 59.00 | +0.267 | 85.536 | 386 | 4.520 | 359 | 7 | 2.424e-05 | 2.930e-04 | 0.28844 |
| 13 | demo_verification_demo_20260918T074952Z/tick_006_req6/resp.6.0 | 5 | 1 | 59.00 | -0.160 | 85.675 | 431 | 5.047 | 374 | 10 | 5.907e-03 | 4.740e-02 | 0.28893 |
| 14 | demo_verification_demo_20260918T074952Z/tick_007_req7/req.7.0 | 42 | 1 | 59.00 | +0.249 | 85.515 | 375 | 4.391 | 384 | 4 | 2.411e-05 | 2.608e-04 | 0.28851 |
| 15 | demo_verification_demo_20260918T074952Z/tick_007_req7/resp.7.0 | 5 | 1 | 59.00 | -0.157 | 85.552 | 414 | 4.848 | 372 | 15 | 5.891e-03 | 4.808e-02 | 0.28905 |
| 16 | demo_verification_demo_20260918T074952Z/tick_008_req8/req.8.0 | 42 | 1 | 59.00 | +0.127 | 85.765 | 361 | 4.227 | 330 | 4 | 2.613e-05 | 2.363e-04 | 0.28967 |
| 17 | demo_verification_demo_20260918T074952Z/tick_008_req8/resp.8.0 | 5 | 1 | 59.00 | -0.103 | 85.792 | 389 | 4.555 | 392 | 10 | 5.913e-03 | 5.034e-02 | 0.28935 |
| 18 | demo_verification_demo_20260918T074952Z/tick_009_req9/req.9.0 | 42 | 1 | 59.00 | -0.106 | 85.257 | 395 | 4.625 | 338 | 12 | 2.591e-05 | 2.121e-04 | 0.28900 |
| 19 | demo_verification_demo_20260918T074952Z/tick_009_req9/resp.9.0 | 5 | 1 | 59.00 | -0.466 | 85.839 | 406 | 4.754 | 374 | 13 | 5.859e-03 | 4.130e-02 | 0.28847 |
| 20 | demo_verification_demo_20260918T074952Z/tick_010_req10/req.10.0 | 42 | 1 | 59.00 | -0.019 | 85.469 | 343 | 4.016 | 344 | 1 | 2.546e-05 | 2.239e-04 | 0.28822 |
| 21 | demo_verification_demo_20260918T074952Z/tick_010_req10/resp.10.0 | 5 | 1 | 59.00 | -0.021 | 85.276 | 361 | 4.227 | 328 | 4 | 5.876e-03 | 4.594e-02 | 0.28868 |
| 22 | demo_verification_demo_20260918T074952Z/tick_011_req11/req.11.0 | 42 | 1 | 59.00 | +0.012 | 85.147 | 408 | 4.777 | 336 | 9 | 2.455e-05 | 2.196e-04 | 0.28808 |
| 23 | demo_verification_demo_20260918T074952Z/tick_011_req11/resp.11.0 | 5 | 1 | 59.00 | +0.171 | 85.154 | 435 | 5.094 | 335 | 7 | 5.871e-03 | 3.788e-02 | 0.28824 |
| 24 | demo_verification_demo_20260918T074952Z/tick_012_req12/req.12.0 | 42 | 1 | 59.00 | -0.244 | 85.401 | 442 | 5.176 | 369 | 13 | 2.457e-05 | 2.383e-04 | 0.28863 |
| 25 | demo_verification_demo_20260918T074952Z/tick_012_req12/resp.12.0 | 5 | 1 | 59.00 | -0.231 | 85.161 | 503 | 5.890 | 332 | 8 | 5.905e-03 | 4.349e-02 | 0.28788 |
| 26 | demo_verification_demo_20260918T074952Z/tick_013_req13/req.13.0 | 42 | 1 | 59.00 | +0.107 | 85.376 | 381 | 4.461 | 354 | 6 | 2.439e-05 | 2.486e-04 | 0.28884 |
| 27 | demo_verification_demo_20260918T074952Z/tick_013_req13/resp.13.0 | 5 | 1 | 59.00 | -0.093 | 85.305 | 389 | 4.555 | 353 | 7 | 5.897e-03 | 3.907e-02 | 0.28862 |
| 28 | demo_verification_demo_20260918T074952Z/tick_014_req14/req.14.0 | 42 | 1 | 59.00 | +0.011 | 85.427 | 399 | 4.672 | 353 | 9 | 2.486e-05 | 2.318e-04 | 0.28836 |
| 29 | demo_verification_demo_20260918T074952Z/tick_014_req14/resp.14.0 | 5 | 1 | 59.00 | -0.005 | 85.241 | 432 | 5.058 | 348 | 7 | 5.888e-03 | 3.340e-02 | 0.28867 |
| 30 | demo_verification_demo_20260918T074952Z/tick_015_req15/req.15.0 | 42 | 1 | 59.00 | +0.144 | 85.239 | 386 | 4.520 | 316 | 6 | 2.437e-05 | 2.174e-04 | 0.28872 |
| 31 | demo_verification_demo_20260918T074952Z/tick_015_req15/resp.15.0 | 5 | 1 | 59.00 | -0.035 | 85.552 | 402 | 4.707 | 382 | 7 | 5.880e-03 | 4.035e-02 | 0.28920 |
| 32 | demo_verification_demo_20260918T074952Z/tick_016_req16/req.16.0 | 42 | 1 | 59.00 | -0.075 | 85.408 | 380 | 4.450 | 358 | 7 | 2.486e-05 | 2.317e-04 | 0.28834 |
| 33 | demo_verification_demo_20260918T074952Z/tick_016_req16/resp.16.0 | 5 | 1 | 59.00 | -0.090 | 85.439 | 424 | 4.965 | 360 | 6 | 5.880e-03 | 4.363e-02 | 0.28849 |
| 34 | demo_verification_demo_20260918T074952Z/tick_017_req17/req.17.0 | 42 | 1 | 59.00 | +0.210 | 85.508 | 374 | 4.379 | 371 | 6 | 2.470e-05 | 2.165e-04 | 0.28813 |
| 35 | demo_verification_demo_20260918T074952Z/tick_017_req17/resp.17.0 | 5 | 1 | 59.00 | +0.020 | 85.289 | 400 | 4.684 | 363 | 15 | 5.866e-03 | 4.591e-02 | 0.28833 |
| 36 | demo_verification_demo_20260904T183832Z/tick_000_req0/req.0.0 | 42 | 1 | 59.00 | +0.071 | 85.341 | 392 | 4.590 | 347 | 8 | 2.287e-05 | 7.079e-04 | 0.28884 |
| 37 | demo_verification_demo_20260904T183832Z/tick_000_req0/resp.0.0 | 7 | 1 | 59.00 | +0.138 | 85.621 | 397 | 4.649 | 379 | 11 | 6.063e-03 | 1.371e-01 | 0.28884 |
| 38 | demo_verification_demo_20260904T183832Z/tick_001_req1/req.1.0 | 42 | 1 | 59.00 | +0.183 | 85.536 | 411 | 4.813 | 357 | 15 | 1.994e-05 | 6.908e-04 | 0.28797 |
| 39 | demo_verification_demo_20260904T183832Z/tick_001_req1/resp.1.0 | 7 | 1 | 59.00 | +0.177 | 85.309 | 386 | 4.520 | 338 | 8 | 6.123e-03 | 1.508e-01 | 0.28889 |
| 40 | demo_verification_demo_20260918T025113Z/tick_000_req0/req.0.0 | 42 | 1 | 59.00 | -0.157 | 85.411 | 389 | 4.555 | 367 | 8 | 2.287e-05 | 7.079e-04 | 0.28875 |
| 41 | demo_verification_demo_20260918T025113Z/tick_000_req0/resp.0.0 | 5 | 1 | 59.00 | +0.050 | 85.536 | 406 | 4.754 | 364 | 14 | 5.968e-03 | 1.433e-01 | 0.28907 |
| 42 | demo_verification_demo_20260918T025113Z/tick_001_req1/req.1.0 | 42 | 1 | 59.00 | +0.083 | 85.348 | 392 | 4.590 | 350 | 12 | 2.279e-05 | 6.426e-04 | 0.28871 |
| 43 | demo_verification_demo_20260918T025113Z/tick_001_req1/resp.1.0 | 5 | 1 | 59.00 | -0.191 | 85.658 | 399 | 4.672 | 395 | 6 | 6.034e-03 | 1.705e-01 | 0.28936 |
| 44 | demo_verification_demo_20260918T025113Z/tick_002_req2/req.2.0 | 42 | 1 | 59.00 | -0.169 | 85.608 | 384 | 4.496 | 350 | 6 | 2.188e-05 | 5.647e-04 | 0.28879 |
| 45 | demo_verification_demo_20260918T025113Z/tick_002_req2/resp.2.0 | 5 | 1 | 59.00 | +0.116 | 85.650 | 372 | 4.356 | 374 | 8 | 5.675e-03 | 1.436e-01 | 0.28896 |
| 46 | demo_verification_demo_20260918T025113Z/tick_003_req3/req.3.0 | 42 | 1 | 59.00 | +0.135 | 85.673 | 359 | 4.204 | 374 | 7 | 3.578e-05 | 1.036e-03 | 0.28865 |
| 47 | demo_verification_demo_20260918T025113Z/tick_003_req3/resp.3.0 | 5 | 1 | 59.00 | +0.155 | 85.734 | 383 | 4.485 | 360 | 7 | 5.618e-03 | 1.529e-01 | 0.28891 |
| 48 | demo_verification_demo_20260918T025113Z/tick_004_req4/req.4.0 | 42 | 1 | 59.00 | -0.099 | 85.515 | 371 | 4.344 | 360 | 8 | 2.963e-05 | 7.458e-04 | 0.28864 |
| 49 | demo_verification_demo_20260918T025113Z/tick_004_req4/resp.4.0 | 5 | 1 | 59.00 | +0.181 | 86.001 | 408 | 4.777 | 363 | 17 | 6.142e-03 | 1.441e-01 | 0.28919 |
| 50 | demo_verification_demo_20260918T025113Z/tick_005_req5/req.5.0 | 42 | 1 | 59.00 | +0.088 | 85.294 | 401 | 4.695 | 315 | 12 | 2.805e-05 | 8.308e-04 | 0.28801 |
| 51 | demo_verification_demo_20260918T025113Z/tick_005_req5/resp.5.0 | 5 | 1 | 59.00 | +0.407 | 85.296 | 385 | 4.508 | 318 | 7 | 5.938e-03 | 1.506e-01 | 0.28852 |
| 52 | demo_verification_demo_20260918T025113Z/tick_006_req6/req.6.0 | 42 | 1 | 59.00 | +0.203 | 85.629 | 390 | 4.567 | 346 | 7 | 1.830e-05 | 5.304e-04 | 0.28927 |
| 53 | demo_verification_demo_20260918T025113Z/tick_006_req6/resp.6.0 | 5 | 1 | 59.00 | -0.015 | 85.517 | 390 | 4.567 | 369 | 4 | 5.718e-03 | 1.500e-01 | 0.28853 |
| 54 | demo_verification_demo_20260918T025113Z/tick_007_req7/req.7.0 | 42 | 1 | 59.00 | -0.037 | 85.243 | 381 | 4.461 | 347 | 9 | 1.950e-05 | 6.447e-04 | 0.28904 |
| 55 | demo_verification_demo_20260918T025113Z/tick_007_req7/resp.7.0 | 5 | 1 | 59.00 | -0.287 | 85.281 | 393 | 4.602 | 310 | 11 | 5.954e-03 | 1.687e-01 | 0.28890 |

Tail counts, pooled (Gaussian expectation uses sigma_pred and the integer continuity correction P(|I| > b) = 2Q((b+0.5)/sigma)):

| z | bound b = floor(z*sigma_pred) | observed #(|I| > b) | Gaussian expectation | observed/expected | sqrt(expected) |
|---|---|---|---|---|---|
| 1 | 85 | 2328239 | 2324974.80 | 1.0014 | 1524.79 |
| 2 | 170 | 337926 | 336795.51 | 1.0034 | 580.34 |
| 3 | 256 | 19818 | 19592.65 | 1.0115 | 139.97 |
| 3.5 | 298 | 3525 | 3476.31 | 1.0140 | 58.96 |
| 4 | 341 | 471 | 467.38 | 1.0077 | 21.62 |
| 4.5 | 384 | 56 | 49.35 | 1.1348 | 7.02 |
| 5 | 427 | 5 | 4.08 | 1.2243 | 2.02 |
| 5.5 | 469 | 1 | 0.28 | 3.5380 | 0.53 |
| 6 | 512 | 0 | 0.01 | 0.0000 | 0.12 |

Normal-quantile comparison, pooled:

| p | empirical quantile of I | sigma_pred * z_p | ratio |
|---|---|---|---|
| 0.5 | 0.0 | 0.0 | - |
| 0.75 | 58.0 | 57.60 | 1.0069 |
| 0.9 | 110.0 | 109.45 | 1.0051 |
| 0.99 | 199.0 | 198.67 | 1.0016 |
| 0.999 | 264.0 | 263.91 | 1.0003 |
| 0.9999 | 318.0 | 317.61 | 1.0012 |
| 0.99999 | 370.0 | 364.23 | 1.0158 |
| 0.999999 | 414.0 | 405.95 | 1.0198 |

histogram in 0.5-sigma_pred bins over [-6, 6] sigma: chi^2 = 17.6 on 19 degrees of freedom (bins with expectation >= 20)

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

ring N = 4096, stock uniform-ternary key with Hamming weight 2706, sigma_pred = 15.0194, R2 = 0.9216, sample-variance inflation 1 + R2 - 0.6(h-1)/h = 1.3218; 50000 ciphertexts per class (source sha256 cf1a32c913982c49, argv `harness/build/overflow_tail_probe --log-ring 12 --cts 50000`)

| class | ciphertexts | towers | noiseScaleDeg | samples | sigma / sigma_pred | max|I| (in sigma) | chi^2 of per-ciphertext variances, correlated SE (dof; upper-tail p) | same with the independent-sample SE |
|---|---|---|---|---|---|---|---|---|
| fresh | 50000 | 9 | 1 | 204800000 | 1.00001 | 86 (5.726) | 50245 (50000; 0.219) | 66416 (0) |
| evaluated | 50000 | 5 | 1 | 204800000 | 1.00003 | 107 (7.124) | 50077 (50000; 0.403) | 66193 (0) |

Tail counts, observed / Gaussian expectation (two-sided Poisson-style reading: `p_hi` = P(count >= observed)):

| class | |I| > 3 sigma (b = 45) | |I| > 3.5 sigma (b = 52) | |I| > 4 sigma (b = 60) | |I| > 4.5 sigma (b = 67) | |I| > 5 sigma (b = 75) | |I| > 5.5 sigma (b = 82) | |I| > 6 sigma (b = 90) |
|---|---|---|---|---|---|---|---|
| fresh | 501139 / 501827.00 (ratio 0.999, p_hi 0.834) | 96522 / 96916.40 (ratio 0.996, p_hi 0.898) | 11264 / 11515.00 (ratio 0.978, p_hi 0.991) | 1406 / 1430.31 (ratio 0.983, p_hi 0.743) | 97 / 102.13 (ratio 0.950, p_hi 0.707) | 10 / 8.10 (ratio 1.235, p_hi 0.296) | 0 / 0.35 (ratio 0.000, p_hi 1) |
| evaluated | 500300 / 501827.00 (ratio 0.997, p_hi 0.985) | 96363 / 96916.40 (ratio 0.994, p_hi 0.962) | 11482 / 11515.00 (ratio 0.997, p_hi 0.622) | 1425 / 1430.31 (ratio 0.996, p_hi 0.559) | 109 / 102.13 (ratio 1.067, p_hi 0.261) | 14 / 8.10 (ratio 1.729, p_hi 0.0372) | 2 / 0.35 (ratio 5.794, p_hi 0.0475) |

Run 1 agrees with the model for both kinds out to 5 sigma (and the excess kurtosis of the two histograms is -0.000443 / -0.000462 against the
-1.2/h = -0.00044 of a sum of h uniforms, skewness 0: moments listed under the pooled table below), but the `evaluated` class has 14 values beyond 5.5 sigma against 8.1 expected, 2 beyond 6 sigma against
0.35, and **one value of 7.12 sigma**. So the run was repeated with a diagnostic build on a new key, `evaluated` only, 60,000 ciphertexts
(`harness/overflow_tail_probe2.cpp`, sha256 `6c9c5f231b34…`, binary `b7cefa9e4942…`, `tail_probe2.jsonl`, `tail_probe2_loadstate.txt`):

| sample | ciphertexts | samples | max|I| in sigma | > 3 sigma: observed / expected (p_hi) | > 3.5 sigma: observed / expected (p_hi) | > 4 sigma: observed / expected (p_hi) | > 4.5 sigma: observed / expected (p_hi) | > 5 sigma: observed / expected (p_hi) | > 5.5 sigma: observed / expected (p_hi) | > 6 sigma: observed / expected (p_hi) |
|---|---|---|---|---|---|---|---|---|---|---|
| fresh, run 1 | 50000 | 204800000 | 5.726 | 501139 / 501827.00 (0.834) | 96522 / 96916.40 (0.898) | 11264 / 11515.00 (0.991) | 1406 / 1430.31 (0.743) | 97 / 102.13 (0.707) | 10 / 8.10 (0.296) | 0 / 0.35 (1) |
| evaluated, run 1 | 50000 | 204800000 | 7.124 | 500300 / 501827.00 (0.985) | 96363 / 96916.40 (0.962) | 11482 / 11515.00 (0.622) | 1425 / 1430.31 (0.559) | 109 / 102.13 (0.261) | 14 / 8.10 (0.0372) | 2 / 0.35 (0.0475) |
| evaluated, run 2 (new key) | 60000 | 245760000 | 5.680 | 727856 / 730141.00 (0.996) | 116688 / 117358.00 (0.975) | 14567 / 14585.90 (0.563) | 1869 / 1900.75 (0.77) | 126 / 144.79 (0.948) | 6 / 8.47 (0.848) | 0 / 0.57 (1) |
| evaluated, runs 1 + 2 | 110000 | 450560000 | 7.124 | 1228156 / 1231968.00 (1) | 213051 / 214274.40 (0.996) | 26049 / 26100.90 (0.627) | 3294 / 3331.06 (0.742) | 235 / 246.92 (0.784) | 20 / 16.57 (0.23) | 2 / 0.91 (0.233) |

The largest value of run 1's `evaluated` class is |I| = 107 = 7.124 sigma. Under the Gaussian model P(|I_j| >= 107) = 1.333e-12 per sample, i.e. an expected 2.73e-04 such values among the 204800000 samples of that class and 6.01e-04 among the 450560000 `evaluated` samples of both runs.
Run 1, `evaluated`, every value with |I| >= 84 (value x count): +107 x 1, +92 x 1, +88 x 1, +87 x 3, -87 x 1, -86 x 1, +85 x 1, +84 x 1; the same for `fresh`: +86 x 1, -86 x 2, +84 x 1

Moments of the run-1 histograms (a sum of h uniforms has excess kurtosis -1.2/h = -0.00044 for this key's h = 2706; standard errors for n independent samples: mean 0.0010, skewness 0.00017, kurtosis 0.00034):
- fresh: mean +0.00085, sigma/sigma_pred 1.00002, skewness +0.000041, excess kurtosis -0.000443; values above +5 sigma: 46, below -5 sigma: 51
- evaluated: mean +0.00015, sigma/sigma_pred 1.00004, skewness -0.000019, excess kurtosis -0.000462; values above +5 sigma: 52, below -5 sigma: 57

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

| sigma_I source | h | sigma_I | K/sigma | B | what the planted sweep measured just above B | P(|I_j| > B) per coefficient | per bootstrap | per tick (480 boots) | mean ticks to first event |
|---|---|---|---|---|---|---|---|---|---|
| h measured on the Mac key (probe C) | 87520 | 85.4015 | 5.9952 | 512 | edge of the fitted range (no measurable effect up to 515-517) | 1.960e-09 | 2.569e-04 | 1.160e-01 | 9 |
| h measured on the Mac key (probe C) | 87520 | 85.4015 | 5.9952 | 518 | F=7 and F=10: median relErrRms first exceeds 2x baseline at 519 | 1.269e-09 | 1.663e-04 | 7.671e-02 | 13 |
| h measured on the Mac key (probe C) | 87520 | 85.4015 | 5.9952 | 521 | F=10: median relErrRms >= 1e-3 from 522 | 1.019e-09 | 1.335e-04 | 6.208e-02 | 16 |
| h measured on the Mac key (probe C) | 87520 | 85.4015 | 5.9952 | 525 | F=7: median relErrRms >= 1e-3 from 526 (F=10: >= 1e-2 from 526) | 7.589e-10 | 9.947e-05 | 4.663e-02 | 21 |
| h measured on the Mac key (probe C) | 87520 | 85.4015 | 5.9952 | 528 | F=7: >= 1e-2 from 529 (F=10: >= 0.1 from 529) | 6.077e-10 | 7.965e-05 | 3.751e-02 | 27 |
| h measured on the Mac key (probe C) | 87520 | 85.4015 | 5.9952 | 531 | F=7: >= 0.1 from 532 | 4.860e-10 | 6.371e-05 | 3.012e-02 | 33 |
| h measured on the Mac key (probe C) | 87520 | 85.4015 | 5.9952 | 534 | F=10: first decode exception at 535 | 3.883e-10 | 5.089e-05 | 2.413e-02 | 41 |
| h measured on the Mac key (probe C) | 87520 | 85.4015 | 5.9952 | 538 | F=7: first decode exception at 539 | 2.872e-10 | 3.765e-05 | 1.791e-02 | 56 |
| h measured on the Mac key (probe C) | 87520 | 85.4015 | 5.9952 | 545 | F=7: every trial throws from 546 | 1.686e-10 | 2.210e-05 | 1.055e-02 | 95 |
| h measured on the Mac key (probe C) | 87520 | 85.4015 | 5.9952 | 552 | F=7 and F=10: the WHOLE ciphertext is destroyed from 553 | 9.837e-11 | 1.289e-05 | 6.170e-03 | 162 |
| h = 2N/3 (expected value) | 87381 | 85.3338 | 6.0000 | 512 | edge of the fitted range (no measurable effect up to 515-517) | 1.904e-09 | 2.495e-04 | 1.129e-01 | 9 |
| h = 2N/3 (expected value) | 87381 | 85.3338 | 6.0000 | 518 | F=7 and F=10: median relErrRms first exceeds 2x baseline at 519 | 1.231e-09 | 1.614e-04 | 7.453e-02 | 13 |
| h = 2N/3 (expected value) | 87381 | 85.3338 | 6.0000 | 521 | F=10: median relErrRms >= 1e-3 from 522 | 9.883e-10 | 1.295e-04 | 6.028e-02 | 17 |
| h = 2N/3 (expected value) | 87381 | 85.3338 | 6.0000 | 525 | F=7: median relErrRms >= 1e-3 from 526 (F=10: >= 1e-2 from 526) | 7.359e-10 | 9.645e-05 | 4.524e-02 | 22 |
| h = 2N/3 (expected value) | 87381 | 85.3338 | 6.0000 | 528 | F=7: >= 1e-2 from 529 (F=10: >= 0.1 from 529) | 5.891e-10 | 7.721e-05 | 3.638e-02 | 27 |
| h = 2N/3 (expected value) | 87381 | 85.3338 | 6.0000 | 531 | F=7: >= 0.1 from 532 | 4.710e-10 | 6.173e-05 | 2.920e-02 | 34 |
| h = 2N/3 (expected value) | 87381 | 85.3338 | 6.0000 | 534 | F=10: first decode exception at 535 | 3.761e-10 | 4.930e-05 | 2.338e-02 | 43 |
| h = 2N/3 (expected value) | 87381 | 85.3338 | 6.0000 | 538 | F=7: first decode exception at 539 | 2.781e-10 | 3.645e-05 | 1.735e-02 | 58 |
| h = 2N/3 (expected value) | 87381 | 85.3338 | 6.0000 | 545 | F=7: every trial throws from 546 | 1.632e-10 | 2.138e-05 | 1.021e-02 | 98 |
| h = 2N/3 (expected value) | 87381 | 85.3338 | 6.0000 | 552 | F=7 and F=10: the WHOLE ciphertext is destroyed from 553 | 9.508e-11 | 1.246e-05 | 5.964e-03 | 168 |
