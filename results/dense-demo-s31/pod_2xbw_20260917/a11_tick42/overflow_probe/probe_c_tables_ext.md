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

