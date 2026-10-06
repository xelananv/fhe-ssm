ring N = 4096, a NEW stock uniform-ternary key (Hamming weight 2814, sigma_pred = 15.3161), class `evaluated` only, source sha256 6c9c5f231b348b4b, argv `harness/build/overflow_tail_probe2 --log-ring 12 --cts 60000 --classes evaluated`; status: complete

| ciphertexts | samples | sigma / sigma_pred | max|I| (in sigma) | chi^2 of per-ciphertext variances, correlated SE (dof; upper-tail p) | |I| > 3 sigma (b = 45) | |I| > 3.5 sigma (b = 53) | |I| > 4 sigma (b = 61) | |I| > 4.5 sigma (b = 68) | |I| > 5 sigma (b = 76) | |I| > 5.5 sigma (b = 84) | |I| > 6 sigma (b = 91) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 60000 | 245760000 | 0.99994 | 87 (5.680) | 59725 (60000; 0.786) | 727856 / 730141.00 (p_hi 0.996) | 116688 / 117358.00 (p_hi 0.975) | 14567 / 14585.90 (p_hi 0.563) | 1869 / 1900.75 (p_hi 0.77) | 126 / 144.79 (p_hi 0.948) | 6 / 8.47 (p_hi 0.848) | 0 / 0.57 (p_hi 1) |

Ciphertexts whose final max|I| reached the logging threshold (5.3 sigma): 22. For each: the overflow at the SAME coefficient one step earlier (`scalar` stage = the ciphertext before the last rescale), the largest max|I| over the 11 earlier stages of that ciphertext, and the decrypt check.

| ct | I at argmax | in sigma | argmax j | that ciphertext's sigma_hat | # |I| > 4 sigma in it | frac at argmax | I at the same j before the last rescale | largest stage max|I| | relErrRms of the decrypted result vs the clear computation |
|---|---|---|---|---|---|---|---|---|---|
| 6246 | +87 | 5.68 | 2959 | 15.26 | 1 | +1.38e-04 | -31 | 67 | 6.0e-14 |
| 1954 | +86 | 5.62 | 3798 | 15.51 | 1 | -3.60e-03 | +7 | 64 | 7.7e-14 |
| 20455 | +86 | 5.62 | 2275 | 15.34 | 1 | +1.84e-03 | -18 | 67 | 6.6e-14 |
| 31300 | -86 | 5.62 | 3427 | 15.64 | 1 | +2.88e-03 | -13 | 69 | 5.9e-14 |
| 53466 | -86 | 5.62 | 2049 | 15.49 | 2 | +3.34e-03 | -8 | 64 | 6.0e-14 |
| 19389 | +85 | 5.55 | 2034 | 15.45 | 1 | -1.91e-03 | +15 | 74 | 5.9e-14 |
| 17226 | +84 | 5.48 | 2666 | 15.40 | 1 | -3.06e-03 | -25 | 65 | 8.1e-14 |
| 41585 | +84 | 5.48 | 3999 | 15.12 | 2 | -3.59e-03 | -12 | 64 | 9.6e-14 |
| 44288 | -84 | 5.48 | 2378 | 15.32 | 1 | -2.83e-03 | +1 | 63 | 7.0e-14 |
| 55036 | -84 | 5.48 | 530 | 15.12 | 1 | -1.54e-03 | +5 | 65 | 7.3e-14 |
| 43004 | +83 | 5.42 | 2090 | 15.32 | 2 | +2.41e-03 | -6 | 61 | 1.1e-13 |
| 48025 | +83 | 5.42 | 3683 | 15.37 | 1 | +1.25e-04 | -1 | 62 | 9.6e-14 |
| 48416 | -83 | 5.42 | 2933 | 15.36 | 2 | -2.35e-03 | -10 | 70 | 6.7e-14 |
| 1573 | +82 | 5.35 | 3602 | 15.43 | 1 | -5.20e-04 | +0 | 64 | 8.3e-14 |
| 12965 | +82 | 5.35 | 2241 | 15.29 | 1 | -1.25e-03 | -8 | 66 | 7.7e-14 |
| 23736 | +82 | 5.35 | 1028 | 15.35 | 1 | -1.51e-03 | -9 | 67 | 6.2e-14 |
| 24718 | +82 | 5.35 | 458 | 15.26 | 1 | -6.84e-05 | -25 | 60 | 6.3e-14 |
| 32055 | +82 | 5.35 | 164 | 15.34 | 1 | +2.88e-03 | -22 | 62 | 7.0e-14 |
| 34587 | +82 | 5.35 | 1066 | 15.41 | 2 | -2.35e-03 | -18 | 62 | 6.9e-14 |
| 35293 | -82 | 5.35 | 3607 | 15.72 | 1 | -2.66e-03 | +24 | 73 | 8.1e-14 |
| 39940 | -82 | 5.35 | 3504 | 15.31 | 1 | +2.53e-03 | -14 | 71 | 8.5e-14 |
| 52437 | +82 | 5.35 | 16 | 15.52 | 1 | -5.15e-03 | -10 | 68 | 8.5e-14 |

signs: 15 positive, 7 negative (two-sided binomial p = 0.134); argmax in the upper half of the ring (j >= N/2): 15 of 22
expected number of ciphertexts with max|I| >= 82 under the Gaussian model: 25.33 (observed 22)
