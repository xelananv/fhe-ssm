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
