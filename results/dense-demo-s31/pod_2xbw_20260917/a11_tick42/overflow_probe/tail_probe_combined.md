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
