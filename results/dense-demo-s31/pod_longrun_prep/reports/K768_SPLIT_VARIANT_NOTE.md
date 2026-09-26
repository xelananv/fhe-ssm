# K = 768 on the pod: why there are now two forms of the switch, and what was checked offline (2026-09-18 evening)

Nothing in this note ran on a GPU. Measured numbers are transcribed from
`../../pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.md` (OpenFHE 1.5.1 CPU bootstrap, rings 2^12 / 2^13); ARITHMETIC is marked.

## 1. What the CPU confirmation changed
The extended table does on the failure side what the arithmetic predicted (old zone 500..766 normal, new edge 775 / 788 / 798). It does NOT
cost "at most 1.5x" in precision: the straight swap (one scalar `pre/(768 N)`) measured **2.43x** the stock table's relErrRms at N = 2^13 / F = 10
(2.45x at F = 7, 3.35x at N = 2^12). One part of that has a found and measured cause:

- `EvalMult(ct, double)` encodes the constant as the INTEGER `round(c x SF)` (SF = the raised level's scaling factor = FirstPrime(59, 2N)). For
  K = 512 the constant is a power of two; for 768 = 3 x 2^8 it is not, and the rounding of the 1/3 is a relative scale error eta of the sine's
  input: every coefficient's overflow I is seen as I(1 + eta), the sine returns I x eta instead of 0, x 2^F in the plaintext. Measured slope at
  N = 2^13: +7.31e-12 (335 of 335 planted trials with the predicted sign); arithmetic +7.2475e-12. At N = 2^12 the arithmetic says the sign
  flips (-3.67e-12) and the measurement flips with it (-3.75e-12, 42 of 42).
- **ARITHMETIC at the demo's ring** (an independent exact-rational check of the probe's table, same values): FirstPrime(59, 2^18) =
  2^59 + 12,058,625; eta = **+9.55e-11 for the scalar 1/(768 N)**, **-2.09e-11 for 1/(512 N) (the stock path is not exact there either) and for
  1/(256 N)**. FIDESlib builds the same constant with the same rounding (`Bootstrap.cu`: `constantEvalMult = pre * (1.0 / (k * cc.N))`,
  `Context.cu` `ElemForEvalMult`: `static_cast<int128>(operand / approxFactor * scFactor + 0.5)`): source reading, not run.
- The fix that was RUN on CPU (variant B): the 3 goes into the CoeffsToSlots precomputation (`scaleEnc = pre / 3`, the place where
  the library folds the whole K of its sparse cases), the scalar becomes `pre/(256 N)`. Measured at N = 2^13: slope -0.6e-13 +- 0.9e-13 (gone),
  relErrRms **1.78x** the stock table's instead of 2.43x, same depth / towers / output level / edge thresholds. What remains is noise on the
  coefficients whose own overflow is 0, +-1, +-2 (rms error 7.4x the control's at I = 0, 1.3x from |I| >= 9; coefficients with I = 0 are 1.9 %
  of all at sigma_I = 21 and about 0.5 % at the demo's sigma_I = 85, so that share should weigh less at 2^17: not measured).
- An estimate of what the straight swap would do on the GPU, NOT a measurement (the factor 2 = q0/Delta is checked on the CPU
  records: variant A minus variant B in quadrature, sqrt(4.49e-7^2 - 3.29e-7^2) = 3.05e-7 measured against 2 x 1.58e-7 = 3.17e-7 predicted at
  N = 2^13 / F = 10): per-coefficient error 2 x eta x sigma_I x 2^F =
  2 x 9.55e-11 x 85.3 x 128 = 2.1e-6, x sqrt(N) = 7.6e-4 rms per slot, about 2.6e-3 as a max over 1,024 channels, against the 2.0e-3 the
  two-card selftest recorded for ONE stock bootstrap last session (`{"selftest":"bootstrap","err":0.00199027}` / `0.00206614`, max-abs). The
  same estimate for the stock and the split constants: 5.8e-4. I.e. the straight swap is expected to be visibly less precise at 2^17, the
  split form about as precise as the stock table plus the Ext table's own 1.3-1.8x.

## 2. What was authored
| piece | what | where |
|---|---|---|
| **O1** (OpenFHE, 29 insertions / 2 deletions, 2 files) | `FIDESLIB_BOOT_UNIFORM_EXT=split` makes `EvalBootstrapSetup` and `EvalBootstrapPrecompute` use `scaleEnc = pre / 3` for `UNIFORM_TERNARY` (the functional-bootstrap setup is left alone); prints `{"openfheBootScaleEnc":"pre/3",…}` once; defines the macro `OPENFHE_FHESSM_BOOT_UNIFORM_SPLIT_ENV` in `ckksrns-fhe.h`. Any other value of the variable: stock behaviour | `patches/openfhe/O1_openfhe_boot_scaleenc_split_env.patch` |
| **K1** (FIDESlib, regenerated) | `=1`: Ext table, `bootK = 768` (as before). `=split`: Ext table, `bootK = 256`, ONLY if the linked OpenFHE carries O1 (the macro); otherwise it falls back to `=1` and prints `splitAskedButOpenfheLacksO1:true`. The JSON line now carries `tableK` (512 / 768: the overflow bound), `bootK` (what the scalar divides by), `scaleEncOddPart` (1 / 3) | `patches/K1_boot_uniform_ext_env.patch` |
| build hook | `OPENFHE_EXTRA_PATCH_DIR=<dir>`: FIDESlib's `deps/build.sh` wipes and re-checks-out the OpenFHE source, so the rider inserts a `git apply` loop INTO it, right after FIDESlib's own OpenFHE patch. A patch that does not apply is skipped loudly and the build goes on (K1 then falls back to `=1` by itself); a `build.sh` without the expected anchor line is fatal before anything compiles. After the OpenFHE install the rider says whether the macro is in the installed headers | `campaign/scripts/rider_build_fideslib.sh` |
| cell 1b | two-card selftest under the three arms (`0`, `1`, `split`), twice each, one table (table / tableK / bootK / scaleEncOddPart / O1 line / degree / selftest bootstrap err / lvlPost / bootRestored / hardFail) and the arm for the long run by a rule written down beforehand: `split` if its line shows `scaleEncOddPart 3`, every repeat passes and its mean err <= 2x the stock arm's; else `1` under the same conditions; else STOP. Writes `/root/demo/BOOT_ARM`, which `longrun.env` sources (precedence: explicit env > BOOT_ARM > `1`) | `pod_scripts/boot_arms_selftest.sh`, `pod_scripts/longrun.env` |

One variable, two readers: the OpenFHE precomputation and FIDESlib's scalar cannot disagree unless O1 is missing from the build, and that case
is decided at compile time by the macro. The remaining way to be wrong (O1 present but its scaling not reaching the GPU plaintexts) makes the
sine's input 3x too large: every bootstrap is garbage and the 40-second selftest hard-fails before anything is paid for.

Why this is expected to work in FIDESlib: its SPARSE path already runs exactly this shape (`RawCiphertext.cu` SPARSE branch: "do not divide by
k as we already did it during precomputation", `bootK = 1.0` with the K folded into OpenFHE's precomputation), and the CoeffsToSlots plaintexts
are imported as OpenFHE made them (`AddBootstrapPlaintexts`: `precom->m_U0hatTPreFFT`). `bootK` is read in exactly two places, both
`constantEvalMult = pre * (1.0 / (k * cc.N))` (`Bootstrap.cu:79-85`, `:222-235`).

## 3. Offline checks (all light; one at a time; no GPU)
- **O1 was written against the pod's exact OpenFHE files.** `vendor/openfhe-fideslib/openfhe-src` holds `ckksrns-fhe.cpp` with git blob `060f9b6e…`
  and `ckksrns-fhe.h` with blob `765a33f4…`: the POST-images named in the index lines of FIDESlib's own `deps/fideslib-ref-1.5.1.1.patch` at the
  pinned commit `fa972864` (`index e2223e8a..060f9b6e`, `index acc0af7a..765a33f4`). `git apply --check` of O1 on those two files: applies. (The
  vendored `vendor/openfhe-development` is a different revision of that file, blob `96c2363e…`; the CPU probe's patches were made against it and
  are NOT what the pod gets.)
- O1's translation unit passes `clang++ -std=c++17 -fsyntax-only -Wall -Werror` against the pod-identical tree (and the same command FAILS on
  a copy with a deliberately undeclared identifier, so the check has teeth). Apple clang, not the pod's GCC.
- The build hook: the inserter run twice on a copy of upstream's `deps/build.sh` (idempotent, `bash -n` clean, the loop lands right after
  `git apply ../fideslib-ref-1.5.1.1.patch`); the inserted loop run on a scratch git tree of the pod-identical files: `OPENFHE-EXTRA-PATCH
  applied`, and on the already-patched tree: `DOES NOT APPLY, SKIPPED`.
- K1's selection block compiled stand-alone (C++20, `-Wall -Werror`, stand-in `lbcrypto::FHECKKSRNS`) with and without the macro and run with
  the variable unset / `0` / `1` / `split`: the eight lines are as designed (`split` without the macro: table 768, `bootK` 768,
  `splitAskedButOpenfheLacksO1:true`; with it: `bootK` 256, `scaleEncOddPart` 3).
- L1 + K1 + K2 still apply together to pristine `fa972864` (`git apply --check` each; they touch different files).
- `boot_arms_selftest.sh`: its decision block run on fabricated console logs for three cases (all pass -> `split`; O1 missing -> `1`; both Ext
  arms garbage -> no arm, STOP). `bash -n` clean. Untested against a real selftest log until the pod.

## 4. What stays unverified until the pod
That nvcc/GCC compile K1 and O1; that FIDESlib's polynomial evaluator takes degree 118 at the same levels (`lvlPost` 19: emulation only,
`k768_ps_level_emulation.txt`); the GPU precision of each arm; the GPU's own overflow edge. Cell 1b answers the first three in about five minutes.
