# K = 768 on CPU: what OpenFHE's own bootstrap does with `g_coefficientsUniformExt` (2026-09-18)

**Scope.** CPU, a PATCHED COPY of the vendored OpenFHE v1.5.1 (`vendor/openfhe-k768-src` -> `vendor/install-k768`; the stock tree and
`vendor/install` are untouched), single-threaded, rings 2^13 and 2^12 (`HEStd_NotSet`), `UNIFORM_TERNARY`, `FLEXIBLEAUTO`, scaling 59 bits, first
modulus 60 bits, level budget {3,3}, full packing, depth 23: the parameters of the previous planted sweeps (`../PROBE_B_REPORT.md`). The probe is
`harness/boot_overflow_probe.cpp` with its planted-overflow instrument unchanged. This is the library whose table and K FIDESlib imports; it is
NOT the GPU bootstrap, and nothing was run at ring 2^17. Every number in the tables is generated from the JSONL records of this directory
by `analyze_k768.py` -> `k768_tables.md` and pasted here by `assemble_k768_report.py`; numbers in the prose are transcribed from those tables.
Columns and paragraphs marked ARITHMETIC are float64 / rational arithmetic on the library's constants, not measurements. No timing is quoted (R6).

## 0. Answer

1. **With the switch off the patched build is the stock library**: the control sweep reproduces the previous thresholds (section 2.1).
2. **With the Ext table the old failure zone is gone**: 275 planted bootstraps with 500 <= |I*| <= 766 at N = 2^13 / F = 10 (every integer
   500..560, every 5 to 600, every 20 to 760, 760..766), 0 decode exceptions, relErrRms 6.97e-05..8.44e-05 against that arm's un-planted median
   of 7.07e-05; the control in the same zone: 79 decode exceptions in 186 bootstraps (section 2.3).
3. **The new edge is where the float arithmetic put it** (N = 2^13, F = 10): first degraded at 775, relErrRms >= 1e-3 from 777, >= 0.1 from 782,
   first decode exception at 788 (all trials from 788), whole ciphertext destroyed from 798; the error at the overflowing coefficient equals the
   float64 re-evaluation of the Ext polynomial x 2^F (section 2.4). Same bootstrap depth, same towers, same output level as the control (2.0).
4. **Two costs that the arithmetic did NOT predict, both measured:**
   (a) **an error that is linear in the coefficient's overflow, I x 7.3e-12 x 2^F at N = 2^13** (335 of 335 planted trials have the predicted
   sign; slope +7.31e-12 +- 0.04e-12; at N = 2^12 it is NEGATIVE, -3.75e-12 +- 0.04e-12, 42 of 42). Both are reproduced to 1-2 % by the
   rounding of ONE scalar constant: the library divides the raised
   ciphertext by K with `EvalMult(ct, pre/(K*N))`, which encodes the constant as the integer round(c x SF); for K = 512 that constant is a power
   of two, for K = 768 = 3 x 2^8 it is not, and the rounding error of 1/3 becomes a relative scale error of the sine's input (section 2.5).
   **FIDESlib builds the same constant with the same rounding (source reading, not run); the same arithmetic at N = 2^17 gives a scale error 13x
   larger than at 2^13.** A second patch (variant B: the factor 3 folded into the CoeffsToSlots plaintexts, scalar constant 1/(256 N)) removes
   the linear error on CPU (slope -0.6e-13 +- 0.9e-13, 108 trials) at the same depth and with the same edge thresholds.
   (b) **a lower baseline precision**: median relErrRms 7.07e-05 (Ext) vs 2.91e-05 (control) at N = 2^13 / F = 10, 40 un-planted trials each:
   2.43x = 1.28 bits (2.45x at F = 7; 3.35x = 1.75 bits at N = 2^12); 1.78x = 0.83 bits with variant B. What remains after the split sits on
   the coefficients whose own overflow is 0, +-1, +-2 (random sign): rms error 7.4x the control's at I = 0, 1.1-1.4x from |I| >= 9 (section 2.2).
5. **What this does not show:** the GPU. FIDESlib imports the same table and K but evaluates the polynomial with its own code; its thresholds, its
   precision with the Ext table and the size of 4(a) there were not run.

## 1. What was built

### 1.1 The patch (`openfhe_k768_env.patch`, `diff -ru` pristine vs copy: one file, `src/pke/lib/scheme/ckksrns/ckksrns-fhe.cpp`)

`OPENFHE_BOOT_UNIFORM_EXT=1`, read with `std::getenv` at every selection site (one build serves both arms; unset or any other value = stock
behaviour), makes the `UNIFORM_TERNARY` path select `g_coefficientsUniformExt` / `K_UNIFORMEXT`. When the switch is on the library prints ONE line
on stderr per process with the degree of the vector it actually uses and the k it actually uses (so each record's `.stderr` file proves its arm):
`[openfhe-k768-patch] EvalBootstrap: OPENFHE_BOOT_UNIFORM_EXT=1 -> Chebyshev degree 118, K = 768`. The header
(`src/pke/include/scheme/ckksrns/ckksrns-fhe.h`) is not modified; the installed headers of `vendor/install-k768` are byte-identical to those of
`vendor/install` (`diff -rq`, in `build_openfhe_k768.log` / checked again in `build_variantB.log`).

Every site where the uniform table or K is chosen, or where K enters a precomputation or a scaling (line numbers of the PRISTINE
`vendor/openfhe-development/src/pke/lib/scheme/ckksrns/ckksrns-fhe.cpp`; found with `grep -n "K_UNIFORM\|g_coefficientsUniform\|R_UNIFORM\|k = \|1.0 / (k"`):

| site (pristine file:line) | what it does | patched? |
|---|---|---|
| `ckksrns-fhe.cpp:626-635` (`EvalBootstrap`, condition at `:628`) | `coefficients = g_coefficientsUniform; k = K_UNIFORM` unless composite scaling at N >= 2^17 | **yes**: the Ext branch is forced when the switch is on |
| `ckksrns-fhe.cpp:638` | `cc->EvalMultInPlace(raised, pre * (1.0 / (k * N)))`: the ONLY place K scales anything for a uniform secret | follows `k` (no edit needed) |
| `ckksrns-fhe.cpp:976-985` (`EvalBootstrapStCFirst`, condition at `:978`) and `:1113-1115` (`normalization = pre * (1.0 / (k * N))`) | the same selection and scaling on the slots-encoding route (`BTSlotsEncoding`; dispatched from `EvalBootstrap` at `:434-435`) | **yes**, same edit. NOT exercised by the probe (it uses the default route) |
| `ckksrns-fhe.cpp:2222-2226` (`GetModDepthInternal`) | depth of the approximate modular reduction = `GetMultiplicativeDepthByCoeffVector(g_coefficientsUniform) + R_UNIFORM`; feeds `GetBootstrapDepth` (`:2206-2211`) and `lDec` in the precomputation (`:212`, `:378-379`) | **yes**: uses the table in use. The value does not change: degree 88 and 118 are both in the bucket 60..119 -> 8 (`ckksrns-utils.cpp:82-104`) |
| `ckksrns-fhe.cpp:177-190` + `:197` (`EvalBootstrapSetup`) and `:343-356` + `:363` (`EvalBootstrapPrecompute`) | `k = 1.0` for `UNIFORM_TERNARY`, `scaleEnc = pre / k`: K is NOT folded into the CoeffsToSlots precomputation for a uniform secret (it is for the sparse ones) | no (nothing depends on K here). Variant B (section 2.5) puts the 3 of 768 here |
| `ckksrns-fhe.cpp:699-705`, `:791`, `:1193`, `:2201-2202` | `R_UNIFORM = 6` double-angle iterations for `UNIFORM_TERNARY`, whatever the table (upstream uses the same R with the Ext table on its composite route) | no |
| `ckksrns-fhe.cpp:2875-2889` + `:2902` (`EvalFBTSetupInternal`) and `:3095-3097` (`EvalMVBPrecomputeInternal`: `k = ... : K_UNIFORM; EvalMultInPlace(raised, 1.0 / (k * N))`) | the FUNCTIONAL bootstrap; it evaluates its own tables (`coeff_cos_25_double`, `coeff_exp_25_double_*`), never `g_coefficientsUniform*` | no: not the CKKS bootstrap, left as upstream wrote it |
| `ckksrns-fhe.h:424`, `:426`, `:428`, `:463`, `:489` | the constants and the two tables (89 and 119 coefficients) | no |

Does anything else key on the degree? `EvalChebyshevSeries` -> `internalEvalChebyPolysPS` (`ckksrns-advancedshe.cpp:763-767`) takes its
Paterson-Stockmeyer parameters from `ComputeDegreesPS(degree)` (`ckksrns-utils.cpp:299-330`): ARITHMETIC on that table, (k, m) = (6, 4) for degree
88 and (8, 4) for degree 118, i.e. baby steps T_1..T_6 / giant steps T_12, T_24, T_48 against T_1..T_8 / T_16, T_32, T_64; depth ceil(log2 k) + m
+ 1 = 8 for both. Measured: `GetBootstrapDepth({3,3}, UNIFORM_TERNARY)` evaluated inside each arm's process is 20 in both, the context has 24
towers in both, and every bootstrap of both arms goes from 2 towers to 5 towers, output level 19, noise scale degree 2 (table T0 below).

### 1.2 Build identity

- Library A (the task's patch): source copy `vendor/openfhe-k768-src` (`ckksrns-fhe.cpp` sha256 `274f014919a9...`, pristine `e292ad178dde...`, header
  `eb18043004b8...` unchanged), build tree `vendor/build-k768`, install `vendor/install-k768`; `libOPENFHEpke.1.5.1.dylib` sha256 `f96529cc99c9...`,
  `libOPENFHEcore` `f8cc9ff0b19c...`. cmake options copied from `harness/build_openfhe.sh` (Release, `WITH_OPENMP=OFF`, no tests / examples /
  benchmarks / extras, Ninja, pip cmake 4.3.4 / ninja 1.13.0, Apple clang 21.0.0); compile flags of `ckksrns-fhe.cpp` identical to the stock build's
  (`-Wall -Werror -DOPENFHE_VERSION=1.5.1 -O3 -DMATHBACKEND=4 -Wno-unknown-pragmas -O3 -DNDEBUG -std=gnu++17 -arch arm64 -fPIC`, compared in both
  `build.ninja`). `tools/memguard.sh 8 ninja -C vendor/build-k768 -j4 install`, 114 steps, no guard trip: `build_openfhe_k768.sh`,
  `build_openfhe_k768.log`. The dylib hashes differ from the stock install's even where the source is identical (the library embeds `__FILE__`
  paths in its exception texts), which is why equivalence is shown by behaviour (section 2.1), not by hash.
- Probe: `harness/boot_overflow_probe.cpp` sha256 `c760781be72b...` (was `da7c0fd6127e...`; additions in 1.3), built by hand with the compile and link
  commands CMake generated for the stock probe (`harness/build/build.ninja`), include / lib / rpath pointed at `vendor/install-k768`:
  `harness/build/boot_overflow_probe_k768`, sha256 `853ce9cd6d6c...` (`build_probe_k768.sh`, `build_boot_overflow_probe_k768.log`: two builds, the
  second is the binary of record; `DYLD_PRINT_LIBRARIES` shows the three dylibs loaded from `vendor/install-k768/lib`). `harness/build`'s own ninja
  graph and `harness/build/boot_overflow_probe` (`bcaaea2a47d6...`, stock install) were not touched.
- Library B / probe B (section 2.5 only): `vendor/openfhe-k768b-src` (`ckksrns-fhe.cpp` `c1a0acbf6455...`) -> `vendor/install-k768b`
  (`libOPENFHEpke` `acc0dc220ca2...`), same probe source -> `harness/build/boot_overflow_probe_k768b` (`c5ae6e9c167d...`): `build_variantB.sh`,
  `build_variantB.log`, patch `openfhe_k768_split_variantB.patch` (diff against library A's source).
- All hashes in one file: `identity_sha256.txt`; they are also printed at the top of every `loadstate_*.txt`.
- Load state: `loadstate_main.txt`, `loadstate_splitB.txt`, `loadstate_extra.txt` (`tools/mem_probe.sh` + `uptime` before and after each battery:
  nothing else that links or compiles OpenFHE; swap 1,953 MB before and after the main battery). One probe process at a time, each under
  `caffeinate -i tools/memguard.sh 6`; footprint of a running ring-2^13 probe 1.0 GB (`footprint_samples.txt`). Light Python analysis ran beside
  the batteries, so `bootMs` in the records is not a clean timing and none is quoted.

### 1.3 What was added to the probe (no change to its logic; diff in `boot_overflow_probe_additions.diff`)

The probe hard-codes `const uint32_t K = 512` for three REPORTING fields (header `K`, `KoverSigma`, the per-trial counter `nOverK`); K enters
neither the planting, nor the sweep range (`--plant lo:hi:step` already was an option), nor any threshold (classification is done by the analysis
scripts). Added:

- `--k-bound N` (default 512): the value of that label. The probe refuses to run (exit 2) when the label contradicts the environment
  (`--k-bound 768` without `OPENFHE_BOOT_UNIFORM_EXT=1`, or the reverse), so a mislabelled record cannot be written.
- header field `envUniformExt`: the variable as the process sees it.
- trial field `topCoefErrI`: the exact overflow I_j of the six coefficients with the largest error (the record already listed those six).
- `--err-by-absI W` (default off): one extra record per trial with the decode-free coefficient error aggregated in bins of |I_j| (count, sum of
  squares, max). It answers "is the error a function of the coefficient's own overflow?", which is what separates the two arms in section 2.2.

## 2. Results

### 2.0 Every record file: arm, depth, sanity (4b-iv)

{{SECTION:## T0.}}

`bootRestored` (output towers > input towers) is true on every one of the bootstraps, `adjust==library` (the probe's exact overflow is computed
on the ciphertext the library really raises) on every one, and `EvalBootstrap` itself never threw. Depth 23, `bootDepth` 20, 24 towers, 2 -> 5
towers, output level 19, noise scale degree 2: identical in the control, the Ext arm and variant B.

### 2.1 CONTROL (4a): with the switch off the patched build reproduces the stock build's thresholds

{{SECTION:## T1.}}

Same argv as the earlier main sweep except for the binary. Thresholds 519 / 522 / 525 / 529 / 534 / 553 against 519 / 522 / 526 / 529 /
535 / 553 (different random keys and ciphertexts, 3 trials per I*); "all trials throw from" differs (536 vs 548) because the previous sweep had
one trial of three that still decoded at 547. The per-I* comparison is table T1b of `k768_tables.md`. The un-planted baseline agrees too (next
table: 2.914e-05 vs 2.854e-05 on the stock build).

### 2.2 Baseline precision without planting (4b-i)

{{SECTION:## T2.}}

Decode-free coefficient error against the coefficient's OWN overflow |I_j| (`errByAbsI` records pooled over the 40 trials of each file; N = 2^13,
F = 10; the same tables for F = 7 and N = 2^12 are in `k768_tables.md`, T2b). Control vs Ext (library A):

{{SECTION:### ctrl_r13_F10_base.jsonl (stock table) vs ext_r13_F10_base.jsonl}}

Control vs Ext + split (variant B, section 2.5):

{{SECTION:### ctrl_r13_F10_base.jsonl (stock table) vs B_split_r13_F10_base.jsonl}}

{{SECTION:## T2c.}}

Reading:

- N = 2^13, F = 10, 40 trials per arm, same binary: median relErrRms **2.914e-05 (stock table) -> 7.066e-05 (Ext table): 2.43x, 1.28 bits**. No
  decode failure in any arm. Variant B (section 2.5) gives 5.179e-05: 1.78x, 0.83 bits. F = 7: 3.598e-06 -> 8.826e-06 (2.45x). N = 2^12, F = 11:
  1.479e-05 -> 4.960e-05 (3.35x, 1.75 bits; variant B 4.335e-05, 2.93x).
- The control's coefficient error does not depend on the coefficient's own overflow (rms 1.85e-07 in every |I_j| bin, 2.02e-07 at I_j = 0). The Ext
  arm's does, in two ways: it **rises with |I_j|** (3.5e-07 at |I_j| 5..8 -> 6.0e-07 at 65..96; this is cost (a), it disappears in variant B: 2.6e-07
  -> 2.3e-07), and it **peaks on the coefficients whose overflow is 0** (1.58e-06 = 7.8x the control; +-1: 5.4e-07; +-2: 4.0e-07), where it stays
  in variant B (1.50e-06 at 0). In the Ext arm 134 of the 240 largest errors of the 40 trials sit on coefficients with I_j = 0 (1.9 % of all
  coefficients); in the control 13 of 240. The signs of those errors are balanced (68 positive, 66 negative in `ext_r13_F10_base.jsonl`): noise, not
  a bias.
- Candidate explanation of the I_j = 0 peak, NOT tested: noise injected before or inside the double-angle iterations is amplified by
  prod 4 a_i |cos(2^i theta_0)| = 4^R a / (2^R |sin theta_0|), theta_0 = 2 pi (I - 1/4) / 2^R, which is largest at I = 0 and falls to about a third at
  |I| = 1 and an eighth at |I| = 2: arithmetic ratios 0.275 and 0.129; measured excesses over the |I| = 5..8 bin (in quadrature) 1.54e-06 at 0 and
  0.261 / 0.118 of that at 1 / 2 (`ext_r13_F10_base`; 0.276 / 0.148 at F = 7, 0.267 / 0.104 with the split, 0.268 / 0.096 at N = 2^12: last table above). Why the degree-118 evaluation injects more of it than the degree-88 one was not investigated (the Paterson-Stockmeyer
  giant steps are T_16, T_32, T_64 instead of T_12, T_24, T_48; at x ~ 0 every |T_2^j| is 1, so each doubling multiplies the noise it carries by 4).
  This part is a property of OpenFHE's polynomial evaluator and says nothing about FIDESlib's.

### 2.3 The OLD failure zone under the Ext table (4b-ii)

{{SECTION:## T5.}}

{{SECTION:## T5b.}}

N = 2^13, F = 10: every integer 500..560, every 5 to 600, every 20 from 100 to 760, 760..766, plus the few in-range points of the negative and
second-half sweeps: **275 bootstraps with 500 <= |I*| <= 766, no decode exception, no trial above 8.44e-05**, where the control throws in 79 of
186 and is destroyed from 553. The same holds at F = 7 (36 bootstraps) and at N = 2^12 (21). The planted trials' median (7.59e-05) sits 7 % above
the un-planted baseline (7.07e-05): the planting instrument narrows the overflow distribution of the OTHER coefficients (sigma_I 16.5 instead of
21.3, column above), which puts more of them at I = 0, where the Ext arm's noise sits (section 2.2): a likely reason, not tested. The reference for
a planted trial is therefore the sweep's own baseline (T3, T5), against which the largest per-I* median is 1.10-1.11x. The only trace of the
planted coefficient itself is cost (a): its own error is 16-25x the floor (4.0e-06 at 511..599) but one coefficient of 8,192 at that level does not
move relErrRms.

### 2.4 The NEW edge (4b-iii)

{{SECTION:## T3.}}

Per I*, N = 2^13, F = 10, rows 766..806 of `ext_r13_F10_planted_edge.jsonl` (the F = 7 and N = 2^12 edges are in `k768_tables.md`, T4e; every full
sweep in T4). `share` = (e_j*^2 + e_(j*+N/2)^2) / sum_j e_j^2 over the decode-free coefficient errors; `floor` = median |coefficient error| over all
N coefficients; `model` = the Ext polynomial re-evaluated in float64 at I*/768, times 2^F (ARITHMETIC):

{{SECTION:### ext_r13_F10_planted_edge.jsonl}}

- N = 2^13, F = 10: relErrRms > 2x baseline from **775**, >= 1e-3 from **777**, >= 1e-2 from 780, >= 0.1 from **782**, first decode exception
  ("approximation error is too high") at **788** and every trial from 788, the whole ciphertext destroyed from **798** (median coefficient error
  1.3e+02 at 798, 2.6e+41 at 803). Against the stock table's 519 / 522 / 525-526 / 529 / 534-535 / 553: first degraded at K + 7 in both, destroyed
  at K + 41 (stock) against K + 30 (Ext): the Ext polynomial leaves its range faster, as the float arithmetic had it
  (`../../boot_overflow_ext_table.txt`: injected error 1e-6 / 1e-4 / 1e-2 of the sine output at 778 / 784 / 790; stock 523 / 530 / 537).
- The error at the overflowing coefficient is the Ext polynomial evaluated outside its range times 2^F: measured / model median 1.22 over
  774 <= I* <= 796 (69 bootstraps); row by row in T4e (e.g. 780: 8.2e-03 vs 7.8e-03, 790: 2.1e+01 vs 1.8e+01, 798: 2.6e+19 vs 8.0e+17).
  Below 772 the model is below the floor and what the column `|coef err at j*|` shows (medians 5.3e-06 to 6.4e-06 for 766..771) is cost (a), not
  the polynomial.
- F = 7, ring 2^12, the negative side and a second-half coefficient: section 2.6.

### 2.5 Not predicted: an error linear in the overflow, its cause, and a fix that was run

{{SECTION:## T6.}}

{{SECTION:## T7.}}

Measured. Inside the fitted range the control's error at the planted coefficient is at the floor whatever I* (ratio to floor 0.7-1.3 from 100 to
510). Under the Ext table it **has the sign of I* x eta in 335 of 335 trials at N = 2^13 / F = 10 and is proportional to I***: 8.9e-07 at 100..199,
4.0e-06 at 511..599, 5.5e-06 at 700..766 (floor 2.2e-07); least-squares slope / 2^F = +7.327e-12 +- 0.061e-12 over 100..760 (102 trials),
+7.303e-12 over 500..560 (183 trials), pooled +7.307e-12 +- 0.041e-12. At F = 7 the same eta (+6.88e-12 +- 0.08e-12, 36 of 36), i.e. an error 8x
smaller in message units. **At N = 2^12 the sign is NEGATIVE: -3.749e-12 +- 0.041e-12, 42 of 42 trials.** The un-planted trials show the same law
on ordinary coefficients (table T2b: the rise with |I_j|).

Cause, ARITHMETIC that reproduces the measurement (`k768_const_rounding.py`, `.txt` below). `EvalBootstrap` divides the raised ciphertext by
K with one scalar, `EvalMultInPlace(raised, pre * (1.0 / (k * N)))` (`ckksrns-fhe.cpp:638`). `EvalMult` by a double encodes the constant as the
INTEGER `static_cast<int128>(c * scFactor + 0.5)` (`ckksrns-leveledshe.cpp:441-...`, `GetElementForEvalMult`, 64-bit build) with scFactor = the
level-0 scaling factor = the last modulus of the chain = FirstPrime(59, 2N) = 2^59 + 16385 at both rings (source reading:
`ckksrns-cryptoparameters.cpp:103`, `ckksrns-parametergeneration.cpp:419-420`; the probe does not print it). For K = 512, N = 2^13, pre = 2^-deg = 1/2
(`deg` 1 in every header record):
c x SF = 2^36 + 0.002, rounding error 0.002 of 6.9e10: eta = -2.8e-14. For K = 768: c x SF = 2^37/3 + 0.0013 = 45812984490.668, rounded UP by 0.332:
**eta = +7.2475e-12** (measured pooled +7.307e-12, ratio 1.008); at N = 2^12 c x SF = 91625968981.336, rounded DOWN by 0.336: **eta = -3.6664e-12**
(measured -3.749e-12, ratio 1.023). The
constant actually applied is c (1 + eta), the sine sees I (1 + eta) instead of I and returns I x eta instead of 0; x 2^F in the plaintext
coefficient. The sign flip between the two rings and both magnitudes come out of that one rounding.

{{FILE:k768_const_rounding.txt}}

What the arithmetic says about N = 2^17 (NOT measured; it assumes scaling 59 bits, first modulus 60 bits, `FLEXIBLEAUTO`, which is what
`hpc_gpu_port/experimental/gpu_real_model_x.cu:1423-1426` sets, hence SF = FirstPrime(59, 2^18) and deg = 1):
eta = +9.55e-11 for K = 768 against -2.09e-11 for K = 512 (the stock constant is no longer exact there either, because FirstPrime(59, 2^18) is
2^59 + 12,058,625); rms over coefficients of I x eta x 2^F with sigma_I = 85.33 and F = 7: 1.04e-06 against 2.29e-07 per plaintext coefficient.
For scale only, and across rings, which is NOT a like-for-like comparison: the whole coefficient error of the CPU control at N = 2^13 / F = 7 is
2.28e-08 rms (median `coefErrRms` of `ctrl_r13_F07_base.jsonl`, table T2). Note what the same arithmetic says about the STOCK constant at 2^17: it
is not exact there either (eta = -2.1e-11), so a linear error of a quarter of the K = 768 one would already be part of the demo's present
precision; the per-bootstrap overflow recorder of the long-run prep can check both on the GPU.

**FIDESlib, by source reading only (`vendor/FIDESlib-K`, the tree that carries the GPU switch `FIDESLIB_BOOT_UNIFORM_EXT`):**
`Bootstrap.cu:95-99` and `:238-247` compute `constantEvalMult = pre * (1.0 / (k * cc.N))` with `k = cc.GetBootK()` and apply it with
`ctxt.multScalar`; `Context.cu:366-412` (`ElemForEvalMult`) encodes it as `static_cast<int128>(operand / approxFactor * scFactor + 0.5)`, the same
rounding, with `scFactor = param.ScalingFactorReal[level]` taken from the OpenFHE context; the CoeffsToSlots plaintexts are imported from OpenFHE's
own precomputation (`openfhe-interface/RawCiphertext.cu:1203`, `precom->m_U0hatTPreFFT`). So the same scale error is EXPECTED on the GPU with
bootK = 768; it was not run, and whether its scFactor at that level equals OpenFHE's was not checked.

The fix that was run (variant B, `openfhe_k768_split_variantB.patch`, `OPENFHE_BOOT_UNIFORM_EXT_SPLIT=1`): 768 = 3 x 2^8; the 3 goes into the
CoeffsToSlots precomputation (`scaleEnc = pre / 3` at `ckksrns-fhe.cpp:177-197` and `:343-363`, exactly where the sparse cases fold their whole K)
and the scalar constant becomes pre/(256 N), a power of two again. Measured at N = 2^13: slope / 2^F = -3.6e-14 +- 1.0e-13 over 100..760 (102
trials; pooled with 765..766: -5.8e-14 +- 9.2e-14, 108 trials; predicted -2.8e-14), error at the planted coefficient back at the floor (ratio to
floor 0.6-1.5 in every window), baseline relErrRms
5.179e-05 instead of 7.066e-05, median coefficient error 7.96e-08 (BELOW the control's 1.26e-07), edge thresholds unchanged (774 / 778 / 780 /
783 / 788 / 798), same depth and towers (T0). With the split switch off library B behaves as library A (6.956e-05, 10 trials) and with both
switches off as the control (2.778e-05, 10 trials). On the GPU route the equivalent would be bootK = 256 in FIDESlib plus the `pre / 3` of
the OpenFHE precomputation it imports: not built, not run.

A smaller residual is visible in the arms whose constant is (nearly) exact: control 500..510: -1.99e-13 +- 0.44e-13 (but +0.96e-13 +- 0.60e-13 over
100..500); split at ring 2^12: -2.45e-13 +- 0.82e-13, where the error at the planted coefficient reaches 3x the floor at 700..766. It is 7-9x the
constant-rounding arithmetic for those arms (-2.8e-14), at or just above the noise level, present with the stock table too, and was not investigated.

### 2.6 Robustness: F = 7, ring 2^12, negative overflow, a coefficient of the second half

Thresholds of these sweeps are in table T3 above, baselines in T2, slopes in T7.

- **F = 7** (`--corr 7`, the value the library's default formula gives at N = 2^17 / 2^16 slots; run at N = 2^13): baseline 3.598e-06 (control) ->
  8.826e-06 (Ext), 2.45x = 1.29 bits, the same ratio as at F = 10. Edge 776 / 779 / 782 / 786 / first exception 790 (all trials from 793) /
  destroyed 799: 1-4 units later than at F = 10, as the 2^F scaling of the same sine-output error implies (the stock table moved the same way:
  519 / 526 / 529 / 532 / 539 / 553 at F = 7, `../planted_r13_F07_pos.jsonl`, first row group of T1). Measured / model 0.994 (69 bootstraps). Old
  zone 510..600 step 10: 30 bootstraps, 0 exceptions, relErrRms 9.02e-06..1.05e-05.
- **Ring 2^12** (F = 11): baseline 1.479e-05 -> 4.960e-05, 3.35x = 1.75 bits (variant B 4.335e-05, 2.93x): the ring with the larger share of
  coefficients at I = 0 (sigma_I 15.0-15.3) pays more. Edge 774 / 777 / 779 / 782 / first exception 786 (all from 788) / destroyed 798. Linear
  error negative, as above.
- **Negative overflow** (N = 2^13, F = 10, -806..-766 step 2, 2 trials): 774 / 776 / 780 / 782 / 788 / 798, measured / model 0.91: -I* behaves as
  +I*, and the linear error changes sign with I* (those trials are among the 335).
- **A coefficient of the second half** (j* = 6000, 765..805 step 5): 775 / 780 / 785 / 785 / 790 / 800 on a grid of 5: the same edge.
- Every planted trial of every file is `plantVerified`, `achieved == target`, `adjust==library`; the largest overflow of any NON-planted
  coefficient in any planted file is 91 (sanity table under T3), so the planted coefficient is the only one near either edge.
- Red-flag check (`zero_relerr_check.txt`): 1,735 trial records, none with relErrRms exactly 0, none with a non-finite decoded slot among the
  non-throwing trials.

## 3. What the data support, and what they do not

Supported (measured; OpenFHE 1.5.1 CPU `EvalBootstrap`, rings 2^12 / 2^13, the parameters above):

- Selecting `g_coefficientsUniformExt` / `K_UNIFORMEXT` for a `UNIFORM_TERNARY`, non-composite context needs no other change in the library: same
  depth budget (20), same towers, same output level, no exception inside `EvalBootstrap`, and the bootstrap is correct (no decode failure in
  any un-planted trial of any Ext run).
- The overflow failure mode moves from just above 512 to just above 768: nothing happens to a ciphertext whose largest overflow is anywhere in
  500..766; the new thresholds are 775 (degraded) / 782 (relErr 0.1) / 788 (decode exception) / 798 (destroyed) at F = 10, and they shift with F by
  the amount the 2^F scaling predicts (section 2.6).
- The Ext polynomial outside its range is what the float64 model says it is (measured / model 1.2, 69 bootstraps).
- K = 768 as the library wires it costs precision: 1.28 bits of relErrRms at N = 2^13 (F = 10 and F = 7), 1.75 bits at N = 2^12. Part of it is the
  rounded scalar constant, an error proportional to each coefficient's overflow that arithmetic on one rounding reproduces in sign and size at both
  rings; moving the factor 3 into CoeffsToSlots removes it without touching depth or keys and recovers 0.45 bits at 2^13 (0.20 at 2^12). The rest
  is random-sign noise on the coefficients with overflow 0, +-1, +-2, in the proportions of the double-angle amplification.

Not shown:

- **FIDESlib's GPU bootstrap.** It imports the same table and K and, by source reading, builds the same rounded constant; its polynomial
  evaluation is its own code. Its edge, its baseline precision with the Ext table, and the linear error were not run. The K = 768 A/B cell of
  the next GPU session should record per-bootstrap precision for both arms (not only failures), and the sign and size of the error against the
  per-coefficient overflow that the flight recorder already computes; the CPU numbers to compare with are T2 / T2b / T7.
- **Ring 2^17.** Every measurement here is at 2^12 / 2^13 with sigma_I = 15 / 21: the natural overflows never came near either K (max |I| 105 in
  all un-planted trials), so everything about the edges comes from the planted instrument, and the precision ratios are for THIS mix of
  overflows. At 2^17 (sigma_I = 85) fewer coefficients sit at I = 0 (0.5 % instead of 1.9 %) and the rounded constant is coarser (arithmetic
  above); how the two costs net out there is not something these runs can say.
- **The failure RATE with K = 768.** The thresholds are measured; the rate is the Gaussian-tail arithmetic of `../../boot_overflow_ext_table.txt`
  (K = 768 is 9.0 sigma at N = 2^17) with the measured edges put in (table T8 below, ARITHMETIC): 1.5e-14 per bootstrap for the first
  degradation (775), 3.6e-15 for the decode exception (788), against 1.6e-04 and 5.3e-05 for the stock table's 519 and 534. The unresolved
  far-tail question of `../PROBE_C_REPORT.md` (replies running high beyond 5 sigma) applies to that extrapolation unchanged.
- Why the degree-118 evaluation is noisier at I = 0 (section 2.2 gives a candidate, untested), and the -2e-13 residual slope (section 2.5).

Table T8 (ARITHMETIC, not a measurement):

{{SECTION:## T8.}}

## 4. Commands and files

- Library A: `bash .../k768/build_openfhe_k768.sh` (copy made with `cp -Rp vendor/openfhe-development vendor/openfhe-k768-src`; all three
  `vendor/` directories are git-ignored: `git check-ignore -v` -> `.gitignore:4:vendor/`). Probe: `bash .../k768/build_probe_k768.sh`. Variant B:
  `bash .../k768/build_variantB.sh`.
- Runs: `bash .../k768/run_k768.sh main | splitB | extra`; the exact command line of every process is in `runlog_main.txt`, `runlog_splitB.txt`,
  `runlog_extra.txt` and in the header record of each JSONL (`cmdline`, `harnessSha256 c760781be72bd1af`, `envUniformExt`). Every process is
  `[OPENFHE_BOOT_UNIFORM_EXT=1 [OPENFHE_BOOT_UNIFORM_EXT_SPLIT=1]] caffeinate -i tools/memguard.sh 6 harness/build/boot_overflow_probe_k768[b] ...`.
  All exits 0, no guard trip; the only stderr content is the library's one-line notice in the Ext arms.
- Analysis: `.venv/bin/python analyze_k768.py` -> `k768_tables.md` (T0-T7, including every per-I* table); `.venv/bin/python
  k768_const_rounding.py` -> `k768_const_rounding.txt`; `.venv/bin/python assemble_k768_report.py` -> this file from `K768_CPU_REPORT.src.md`.
- `dev_smoke/`: the first runs of the first probe build (no table uses them; `smoke_r12_ext_600.jsonl` is where the negative sign at ring 2^12
  was first seen).
- Nothing was committed.
