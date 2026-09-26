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

`arm` is read from what the LIBRARY printed on stderr (one line per process, only when a switch is on): A-ext = `[openfhe-k768-patch] EvalBootstrap: OPENFHE_BOOT_UNIFORM_EXT=1 -> Chebyshev degree 118, K = 768`; B-split = `... -> Chebyshev degree 118, K in the scalar constant = 256 (OPENFHE_BOOT_UNIFORM_EXT_SPLIT=1: the factor 3 is in CoeffsToSlots)`; B-nosplit = `... K in the scalar constant = 768 (no split)`; off = empty stderr (stock selection). `other stderr` = anything else in the stderr file.

| file | arm (from stderr) | OPENFHE_BOOT_UNIFORM_EXT (header) | K label | N | depth | bootDepth (library's GetBootstrapDepth) | towersQ | F | harnessSha256 | trials | adjust==library | bootRestored | EvalBootstrap threw | towersIn -> towersOut | outLevel | outNoiseScaleDeg | other stderr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_ctrl_r13_F10_base.jsonl | off (B library) | unset | 512 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 10 | 10/10 | 10/10 | 0 | [2] -> [5] | [19] | [2] | none |
| B_nosplit_r13_F10_base.jsonl | B-nosplit | 1 | 768 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 10 | 10/10 | 10/10 | 0 | [2] -> [5] | [19] | [2] | none |
| B_split_r12_F11_base.jsonl | B-split | 1 | 768 | 4096 | 23 | 20 | 24 | 11 | c760781be72bd1af | 40 | 40/40 | 40/40 | 0 | [2] -> [5] | [19] | [2] | none |
| B_split_r12_F11_planted_wide.jsonl | B-split | 1 | 768 | 4096 | 23 | 20 | 24 | 11 | c760781be72bd1af | 36 | 36/36 | 36/36 | 0 | [2] -> [5] | [19] | [2] | none |
| B_split_r13_F10_base.jsonl | B-split | 1 | 768 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 40 | 40/40 | 40/40 | 0 | [2] -> [5] | [19] | [2] | none |
| B_split_r13_F10_planted_edge.jsonl | B-split | 1 | 768 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 123 | 123/123 | 123/123 | 0 | [2] -> [5] | [19] | [2] | none |
| B_split_r13_F10_planted_wide.jsonl | B-split | 1 | 768 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 102 | 102/102 | 102/102 | 0 | [2] -> [5] | [19] | [2] | none |
| ctrl_r12_F11_base.jsonl | off | unset | 512 | 4096 | 23 | 20 | 24 | 11 | c760781be72bd1af | 40 | 40/40 | 40/40 | 0 | [2] -> [5] | [19] | [2] | none |
| ctrl_r13_F07_base.jsonl | off | unset | 512 | 8192 | 23 | 20 | 24 | 7 | c760781be72bd1af | 40 | 40/40 | 40/40 | 0 | [2] -> [5] | [19] | [2] | none |
| ctrl_r13_F10_base.jsonl | off | unset | 512 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 40 | 40/40 | 40/40 | 0 | [2] -> [5] | [19] | [2] | none |
| ctrl_r13_F10_planted.jsonl | off | unset | 512 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 183 | 183/183 | 183/183 | 0 | [2] -> [5] | [19] | [2] | none |
| ctrl_r13_F10_planted_wide.jsonl | off | unset | 512 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 63 | 63/63 | 63/63 | 0 | [2] -> [5] | [19] | [2] | none |
| ext_r12_F11_base.jsonl | A-ext | 1 | 768 | 4096 | 23 | 20 | 24 | 11 | c760781be72bd1af | 40 | 40/40 | 40/40 | 0 | [2] -> [5] | [19] | [2] | none |
| ext_r12_F11_planted_edge.jsonl | A-ext | 1 | 768 | 4096 | 23 | 20 | 24 | 11 | c760781be72bd1af | 108 | 108/108 | 108/108 | 0 | [2] -> [5] | [19] | [2] | none |
| ext_r12_F11_planted_wide.jsonl | A-ext | 1 | 768 | 4096 | 23 | 20 | 24 | 11 | c760781be72bd1af | 36 | 36/36 | 36/36 | 0 | [2] -> [5] | [19] | [2] | none |
| ext_r13_F07_base.jsonl | A-ext | 1 | 768 | 8192 | 23 | 20 | 24 | 7 | c760781be72bd1af | 40 | 40/40 | 40/40 | 0 | [2] -> [5] | [19] | [2] | none |
| ext_r13_F07_planted_edge.jsonl | A-ext | 1 | 768 | 8192 | 23 | 20 | 24 | 7 | c760781be72bd1af | 123 | 123/123 | 123/123 | 0 | [2] -> [5] | [19] | [2] | none |
| ext_r13_F07_planted_old.jsonl | A-ext | 1 | 768 | 8192 | 23 | 20 | 24 | 7 | c760781be72bd1af | 30 | 30/30 | 30/30 | 0 | [2] -> [5] | [19] | [2] | none |
| ext_r13_F10_base.jsonl | A-ext | 1 | 768 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 40 | 40/40 | 40/40 | 0 | [2] -> [5] | [19] | [2] | none |
| ext_r13_F10_planted_edge.jsonl | A-ext | 1 | 768 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 213 | 213/213 | 213/213 | 0 | [2] -> [5] | [19] | [2] | none |
| ext_r13_F10_planted_imag.jsonl | A-ext | 1 | 768 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 27 | 27/27 | 27/27 | 0 | [2] -> [5] | [19] | [2] | none |
| ext_r13_F10_planted_neg.jsonl | A-ext | 1 | 768 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 42 | 42/42 | 42/42 | 0 | [2] -> [5] | [19] | [2] | none |
| ext_r13_F10_planted_old.jsonl | A-ext | 1 | 768 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 183 | 183/183 | 183/183 | 0 | [2] -> [5] | [19] | [2] | none |
| ext_r13_F10_planted_old2.jsonl | A-ext | 1 | 768 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 24 | 24/24 | 24/24 | 0 | [2] -> [5] | [19] | [2] | none |
| ext_r13_F10_planted_wide.jsonl | A-ext | 1 | 768 | 8192 | 23 | 20 | 24 | 10 | c760781be72bd1af | 102 | 102/102 | 102/102 | 0 | [2] -> [5] | [19] | [2] | none |

`bootRestored` (output towers > input towers) is true on every one of the bootstraps, `adjust==library` (the probe's exact overflow is computed
on the ciphertext the library really raises) on every one, and `EvalBootstrap` itself never threw. Depth 23, `bootDepth` 20, 24 towers, 2 -> 5
towers, output level 19, noise scale degree 2: identical in the control, the Ext arm and variant B.

### 2.1 CONTROL (4a): with the switch off the patched build reproduces the stock build's thresholds

| sweep | OPENFHE_BOOT_UNIFORM_EXT | K (label) | N | F | I* grid | baseline relErrRms (median over |I*| <= K-2) | median relErrRms > 2x baseline | median >= 1e-3 | median >= 1e-2 | median >= 1e-1 | first decode exception | all trials throw from | whole ciphertext destroyed (median coefficient error > 1) | measured / model at j*, median over K+6 <= |I*| <= K+28 (min..max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PREVIOUS (stock build) ../planted_r12_F11_pos.jsonl | unset | 512 | 4096 | 11 | 500..560 step 1 | 1.403e-05 (n = 33) | 518 | 522 | 525 | 528 | 534 | 534 | 553 | 1.001 (0.758..1.160), n = 69 |
| PREVIOUS (stock build) ../planted_r13_F07_pos.jsonl | unset | 512 | 8192 | 7 | 500..560 step 1 | 3.540e-06 (n = 33) | 519 | 526 | 529 | 532 | 539 | 546 | 553 | 0.995 (0.303..1.666), n = 69 |
| PREVIOUS (stock build) ../planted_r13_F10_pos.jsonl | unset | 512 | 8192 | 10 | 500..560 step 1 | 2.914e-05 (n = 33) | 519 | 522 | 526 | 529 | 535 | 548 | 553 | 1.116 (0.218..2.674), n = 69 |
| CONTROL (k768 build, switch off) ctrl_r13_F10_planted.jsonl | unset | 512 | 8192 | 10 | 500..560 step 1 | 2.790e-05 (n = 33) | 519 | 522 | 525 | 529 | 534 | 536 | 553 | 1.108 (0.030..4.304), n = 69 |
| CONTROL (k768 build, switch off) ctrl_r13_F10_planted_wide.jsonl | unset | 512 | 8192 | 10 | 100..500 step 20 | 2.986e-05 (n = 63) | None | None | None | None | None | None | None | - |

Same argv as the earlier main sweep except for the binary. Thresholds 519 / 522 / 525 / 529 / 534 / 553 against 519 / 522 / 526 / 529 /
535 / 553 (different random keys and ciphertexts, 3 trials per I*); "all trials throw from" differs (536 vs 548) because the previous sweep had
one trial of three that still decoded at 547. The per-I* comparison is table T1b of `k768_tables.md`. The un-planted baseline agrees too (next
table: 2.914e-05 vs 2.854e-05 on the stock build).

### 2.2 Baseline precision without planting (4b-i)

| file | OPENFHE_BOOT_UNIFORM_EXT | K label | N | F | trials | decode fails | max|I| over all trials | sigma_I measured (mean) | relErrRms median | relErrRms min..max | relErrMax median | coefErrRms median | floor = median |coef err| (median over trials) | max |coef err| (median over trials) | max/floor (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PREVIOUS (stock build) ../sound_r13_stock.jsonl | unset | 512 | 8192 | 10 | 40 | 0 | 98 | 21.289 | 2.854e-05 | 2.777e-05..2.930e-05 | 9.135e-05 | 1.814e-07 | 1.225e-07 | 7.046e-07 | 5.7 |
| B_ctrl_r13_F10_base.jsonl | unset | 512 | 8192 | 10 | 10 | 0 | 90 | 21.277 | 2.778e-05 | 2.763e-05..2.849e-05 | 1.039e-04 | 1.770e-07 | 1.192e-07 | 6.937e-07 | 5.8 |
| B_nosplit_r13_F10_base.jsonl | 1 | 768 | 8192 | 10 | 10 | 0 | 97 | 21.254 | 6.956e-05 | 6.529e-05..7.681e-05 | 2.181e-04 | 4.440e-07 | 2.290e-07 | 6.264e-06 | 27.4 |
| B_split_r12_F11_base.jsonl | 1 | 768 | 4096 | 11 | 40 | 0 | 70 | 15.267 | 4.335e-05 | 3.623e-05..5.117e-05 | 8.857e-05 | 3.865e-07 | 7.002e-08 | 5.869e-06 | 82.5 |
| B_split_r13_F10_base.jsonl | 1 | 768 | 8192 | 10 | 40 | 0 | 97 | 21.294 | 5.179e-05 | 4.776e-05..5.733e-05 | 1.145e-04 | 3.290e-07 | 7.964e-08 | 6.084e-06 | 77.3 |
| ctrl_r12_F11_base.jsonl | unset | 512 | 4096 | 11 | 40 | 0 | 73 | 15.182 | 1.479e-05 | 1.429e-05..1.530e-05 | 7.422e-05 | 1.321e-07 | 8.917e-08 | 4.923e-07 | 5.6 |
| ctrl_r13_F07_base.jsonl | unset | 512 | 8192 | 7 | 40 | 0 | 101 | 21.392 | 3.598e-06 | 3.499e-06..3.679e-06 | 2.168e-05 | 2.284e-08 | 1.541e-08 | 8.896e-08 | 5.8 |
| ctrl_r13_F10_base.jsonl | unset | 512 | 8192 | 10 | 40 | 0 | 96 | 21.393 | 2.914e-05 | 2.857e-05..2.994e-05 | 1.527e-04 | 1.855e-07 | 1.256e-07 | 7.243e-07 | 5.8 |
| ext_r12_F11_base.jsonl | 1 | 768 | 4096 | 11 | 40 | 0 | 73 | 14.974 | 4.960e-05 | 4.137e-05..5.398e-05 | 1.029e-04 | 4.433e-07 | 1.768e-07 | 6.119e-06 | 34.6 |
| ext_r13_F07_base.jsonl | 1 | 768 | 8192 | 7 | 40 | 0 | 102 | 21.241 | 8.826e-06 | 8.157e-06..9.512e-06 | 2.506e-05 | 5.618e-08 | 2.911e-08 | 8.782e-07 | 29.4 |
| ext_r13_F10_base.jsonl | 1 | 768 | 8192 | 10 | 40 | 0 | 105 | 21.295 | 7.066e-05 | 6.630e-05..7.469e-05 | 2.747e-04 | 4.488e-07 | 2.281e-07 | 6.957e-06 | 30.5 |

Ratios to the control of the same ring and F (medians of the table above). `ext_*` / `ctrl_*` pairs share binary (library A) and trial count (40); the `B_*` rows come from the variant-B binary (10 trials for `B_nosplit` and `B_ctrl`) and are compared with library A's control:

| pair | relErrRms Ext / control | bits of precision lost = log2 of that | coefErrRms Ext / control | floor Ext / control |
|---|---|---|---|---|
| ext_r12_F11_base.jsonl / ctrl_r12_F11_base.jsonl | 3.353 | +1.75 | 3.357 | 1.982 |
| B_split_r12_F11_base.jsonl / ctrl_r12_F11_base.jsonl | 2.930 | +1.55 | 2.926 | 0.785 |
| ext_r13_F07_base.jsonl / ctrl_r13_F07_base.jsonl | 2.453 | +1.29 | 2.460 | 1.890 |
| ext_r13_F10_base.jsonl / ctrl_r13_F10_base.jsonl | 2.425 | +1.28 | 2.419 | 1.815 |
| B_split_r13_F10_base.jsonl / ctrl_r13_F10_base.jsonl | 1.777 | +0.83 | 1.773 | 0.634 |
| B_nosplit_r13_F10_base.jsonl / ctrl_r13_F10_base.jsonl | 2.387 | +1.26 | 2.393 | 1.823 |
| B_ctrl_r13_F10_base.jsonl / ctrl_r13_F10_base.jsonl | 0.953 | -0.07 | 0.954 | 0.948 |

Decode-free coefficient error against the coefficient's OWN overflow |I_j| (`errByAbsI` records pooled over the 40 trials of each file; N = 2^13,
F = 10; the same tables for F = 7 and N = 2^12 are in `k768_tables.md`, T2b). Control vs Ext (library A):

| |I_j| | control: coefficients | control: rms coef error | control: max |coef error| | Ext: coefficients | Ext: rms coef error | Ext: max |coef error| | rms Ext / control |
|---|---|---|---|---|---|---|---|
| 0 | 6060 | 2.016e-07 | 7.437e-07 | 6152 | 1.582e-06 | 1.563e-05 | 7.846 |
| 1 | 12201 | 1.863e-07 | 8.703e-07 | 12187 | 5.353e-07 | 5.467e-06 | 2.873 |
| 2 | 12210 | 1.857e-07 | 7.022e-07 | 12179 | 3.978e-07 | 6.264e-06 | 2.142 |
| 3..4 | 24059 | 1.859e-07 | 7.788e-07 | 24352 | 3.707e-07 | 9.552e-06 | 1.994 |
| 5..8 | 46617 | 1.844e-07 | 7.753e-07 | 46989 | 3.534e-07 | 6.111e-06 | 1.916 |
| 9..16 | 82139 | 1.853e-07 | 8.063e-07 | 82263 | 3.614e-07 | 8.744e-06 | 1.950 |
| 17..24 | 61905 | 1.849e-07 | 8.219e-07 | 61919 | 3.884e-07 | 1.294e-05 | 2.100 |
| 25..32 | 40310 | 1.854e-07 | 8.305e-07 | 40033 | 4.072e-07 | 8.724e-06 | 2.196 |
| 33..48 | 34462 | 1.847e-07 | 7.715e-07 | 34123 | 4.544e-07 | 9.570e-06 | 2.461 |
| 49..64 | 6889 | 1.876e-07 | 7.467e-07 | 6685 | 5.323e-07 | 8.293e-06 | 2.837 |
| 65..96 | 828 | 1.893e-07 | 7.212e-07 | 797 | 5.954e-07 | 1.923e-06 | 3.146 |
| >= 97 | 0 | - | - | 1 | 8.722e-07 | 8.722e-07 | - |

Control vs Ext + split (variant B, section 2.5):

| |I_j| | control: coefficients | control: rms coef error | control: max |coef error| | Ext: coefficients | Ext: rms coef error | Ext: max |coef error| | rms Ext / control |
|---|---|---|---|---|---|---|---|
| 0 | 6060 | 2.016e-07 | 7.437e-07 | 6203 | 1.496e-06 | 1.068e-05 | 7.424 |
| 1 | 12201 | 1.863e-07 | 8.703e-07 | 12373 | 4.724e-07 | 6.862e-06 | 2.536 |
| 2 | 12210 | 1.857e-07 | 7.022e-07 | 12155 | 3.035e-07 | 6.141e-06 | 1.634 |
| 3..4 | 24059 | 1.859e-07 | 7.788e-07 | 24412 | 2.629e-07 | 6.943e-06 | 1.414 |
| 5..8 | 46617 | 1.844e-07 | 7.753e-07 | 46526 | 2.622e-07 | 7.593e-06 | 1.422 |
| 9..16 | 82139 | 1.853e-07 | 8.063e-07 | 82320 | 2.417e-07 | 7.259e-06 | 1.304 |
| 17..24 | 61905 | 1.849e-07 | 8.219e-07 | 61836 | 2.513e-07 | 9.992e-06 | 1.359 |
| 25..32 | 40310 | 1.854e-07 | 8.305e-07 | 40440 | 2.405e-07 | 6.830e-06 | 1.297 |
| 33..48 | 34462 | 1.847e-07 | 7.715e-07 | 33922 | 2.430e-07 | 5.945e-06 | 1.316 |
| 49..64 | 6889 | 1.876e-07 | 7.467e-07 | 6670 | 2.137e-07 | 4.106e-06 | 1.139 |
| 65..96 | 828 | 1.893e-07 | 7.212e-07 | 822 | 2.323e-07 | 3.257e-06 | 1.228 |
| >= 97 | 0 | - | - | 1 | 5.253e-08 | 5.253e-08 | - |

Overflow I_j of the 6 coefficients with the largest error in each un-planted trial (`topCoefErrI`), pooled per file:

| file | trials | top-6 entries | entries with |I_j| = 0 | <= 2 | <= 8 | > 8 | median |I_j| of the top-6 | for reference: fraction of ALL coefficients with |I_j| = 0 / <= 2 / <= 8 (from errByAbsI) |
|---|---|---|---|---|---|---|---|---|
| B_ctrl_r13_F10_base.jsonl | 10 | 60 | 4 | 5 | 17 | 43 | 16.5 | 0.0184 / 0.0932 / 0.3119 |
| B_nosplit_r13_F10_base.jsonl | 10 | 60 | 26 | 28 | 39 | 21 | 3.5 | 0.0187 / 0.0934 / 0.3088 |
| B_split_r12_F11_base.jsonl | 40 | 240 | 138 | 149 | 186 | 54 | 0.0 | 0.0264 / 0.1302 / 0.4208 |
| B_split_r13_F10_base.jsonl | 40 | 240 | 119 | 128 | 155 | 85 | 1.0 | 0.0189 / 0.0938 / 0.3103 |
| ctrl_r12_F11_base.jsonl | 40 | 240 | 39 | 59 | 114 | 126 | 9.5 | 0.0265 / 0.1309 / 0.4248 |
| ctrl_r13_F07_base.jsonl | 40 | 240 | 14 | 24 | 73 | 167 | 16.0 | 0.0191 / 0.0934 / 0.3095 |
| ctrl_r13_F10_base.jsonl | 40 | 240 | 13 | 30 | 74 | 166 | 14.0 | 0.0185 / 0.0930 / 0.3087 |
| ext_r12_F11_base.jsonl | 40 | 240 | 133 | 148 | 178 | 62 | 0.0 | 0.0265 / 0.1309 / 0.4283 |
| ext_r13_F07_base.jsonl | 40 | 240 | 130 | 140 | 158 | 82 | 0.0 | 0.0192 / 0.0939 / 0.3110 |
| ext_r13_F10_base.jsonl | 40 | 240 | 134 | 141 | 161 | 79 | 0.0 | 0.0188 / 0.0931 / 0.3108 |

Sign of the entries of `topCoefErr` that sit on a coefficient with I_j = 0 (a bias would have one sign):

| file | entries at I_j = 0 | positive | negative |
|---|---|---|---|
| B_ctrl_r13_F10_base.jsonl | 4 | 2 | 2 |
| B_nosplit_r13_F10_base.jsonl | 26 | 11 | 15 |
| B_split_r12_F11_base.jsonl | 138 | 74 | 64 |
| B_split_r13_F10_base.jsonl | 119 | 59 | 60 |
| ctrl_r12_F11_base.jsonl | 39 | 15 | 24 |
| ctrl_r13_F07_base.jsonl | 14 | 7 | 7 |
| ctrl_r13_F10_base.jsonl | 13 | 9 | 4 |
| ext_r12_F11_base.jsonl | 133 | 66 | 67 |
| ext_r13_F07_base.jsonl | 130 | 65 | 65 |
| ext_r13_F10_base.jsonl | 134 | 68 | 66 |

Shape of the peak at small |I_j|: excess of the rms error over the |I_j| = 5..8 bin of the same file, in quadrature, relative to the excess at I_j = 0; next to the ARITHMETIC ratio of the double-angle amplification A(I) = 1 / (2^R |sin(2 pi (I - 1/4) / 2^R)|), R = 6, rms over +I and -I (noise injected before the double-angle iterations is multiplied by prod_i 4 a_i |cos(2^i theta_0)| = const * A(I)):

| file | excess at 0 | excess at |I| = 1 (ratio to 0) | excess at |I| = 2 (ratio to 0) | arithmetic A(1)/A(0) | arithmetic A(2)/A(0) |
|---|---|---|---|---|---|
| B_ctrl_r13_F10_base.jsonl | 7.831e-08 | - | 2.049e-08 (0.262) | 0.275 | 0.129 |
| B_nosplit_r13_F10_base.jsonl | 1.491e-06 | 3.874e-07 (0.260) | 1.246e-07 (0.084) | 0.275 | 0.129 |
| B_split_r12_F11_base.jsonl | 1.501e-06 | 4.029e-07 (0.268) | 1.435e-07 (0.096) | 0.275 | 0.129 |
| B_split_r13_F10_base.jsonl | 1.473e-06 | 3.929e-07 (0.267) | 1.528e-07 (0.104) | 0.275 | 0.129 |
| ctrl_r12_F11_base.jsonl | 8.149e-08 | - | - | 0.275 | 0.129 |
| ctrl_r13_F07_base.jsonl | 9.909e-09 | 2.451e-09 (0.247) | 2.034e-09 (0.205) | 0.275 | 0.129 |
| ctrl_r13_F10_base.jsonl | 8.138e-08 | 2.635e-08 (0.324) | 2.161e-08 (0.266) | 0.275 | 0.129 |
| ext_r12_F11_base.jsonl | 1.510e-06 | 3.906e-07 (0.259) | 1.789e-07 (0.118) | 0.275 | 0.129 |
| ext_r13_F07_base.jsonl | 1.839e-07 | 5.074e-08 (0.276) | 2.717e-08 (0.148) | 0.275 | 0.129 |
| ext_r13_F10_base.jsonl | 1.542e-06 | 4.020e-07 (0.261) | 1.826e-07 (0.118) | 0.275 | 0.129 |

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

| file | I* range | bootstraps | decode exceptions | relErrRms: median of per-I* medians (min..max over ALL trials) | the same file's own baseline (|I*| <= 510) | largest per-I* median / baseline | |coef err at j*|: median (max) over all trials | floor: median |
|---|---|---|---|---|---|---|---|---|
| B_split_r12_F11_planted_wide.jsonl | 100..760 | 36 | 0 | 5.191e-05 (4.528e-05..6.292e-05) | 5.493e-05 | 1.03 | 1.415e-07 (2.783e-06) | 8.598e-08 |
| B_split_r13_F10_planted_wide.jsonl | 100..760 | 102 | 0 | 6.018e-05 (5.051e-05..7.164e-05) | 5.867e-05 | 1.11 | 8.913e-08 (4.596e-06) | 8.676e-08 |
| ext_r12_F11_planted_wide.jsonl | 100..760 | 36 | 0 | 5.912e-05 (5.067e-05..6.563e-05) | 6.005e-05 | 1.05 | 3.327e-06 (6.666e-06) | 1.845e-07 |
| ext_r13_F07_planted_old.jsonl | 510..600 | 30 | 0 | 9.457e-06 (9.019e-06..1.046e-05) | 9.612e-06 | 1.02 | 4.719e-07 (5.971e-07) | 2.789e-08 |
| ext_r13_F10_planted_old.jsonl | 500..560 | 183 | 0 | 7.599e-05 (6.965e-05..8.359e-05) | 7.554e-05 | 1.10 | 3.927e-06 (8.274e-06) | 2.248e-07 |
| ext_r13_F10_planted_old2.jsonl | 565..600 | 24 | 0 | 7.565e-05 (7.067e-05..8.050e-05) | - | - | 4.520e-06 (5.111e-06) | 2.267e-07 |
| ext_r13_F10_planted_wide.jsonl | 100..760 | 102 | 0 | 7.471e-05 (6.803e-05..8.443e-05) | 7.390e-05 | 1.11 | 3.221e-06 (5.886e-06) | 2.255e-07 |

| arm | N | F | files | bootstraps | decode exceptions | EvalBootstrap threw | relErrRms min / median / max | sigma_I of the planted ciphertexts (mean of `sigmaIhat`) | for comparison: the un-planted baseline of the same arm, ring and F (median, from T2; its sigma_I is in T2) |
|---|---|---|---|---|---|---|---|---|---|
| Ext (variant A) | 4096 | 11 | 2 | 21 | 0 | 0 | 4.691e-05 / 5.662e-05 / 6.403e-05 | 15.47 | 4.960e-05 |
| Ext (variant A) | 8192 | 7 | 2 | 36 | 0 | 0 | 9.019e-06 / 9.464e-06 / 1.046e-05 | 16.41 | 8.826e-06 |
| Ext (variant A) | 8192 | 10 | 6 | 275 | 0 | 0 | 6.965e-05 / 7.592e-05 / 8.443e-05 | 16.45 | 7.066e-05 |
| Ext + split (variant B) | 4096 | 11 | 1 | 15 | 0 | 0 | 4.528e-05 / 5.032e-05 / 5.628e-05 | 14.79 | 4.335e-05 |
| Ext + split (variant B) | 8192 | 10 | 2 | 48 | 0 | 0 | 5.534e-05 / 6.114e-05 / 7.164e-05 | 16.75 | 5.179e-05 |
| stock table (switch off) | 8192 | 10 | 2 | 186 | 79 | 0 | 2.684e-05 / 3.046e-05 / 6.573e+00 | 16.47 | 2.914e-05 |

N = 2^13, F = 10: every integer 500..560, every 5 to 600, every 20 from 100 to 760, 760..766, plus the few in-range points of the negative and
second-half sweeps: **275 bootstraps with 500 <= |I*| <= 766, no decode exception, no trial above 8.44e-05**, where the control throws in 79 of
186 and is destroyed from 553. The same holds at F = 7 (36 bootstraps) and at N = 2^12 (21). The planted trials' median (7.59e-05) sits 7 % above
the un-planted baseline (7.07e-05): the planting instrument narrows the overflow distribution of the OTHER coefficients (sigma_I 16.5 instead of
21.3, column above), which puts more of them at I = 0, where the Ext arm's noise sits (section 2.2): a likely reason, not tested. The reference for
a planted trial is therefore the sweep's own baseline (T3, T5), against which the largest per-I* median is 1.10-1.11x. The only trace of the
planted coefficient itself is cost (a): its own error is 16-25x the floor (4.0e-06 at 511..599) but one coefficient of 8,192 at that level does not
move relErrRms.

### 2.4 The NEW edge (4b-iii)

| sweep | OPENFHE_BOOT_UNIFORM_EXT | K (label) | N | F | I* grid | baseline relErrRms (median over |I*| <= K-2) | median relErrRms > 2x baseline | median >= 1e-3 | median >= 1e-2 | median >= 1e-1 | first decode exception | all trials throw from | whole ciphertext destroyed (median coefficient error > 1) | measured / model at j*, median over K+6 <= |I*| <= K+28 (min..max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_split_r12_F11_planted_wide.jsonl [Ext + split (variant B)] | 1 | 768 | 4096 | 11 | 100..760 step 60 | 5.267e-05 (n = 36) | None | None | None | None | None | None | None | - |
| B_split_r13_F10_planted_edge.jsonl [Ext + split (variant B)] | 1 | 768 | 8192 | 10 | 765..805 step 1 | 6.226e-05 (n = 6) | 774 | 778 | 780 | 783 | 788 | 789 | 798 | 1.059 (0.122..215.235), n = 69 |
| B_split_r13_F10_planted_wide.jsonl [Ext + split (variant B)] | 1 | 768 | 8192 | 10 | 100..760 step 20 | 6.004e-05 (n = 102) | None | None | None | None | None | None | None | - |
| ext_r12_F11_planted_edge.jsonl [Ext (variant A)] | 1 | 768 | 4096 | 11 | 765..800 step 1 | 5.462e-05 (n = 6) | 774 | 777 | 779 | 782 | 786 | 788 | 798 | 1.055 (0.001..483.156), n = 69 |
| ext_r12_F11_planted_wide.jsonl [Ext (variant A)] | 1 | 768 | 4096 | 11 | 100..760 step 60 | 5.956e-05 (n = 36) | None | None | None | None | None | None | None | - |
| ext_r13_F07_planted_edge.jsonl [Ext (variant A)] | 1 | 768 | 8192 | 7 | 765..805 step 1 | 9.447e-06 (n = 6) | 776 | 779 | 782 | 786 | 790 | 793 | 799 | 0.994 (0.058..8.664), n = 69 |
| ext_r13_F07_planted_old.jsonl [Ext (variant A)] | 1 | 768 | 8192 | 7 | 510..600 step 10 | 9.465e-06 (n = 30) | None | None | None | None | None | None | None | - |
| ext_r13_F10_planted_edge.jsonl [Ext (variant A)] | 1 | 768 | 8192 | 10 | 760..830 step 1 | 7.552e-05 (n = 21) | 775 | 777 | 780 | 782 | 788 | 788 | 798 | 1.220 (0.020..4887.536), n = 69 |
| ext_r13_F10_planted_imag.jsonl [Ext (variant A)] | 1 | 768 | 8192 | 10 | 765..805 step 5 | 7.805e-05 (n = 3) | 775 | 780 | 785 | 785 | 790 | 790 | 800 | 1.346 (0.157..4.232), n = 15 |
| ext_r13_F10_planted_neg.jsonl [Ext (variant A)] | 1 | 768 | 8192 | 10 | 766..806 step 2 | 7.316e-05 (n = 2) | 774 | 776 | 780 | 782 | 788 | 790 | 798 | 0.906 (0.027..11796.696), n = 24 |
| ext_r13_F10_planted_old.jsonl [Ext (variant A)] | 1 | 768 | 8192 | 10 | 500..560 step 1 | 7.591e-05 (n = 183) | None | None | None | None | None | None | None | - |
| ext_r13_F10_planted_old2.jsonl [Ext (variant A)] | 1 | 768 | 8192 | 10 | 565..600 step 5 | 7.616e-05 (n = 24) | None | None | None | None | None | None | None | - |
| ext_r13_F10_planted_wide.jsonl [Ext (variant A)] | 1 | 768 | 8192 | 10 | 100..760 step 20 | 7.456e-05 (n = 102) | None | None | None | None | None | None | None | - |

Sanity of the planted sweeps (every file of this directory):

| file | bootstraps | plantVerified | achieved == target | adjust==library | bootRestored | EvalBootstrap threw | largest |I| of any NON-planted coefficient |
|---|---|---|---|---|---|---|---|
| B_split_r12_F11_planted_wide.jsonl | 36 | 36/36 | 36/36 | 36/36 | 36/36 | 0 | 42 |
| B_split_r13_F10_planted_edge.jsonl | 123 | 123/123 | 123/123 | 123/123 | 123/123 | 0 | 78 |
| B_split_r13_F10_planted_wide.jsonl | 102 | 102/102 | 102/102 | 102/102 | 102/102 | 0 | 76 |
| ctrl_r13_F10_planted.jsonl | 183 | 183/183 | 183/183 | 183/183 | 183/183 | 0 | 79 |
| ctrl_r13_F10_planted_wide.jsonl | 63 | 63/63 | 63/63 | 63/63 | 63/63 | 0 | 91 |
| ext_r12_F11_planted_edge.jsonl | 108 | 108/108 | 108/108 | 108/108 | 108/108 | 0 | 55 |
| ext_r12_F11_planted_wide.jsonl | 36 | 36/36 | 36/36 | 36/36 | 36/36 | 0 | 45 |
| ext_r13_F07_planted_edge.jsonl | 123 | 123/123 | 123/123 | 123/123 | 123/123 | 0 | 75 |
| ext_r13_F07_planted_old.jsonl | 30 | 30/30 | 30/30 | 30/30 | 30/30 | 0 | 69 |
| ext_r13_F10_planted_edge.jsonl | 213 | 213/213 | 213/213 | 213/213 | 213/213 | 0 | 73 |
| ext_r13_F10_planted_imag.jsonl | 27 | 27/27 | 27/27 | 27/27 | 27/27 | 0 | 65 |
| ext_r13_F10_planted_neg.jsonl | 42 | 42/42 | 42/42 | 42/42 | 42/42 | 0 | 72 |
| ext_r13_F10_planted_old.jsonl | 183 | 183/183 | 183/183 | 183/183 | 183/183 | 0 | 73 |
| ext_r13_F10_planted_old2.jsonl | 24 | 24/24 | 24/24 | 24/24 | 24/24 | 0 | 68 |
| ext_r13_F10_planted_wide.jsonl | 102 | 102/102 | 102/102 | 102/102 | 102/102 | 0 | 82 |

Per I*, N = 2^13, F = 10, rows 766..806 of `ext_r13_F10_planted_edge.jsonl` (the F = 7 and N = 2^12 edges are in `k768_tables.md`, T4e; every full
sweep in T4). `share` = (e_j*^2 + e_(j*+N/2)^2) / sum_j e_j^2 over the decode-free coefficient errors; `floor` = median |coefficient error| over all
N coefficients; `model` = the Ext polynomial re-evaluated in float64 at I*/768, times 2^F (ARITHMETIC):

| I* | n | decode exceptions | relErrRms (min / median / max over non-throwing) | share of squared coef error on {j*, j*+N/2} (median) | |coef err at j*| (min / median / max) | |coef err at j*+N/2| median | floor | model |g(I/K)|*2^F (arithmetic) |
|---|---|---|---|---|---|---|---|---|
| 766 | 3 | 0 | 7.491e-05 / 7.552e-05 / 7.818e-05 | 0.0176 | 5.629e-06 / 5.667e-06 / 5.870e-06 | 2.319e-07 | 2.201e-07 | 5.458e-10 |
| 767 | 3 | 0 | 7.346e-05 / 7.355e-05 / 7.711e-05 | 0.0186 | 5.072e-06 / 5.664e-06 / 5.894e-06 | 1.933e-07 | 2.266e-07 | 1.478e-10 |
| 768 | 3 | 0 | 7.288e-05 / 7.777e-05 / 7.905e-05 | 0.0170 | 5.492e-06 / 5.638e-06 / 5.948e-06 | 4.067e-07 | 2.240e-07 | 1.298e-09 |
| 769 | 3 | 0 | 7.411e-05 / 7.562e-05 / 7.717e-05 | 0.0150 | 4.691e-06 / 5.335e-06 / 5.667e-06 | 3.330e-07 | 2.233e-07 | 1.070e-08 |
| 770 | 3 | 0 | 7.518e-05 / 7.621e-05 / 8.031e-05 | 0.0169 | 5.418e-06 / 5.937e-06 / 6.347e-06 | 3.879e-07 | 2.262e-07 | 1.006e-08 |
| 771 | 3 | 0 | 7.708e-05 / 7.713e-05 / 8.049e-05 | 0.0220 | 5.155e-06 / 6.386e-06 / 7.070e-06 | 9.979e-07 | 2.265e-07 | 2.219e-07 |
| 772 | 3 | 0 | 7.325e-05 / 7.381e-05 / 7.715e-05 | 0.0391 | 4.377e-06 / 8.367e-06 / 8.605e-06 | 3.829e-06 | 2.249e-07 | 1.275e-06 |
| 773 | 3 | 0 | 7.994e-05 / 7.998e-05 / 9.414e-05 | 0.1337 | 7.591e-06 / 1.600e-05 / 2.551e-05 | 5.429e-06 | 2.271e-07 | 5.359e-06 |
| 774 | 3 | 0 | 8.721e-05 / 9.403e-05 / 1.537e-04 | 0.2273 | 9.846e-06 / 2.564e-05 / 6.292e-05 | 2.113e-05 | 2.247e-07 | 1.902e-05 |
| 775 | 3 | 0 | 1.764e-04 / 2.745e-04 / 6.074e-04 | 0.9206 | 6.773e-05 / 1.335e-04 / 2.888e-04 | 6.835e-05 | 2.247e-07 | 6.046e-05 |
| 776 | 3 | 0 | 3.218e-04 / 4.964e-04 / 8.220e-04 | 0.9736 | 1.532e-04 / 2.721e-04 / 4.346e-04 | 9.234e-05 | 2.263e-07 | 1.773e-04 |
| 777 | 3 | 0 | 1.130e-03 / 1.872e-03 / 2.031e-03 | 0.9982 | 2.777e-04 / 6.973e-04 / 8.824e-04 | 7.742e-04 | 2.231e-07 | 4.887e-04 |
| 778 | 3 | 0 | 2.402e-03 / 3.128e-03 / 3.640e-03 | 0.9994 | 1.362e-03 / 1.768e-03 / 1.793e-03 | 3.257e-04 | 2.270e-07 | 1.281e-03 |
| 779 | 3 | 0 | 8.217e-03 / 9.671e-03 / 1.057e-02 | 0.9999 | 4.628e-03 / 5.515e-03 / 5.907e-03 | 8.776e-04 | 2.278e-07 | 3.220e-03 |
| 780 | 3 | 0 | 1.519e-02 / 1.541e-02 / 2.315e-02 | 1.0000 | 6.872e-03 / 8.227e-03 / 1.211e-02 | 5.331e-03 | 2.242e-07 | 7.817e-03 |
| 781 | 3 | 0 | 6.867e-03 / 3.748e-02 / 4.340e-02 | 1.0000 | 3.967e-03 / 1.935e-02 / 2.021e-02 | 8.529e-03 | 2.275e-07 | 1.842e-02 |
| 782 | 3 | 0 | 3.281e-02 / 1.115e-01 / 1.292e-01 | 1.0000 | 1.880e-02 / 6.206e-02 / 6.310e-02 | 1.409e-02 | 2.249e-07 | 4.231e-02 |
| 783 | 3 | 0 | 7.829e-02 / 1.819e-01 / 2.063e-01 | 1.0000 | 4.356e-02 / 1.035e-01 / 1.118e-01 | 1.129e-02 | 2.215e-07 | 9.507e-02 |
| 784 | 3 | 0 | 3.024e-01 / 4.730e-01 / 5.489e-01 | 1.0000 | 1.726e-01 / 2.557e-01 / 3.013e-01 | 7.482e-02 | 2.215e-07 | 2.096e-01 |
| 785 | 3 | 0 | 4.456e-02 / 1.034e+00 / 1.392e+00 | 1.0000 | 8.988e-03 / 5.608e-01 / 7.891e-01 | 2.394e-02 | 2.233e-07 | 4.544e-01 |
| 786 | 3 | 0 | 1.349e+00 / 1.983e+00 / 2.947e+00 | 1.0000 | 4.669e-01 / 1.152e+00 / 1.569e+00 | 6.234e-01 | 2.269e-07 | 9.715e-01 |
| 787 | 3 | 0 | 2.626e+00 / 2.938e+00 / 6.171e+00 | 1.0000 | 1.280e+00 / 1.654e+00 / 3.545e+00 | 1.062e-01 | 2.265e-07 | 2.053e+00 |
| 788 | 3 | 3 | - | 1.0000 | 5.321e+00 / 5.325e+00 / 5.444e+00 | 4.568e-01 | 2.242e-07 | 4.297e+00 |
| 789 | 3 | 3 | - | 1.0000 | 9.058e+00 / 1.091e+01 / 1.375e+01 | 6.686e-01 | 2.264e-07 | 8.925e+00 |
| 790 | 3 | 3 | - | 1.0000 | 9.644e+00 / 2.061e+01 / 2.368e+01 | 2.419e+00 | 2.280e-07 | 1.843e+01 |
| 791 | 3 | 3 | - | 1.0000 | 4.168e+01 / 5.486e+01 / 5.978e+01 | 3.967e+00 | 2.266e-07 | 3.777e+01 |
| 792 | 3 | 3 | - | 1.0000 | 7.074e+01 / 7.729e+01 / 1.049e+02 | 2.053e+00 | 2.223e-07 | 7.581e+01 |
| 793 | 3 | 3 | - | 1.0000 | 1.270e+02 / 1.480e+02 / 1.502e+02 | 2.466e+01 | 2.208e-07 | 1.379e+02 |
| 794 | 3 | 3 | - | 1.0000 | 3.737e+01 / 7.062e+01 / 1.532e+02 | 3.137e+01 | 2.220e-07 | 1.369e+02 |
| 795 | 3 | 3 | - | 1.0000 | 5.762e+01 / 1.635e+02 / 3.011e+02 | 2.108e+02 | 2.240e-07 | 1.627e+02 |
| 796 | 3 | 3 | - | 1.0000 | 1.040e+02 / 3.119e+03 / 8.178e+04 | 1.033e+04 | 2.217e-07 | 1.673e+01 |
| 797 | 3 | 3 | - | 1.0000 | 2.281e+10 / 7.115e+10 / 2.398e+12 | 1.834e+11 | 9.663e-07 | 2.965e+11 |
| 798 | 3 | 3 | - | 1.0000 | 4.570e+17 / 2.557e+19 / 2.907e+19 | 1.308e+19 | 1.325e+02 | 7.974e+17 |
| 799 | 3 | 3 | - | 1.0000 | 2.841e+22 / 7.247e+24 / 3.833e+25 | 8.653e+24 | 5.292e+07 | 9.454e+23 |
| 800 | 3 | 3 | - | 1.0000 | 5.132e+27 / 3.628e+31 / 1.583e+34 | 3.183e+32 | 1.491e+15 | 5.267e+30 |
| 801 | 3 | 3 | - | 1.0000 | 1.486e+39 / 9.825e+39 / 4.564e+40 | 2.344e+39 | 4.750e+22 | 3.855e+38 |
| 802 | 3 | 3 | - | 1.0000 | 2.702e+39 / 1.393e+49 / 1.902e+52 | 3.387e+48 | 6.733e+31 | 5.603e+47 |
| 803 | 3 | 3 | - | 1.0000 | 9.972e+52 / 1.184e+53 / 1.586e+53 | 1.266e+53 | 2.584e+41 | 1.478e+58 |
| 804 | 3 | 3 | - | 0.0002 | 9.988e+51 / 6.025e+52 / 1.175e+53 | 1.024e+53 | 9.608e+52 | 4.418e+69 |
| 805 | 3 | 3 | - | 0.0002 | 3.551e+52 / 6.617e+52 / 1.813e+53 | 7.205e+52 | 9.456e+52 | 7.777e+81 |
| 806 | 3 | 3 | - | 0.0000 | 3.762e+52 / 4.129e+52 / 1.493e+53 | 5.280e+52 | 9.527e+52 | 4.227e+94 |

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

Pooled over I* windows; `ratio to floor` = median over trials of |coef err at j*| / (that trial's median |coefficient error|). Only trials with |I*| <= K - 2 of the file's arm are used (the edge is in T3/T4).

| arm | file(s) | N | F | I* window | trials | |coef err at j*|: median | rms | max | floor (median over trials) | ratio to floor: median | fraction of trials with |coef err at j*| > 5x floor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ext (variant A) | 1 file(s) | 4096 | 11 | 100..199 | 6 | 9.975e-07 | 9.492e-07 | 1.102e-06 | 1.863e-07 | 5.56 | 0.667 |
| Ext (variant A) | 1 file(s) | 4096 | 11 | 200..299 | 6 | 1.818e-06 | 1.820e-06 | 2.244e-06 | 1.859e-07 | 9.67 | 1.000 |
| Ext (variant A) | 1 file(s) | 4096 | 11 | 300..399 | 3 | 2.519e-06 | 2.672e-06 | 3.040e-06 | 1.852e-07 | 13.63 | 1.000 |
| Ext (variant A) | 1 file(s) | 4096 | 11 | 400..510 | 6 | 3.327e-06 | 3.283e-06 | 3.633e-06 | 1.851e-07 | 18.03 | 1.000 |
| Ext (variant A) | 1 file(s) | 4096 | 11 | 511..599 | 6 | 4.269e-06 | 4.266e-06 | 4.778e-06 | 1.814e-07 | 23.30 | 1.000 |
| Ext (variant A) | 1 file(s) | 4096 | 11 | 600..699 | 3 | 5.082e-06 | 4.974e-06 | 5.278e-06 | 1.820e-07 | 27.92 | 1.000 |
| Ext (variant A) | 2 file(s) | 4096 | 11 | 700..766 | 12 | 5.703e-06 | 5.769e-06 | 6.666e-06 | 1.795e-07 | 32.38 | 1.000 |
| Ext (variant A) | 1 file(s) | 8192 | 7 | 400..510 | 3 | 4.343e-07 | 4.384e-07 | 4.590e-07 | 2.809e-08 | 15.43 | 1.000 |
| Ext (variant A) | 1 file(s) | 8192 | 7 | 511..599 | 24 | 4.719e-07 | 4.772e-07 | 5.894e-07 | 2.788e-08 | 16.75 | 1.000 |
| Ext (variant A) | 1 file(s) | 8192 | 7 | 600..699 | 3 | 5.675e-07 | 5.675e-07 | 5.971e-07 | 2.790e-08 | 20.54 | 1.000 |
| Ext (variant A) | 1 file(s) | 8192 | 7 | 700..766 | 6 | 6.991e-07 | 7.023e-07 | 7.322e-07 | 2.755e-08 | 25.11 | 1.000 |
| Ext (variant A) | 1 file(s) | 8192 | 10 | 100..199 | 15 | 8.938e-07 | 1.013e-06 | 1.541e-06 | 2.255e-07 | 3.96 | 0.267 |
| Ext (variant A) | 1 file(s) | 8192 | 10 | 200..299 | 15 | 1.795e-06 | 1.857e-06 | 2.337e-06 | 2.281e-07 | 7.93 | 1.000 |
| Ext (variant A) | 1 file(s) | 8192 | 10 | 300..399 | 15 | 2.662e-06 | 2.718e-06 | 3.105e-06 | 2.243e-07 | 11.67 | 1.000 |
| Ext (variant A) | 2 file(s) | 8192 | 10 | 400..510 | 51 | 3.638e-06 | 3.792e-06 | 8.274e-06 | 2.248e-07 | 16.16 | 1.000 |
| Ext (variant A) | 3 file(s) | 8192 | 10 | 511..599 | 183 | 4.022e-06 | 4.079e-06 | 6.213e-06 | 2.251e-07 | 17.82 | 1.000 |
| Ext (variant A) | 2 file(s) | 8192 | 10 | 600..699 | 18 | 4.789e-06 | 4.778e-06 | 5.161e-06 | 2.268e-07 | 21.23 | 1.000 |
| Ext (variant A) | 4 file(s) | 8192 | 10 | 700..766 | 38 | 5.484e-06 | 5.507e-06 | 5.918e-06 | 2.241e-07 | 24.53 | 1.000 |
| Ext + split (variant B) | 1 file(s) | 4096 | 11 | 100..199 | 6 | 1.419e-07 | 4.539e-07 | 1.077e-06 | 8.559e-08 | 1.68 | 0.167 |
| Ext + split (variant B) | 1 file(s) | 4096 | 11 | 200..299 | 6 | 6.371e-08 | 1.701e-07 | 3.952e-07 | 8.961e-08 | 0.70 | 0.000 |
| Ext + split (variant B) | 1 file(s) | 4096 | 11 | 300..399 | 3 | 4.523e-08 | 8.955e-08 | 1.465e-07 | 8.852e-08 | 0.52 | 0.000 |
| Ext + split (variant B) | 1 file(s) | 4096 | 11 | 400..510 | 6 | 9.169e-08 | 1.352e-07 | 2.545e-07 | 8.919e-08 | 1.03 | 0.000 |
| Ext + split (variant B) | 1 file(s) | 4096 | 11 | 511..599 | 6 | 1.508e-07 | 2.661e-07 | 5.575e-07 | 8.507e-08 | 1.72 | 0.167 |
| Ext + split (variant B) | 1 file(s) | 4096 | 11 | 600..699 | 3 | 2.612e-07 | 2.600e-07 | 3.209e-07 | 8.082e-08 | 3.16 | 0.000 |
| Ext + split (variant B) | 1 file(s) | 4096 | 11 | 700..766 | 6 | 2.630e-07 | 1.160e-06 | 2.783e-06 | 7.832e-08 | 3.28 | 0.167 |
| Ext + split (variant B) | 1 file(s) | 8192 | 10 | 100..199 | 15 | 7.041e-08 | 1.410e-07 | 3.310e-07 | 8.407e-08 | 0.82 | 0.000 |
| Ext + split (variant B) | 1 file(s) | 8192 | 10 | 200..299 | 15 | 5.165e-08 | 2.797e-07 | 8.777e-07 | 8.494e-08 | 0.61 | 0.067 |
| Ext + split (variant B) | 1 file(s) | 8192 | 10 | 300..399 | 15 | 8.703e-08 | 1.231e-07 | 2.647e-07 | 8.641e-08 | 1.01 | 0.000 |
| Ext + split (variant B) | 1 file(s) | 8192 | 10 | 400..510 | 18 | 9.295e-08 | 1.094e-06 | 4.596e-06 | 8.723e-08 | 1.04 | 0.056 |
| Ext + split (variant B) | 1 file(s) | 8192 | 10 | 511..599 | 12 | 9.195e-08 | 1.440e-07 | 2.546e-07 | 8.778e-08 | 1.06 | 0.000 |
| Ext + split (variant B) | 1 file(s) | 8192 | 10 | 600..699 | 15 | 1.046e-07 | 1.398e-07 | 2.951e-07 | 8.766e-08 | 1.19 | 0.000 |
| Ext + split (variant B) | 2 file(s) | 8192 | 10 | 700..766 | 18 | 1.304e-07 | 2.717e-07 | 9.301e-07 | 8.752e-08 | 1.45 | 0.056 |
| stock table (switch off) | 1 file(s) | 8192 | 10 | 100..199 | 15 | 8.866e-08 | 1.419e-07 | 2.809e-07 | 1.285e-07 | 0.68 | 0.000 |
| stock table (switch off) | 1 file(s) | 8192 | 10 | 200..299 | 15 | 1.660e-07 | 1.941e-07 | 4.710e-07 | 1.292e-07 | 1.29 | 0.000 |
| stock table (switch off) | 1 file(s) | 8192 | 10 | 300..399 | 15 | 1.091e-07 | 1.620e-07 | 3.169e-07 | 1.277e-07 | 0.86 | 0.000 |
| stock table (switch off) | 2 file(s) | 8192 | 10 | 400..510 | 51 | 9.907e-08 | 1.548e-07 | 3.898e-07 | 1.212e-07 | 0.83 | 0.000 |

Least-squares line through the origin, signed `coefErrAtArgmaxI` = slope * I*, over the trials with 100 <= |I*| <= K - 2 whose planted coefficient is the arg max |I|; `eta measured` = slope / 2^F. `eta predicted` = (round(c*SF) - c*SF)/(c*SF) for the scalar constant c = pre/(K_scalar*N) the arm multiplies the raised ciphertext by (K_scalar = 512 stock, 768 Ext, 256 Ext + split), SF = FirstPrime(59, 2N): ARITHMETIC on the library's rules (`k768_const_rounding.py`), not a measurement. `residual rms` = rms of (error - slope * I*), to be compared with the floor.

| arm | file | N | F | trials | I* range | same sign as I* * eta_pred | eta measured = slope / 2^F | standard error | eta predicted (arithmetic) | measured / predicted | residual rms | floor (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ext + split (variant B) | B_split_r12_F11_planted_wide.jsonl | 4096 | 11 | 36 | 100..760 | 28/36 | -2.4549e-13 | 8.2e-14 | -2.8423e-14 | 8.637 | 4.765e-07 | 8.598e-08 |
| Ext + split (variant B) | B_split_r13_F10_planted_edge.jsonl | 8192 | 10 | 6 | 765..766 | 5/6 | -2.0430e-13 | 5.9e-14 | -2.8423e-14 | 7.188 | 1.040e-07 | 8.645e-08 |
| Ext + split (variant B) | B_split_r13_F10_planted_wide.jsonl | 8192 | 10 | 102 | 100..760 | 73/102 | -3.5850e-14 | 1.0e-13 | -2.8423e-14 | 1.261 | 4.936e-07 | 8.676e-08 |
| stock table (switch off) | ctrl_r13_F10_planted.jsonl | 8192 | 10 | 33 | 500..510 | 28/33 | -1.9855e-13 | 4.4e-14 | -2.8423e-14 | 6.986 | 1.290e-07 | 1.196e-07 |
| stock table (switch off) | ctrl_r13_F10_planted_wide.jsonl | 8192 | 10 | 63 | 100..500 | 26/63 | +9.5638e-14 | 6.0e-14 | -2.8423e-14 | -3.365 | 1.555e-07 | 1.280e-07 |
| Ext (variant A) | ext_r12_F11_planted_edge.jsonl | 4096 | 11 | 6 | 765..766 | 6/6 | -3.6784e-12 | 4.3e-14 | -3.6664e-12 | 1.003 | 1.503e-07 | 1.767e-07 |
| Ext (variant A) | ext_r12_F11_planted_wide.jsonl | 4096 | 11 | 36 | 100..760 | 36/36 | -3.7795e-12 | 5.2e-14 | -3.6664e-12 | 1.031 | 2.989e-07 | 1.845e-07 |
| Ext (variant A) | ext_r13_F07_planted_edge.jsonl | 8192 | 7 | 6 | 765..766 | 6/6 | +7.1651e-12 | 8.5e-14 | +7.2475e-12 | 0.989 | 1.852e-08 | 2.755e-08 |
| Ext (variant A) | ext_r13_F07_planted_old.jsonl | 8192 | 7 | 30 | 510..600 | 30/30 | +6.7738e-12 | 9.9e-14 | +7.2475e-12 | 0.935 | 3.804e-08 | 2.789e-08 |
| Ext (variant A) | ext_r13_F10_planted_edge.jsonl | 8192 | 10 | 21 | 760..766 | 21/21 | +7.1085e-12 | 6.4e-14 | +7.2475e-12 | 0.981 | 2.220e-07 | 2.230e-07 |
| Ext (variant A) | ext_r13_F10_planted_old.jsonl | 8192 | 10 | 183 | 500..560 | 183/183 | +7.3032e-12 | 6.8e-14 | +7.2475e-12 | 1.008 | 4.955e-07 | 2.248e-07 |
| Ext (variant A) | ext_r13_F10_planted_old2.jsonl | 8192 | 10 | 24 | 565..600 | 24/24 | +7.6761e-12 | 9.1e-14 | +7.2475e-12 | 1.059 | 2.609e-07 | 2.267e-07 |
| Ext (variant A) | ext_r13_F10_planted_wide.jsonl | 8192 | 10 | 102 | 100..760 | 102/102 | +7.3271e-12 | 6.1e-14 | +7.2475e-12 | 1.011 | 2.967e-07 | 2.255e-07 |

Pooled over the files of one arm, ring and F (same selection of trials):

| arm | N | F | trials | same sign as I* * eta_pred | eta measured | standard error | eta predicted (arithmetic) | measured / predicted |
|---|---|---|---|---|---|---|---|---|
| Ext (variant A) | 4096 | 11 | 42 | 42/42 | -3.7492e-12 | 4.1e-14 | -3.6664e-12 | 1.023 |
| Ext (variant A) | 8192 | 7 | 36 | 36/36 | +6.8814e-12 | 8.4e-14 | +7.2475e-12 | 0.949 |
| Ext (variant A) | 8192 | 10 | 335 | 335/335 | +7.3066e-12 | 4.1e-14 | +7.2475e-12 | 1.008 |
| Ext + split (variant B) | 4096 | 11 | 36 | 28/36 | -2.4549e-13 | 8.2e-14 | -2.8423e-14 | 8.637 |
| Ext + split (variant B) | 8192 | 10 | 108 | 78/108 | -5.8367e-14 | 9.2e-14 | -2.8423e-14 | 2.053 |
| stock table (switch off) | 8192 | 10 | 96 | 54/96 | -6.9314e-14 | 4.0e-14 | -2.8423e-14 | 2.439 |

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

```
scalingModSize 59, firstModSize 60 (deg = 1, pre = 1/2), FLEXIBLEAUTO; SF = FirstPrime(59, 2N)
       N    K    SF - 2^59                 c*SF  round - c*SF   eta = relative error of the constant  sigma_I (h = 2N/3)  rms over coefficients of I*eta          ... * 2^F (F)
    2^12  512        16385    137438953472.0039       -0.0039                            -2.8423e-14               15.09                       4.288e-13        8.783e-10 (11)
    2^12  768        16385     91625968981.3359       -0.3359                            -3.6664e-12               15.09                       5.532e-11        1.133e-07 (11)
    2^13  512        16385     68719476736.0020       -0.0020                            -2.8423e-14               21.34                       6.064e-13        6.210e-10 (10)
    2^13  768        16385     45812984490.6680       +0.3320                            +7.2475e-12               21.34                       1.546e-10        1.583e-07 (10)
    2^14  512      1015809     34359738368.0605       -0.0605                            -1.7621e-12               30.17                       5.317e-11        2.722e-08 (9)
    2^14  768      1015809     22906492245.3737       -0.3737                            -1.6314e-11               30.17                       4.922e-10        2.520e-07 (9)
    2^15  512      4849665     17179869184.1445       -0.1445                            -8.4128e-12               42.67                       3.590e-10        9.189e-08 (8)
    2^15  768      4849665     11453246122.7630       +0.2370                            +2.0691e-11               42.67                       8.828e-10        2.260e-07 (8)
    2^16  512      4849665      8589934592.0723       -0.0723                            -8.4128e-12               60.34                       5.076e-10        1.300e-07 (8)
    2^16  768      4849665      5726623061.3815       -0.3815                            -6.6620e-11               60.34                       4.020e-09        1.029e-06 (8)
    2^17  512     12058625      4294967296.0898       -0.0898                            -2.0918e-11               85.33                       1.785e-09        2.285e-07 (7)
    2^17  768     12058625      2863311530.7266       +0.2734                            +9.5497e-11               85.33                       8.149e-09        1.043e-06 (7)

the split used by variant B (OPENFHE_BOOT_UNIFORM_EXT_SPLIT=1): scalar constant pre/(256*N), the factor 1/3 inside the CoeffsToSlots plaintexts
    2^12  256        16385    274877906944.0078       -0.0078                            -2.8423e-14
    2^13  256        16385    137438953472.0039       -0.0039                            -2.8423e-14
    2^17  256     12058625      8589934592.1797       -0.1797                            -2.0918e-11
```

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

P(max_j |I_j| >= B) = 1 - (1 - 2 Q((B - 0.5) / sigma_I))^N with sigma_I = sqrt((2N/3 + 1)/12) = 85.33, as in `../demo_rate_from_thresholds.py`; thresholds B are the ones measured on CPU at N = 2^13 (T1 / T3), NOT at 2^17, and the tail law itself is the extrapolation discussed in `../PROBE_C_REPORT.md`.

| sweep (arm) | F of the sweep | threshold | B | B / sigma_I | P per bootstrap | P per 480-bootstrap tick |
|---|---|---|---|---|---|---|
| ctrl_r13_F10_planted.jsonl [stock table (switch off)] | 10 | median relErrRms > 2x baseline | 519 | 6.08 | 1.61e-04 | 7.45e-02 |
| ctrl_r13_F10_planted.jsonl [stock table (switch off)] | 10 | median relErrRms >= 0.1 | 529 | 6.20 | 7.72e-05 | 3.64e-02 |
| ctrl_r13_F10_planted.jsonl [stock table (switch off)] | 10 | first decode exception | 534 | 6.26 | 5.31e-05 | 2.52e-02 |
| ctrl_r13_F10_planted.jsonl [stock table (switch off)] | 10 | whole ciphertext destroyed | 553 | 6.48 | 1.25e-05 | 5.96e-03 |
| ext_r13_F07_planted_edge.jsonl [Ext (variant A)] | 7 | median relErrRms > 2x baseline | 776 | 9.09 | 1.32e-14 | 6.36e-12 |
| ext_r13_F07_planted_edge.jsonl [Ext (variant A)] | 7 | median relErrRms >= 0.1 | 786 | 9.21 | 4.48e-15 | 2.15e-12 |
| ext_r13_F07_planted_edge.jsonl [Ext (variant A)] | 7 | first decode exception | 790 | 9.26 | 2.89e-15 | 1.39e-12 |
| ext_r13_F07_planted_edge.jsonl [Ext (variant A)] | 7 | whole ciphertext destroyed | 799 | 9.36 | 1.07e-15 | 5.14e-13 |
| ext_r13_F10_planted_edge.jsonl [Ext (variant A)] | 10 | median relErrRms > 2x baseline | 775 | 9.08 | 1.47e-14 | 7.08e-12 |
| ext_r13_F10_planted_edge.jsonl [Ext (variant A)] | 10 | median relErrRms >= 0.1 | 782 | 9.16 | 6.92e-15 | 3.32e-12 |
| ext_r13_F10_planted_edge.jsonl [Ext (variant A)] | 10 | first decode exception | 788 | 9.23 | 3.60e-15 | 1.73e-12 |
| ext_r13_F10_planted_edge.jsonl [Ext (variant A)] | 10 | whole ciphertext destroyed | 798 | 9.35 | 1.20e-15 | 5.75e-13 |

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
