# K = 768 in the GPU bootstrap, and a bootstrap flight recorder — code reading, patches, pod checks (2026-09-18)

**Status: authored offline. Nothing here was compiled or run** (no CUDA toolchain on this machine, the one heavy slot was
taken). What WAS executed is Python only: an emulation of FIDESlib's level bookkeeping (`k768_ps_level_emulation.py/.txt`)
and a transliteration of the recorder's arithmetic against exact integers (`k3_recorder_arithmetic_check.py/.txt`), both
beside this file. Every FIDESlib `file:line` below is the pristine upstream tree `vendor/FIDESlib-upstream` at
`fa972864ae8d624e77d3ac6ad31a1d40ef1c4d0c` (the pod's commit: `pod_2xbw_20260917/pod_pull/s37/selftest_2dev/BUILD_IDENTITY.txt`);
OpenFHE lines are `vendor/openfhe-development` (1.5.1). Measured numbers are transcribed from the file named beside them.

## 0. Deliverables
| file | what |
|---|---|
| `../patches/K1_boot_uniform_ext_env.patch` | FIDESlib, `RawCiphertext.cu` (+17): `FIDESLIB_BOOT_UNIFORM_EXT=1` selects `g_coefficientsUniformExt` / `K_UNIFORMEXT`; one line at context creation names the active table/K/R in BOTH arms. Default: today's table. |
| `../patches/K2_boot_preraise_hook.patch` | FIDESlib, `Bootstrap.cuh` (+17) and `Bootstrap.cu` (+25): `g_bootPreRaiseHook`, a `std::function`, empty by default. 42 added lines, no default-path behaviour change. |
| `../patches/harness/K3_harness_boot_record.patch` | copy of `git diff -- hpc_gpu_port/experimental/gpu_real_model_x.cu` (the edit is in the working tree, uncommitted): `--boot-record --boot-record-secret PATH|self`. 262 insertions, 1 modified line. |

`git -C vendor/FIDESlib-upstream apply --check` passes for K1 and K2, separately and together; they touch no file that
`L1_mgpu_ks_table_free.patch` or the pod's `patch_peerutils.py` / `patch_mgpu_key_decomp.py` touch.
**K3 sits in `patches/harness/` on purpose:** `campaign/scripts/rider_build_fideslib.sh:75-82` git-applies EVERY `*.patch` of
`FIDESLIB_EXTRA_PATCH_DIR` (= `patches/`, `POD_RUN_PLAN.md:45`) to the FIDESlib checkout and dies on one that neither applies
nor is already applied. A harness diff in that directory is such a patch (checked on a pristine copy: both `apply --check`
and `apply -R --check` fail, "No such file"), so it would have stopped the pod build at its first step. With K3 moved, the
same loop applies K1, K2 and L1 to a pristine tree. The copy committed in `f921ac1` is still at the old path and is stale
(it predates the loader change and `nonCanon`): delete it when committing.

## 1. Part 1 — K = 768: what the code says
**1.1 Where the table and K enter.** The harness sets `UNIFORM_TERNARY` (`gpu_real_model_x.cu` `SetSecretKeyDist`) and goes through
the api: `api/CryptoContext.cpp:209-219` maps it to `FIDESlib::UNIFORM`, `:221` calls `GetRawParams`, whose `UNIFORM` branch
(`RawCiphertext.cu:576-581`) copies `g_coefficientsUniform`, `K_UNIFORM`, `R_UNIFORM` into `RawParams`
(`RawCiphertext.cuh:80-82`; `bootK` is `uint32_t`). Consumers, exhaustively (grep `bootK|GetBootK|coefficientsCheby|GetCoeffsChebyshev`
over `src/` and `api/`): `Context.cu:556-569` (getters; `GetBootK` returns `int`), `Bootstrap.cu:79` and `:222-225`
(`double k = cc.GetBootK(); constantEvalMult = pre * (1.0 / (k * cc.N))`, applied at `:235`), `ApproxModEval.cu:35-36,:53-55,:100,:110`.
768 fits every type on the path. No literal 512, 88, 89, 118 or 119 exists in `src/` or `api/`. **K is not folded into anything
precomputed** for a uniform secret: OpenFHE's precompute uses `k = 1.0` for `UNIFORM_TERNARY` (`ckksrns-fhe.cpp:179-180`, `:345-346`)
and divides by K only on the raised ciphertext (`:638`), which is the statement FIDESlib mirrors at `Bootstrap.cu:225`. So the
CtS/StC plaintexts, the keys and the serialized context do not depend on K.
`api/CryptoContext.cpp:357-364` also names `g_coefficientsUniform`, but only to compute `modall =
GetMultiplicativeDepthByCoeffVector(...) + R_UNIFORM`; degree 88 and 118 share the bucket `60..119 -> 8`
(`ckksrns-utils.cpp:82-110`, row at `:95`), so `modall` = 14 either way. K1 leaves that file alone: the CPU-side context is
bit-for-bit today's.

**1.2 Is the evaluator generic in the degree? Yes.** `ApproxModEval.cu:392` takes `n = Degree(coefficients)`, `:403-405` takes
`(k, m) = lbcrypto::ComputeDegreesPS(n)`; the hard-wired `(12,2)/(13,3)` block at `:406-414` is under `if (false)`. The power
basis, the `T_{k 2^i}` chain and the scratch set are sized from `k` and `m` (`:437-447`, `:484-511`, `:604-615`, `:644-670`), the
recursion `innerEvalChebyshevPS` (`:138-366`) takes them as arguments, and the weighted-sum kernel loops over a run-time `n`
(`Ciphertext.cpp:1169-1229`, `LimbPartition.cu:2340-2368`, `ElemenwiseBatchKernels.cu:187-200`). `ComputeDegreesPS`
(`ckksrns-utils.cpp:299-330`, row `{239, 4}` at `:313`): degree 88 -> m = 4, k = floor(88/15)+1 = **6**; degree 118 -> m = 4,
k = floor(118/15)+1 = **8**. The one m-dependent piece of FIDESlib's recursion (the in-place level adjustment of `T2[m-1]` only
when `max_m - m <= 1`, `:351-353`, with the comment at `:354` that m > 3 would need caching) is therefore exercised by today's
degree-88 path already, with the same m = 4; degree 118 changes **k only**.

**1.3 Levels, counted from the code.** `k768_ps_level_emulation.py` replays `evalChebyshevSeries` + `applyDoubleAngleIterations`
on (level, NoiseLevel) pairs with each `Ciphertext` member transcribed from `Ciphertext.cpp` (FLEXIBLEAUTO arms: `rescale :426`,
`mult :458`, `square :679`, `multScalar :764`, `add :164`, `dropToLevel :1141`, `evalLinearWSumMutable :1169`,
`adjustScaleAndLevel :1374`, `adjustForAddOrSub :1465`, `adjustForMult :1500`), keeps the in-place side effects on `T[]`/`T2[]`,
and does the Chebyshev long divisions on the library's real tables. Output (`k768_ps_level_emulation.txt`), input (E0, NoiseLevel 1):

| | degree 88, K 512 | degree 118, K 768 |
|---|---|---|
| (k, m) | (6, 4) | (8, 4) |
| power basis T_1..T_k, level - E0 | -1,-2,-2,-2,-3,-3 | -1,-2,-2,-2,-3,-3,-3,-3 |
| 7 recursion nodes (m, offset) -> output level - E0 | -4,-5,-5,-5,-6,-6,-6 | identical |
| series output / after 6 double angles | (-6, 2) / (-12, 2) | identical |
| levels consumed (settled in -> settled out) | 13 = 7 + 6 | 13 = 7 + 6 |
| ct x ct multiplications per ciphertext | 18 + 6 = 24 | 20 + 6 = 26 |
| rescales / weighted-sum inputs per ciphertext | 43 / 82 | 45 / 112 |

The same holds for an input with a pending rescale. `T_7 = T_3 T_4` and `T_8 = T_4^2` are built from operands already at E0-2 and
land where `T_5`, `T_6` land, so no new level and no new (level, scale-degree) pair is touched. **The bootstrap's output level
does not move.** Consistency check of the emulation against a record: 1 (the `1/(K N)` scalar, `Bootstrap.cu:235`) + 3 (CtS,
budget {3,3}) + 13 + 3 (StC) - 1 (the result leaves with its last rescale pending) = 19 = `lvlPost` on 1664/1664 traced
bootstraps at depth 41 (`pod_2xbw_20260917/pod_pull/s37/serve_L7/logs/server.jsonl`). Hence the harness's schedule, the
compressed store (keyed by level) and its stamp, and the key set are untouched. This is an emulation of the bookkeeping, not a
run: the pod check is `lvlPost 19` and a warm tick without store misses under K = 768 (section 4).

**1.4 Precision.** The table itself costs nothing visible: inside its range it reproduces the target to 8.60e-12 (K = 768) vs
1.73e-12 (K = 512) of the sine output (`a11_tick42/boot_overflow_ext_table.txt`). The pre-scaling does: the message enters the
series as `m/(q0 K)`, 1.5x smaller against whatever error CtS and the series add, and the evaluated function's slope is
proportional to K, so an error made before or inside the modular reduction is amplified 768/512 = 1.5x more: **up to 0.58 bit
of bootstrap precision** (upper bound; errors made after the reduction are unaffected). Not measured. The selftest bootstrap
reads 0.00206614 at K = 512 (`selftest_2dev/lines.jsonl`) against the harness's pass threshold of 5e-3 (the
`verdict("bootstrap", ... < 5e-3` line), so a 1.5x arm still passes but with less margin; the decisive number is the per-tick `relErrRms` A/B.

**1.5 Predicted cost.** Fully packed slots evaluate the series on two ciphertexts (`Bootstrap.cu:281-291`,
`ApproxModEval.cu:34-36`), so ct x ct multiplications per bootstrap go **48 -> 52** (the 48 re-derived here agrees with
`MEMORY_RESIDUE_TRACE_20260918.md:149`). The two new products per ciphertext sit near the top of the modular reduction's
level range; under a cost linear to quadratic in the limb count that is about +9 to +10 % of the modular reduction's
multiplication work (projection), plus 2 rescales and 30 more weighted-sum inputs per ciphertext and two more scratch
ciphertexts (k + m: 10 -> 12, transient). How much of a tick that is cannot be read from the serve lines (`reqBootMs` is an
async-dispatch figure without `--sync-timers`: the `boot` lambda's own comment): measure `reqLayerLoopMs`, both arms on the SAME binary — the env switch makes that possible (R6).
Memory residue, if `L1` is not applied: 144 B x (52 x 480 + 1,706) = 144 x 26,666 = **3,839,904 B** per request per card, against
3,563,424 B today (`MEMORY_RESIDUE_TRACE_20260918.md:13`): +276,480 B.

## 2. K1 — the patch
Inside the `UNIFORM` branch, after today's four assignments: read `FIDESLIB_BOOT_UNIFORM_EXT`; iff it is exactly `1`, overwrite
`coefficientsCheby` with `lbcrypto::FHECKKSRNS::g_coefficientsUniformExt` and `bootK` with `K_UNIFORMEXT` (both public in the
patched OpenFHE the pod builds: `deps/fideslib-ref-1.5.1.1.patch` puts `public:` before `K_SPARSE`; the same function already
reads `lbcrypto::FHECKKSRNS::R_UNIFORM` and `::g_coefficientsUniform` that way); then print, in both arms,
`{"fideslibBootTable":"g_coefficientsUniform"|"g_coefficientsUniformExt","bootK":512|768,"doubleAngleIts":6,"chebyDegree":88|118,"envBootUniformExt":false|true}`.
The variable is read once, at `LoadContext`, so it must be in the SERVER's environment at launch. A proper
`BOOT_CONFIG::UNIFORM_EXT` would be the cleaner library feature but is not equally small: enum (`forwardDefs.cuh:38`), a branch
in `GetRawParams`, and a selector the api does not have (`keyDist` has no such value), i.e. an api or harness change.

## 3. Part 2 — the flight recorder
**3.1 The point.** `ModRaise` (`Bootstrap.cu:393`): pending rescale `:422-423`; FLEXIBLEAUTO adjustment `multScalar(adjustmentFactor)`
`:478` -> `rescale()` `:492` -> `dropToLevel(0, true)` `:493` (FIDESlib does perform OpenFHE's `AdjustCiphertext`; prescaled arm
`:519-524`, FIXED* arm `:543-557`); sparse-encapsulation key switch `:592-602` (off for `UNIFORM`, `RawCiphertext.cu:581`);
then per polynomial INTT (`:618` c0, `:673` c1) -> `grow` (`:632`/`:685`) -> `broadcastLimb0` (`:649`/`:701`) -> NTT
(`:661`/`:712`). `dropToLevel` only lowers `level` (`RNSPoly.cpp:838-850`), so the ciphertext is one limb there. The broadcast
lifts with `SwitchModulus` (`ElemenwiseBatchKernels.cu:150-158`, `Rescale.cuh:44-64`): **centered**, `v > q0>>1 -> v - q0`, as
OpenFHE. Limb 0 lives on partition `cc.limbGPUid[0].x` = the first device of the list (`Context.cu:248`, `dev = i % GPUid.size()`); `RNSPoly::store`
(`RNSPoly.cpp:201-208`) already locates each limb's device and copies it with a stream-synchronized `cudaMemcpyAsync`
(`Limb.cu:66-76`); it is what `Ciphertext::store` and the harness's `hostCt` use on the same two-card box. After the INTT the data
is in COEFFICIENT representation in natural order: `multMonomial` loads `coefs[power] = 1` and calls the same `NTT`
(`Ciphertext.cpp:1627-1682`: `coefs[power] = 1` at `:1639`, `NTT` at `:1679`), and `ModRaise`/rescale themselves need `NTT(lift(INTT(x)))` to be exact; values are in `[0, q0)`:
the INTT ends in a `modmult` with a final conditional subtraction (`NTTfusions.cuh:414`, `ModMult.cuh:90-95`) and `SwitchModulus` assumes it.

**3.2 K2.** `Bootstrap.cuh`: `#define FIDESLIB_HAS_BOOT_PRERAISE_HOOK 1`, `using BootPreRaiseHook = std::function<void(const
uint64_t* c0, const uint64_t* c1, size_t N, uint64_t q0, bool evalRepresentation)>`, `extern BootPreRaiseHook g_bootPreRaiseHook`.
`Bootstrap.cu`: the definition, a 10-line static helper (synchronize every device of the polynomial, `poly.store(out)`, restore
the current device), and two insertions in `ModRaise`: right after `c0.INTT` fetch limb 0 of c0; right after `c1.INTT` fetch
limb 0 of c1 and call the hook with `evalRepresentation = false`. The arrays handed over are exactly the ones `broadcastLimb0`
reads next. Hook empty (default): one empty `std::vector` and two `if`s per bootstrap.

**3.3 K3.** `--boot-record --boot-record-secret PATH` (PATH = the OpenFHE BINARY `secret.key` of the pod TEST key set; the word
`self` takes the secret the process already holds, for smoke tests with in-process keys). Refused unless both are given, in a
client-crypto process, and in a binary without `FHE_SSM_DEMO_SER` or built against a FIDESlib without K2 (the macro; the
harness therefore compiles against a stock FIDESlib too). Setup: load the key with the loader `--allow-secret` already uses
(`fideslib::Serial::DeserializeFromFile`, no new cereal instantiation in the harness), **refuse if its key tag differs from the
loaded public key**, take tower 0 of the secret to coefficient form (`SetFormat`, the idiom `seeded_keys.hpp:178` already compiles),
require {0, 1, q0-1}, precompute `FFT(twist(s))`, print a loud stderr banner and `{"bootRecordArmed":true,...,"hamming":h,"sigmaIPred":sqrt((h+1)/12)}`.
Callback, inline on the calling thread: center both limbs, `x = c0/q0 + negacyclic(s, c1/q0)` by a self-contained radix-2
complex FFT with the twist `exp(i pi k/N)`, `I = nearbyint(x)`; one line per bootstrap
`{"bootRecord":true,"n":<boot index = bootTrace's "boot">,"maxI":..,"rmsI":..,"coeffsOver512":..,"maxFrac":..,"nonCanon":..,"stage":"..","ms":..}`
(`nonCanon` = words handed over that are not in `[0, q0)`; the device lift assumes none, expect 0);
the serve line gains `reqBootRecN, reqBootMaxI, reqBootOver512, reqBootOver530, reqBootOver537, reqBootOver768, reqBootRecMs`
(counts = bootstraps of the request whose max|I| exceeded the bound); a `{"bootRecordSummary":true,...}` line precedes the
final summary. With the flag off the added string is empty and every output line is byte-identical to HEAD's.
The arithmetic was checked in Python, loop for loop: 0 mismatches of I against exact integers on all coefficients at N = 256,
1024, 4096 and on 64 probed coefficients at N = 2^17 with a 60-bit modulus; worst float-vs-exact fraction 1.78e-13; rms I
85.208 against sqrt((h+1)/12) = 85.267 (`k3_recorder_arithmetic_check.txt`).

**3.4 Cost.** Per bootstrap: two 1 MiB device-to-host copies, two all-device synchronizations, two FFTs of 131,072 points and
three linear passes on one core: an estimated 10-20 ms (unmeasured; every line carries its `ms`), i.e. roughly 5-10 s per
480-bootstrap request, 3-6 % of a 182 s tick (`serve_A11/logs/server.jsonl`, `reqLayerLoopMs` 182431). The synchronizations also
remove host/device overlap, which `reqBootRecMs` does not see: **a run with `--boot-record` is not a timing cell**; compare walls
only between runs with the same recorder state. A worker thread was deliberately not written: `main` has dozens of early
`return`s, a joinable `std::thread` at that scope terminates the process on any of them, and the inline cost is bounded and
self-reported. If `reqBootRecMs` proves too large: a RAII-joined single consumer fed with copies of the two limbs, its lines
buffered and printed by the main thread at the serve line.

**3.5 What it proves, and what it does not.** With `maxFrac` small (it is the message `m/q0`: about |msg|/128 if the correction factor is the library formula's 7 and
the first modulus 60 bits — derived, not measured; a wrong key, modulus or coefficient order gives ~0.5) and `rmsI` near `sigmaIPred`, `maxI` is the exact overflow of the pair
FIDESlib lifted, because the hook sits after every operation that changes c0/c1 before the raise — `:422-423`, `:478`, `:492`,
`:493` (or `:519-524`), `:592-602`, the INTTs `:618`/`:673` — and before `:632`/`:649` (`:685`/`:701`). Anything the harness
does before `EvalBootstrap` is upstream of that and included. It does NOT prove that an overflowing bootstrap corrupted a reply:
that is the join of `bootRecord` lines with the request's decode result (dose and response, per request). It does not see the
reduction's actual input `(I + m/q0)/K` after CtS (CtS error is not in it), says nothing about another key set except
statistically, and under sparse encapsulation it would hold the wrong secret (it would say so: `maxFrac` ~ 0.5).

## 4. Pod verification, minutes each (the 39 s two-card selftest cell, `selftest_2dev/CMD`, in-process keys; run it with `gpu_real_model_x`)
1. Build: apply K1 + K2 in the FIDESlib checkout, rebuild AND reinstall FIDESlib, rebuild `gpu_real_model_x`. A stale install
   shows as `g_bootPreRaiseHook` undefined at link (header new, library old) or as the parse-time refusal (header old).
2. Control: the selftest cell prints `"fideslibBootTable":"g_coefficientsUniform","bootK":512,...,"chebyDegree":88` and
   `{"selftest":"bootstrap","result":"ok","err":~0.0021}`.
3. K = 768: same cell with `FIDESLIB_BOOT_UNIFORM_EXT=1`: the line reads `g_coefficientsUniformExt`, 768, 6, 118; bootstrap
   verdict ok, `err` within ~1.5x of the control. With `--trace-boots`: `lvlPost` 19, `bootRestored:true`, as the control.
4. Recorder: same cell + `--boot-record --boot-record-secret self`: one `bootRecordArmed` line (`hamming` near 87,381,
   `sigmaIPred` near 85.3), then per bootstrap `rmsI` ~ 85, `maxFrac` << 0.5, `maxI` typically 370-450
   (`a11_tick42/overflow_tail_montecarlo.txt`: P(max|I| > 380) = 0.6623 observed over 3,000 synthetic ciphertexts). `maxFrac` ~ 0.5 = instrument wrong: stop and read 3.5.
5. Serving A/B (brief section 3, cell 3): a warm K = 768 tick must show `reqBoots` 480, `reqEncPtMs` 0 (no store miss) and
   `reqHostPtEncodeCount` 2230, as `serve_A11`'s serve lines do: the level trajectory did not move. Then `relErrRms`/top-1 over 4 ticks and `reqLayerLoopMs`,
   recorder OFF in both arms. Long run: recorder ON with the pod test `secret.key`; `reqBootRecN` must equal `reqBoots`.

## 5. Rollback
Unset the variable / drop the flags: both levers are off by default. To remove the code: `git apply -R` K2 then K1 in the
FIDESlib checkout and rebuild; `git apply -R results/dense-demo-s31/pod_longrun_prep/patches/harness/K3_harness_boot_record.patch`
in the repo (the harness edit is uncommitted, so `git checkout --` also works today).

## 6. Hostile re-read: residual risks (none could be compiled away here)
- K3 uses `lbcrypto::NativePoly`, `GetElementAtIndex(0)`, `SetFormat`, `SetValuesToZero` (public, `poly.h:169`), `operator[]`,
  `NativeInteger(uint64_t)`, `ConvertToInt<uint64_t>()`, `GetPrivateElement`, `GetKeyTag`, and `std::any_cast` on
  `keys.publicKey->pimpl` / `keys.secretKey->pimpl` (the api's own pattern, `api/CryptoContext.cpp:225`, `api/Serialize.cpp:89`).
  All but `SetValuesToZero` already compile in `seeded_keys.hpp` or the harness. First suspect on a compile error: that one line
  (deleting it is harmless).
- K3 adds `#include "CKKS/Bootstrap.cuh"` to the DEMO_SER block; it pulls `"pke/openfhe.h"`, resolved by the existing
  `-I$INSTALL/include/openfhe` flag (`DEMO_README.md:161`).
- The callback lambda captures main-scope objects by reference and is held by the library until exit; nothing local to the
  setup block is referenced inside it (checked name by name). It never touches CUDA.
- K2: `RNSPoly`, `LimbPartition::device`, `cc.prime[0].p` are used exactly as the surrounding code uses them; the helper is
  host code in a `.cu`. This work did not test `store` on a level-0, coefficient-representation polynomial (the code path is
  representation-blind: it copies `level + 1` limbs).
- K1: `getenv`/`strcmp`/`cout` with explicit includes; the conditional of two string literals decays to `const char*`.
- Brace/parenthesis balance of the two new harness blocks was counted mechanically (22/22, 74/74; 19/19, 73/73).

## 7. Uncertain / not done
The K = 768 precision cost (bounded at 1.5x, unmeasured) and wall cost (projection only); level-trajectory equality rests on the
emulation plus the 19-level cross-check, not on a run; the recorder's per-bootstrap cost is an estimate; no worker thread;
`hpc_gpu_port/experimental/README.md` was not edited (only the `.cu` was in scope); nothing was committed.
