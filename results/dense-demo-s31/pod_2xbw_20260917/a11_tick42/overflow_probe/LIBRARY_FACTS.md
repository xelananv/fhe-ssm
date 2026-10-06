# Library facts verified by reading the vendored OpenFHE v1.5.1 source (2026-09-18)

Tree: `vendor/openfhe-development/` (no `.git`; `CMakeLists.txt:30-33` sets version 1.5.1). The binary every probe here links is
`vendor/install/lib/libOPENFHEpke.1.5.1.dylib` (sha256 `b0e96386b7fa…`, built 2026-08-29) and `libOPENFHEcore.1.5.1.dylib`
(sha256 `3b93e1d1298d…`), `WITH_OPENMP=OFF` (`vendor/build/CMakeCache.txt:290`). `find vendor/openfhe-development/src -newer
vendor/openfhe-development/CMakeLists.txt` lists ONE file, `ckksrns-schemeswitching.cpp` (the S4.0 work); `ckksrns-fhe.cpp`
(sha256 `e292ad178dde…`, mtime Apr 10) and `ckksrns-fhe.h` (sha256 `eb18043004b8…`) predate the library build. The installed header
`vendor/install/include/openfhe/pke/scheme/ckksrns/ckksrns-fhe.h` is identical to the source header over its first 400 lines (diffed).

Paths below are relative to `vendor/openfhe-development/src/`.

| fact | where |
|---|---|
| `K_UNIFORM = 512` ("upper bound for the number of overflows in the uniform secret case") | `pke/include/scheme/ckksrns/ckksrns-fhe.h:424` |
| `K_UNIFORMEXT = 768` (composite degree > 2) | `ckksrns-fhe.h:426` |
| `R_UNIFORM = 6` double-angle iterations | `ckksrns-fhe.h:428` |
| `g_coefficientsUniform` (89 coefficients, degree 88) | `ckksrns-fhe.h:463`; the K = 768 table `g_coefficientsUniformExt` at `:489` |
| table selection in `EvalBootstrap`: not `SPARSE_TERNARY` (`:617`), not `SPARSE_ENCAPSULATED` (`:622`), and `(compositeDegree == 1) \|\| ((compositeDegree == 2) && (N < (1 << 17)))` -> `coefficients = g_coefficientsUniform; k = K_UNIFORM;` else the Ext table with `K_UNIFORMEXT` | `pke/lib/scheme/ckksrns/ckksrns-fhe.cpp:626-635` (condition at `:628`); comment at `:627`: "For larger composite degrees, larger K used to achieve a reasonable probability of failure" |
| so for a non-composite (`FLEXIBLEAUTO`, compositeDegree 1) `UNIFORM_TERNARY` context the ring dimension does NOT enter the choice: K = 512 at every N, including 2^17 | same lines |
| in `EvalBootstrapSetup`/`EvalBootstrapPrecompute` the `UNIFORM_TERNARY` case uses `k = 1.0` for the encoding matrices (`scaleEnc = pre / k`), i.e. the division by K is NOT folded into the precomputation (it is for the sparse cases) | `ckksrns-fhe.cpp:177-190`, `:343-356`, `:197`, `:363` |
| the division by K happens on the raised ciphertext: `cc->EvalMultInPlace(raised, pre * (1.0 / (k * N)))` with `k = K_UNIFORM` | `ckksrns-fhe.cpp:638` |
| path from `EvalBootstrap`'s argument to ModRaise: `raised = ciphertext->Clone()`; `ModReduceInternalInPlace(raised, compositeDegree * (noiseScaleDeg - 1))`; `AdjustCiphertext(raised, 2^-correction, lvl)` | `ckksrns-fhe.cpp:558-562`; `correction = m_correctionFactor - deg`, `deg = round(log2(q0 / 2^p))` at `:532-541` |
| `AdjustCiphertext` (FLEXIBLE*): `adjustmentFactor = (targetSF/sourceSF) * (modToDrop/sourceSF) * correction`; `EvalMultInPlace`; `ModReduceInternalInPlace`; `SetScalingFactor(targetSF)` -- deterministic, reads nothing but its arguments | `ckksrns-fhe.cpp:2228-2263` |
| ModRaise itself: for every ciphertext element, `SetFormat(COEFFICIENT)`, `DCRTPoly tmp(dcrt.GetElementAtIndex(0), elementParamsRaisedPtr)`, `SetFormat(EVALUATION)` -- "Only level 0 ciphertext used here. Other towers ignored" | `ckksrns-fhe.cpp:591-601` |
| that constructor copies the q0 tower into every tower and calls `SwitchModulus` on towers 1.. | `core/include/lattice/hal/default/dcrtpoly-impl.h:87-93` |
| `SwitchModulus` is the CENTERED lift: a value `v > q0 >> 1` is treated as `v - q0` | `core/lib/math/hal/intnat/mubintvecnat.cpp:99-121` |
| hence the raised ciphertext decrypts to `t = [c0]_q0 + [c1]_q0 * s` over the integers `= m' + q0 * I`, and the quantity the polynomial sees is `x_j = t_j / (q0 * K) ~ I_j / K` (each of the N coefficients is its own real slot: `ctxtEnc = enc + conj`, `ctxtEncI = (enc - conj) * X^(3N/2)`) | `ckksrns-fhe.cpp:664-671` |
| Chebyshev evaluation on `[-1, 1]` (`coeffLowerBound = -1.0`, `coeffUpperBound = 1.0`) | `ckksrns-fhe.cpp:640-642`, `:691-692` |
| double angle: `numIter = R_UNIFORM` iff `UNIFORM_TERNARY`; `y <- 2 y^2 - (2 pi)^(-2^i)`, `i = 1-R .. 0` | `ckksrns-fhe.cpp:699-705`, `:2366-2377` |
| the message is scaled back by `2^deg` (`MultByIntegerInPlace(ctxtEnc, scalar)`) and by `2^correction` (`corFactor`) -> an error `eps` of the sine output becomes `eps * 2^F` in a plaintext coefficient, `F = m_correctionFactor` | `ckksrns-fhe.cpp:710-713`, `:821-825` |
| default correction factor (FLEXIBLE*): `round(-0.2419 * (2 log2(M/2) + log2(slots)) + 19.081)` clamped to `[7, 14]` | `ckksrns-fhe.cpp:101-111`. Arithmetic, not a measurement: N = 2^17, slots = 2^16 gives `round(6.986) = 7`; measured here: 10 at N = 2^13/4096 slots, 11 at N = 2^12/2048 slots (the `correctionFactor` field of the probe's header records) |
| silent no-op guard: if the output has no more towers than the input, `EvalBootstrap` returns the input clone | `ckksrns-fhe.cpp:833-836` (the probe records `towersIn`, `towersOut`, `bootRestored`) |
| the decode check that throws: REAL data type and `logstd > p - 5.0` (the imaginary-part noise estimate leaves fewer than 5 bits) -> "The decryption failed because the approximation error is too high. Check the parameters." | `pke/lib/encoding/ckkspackedencoding.cpp:426-428` (estimate), `:451-455` (condition at `:452`; the exception text names `l.455`) |
| the same decode ADDS Gaussian noise of the estimated standard deviation to every slot (`std::normal_distribution<> d(0, stddev)`), so slot-domain error shapes are not the raw error; the probe therefore also reads the plaintext polynomial coefficient by coefficient without the decoder | `ckkspackedencoding.cpp:463-486` (`std::normal_distribution<> d(0, stddev)` at `:468`) |
| public/secret key layout: `s` sampled, `a` uniform, `e` Gaussian, `b = ns*e - a*s`, public elements `{b, a}`, key tag copied from the secret key | `pke/lib/schemebase/base-pke.cpp:47-96` (mirrored by the probe's `handKeys`) |
| the serialized scheme carries `m_correctionFactor` (`corFactor`) and the bootstrap parameter map | `ckksrns-fhe.h:266-279` |
