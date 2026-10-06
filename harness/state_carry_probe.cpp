// state_carry_probe.cpp — STEP-1 FALSIFIER for stateful decode (2026-08-21).
//
// THE QUESTION. hpc_gpu_port/gpu_real_model.cu:2361 zero-initialises the
// per-layer scan state on every invocation:
//
//     // per-layer carried scan state (zero-initialised, one ciphertext each)
//     std::vector<TrackedCt> scanState(L);
//     { std::vector<double> z(SLOTS, 0.0);
//       for (int l = 0; l < L; l++) scanState[l] = encCh(z); }
//
// so ml-eval/fhe_client.py must resend the whole prefix each step and the
// server rebuilds the same state from scratch (fhe_client.py:25,
// hpc_gpu_port/DEMO_README.md:252). Generating N tokens costs N(N+1)/2 scan
// steps instead of N. Making the server STATEFUL means persisting that
// ciphertext across a process boundary. This probe asks whether that is
// numerically legitimate BEFORE any plumbing is written.
//
// WHY ON CPU. Same argument cpu_real_model.cpp makes in its own header:
// FIDESlib's high-level API is deliberately OpenFHE-compatible, so circuit
// logic + OpenFHE serialization semantics are fully testable on the author's
// arm64 Mac at a small ring, in seconds, for free. Any defect found here is
// the SAME defect in the CUDA harness. Only FIDESlib API specifics need
// NVIDIA hardware.
//
// THE SCAN UNDER TEST mirrors gpu_real_model.cu:2601 (the --serial-carry
// path, which is the DEFAULT since the two-level scan regressed 2026-07-31):
//
//     refresh(scanState[l], 3); refresh(x[t], 3);
//     mulPt(scanState[l], decPk);            // decay (.) s
//     TrackedCt bx = x[t]; mulPt(bx, bPk);   // b (.) x_t
//     addAligned(scanState[l], bx);
//     S[t] = scanState[l];
//
// FOUR ARMS.
//   A  contiguous  — one process, N tokens in one loop. Today's T=N pass.
//   B  stateful, state carried IN MEMORY across N single-token "invocations".
//      Isolates "does restarting the loop per token change the numerics"
//      from "does serialization change them".
//   C  stateful, state carried THROUGH A FILE (OpenFHE Serial round trip)
//      every step. This is the real proposal.
//   D  stateless replay — for each k, rebuild from zero over tokens 0..k-1.
//      This is what the demo does TODAY. Arm C must reproduce it exactly, or
//      stateful decode is not a drop-in.
//
// THE SPECIFIC LANDMINE THIS PROBE EXISTS TO CATCH. gpu_real_model.cu:893
// `loadCt` returns `TrackedCt{fc, 0}` — tracked level hard-reset to 0. That
// is correct for --ct-in (client ciphertexts are always fresh) and would be
// WRONG for a mid-computation state, which is at whatever level the scan
// left it. TrackedCt::level() reads ct->GetLevel() and `used` is marked
// VESTIGIAL (:680), so the level SHOULD survive inside the ciphertext — but
// under FLEXIBLEAUTO a ciphertext also carries a scaling factor and a noise
// scale degree, and every later op is encoded against them. Arm C asserts
// level, noiseScaleDeg and scalingFactor are all identical across the round
// trip. This is the harness's most productive bug genus (cf. the two
// level-alignment bugs documented at gpu_real_model.cu:2313 ff., one of
// which encoded 1024 diagonals at the wrong level and produced an
// undecryptable result), so it is asserted, not assumed.
//
// EvalBootstrap IS A SILENT NO-OP UNTIL DEEP ENOUGH (the repository conventions, "instruments
// that lie"). The first version of this probe forced a bootstrap every step on
// an undepleted ciphertext, got zero real boots, and reported every arm green
// with bootstrap-free error (3.4e-12) — measuring nothing. Fixed two ways:
// `--drain K` burns K extra levels per step so refresh() fires GENUINE boots
// (the real scan step sits inside a layer that consumes many levels, so this is
// the faithful mirror, not a hack), and every boot is classified real-vs-no-op
// by comparing the level either side. A bootstrapped run with bootReal == 0 now
// FAILS instead of looking clean.
//
// R3: relErrRms is the reported figure; relErrMax is carried alongside but
// is not the verdict. R5: every number this prints is written to the JSONL
// the runner redirects into results/.
//
// Build: cmake --build harness/build -j --target state_carry_probe
// Run:   ./harness/build/state_carry_probe --tokens 8
//        ./harness/build/state_carry_probe --tokens 8 --no-bootstrap  (fast algebra-only)
//        ./harness/build/state_carry_probe --tokens 8 --drain 6        (real boots)

#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "run_stamp.h"

using namespace lbcrypto;

static int g_hard = 0, g_soft = 0;

static void verdict(const std::string& name, bool ok, double err, bool hard,
                    const std::string& extra = "") {
    std::cout << "{\"test\":\"" << name << "\",\"result\":\"" << (ok ? "ok" : "FAIL")
              << "\",\"relErrRms\":" << err << extra << "}" << std::endl;
    if (!ok) (hard ? g_hard : g_soft)++;
}

// R3 — rms is the reported figure. Also returns max, and flags the two
// documented liar-modes: an EXACT zero (std::max(a,b) returns a when b is
// NaN, so relErr == 0 is a red flag, not a perfect result) and any NaN.
struct Err { double rms = 0, max = 0; bool nan = false, exactZero = false; };

static Err relErr(const std::vector<double>& got, const std::vector<double>& ref) {
    Err e;
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double diff = got[i] - ref[i];
        if (std::isnan(diff) || std::isnan(got[i])) e.nan = true;
        num += diff * diff;
        den += ref[i] * ref[i];
        const double a = std::abs(diff);
        if (!(a <= e.max)) e.max = a;   // NaN-safe: NaN fails the <= and is caught above
    }
    e.rms = (den > 0) ? std::sqrt(num / den) : std::sqrt(num);
    e.exactZero = (e.rms == 0.0);
    return e;
}

static std::string errFields(const Err& e) {
    std::ostringstream o;
    o << ",\"relErrMax\":" << e.max
      << ",\"nan\":" << (e.nan ? "true" : "false")
      << ",\"exactZero\":" << (e.exactZero ? "true" : "false");
    return o.str();
}

int main(int argc, char** argv) {
    uint32_t logRing = 16;      // bootstrapping below 2^16 trips an OpenFHE
                                // correction-factor constraint (cpu_real_model.cpp:63)
    uint32_t N = 8;             // tokens
    bool doBootstrap = true;    // exercise the real risk: state through boots
    uint32_t drain = 4;         // extra levels burned per step (see scanStep)
    bool doReplay = true;       // arm D (quadratic; the expensive one)
    std::string tmpDir = ".";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--log-ring") logRing = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--tokens") N = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--no-bootstrap") doBootstrap = false;
        else if (a == "--drain") drain = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--no-replay") doReplay = false;
        else if (a == "--tmp-dir") tmpDir = argv[++i];
    }
    if (doBootstrap && logRing < 16) logRing = 16;
    // Draining only makes sense when a bootstrap can reclaim the levels; with
    // --no-bootstrap it just walks the ciphertext off the end of the depth.
    const uint32_t levelsAfterConst = 10;   // must match levelsAfter below
    if (!doBootstrap) drain = 0;
    if (doBootstrap && drain + 2 + 3 > levelsAfterConst) {
        std::cout << "{\"fatal\":\"drain too large: need(" << (drain + 2)
                  << ")+BOOT_FLOOR(3) exceeds levelsAfter(" << levelsAfterConst
                  << ") -- refresh would boot unconditionally forever\"}" << std::endl;
        return 2;
    }

    const uint32_t d = 48;
    uint32_t D = 1; while (D < d) D <<= 1;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1u << logRing);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    std::vector<uint32_t> levelBudget = {4, 4};
    const uint32_t levelsAfter = 10;
    const uint32_t depth = doBootstrap ? levelsAfter + 19 : 30;
    parameters.SetMultiplicativeDepth(depth);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE); if (doBootstrap) cc->Enable(FHE);
    const uint32_t SLOTS = cc->GetRingDimension() / 2;
    const uint32_t REP = SLOTS / D;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    if (doBootstrap) {
        cc->EvalBootstrapSetup(levelBudget, {0, 0}, SLOTS);
        cc->EvalBootstrapKeyGen(keys.secretKey, SLOTS);
    }

    const uint32_t BOOT_FLOOR = 3;   // gpu_real_model.cu --boot-floor default in the demo cmdline

    std::cout << "{\"harness\":\"state_carry_probe\",\"ringDim\":" << cc->GetRingDimension()
              << ",\"slots\":" << SLOTS << ",\"D\":" << D << ",\"REP\":" << REP
              << ",\"d\":" << d << ",\"depth\":" << depth << ",\"tokens\":" << N
              << ",\"bootstrap\":" << (doBootstrap ? "true" : "false")
              << ",\"drain\":" << drain << "}" << std::endl;

    struct TrackedCt {                        // mirrors gpu_real_model.cu:678
        Ciphertext<DCRTPoly> ct;
        uint32_t level() const { return ct ? (uint32_t)ct->GetLevel() : 0u; }
    };

    long long bootCount = 0, bootReal = 0, bootNoop = 0;
    long long stepCount = 0;

    auto packCh = [&](const std::vector<double>& v) {
        std::vector<double> s(SLOTS, 0.0);
        for (uint32_t r = 0; r < REP; r++)
            for (size_t i = 0; i < v.size() && i < D; i++) s[r * D + i] = v[i];
        return s;
    };
    auto enc = [&](const std::vector<double>& packed) {
        return TrackedCt{cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(packed))};
    };
    auto dec = [&](const TrackedCt& c) {
        Plaintext p; cc->Decrypt(keys.secretKey, c.ct, &p);
        p->SetLength(SLOTS);
        auto full = p->GetRealPackedValue();
        return std::vector<double>(full.begin(), full.begin() + d);
    };
    auto ptAt = [&](const std::vector<double>& packed, uint32_t lvl) {
        return cc->MakeCKKSPackedPlaintext(packed, 1, lvl);
    };
    auto mulPt = [&](TrackedCt& c, const std::vector<double>& packed) {
        c.ct = cc->EvalMult(c.ct, ptAt(packed, c.level()));
        cc->RescaleInPlace(c.ct);
    };
    auto alignTo = [&](TrackedCt& c, uint32_t target) {
        std::vector<double> one(SLOTS, 1.0);
        while (c.level() < target) {
            c.ct = cc->EvalMult(c.ct, ptAt(one, c.level()));
            cc->RescaleInPlace(c.ct);
        }
    };
    auto addAligned = [&](TrackedCt& a, TrackedCt& b) {   // gpu_real_model.cu addAligned
        const uint32_t tgt = std::max(a.level(), b.level());
        alignTo(a, tgt); alignTo(b, tgt);
        a.ct = cc->EvalAdd(a.ct, b.ct);
    };
    // EvalBootstrap is a SILENT NO-OP until the ciphertext is deep enough
    // (the repository conventions, "instruments that lie"). A probe that calls it on a fresh
    // ciphertext measures nothing and reports success. So every call records
    // the level either side and classifies itself: a REAL boot restores levels
    // (consumed-level count goes DOWN), a no-op leaves it unchanged.
    auto boot = [&](TrackedCt& c) {
        if (!doBootstrap) return;
        const uint32_t before = c.level();
        c.ct = cc->EvalBootstrap(c.ct);
        const uint32_t after = c.level();
        bootCount++;
        if (after < before) bootReal++; else bootNoop++;
    };
    auto refresh = [&](TrackedCt& c, uint32_t need) {      // gpu_real_model.cu:1078
        if (c.level() > depth) { boot(c); return; }
        const uint32_t remaining = depth - c.level();
        if (remaining < need + BOOT_FLOOR) boot(c);
    };

    // ---- the single scan step under test (gpu_real_model.cu:2601) --------
    auto scanStep = [&](TrackedCt& s, TrackedCt x,
                        const std::vector<double>& decPk, const std::vector<double>& bPk) {
        // The `need` MUST cover what this step actually consumes (1 mulPt +
        // `drain`), not a nominal 3. Understating it walks the ciphertext off
        // the bottom -- the DropLastElement crash this probe hit at drain=6.
        // And need+BOOT_FLOOR must stay <= levelsAfter, or the condition is
        // UNSATISFIABLE and every refresh boots forever (the documented
        // bootstrap explosion, gpu_real_model.cu:1088).
        refresh(s, drain + 2); refresh(x, 3);
        mulPt(s, decPk);
        TrackedCt bx = x; mulPt(bx, bPk);
        addAligned(s, bx);
        // DRAIN. In the real harness the scan step is embedded in a LAYER that
        // consumes many levels; the bare recurrence consumes ~1. Draining
        // `drain` extra levels per step is the faithful mirror, and it is what
        // makes refresh() trigger GENUINE bootstraps within a short run instead
        // of silent no-ops on an undepleted ciphertext.
        for (uint32_t k = 0; k < drain; k++) alignTo(s, s.level() + 1);
        stepCount++;
    };

    // ---- workload -------------------------------------------------------
    std::vector<double> a(d), b(d);
    for (uint32_t i = 0; i < d; i++) { a[i] = 0.90 + 0.09 * (i % 7) / 7.0; b[i] = 1.0 - a[i]; }
    const std::vector<double> decPk = packCh(a), bPk = packCh(b);

    std::vector<std::vector<double>> xs(N, std::vector<double>(d));
    for (uint32_t t = 0; t < N; t++)
        for (uint32_t i = 0; i < d; i++) xs[t][i] = std::sin(0.3 * t + 0.11 * i) * 0.4;

    // plaintext ground truth: ref_t = a (.) ref_{t-1} + b (.) x_t
    std::vector<std::vector<double>> refS(N, std::vector<double>(d, 0.0));
    { std::vector<double> r(d, 0.0);
      for (uint32_t t = 0; t < N; t++) {
          for (uint32_t i = 0; i < d; i++) r[i] = a[i] * r[i] + b[i] * xs[t][i];
          refS[t] = r;
      } }

    const std::vector<double> zero(SLOTS, 0.0);

    // ==== CONTROL — is decryption bit-deterministic? (it is NOT) ==========
    // Establishes the floor below which no serialization test can resolve.
    // Two decrypts of the SAME ciphertext, no serialization in between.
    {
        TrackedCt c = enc(packCh(xs[0]));
        const Err e = relErr(dec(c), dec(c));
        std::cout << "{\"control\":\"decryptJitter_sameCiphertext\",\"relErrRms\":" << e.rms
                  << ",\"relErrMax\":" << e.max
                  << ",\"note\":\"nonzero => decrypted-value equality CANNOT test serialization;"
                     " arm C asserts RNS-polynomial bit-identity instead\"}" << std::endl;
    }

    // ================= ARM A — contiguous, one process ===================
    std::vector<std::vector<double>> outA(N);
    std::vector<uint32_t> lvlA(N);
    {
        TrackedCt s = enc(zero);
        for (uint32_t t = 0; t < N; t++) {
            scanStep(s, enc(packCh(xs[t])), decPk, bPk);
            outA[t] = dec(s); lvlA[t] = s.level();
        }
        Err e = relErr(outA[N - 1], refS[N - 1]);
        std::ostringstream x; x << errFields(e) << ",\"finalLevel\":" << lvlA[N - 1];
        verdict("A_contiguous_vs_plaintext", e.rms < 1e-2 && !e.nan && !e.exactZero,
                e.rms, true, x.str());
    }

    // ================= ARM B — stateful, in-memory carry ==================
    std::vector<std::vector<double>> outB(N);
    {
        TrackedCt s = enc(zero);
        for (uint32_t t = 0; t < N; t++) {
            // one "invocation": exactly one token, state survives in memory
            scanStep(s, enc(packCh(xs[t])), decPk, bPk);
            outB[t] = dec(s);
        }
        Err ep = relErr(outB[N - 1], refS[N - 1]);
        Err ea = relErr(outB[N - 1], outA[N - 1]);
        std::ostringstream x; x << errFields(ep) << ",\"vsArmA_relErrRms\":" << ea.rms;
        verdict("B_stateful_memory_vs_plaintext", ep.rms < 1e-2 && !ep.nan, ep.rms, true, x.str());
    }

    // ================= ARM C — stateful, serialized carry =================
    // The real proposal: state leaves the process every step.
    std::vector<std::vector<double>> outC(N);
    long long stateBytes = 0;
    int serMismatch = 0;
    std::vector<std::pair<uint32_t, long long>> stateBytesAtLevel;
    {
        TrackedCt s = enc(zero);
        for (uint32_t t = 0; t < N; t++) {
            scanStep(s, enc(packCh(xs[t])), decPk, bPk);

            // --- serialize -> file -> deserialize, exactly as a server restart would
            const std::string path = tmpDir + "/state_carry_probe.state.bin";
            const uint32_t lvlBefore  = s.level();
            const uint32_t degBefore  = (uint32_t)s.ct->GetNoiseScaleDeg();
            const double   sfBefore   = s.ct->GetScalingFactor();
            const auto     elemBefore = s.ct->GetElements();

            if (!Serial::SerializeToFile(path, s.ct, SerType::BINARY)) {
                std::cout << "{\"fatal\":\"state serialize failed\"}" << std::endl; return 3;
            }
            { std::FILE* f = std::fopen(path.c_str(), "rb");
              if (f) { std::fseek(f, 0, SEEK_END); stateBytes = std::ftell(f); std::fclose(f); } }
            stateBytesAtLevel.push_back({lvlBefore, stateBytes});

            Ciphertext<DCRTPoly> back;
            if (!Serial::DeserializeFromFile(path, back, SerType::BINARY)) {
                std::cout << "{\"fatal\":\"state deserialize failed\"}" << std::endl; return 3;
            }
            s.ct = back;   // the state the "next invocation" would resume from

            // EXACTNESS TEST. Decryption is NOT bit-deterministic in OpenFHE
            // CKKS (measured: two decrypts of the SAME ciphertext differ by
            // ~1.9e-12 -- see the decryptJitter control below), so comparing
            // DECRYPTED values cannot test serialization. The correct test is
            // bit-identity of the underlying RNS polynomials, plus the three
            // pieces of FLEXIBLEAUTO metadata every later op is encoded
            // against (level / noise scale degree / scaling factor).
            const auto elemAfter = s.ct->GetElements();
            bool bitIdentical = (elemBefore.size() == elemAfter.size());
            for (size_t k = 0; k < elemBefore.size() && bitIdentical; k++)
                bitIdentical = (elemBefore[k] == elemAfter[k]);

            const uint32_t lvlAfter = s.level();
            const uint32_t degAfter = (uint32_t)s.ct->GetNoiseScaleDeg();
            const double   sfAfter  = s.ct->GetScalingFactor();

            const bool ok = (lvlBefore == lvlAfter) && (degBefore == degAfter)
                            && (sfBefore == sfAfter) && bitIdentical;
            if (!ok) {
                serMismatch++;
                std::cout << "{\"serRoundTrip\":\"MISMATCH\",\"t\":" << t
                          << ",\"levelBefore\":" << lvlBefore << ",\"levelAfter\":" << lvlAfter
                          << ",\"noiseDegBefore\":" << degBefore << ",\"noiseDegAfter\":" << degAfter
                          << ",\"scaleBefore\":" << sfBefore << ",\"scaleAfter\":" << sfAfter
                          << ",\"bitIdentical\":" << (bitIdentical ? "true" : "false") << "}"
                          << std::endl;
            }
            outC[t] = dec(s);
        }
        Err ep = relErr(outC[N - 1], refS[N - 1]);
        Err ea = relErr(outC[N - 1], outA[N - 1]);
        std::ostringstream x;
        x << errFields(ep) << ",\"vsArmA_relErrRms\":" << ea.rms
          << ",\"serMismatches\":" << serMismatch
          << ",\"stateBytesPerLayer\":" << stateBytes;
        verdict("C_stateful_serialized_vs_plaintext",
                ep.rms < 1e-2 && !ep.nan && serMismatch == 0, ep.rms, true, x.str());
        verdict("C_serialized_roundtrip_preserves_level_scale", serMismatch == 0,
                (double)serMismatch, true, "");

        // STATE SIZE vs LEVEL. A ciphertext at level L carries (depth-L+1)
        // RNS limbs, so the state blob SHRINKS as the scan depletes it. This
        // sets where in the step the state should cross the process boundary,
        // and whether the blob can live client-side at all (step 3).
        for (auto& pr : stateBytesAtLevel)
            std::cout << "{\"stateSize\":true,\"level\":" << pr.first
                      << ",\"limbs\":" << (depth - pr.first + 1)
                      << ",\"bytes\":" << pr.second
                      << ",\"MB\":" << (double)pr.second / 1048576.0 << "}" << std::endl;
    }

    // ============ ARM D — stateless replay (what the demo does today) =====
    if (doReplay) {
        const long long before = stepCount;
        std::vector<std::vector<double>> outD(N);
        for (uint32_t k = 1; k <= N; k++) {
            TrackedCt s = enc(zero);
            for (uint32_t t = 0; t < k; t++) scanStep(s, enc(packCh(xs[t])), decPk, bPk);
            outD[k - 1] = dec(s);
        }
        const long long replaySteps = stepCount - before;
        Err ec = relErr(outC[N - 1], outD[N - 1]);
        std::ostringstream x;
        x << errFields(ec)
          << ",\"replayScanSteps\":" << replaySteps
          << ",\"statefulScanSteps\":" << N
          << ",\"workRatio\":" << (double)replaySteps / (double)N;
        verdict("D_stateful_C_reproduces_stateless_replay",
                ec.rms < 1e-2 && !ec.nan, ec.rms, true, x.str());
    }

    // A bootstrapped run that produced ZERO real boots measured nothing about
    // state-through-bootstrap, however green its other verdicts look.
    if (doBootstrap)
        verdict("bootstraps_were_real_not_silent_noops", bootReal > 0, (double)bootNoop, true,
                ",\"bootReal\":" + std::to_string(bootReal) + ",\"bootNoop\":" + std::to_string(bootNoop));

    std::cout << "{\"summary\":true,\"hardFail\":" << g_hard << ",\"softFail\":" << g_soft
              << ",\"bootstraps\":" << bootCount << ",\"bootReal\":" << bootReal
              << ",\"bootNoop\":" << bootNoop << ",\"scanSteps\":" << stepCount
              << fhe_ssm::runStamp(argc, argv) << "}" << std::endl;
    return g_hard ? 3 : 0;
}
