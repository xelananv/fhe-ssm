// cpu_real_model.cpp — CPU/OpenFHE VALIDATOR for the real-model circuit.
//
// WHY THIS EXISTS (2026-07-18): hpc_gpu_port/gpu_real_model.cu was authored
// without GPU access. Its risk splits in two:
//   (a) CIRCUIT LOGIC — slot packing, rotation conventions, BSGS matvec
//       algebra, RMSNorm reduction + Newton rsqrt, level accounting;
//   (b) FIDESlib API specifics — names/signatures on the CUDA backend.
// FIDESlib's high-level API is deliberately OpenFHE-compatible, so (a) is
// fully testable here on CPU OpenFHE — which runs natively on the author's
// arm64 Mac — at a small ring, in seconds, for free. Any bug found here is
// the SAME bug in the CUDA harness. Only (b) needs NVIDIA hardware.
//
// It already earned its keep: writing it exposed two real defects in the
// first gpu_real_model.cu draft, both fixed here and ported back:
//   1. BSGS giant-step: the plaintext diagonal must be pre-rotated by -g,
//      because the giant rotation is applied AFTER the multiply. Deriving
//      it: rot(inner_g, g)[i] = sum_j d'_j[i+g] u[i+g+j], and we need
//      sum_j diag_{g+j}[i] u[i+g+j], hence d'_j = rot(diag_{g+j}, -g).
//      The draft used diag_k directly -> every matvec silently wrong.
//   2. PERIOD vs MODULUS (the deep one, caught by matvec_square FAIL with
//      err 1.17). The diagonal identity
//          y[i] = sum_k diag_k[i] * u[(i+k) mod cols],  diag_k[i]=W[i][(i+k)%cols]
//      needs rotation modulo **cols**. A ciphertext rotation is modulo the
//      packing period **D**. The draft had D=1024 with cols=768, so for
//      i+k >= 768 the rotation read PADDING ZEROS instead of wrapping —
//      silently wrong output, no error raised. Fix: the packing period must
//      EQUAL the padded input dimension (D == cols), and the input must be
//      replicated with that period.
//   3. Consequently rectangular matvecs must be CHUNKED: a d->dff map
//      (rows > D) is ceil(rows/D) separate D-column matvecs, each producing
//      one output ciphertext; dff->d is the transpose, ceil(cols/D) matvecs
//      summed. This also removes the draft's mixed-period defect (period
//      1024 in and 4096 out inside one ciphertext). Elementwise ops (poly
//      gates) act on each chunk independently — no cross-channel coupling —
//      so chunking is exact, and the FLOP count is inherently the same.
//
// Build:  cmake --build harness/build -j   (target cpu_real_model)
// Run:    ./harness/build/cpu_real_model            # all logic tests
//         ./harness/build/cpu_real_model --bootstrap  # + a bootstrap test
// Output: one JSON verdict per test; exit 0 = all pass, 3 = a hard failure.

#include "openfhe.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <set>
#include <string>
#include <vector>

// Provenance stamp (harness file + sha256, git commit, build time, exact argv).
#include "run_stamp.h"

using namespace lbcrypto;

// S3.7 V2 (2026-09-05): the timer shapes of gpu_real_model.cu (normBootMs, the
// layer-loop stage windows, the host-encode counters), mirrored here with the
// SAME names so the accounting compiles and runs on the Mac -- the .cu cannot be
// compiled here. Pure instrumentation: no test's arithmetic changes.
using Clock = std::chrono::steady_clock;
static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static int g_hard = 0, g_soft = 0;
static void verdict(const std::string& name, bool ok, double err, bool hard) {
    std::cout << "{\"test\":\"" << name << "\",\"result\":\"" << (ok ? "ok" : "FAIL")
              << "\",\"err\":" << err << "}" << std::endl;
    if (!ok) (hard ? g_hard : g_soft)++;
}

int main(int argc, char** argv) {
    bool doBootstrap = false;
    uint32_t logRing = 14;      // 2^14 is fine for the algebra tests; small-ring
                                // bootstrapping trips an OpenFHE correction-factor
                                // constraint, so --bootstrap wants >= 2^16.
    // ---- S3.3 (2026-09-03) validation cells, each OFF unless named by --cell:
    //   shared-babies : the BSGS baby set computed once and reused across the
    //                   chunk calls gives BIT-IDENTICAL ciphertexts (polynomials
    //                   compared element-wise, not just decrypted values).
    //   newton-depth2 : y' = 1.5y + ((-0.5 ms) y)(y y) against the stock chain
    //                   at the same seed/iterations: relErr of both vs exact, the
    //                   difference between them, and the levels each consumes.
    //   pf-trace      : ONE full layer (T = 1, h = 1) under the parent-first
    //                   placements with REAL OpenFHE bootstraps (needs --bootstrap,
    //                   2^16, level budget {3,3}); prints (site, GetLevel,
    //                   remaining) at every refresh/boot so the sequence can be
    //                   compared with results/theory/tick_level_trace.py
    //                   --trace-levels at the same (depth, lam, F, m, I, Dpad, dff).
    // The pre-existing tests above/below run exactly as before (their output is
    // unchanged when no --cell is given).
    std::set<std::string> cells;
    std::string pfEntry = "loop"; bool cellNd2 = false; int cellIters = 3; int cellF = 3, cellM = 4;
    std::string cellSchedule = "parent-first";
    // S3.7 V1 (2026-09-05, S3.6 T18 §3): the norm's eps. The GPU harness (and this twin)
    // accepted an `eps` argument and never read it, so the circuit normalised by
    // 1/sqrt(mean(x^2)) where the model uses mean + 1e-5. DEFAULT ON (the model's
    // frame); --no-norm-eps restores the recorded no-eps chain. Mirrors
    // gpu_real_model.cu's normEpsOn / normEpsUsed / the summary field normEps.
    bool normEpsOn = true; double normEpsUsed = 0.0;
    // S3.7 V1: --norm-seed A B ITERS (default OFF) runs the norm once more with a REAL
    // bundle seed, the CPU twin of gpu_real_model.cu's selftest 4b ("rmsnormSeeded").
    double normSeedA = 0.0, normSeedB = 0.0; int normSeedIters = 0;
    // S3.7 V2 (2026-09-05): --no-stage-timers mirrors the .cu flag (default ON; OFF =
    // no stage windows, the stage fields print 0). --allow-small-ring-boot lets
    // --bootstrap run below 2^16 (a Mac RAM aid for the pf-trace cell: at 2^16 the
    // cell peaked at 7,414,693,888 B RSS = 6.9 GiB, above the 6 GB cap -- record
    // results/theory/s36_20260905/t03_bootcount/cpu_pftrace_pf_nd2_i1_control.time;
    // at 2^15 the level trace is the same circuit (depth 30, budget {3,3}, full
    // packing) and the caller owns the small-ring bootstrap precision).
    bool stageTimers = true, allowSmallRingBoot = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--bootstrap") doBootstrap = true;
        else if (a == "--no-norm-eps") normEpsOn = false;
        else if (a == "--norm-seed") { normSeedA = std::stod(argv[++i]); normSeedB = std::stod(argv[++i]); normSeedIters = std::stoi(argv[++i]); }
        else if (a == "--no-stage-timers") stageTimers = false;
        else if (a == "--stage-timers") stageTimers = true;
        else if (a == "--allow-small-ring-boot") allowSmallRingBoot = true;
        else if (a == "--log-ring") logRing = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--cell") cells.insert(argv[++i]);
        else if (a == "--pf-entry") pfEntry = argv[++i];
        else if (a == "--newton-depth2") cellNd2 = true;
        else if (a == "--schedule") cellSchedule = argv[++i];
        else if (a == "--iters") cellIters = std::stoi(argv[++i]);
        else if (a == "--boot-floor") cellF = std::stoi(argv[++i]);
        else if (a == "--margin") cellM = std::stoi(argv[++i]);
    }
    if (cells.count("pf-trace") && !doBootstrap) { doBootstrap = true; }   // the trace needs real boots
    if (doBootstrap && logRing < 16 && !allowSmallRingBoot) logRing = 16;

    // ---- small parameters: this validates ALGEBRA, not performance -------
    const uint32_t d = 48;         // model channels (deliberately NOT a power
                                   // of 2, to exercise the padding path)
    const uint32_t dff = 128;      // ffn channels -> chunked into dff/D maps
    uint32_t D = 1;                // packing period == PADDED INPUT DIM.
    while (D < d) D <<= 1;         // = 64 here; rotation mod D == mod cols
    const uint32_t NCHUNK = (dff + D - 1) / D;   // ffn output chunks (=2)

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1u << logRing);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    std::vector<uint32_t> levelBudget = {4, 4};
    if (cells.count("pf-trace")) levelBudget = {3, 3};     // the demo budget (HARNESS_BUGS B4: {4,4} consumes 22)
    const uint32_t levelsAfter = 10;
    const uint32_t depth = doBootstrap ? (cells.count("pf-trace") ? 30u : levelsAfter + 19) : 30;
    parameters.SetMultiplicativeDepth(depth);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE); if (doBootstrap) cc->Enable(FHE);
    const uint32_t SLOTS = cc->GetRingDimension() / 2;
    const uint32_t REP = SLOTS / D;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    const int BS = (int)std::ceil(std::sqrt((double)D));
    {
        std::vector<int32_t> rots;
        for (int j = 1; j < BS; j++) rots.push_back(j);
        for (int g = 0; g < (int)D; g += BS) if (g) rots.push_back(g);
        for (int p = 1; p < (int)D; p <<= 1) rots.push_back(p);
        std::sort(rots.begin(), rots.end());
        rots.erase(std::unique(rots.begin(), rots.end()), rots.end());
        cc->EvalRotateKeyGen(keys.secretKey, rots);
    }
    if (doBootstrap) { cc->EvalBootstrapSetup(levelBudget, {0, 0}, SLOTS);
                       cc->EvalBootstrapKeyGen(keys.secretKey, SLOTS); }

    std::cout << "{\"harness\":\"cpu_real_model\",\"ringDim\":" << cc->GetRingDimension()
              << ",\"slots\":" << SLOTS << ",\"D\":" << D << ",\"REP\":" << REP
              << ",\"d\":" << d << ",\"dff\":" << dff << ",\"nChunk\":" << NCHUNK
              << ",\"BS\":" << BS << ",\"depth\":" << depth << "}" << std::endl;

    struct TrackedCt { Ciphertext<DCRTPoly> ct; uint32_t used = 0; };

    // ---- S3.7 V2 (2026-09-05): the .cu's timer accumulators, same names, same
    // arithmetic (gpu_real_model.cu, the block after syncDev). evalMs/bootMs/encMs/
    // decMs are the historical windows (matvec + rmsnorm; boot; enc; dec);
    // normBootMs = the rmsnorm windows' own boots (in evalMs AND bootMs, moved
    // out at the window exits); the stage windows bill wall minus the timed delta
    // inside them; hostPtEncode* counts every ptAt (the twin's mkPt). syncDev is a
    // no-op here: the CPU twin has no device to drain.
    double evalMs = 0, bootMs = 0, encMs = 0, decMs = 0, normBootMs = 0;
    double hostPtEncodeMs = 0, hostPtEncodeStageMs = 0;
    long long hostPtEncodeCount = 0, hostPtEncodeStageCount = 0;
    int stageOpenDepth = 0;
    double layerLoopMs = 0, layerLoopTimedMs = 0;
    double shiftMixMs = 0, scanMs = 0, gateMs = 0, hiddenMs = 0, residualMs = 0;
    auto syncDev = [&] { /* CPU twin: no device */ };
    struct StageMark { Clock::time_point t; double ev, nb, bo, en, de; };
    auto stageOpen = [&]() -> StageMark {
        if (!stageTimers) return StageMark{};
        syncDev();
        stageOpenDepth++;
        return StageMark{Clock::now(), evalMs, normBootMs, bootMs, encMs, decMs};
    };
    auto stageClose = [&](const StageMark& m, double& acc) {
        if (!stageTimers) return;
        syncDev();
        stageOpenDepth--;
        const double wall = msSince(m.t);
        const double timed = ((evalMs - m.ev) - (normBootMs - m.nb)) + (bootMs - m.bo)
                           + (encMs - m.en) + (decMs - m.de);
        acc += wall - timed;
    };

    // pack n channel values, replicated with period D across all slots
    auto packCh = [&](const std::vector<double>& v) {
        std::vector<double> s(SLOTS, 0.0);
        for (uint32_t r = 0; r < REP; r++)
            for (size_t i = 0; i < v.size() && i < D; i++) s[r * D + i] = v[i];
        return s;
    };
    auto enc = [&](const std::vector<double>& packed) {
        auto t = Clock::now();                                  // S3.7 V2: encCh's window
        auto pt = cc->MakeCKKSPackedPlaintext(packed);
        TrackedCt r{cc->Encrypt(keys.publicKey, pt), 0};
        encMs += msSince(t); return r;
    };
    auto dec = [&](TrackedCt& c) {
        auto t = Clock::now();                                  // S3.7 V2: decCh's window
        Plaintext p; cc->Decrypt(keys.secretKey, c.ct, &p);
        p->SetLength(SLOTS); auto v = p->GetRealPackedValue();
        decMs += msSince(t); return v;
    };
    auto ptAt = [&](const std::vector<double>& packed, uint32_t used) {
        auto tPt = Clock::now();                                // S3.7 V2: mkPt's host-encode counters
        Plaintext pt = cc->MakeCKKSPackedPlaintext(packed, 1, used);
        const double ptMs = msSince(tPt);
        hostPtEncodeMs += ptMs; hostPtEncodeCount++;
        if (stageOpenDepth > 0) { hostPtEncodeStageMs += ptMs; hostPtEncodeStageCount++; }
        return pt;
    };
    auto mulPt = [&](TrackedCt& c, const std::vector<double>& packed) {
        auto wpt = ptAt(packed, c.used);
        c.ct = cc->EvalMult(c.ct, wpt);
        cc->RescaleInPlace(c.ct); c.used++;
    };
    auto alignTo = [&](TrackedCt& c, uint32_t target) {
        std::vector<double> one(SLOTS, 1.0);
        while (c.used < target) { auto opt1 = ptAt(one, c.used); c.ct = cc->EvalMult(c.ct, opt1);
                                  cc->RescaleInPlace(c.ct); c.used++; }
    };

    // deterministic test vector
    std::vector<double> v(d);
    for (uint32_t i = 0; i < d; i++) v[i] = std::sin(0.37 * i) * 0.5;

    // ---- 1. pack / encrypt / decrypt roundtrip ---------------------------
    {
        auto c = enc(packCh(v)); auto o = dec(c);
        double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - v[i]));
        verdict("encdec", e < 1e-6, e, true);
    }

    // ---- 2. rotation convention: EvalRotate(ct,k)[i] == x[i+k] -----------
    {
        auto c = enc(packCh(v));
        TrackedCt r{cc->EvalRotate(c.ct, 1), 0}; auto o = dec(r);
        double e = 0; for (uint32_t i = 0; i + 1 < d; i++) e = std::max(e, std::abs(o[i] - v[i + 1]));
        verdict("rotate_left_convention", e < 1e-4, e, true);
    }

    // ---- 3. BSGS dense matvec vs plaintext -------------------------------
    // y[row] = sum_col W[row][col] * u[col], via y = sum_k diag_k (.) rot(u,k)
    // with diag_k[row] = W[row][(row+k) mod cols], BSGS over k = g + j and
    // the plaintext diagonal PRE-ROTATED by -g (the defect this file found).
    // CONTRACT: the input ciphertext is packed with period D and the matrix
    // is treated as (rows x D) with columns >= inCols zero-padded, so the
    // rotation modulus (D) EQUALS the diagonal modulus. rows must be <= D;
    // taller maps are chunked by matvecChunked below.
    // W is given row-major (rows x inCols); inCols <= D.
    const std::vector<Ciphertext<DCRTPoly>>* babyShared = nullptr;   // S3.3 shared-babies cell: set by the caller
    auto matvec = [&](TrackedCt u, const std::vector<double>& W,
                      uint32_t rows, uint32_t inCols, uint32_t rowOff) -> TrackedCt {
        auto t = Clock::now();                                  // S3.7 V2: the matvec evalMs window (no boot inside)
        std::vector<Ciphertext<DCRTPoly>> baby(BS);
        if (babyShared) baby = *babyShared;                       // reuse (S3.3); else compute as before
        else {
        baby[0] = u.ct;
        for (int j = 1; j < BS; j++) baby[j] = cc->EvalRotate(u.ct, j);
        }
        TrackedCt acc; bool have = false;
        for (uint32_t g = 0; g < D; g += BS) {
            Ciphertext<DCRTPoly> inner; bool hi = false;
            for (int j = 0; j < BS && g + (uint32_t)j < D; j++) {
                uint32_t k = g + j;
                std::vector<double> diag(SLOTS, 0.0);
                bool nonzero = false;
                for (uint32_t row = 0; row < rows; row++) {
                    uint32_t col = (row + k) % D;              // modulus == D
                    if (col >= inCols) continue;               // zero-padded column
                    double wv = W[(size_t)(row + rowOff) * inCols + col];
                    // pre-rotate by -g: the value for output `row` sits at
                    // slot (row+g) mod D so the giant rot(.,g) lands it back.
                    uint32_t slot = (row + g) % D;
                    for (uint32_t r = 0; r < REP; r++) diag[r * D + slot] = wv;
                    nonzero = true;
                }
                if (!nonzero) continue;                        // skip empty diagonal
                auto dpt = ptAt(diag, u.used);
                auto term = cc->EvalMult(baby[j], dpt);
                if (!hi) { inner = term; hi = true; } else inner = cc->EvalAdd(inner, term);
            }
            if (!hi) continue;
            if (g > 0) inner = cc->EvalRotate(inner, g);
            if (!have) { acc.ct = inner; have = true; } else acc.ct = cc->EvalAdd(acc.ct, inner);
        }
        cc->RescaleInPlace(acc.ct); acc.used = u.used + 1;
        evalMs += msSince(t);                                   // S3.7 V2
        return acc;
    };
    // rows > D: ceil(rows/D) chunks, one output ciphertext each.
    auto matvecChunked = [&](TrackedCt u, const std::vector<double>& W,
                             uint32_t rows, uint32_t inCols) {
        std::vector<TrackedCt> out;
        for (uint32_t off = 0; off < rows; off += D)
            out.push_back(matvec(u, W, std::min(D, rows - off), inCols, off));
        return out;
    };
    {
        std::vector<double> W((size_t)d * d);
        for (uint32_t i = 0; i < d; i++)
            for (uint32_t j = 0; j < d; j++)
                W[(size_t)i * d + j] = std::cos(0.11 * i + 0.23 * j) * 0.3;
        auto c = enc(packCh(v));
        auto y = matvec(c, W, d, d, 0);
        auto o = dec(y);
        std::vector<double> ref(d, 0.0);
        for (uint32_t i = 0; i < d; i++)
            for (uint32_t j = 0; j < d; j++) ref[i] += W[(size_t)i * d + j] * v[j];
        double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - ref[i]));
        verdict("matvec_square", e < 1e-3, e, true);
    }
    // ---- 3b. rectangular matvec d -> dff (the mixed-period case) ---------
    {
        std::vector<double> W((size_t)dff * d);
        for (uint32_t i = 0; i < dff; i++)
            for (uint32_t j = 0; j < d; j++)
                W[(size_t)i * d + j] = std::sin(0.07 * i + 0.19 * j) * 0.2;
        auto c = enc(packCh(v));
        auto ys = matvecChunked(c, W, dff, d);          // NCHUNK ciphertexts
        std::vector<double> ref(dff, 0.0);
        for (uint32_t i = 0; i < dff; i++)
            for (uint32_t j = 0; j < d; j++) ref[i] += W[(size_t)i * d + j] * v[j];
        double e = 0;
        for (uint32_t ch = 0; ch < ys.size(); ch++) {
            auto o = dec(ys[ch]);
            uint32_t n = std::min(D, dff - ch * D);
            for (uint32_t i = 0; i < n; i++) e = std::max(e, std::abs(o[i] - ref[ch * D + i]));
        }
        verdict("matvec_rect_d_to_dff_chunked", e < 1e-3 && ys.size() == NCHUNK, e, true);
    }
    // ---- 3c. transpose direction dff -> d: chunks summed ------------------
    {
        std::vector<double> W((size_t)d * dff);
        for (uint32_t i = 0; i < d; i++)
            for (uint32_t j = 0; j < dff; j++)
                W[(size_t)i * dff + j] = std::cos(0.05 * i + 0.13 * j) * 0.15;
        std::vector<double> hid(dff);
        for (uint32_t j = 0; j < dff; j++) hid[j] = std::sin(0.21 * j) * 0.4;
        // encrypt hidden as NCHUNK period-D ciphertexts
        TrackedCt accum; bool have = false;
        for (uint32_t ch = 0; ch < NCHUNK; ch++) {
            uint32_t n = std::min(D, dff - ch * D);
            std::vector<double> part(hid.begin() + ch * D, hid.begin() + ch * D + n);
            auto cpart = enc(packCh(part));
            // column-block of W for this chunk, as a (d x D) row-major slice
            std::vector<double> Wb((size_t)d * D, 0.0);
            for (uint32_t i = 0; i < d; i++)
                for (uint32_t j = 0; j < n; j++) Wb[(size_t)i * D + j] = W[(size_t)i * dff + ch * D + j];
            auto y = matvec(cpart, Wb, d, D, 0);
            if (!have) { accum = y; have = true; }
            else { alignTo(accum, y.used); alignTo(y, accum.used);
                   accum.ct = cc->EvalAdd(accum.ct, y.ct); }
        }
        auto o = dec(accum);
        std::vector<double> ref(d, 0.0);
        for (uint32_t i = 0; i < d; i++)
            for (uint32_t j = 0; j < dff; j++) ref[i] += W[(size_t)i * dff + j] * hid[j];
        double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - ref[i]));
        verdict("matvec_dff_to_d_summed", e < 1e-3, e, true);
    }

    // ---- 3d. BATCHED matvec: encode each diagonal ONCE, apply to all tokens.
    // This is the throughput fix in gpu_real_model.cu (matvecBatch). It is a
    // pure loop reordering — the per-token math is untouched — but this test
    // PROVES it: several DISTINCT token vectors are pushed through the batched
    // path and each must match its own per-token result exactly. If any token
    // leaked into another (the hazard with slot-batching), these differ.
    {
        const int NT = 5;
        std::vector<std::vector<double>> toks(NT, std::vector<double>(d));
        for (int t = 0; t < NT; t++)
            for (uint32_t i = 0; i < d; i++)
                toks[t][i] = std::sin(1.7 * (t + 1) + 0.29 * i) * (0.2 + 0.1 * t);
        std::vector<double> W((size_t)d * d);
        for (uint32_t i = 0; i < d; i++)
            for (uint32_t j = 0; j < d; j++)
                W[(size_t)i * d + j] = std::cos(0.17 * i + 0.31 * j) * 0.25;
        // per-token reference path (one matvec call each)
        std::vector<std::vector<double>> perTok(NT);
        for (int t = 0; t < NT; t++) {
            auto c = enc(packCh(toks[t]));
            auto y = matvec(c, W, d, d, 0);
            perTok[t] = dec(y);
        }
        // batched path: diagonal encoded once, reused across all NT tokens
        std::vector<TrackedCt> us; us.reserve(NT);
        for (int t = 0; t < NT; t++) us.push_back(enc(packCh(toks[t])));
        std::vector<Ciphertext<DCRTPoly>> accB(NT); std::vector<bool> haveB(NT, false);
        uint32_t lvl = us[0].used;
        std::vector<std::vector<Ciphertext<DCRTPoly>>> baby(NT);
        for (int t = 0; t < NT; t++) {
            baby[t].resize(BS); baby[t][0] = us[t].ct;
            for (int j = 1; j < BS; j++) baby[t][j] = cc->EvalRotate(us[t].ct, j);
        }
        for (uint32_t g = 0; g < D; g += BS) {
            std::vector<Ciphertext<DCRTPoly>> inner(NT); std::vector<bool> hi(NT, false);
            for (int j = 0; j < BS && g + j < D; j++) {
                uint32_t k = g + j;
                std::vector<double> diag(SLOTS, 0.0); bool nz = false;
                for (uint32_t row = 0; row < d; row++) {
                    uint32_t col = (row + k) % D; if (col >= d) continue;
                    uint32_t slot = (row + g) % D;
                    for (uint32_t r = 0; r < REP; r++) diag[r * D + slot] = W[(size_t)row * d + col];
                    nz = true;
                }
                if (!nz) continue;
                auto dpt = ptAt(diag, lvl);                    // *** encoded ONCE ***
                for (int t = 0; t < NT; t++) {
                    auto term = cc->EvalMult(baby[t][j], dpt);
                    if (!hi[t]) { inner[t] = term; hi[t] = true; }
                    else inner[t] = cc->EvalAdd(inner[t], term);
                }
            }
            for (int t = 0; t < NT; t++) {
                if (!hi[t]) continue;
                auto blk = (g > 0) ? cc->EvalRotate(inner[t], g) : inner[t];
                if (!haveB[t]) { accB[t] = blk; haveB[t] = true; }
                else accB[t] = cc->EvalAdd(accB[t], blk);
            }
        }
        double e = 0, crossMin = 1e9;
        for (int t = 0; t < NT; t++) {
            cc->RescaleInPlace(accB[t]);
            TrackedCt yb{accB[t], lvl + 1}; auto ob = dec(yb);
            for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(ob[i] - perTok[t][i]));
            // sanity: tokens must be genuinely DIFFERENT, else the test is vacuous
            for (int t2 = 0; t2 < NT; t2++) if (t2 != t) {
                double diff = 0;
                for (uint32_t i = 0; i < d; i++) diff = std::max(diff, std::abs(perTok[t][i] - perTok[t2][i]));
                crossMin = std::min(crossMin, diff);
            }
        }
        verdict("matvec_batched_no_token_bleed", e < 1e-3 && crossMin > 1e-2, e, true);
    }

    // ---- 4. RMSNorm: rotate-sum reduction + Newton rsqrt -----------------
    // mean(x^2) over the d real channels; padding slots are 0, so summing the
    // whole D-block and scaling by 1/d is exact.
    auto rmsnorm = [&](TrackedCt x, const std::vector<double>& gpk,
                       double a0, double b0, int iters, double eps) -> TrackedCt {
        auto t = Clock::now();                                  // S3.7 V2: the rmsnorm evalMs window ...
        const double normBoot0 = bootMs;                        // ... and the .cu's snapshot (no boot in this variant: delta 0)
        TrackedCt sq{cc->EvalMult(x.ct, x.ct), x.used};
        cc->RescaleInPlace(sq.ct); sq.used = x.used + 1;
        Ciphertext<DCRTPoly> red = sq.ct;
        for (uint32_t p = 1; p < D; p <<= 1) red = cc->EvalAdd(red, cc->EvalRotate(red, p));
        std::vector<double> invd(SLOTS, 1.0 / (double)d);
        auto invdpt = ptAt(invd, sq.used);
        TrackedCt ms{cc->EvalMult(red, invdpt), sq.used};
        cc->RescaleInPlace(ms.ct); ms.used = sq.used + 1;
        // S3.7 V1 (2026-09-05, S3.6 T18 §3): ms = mean(x^2) + eps, as the model defines
        // it (invd is 1/d here, so sigma = 1 and the constant is eps itself). One
        // plaintext add at ms's own level: zero levels, `used` unchanged. This is the
        // same hunk as gpu_real_model.cu's (there: sigma*eps). --no-norm-eps skips it.
        const double epsUsed = normEpsOn ? eps : 0.0;
        normEpsUsed = epsUsed;
        if (epsUsed != 0.0) {
            std::vector<double> epsv(SLOTS, epsUsed);
            auto epspt = ptAt(epsv, ms.used); ms.ct = cc->EvalAdd(ms.ct, epspt);
        }
        std::vector<double> b0v(SLOTS, b0), a0v(SLOTS, a0);
        auto b0pt = ptAt(b0v, ms.used);
        TrackedCt y{cc->EvalMult(ms.ct, b0pt), ms.used};
        cc->RescaleInPlace(y.ct); y.used = ms.used + 1;
        { auto a0pt = ptAt(a0v, y.used); y.ct = cc->EvalAdd(y.ct, a0pt); }
        for (int it = 0; it < iters; it++) {
            TrackedCt y2{cc->EvalMult(y.ct, y.ct), y.used};
            cc->RescaleInPlace(y2.ct); y2.used = y.used + 1;
            TrackedCt msA = ms; alignTo(msA, y2.used);
            TrackedCt t{cc->EvalMult(msA.ct, y2.ct), y2.used};
            cc->RescaleInPlace(t.ct); t.used = y2.used + 1;
            std::vector<double> mhalf(SLOTS, -0.5), oneP5(SLOTS, 1.5);
            { auto mh = ptAt(mhalf, t.used); t.ct = cc->EvalMult(t.ct, mh); }
            cc->RescaleInPlace(t.ct); t.used++;
            { auto o15 = ptAt(oneP5, t.used); t.ct = cc->EvalAdd(t.ct, o15); }
            alignTo(y, t.used);
            y.ct = cc->EvalMult(y.ct, t.ct); cc->RescaleInPlace(y.ct); y.used++;
        }
        alignTo(x, y.used);
        TrackedCt out{cc->EvalMult(x.ct, y.ct), x.used};
        cc->RescaleInPlace(out.ct); out.used = x.used + 1;
        mulPt(out, gpk);
        evalMs += msSince(t);                                   // S3.7 V2
        normBootMs += bootMs - normBoot0;
        return out;
    };
    {
        double ms = 0; for (uint32_t i = 0; i < d; i++) ms += v[i] * v[i]; ms /= d;
        double eps = 1e-5, inv = 1.0 / std::sqrt(ms + eps);
        // linear seed on [ms/2, 2ms] (mirrors linear_rsqrt_seed in Python)
        double a0 = 1.5 / std::sqrt(ms), b0 = -0.5 / (ms * std::sqrt(ms));
        std::vector<double> g(d, 1.0);
        auto c = enc(packCh(v));
        auto y = rmsnorm(c, packCh(g), a0, b0, 3, eps);
        auto o = dec(y);
        double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - v[i] * inv));
        verdict("rmsnorm_newton", e < 5e-3, e, false);
    }
    // ---- 4b. S3.7 V1 (2026-09-05): the norm with the seed the demo runs ---------
    // Mirrors gpu_real_model.cu selftest 4b. --norm-seed A B ITERS takes a bundle
    // site's (a, b, iters) (pbd430a L0.tm: 53.12723596752366 -13744.462664572491 5) and
    // scales v so that mean(v^2) = -a/(16 b): inside the seed's design window near its
    // low end, where eps/ms ~ 4 %. Reference = the model's 1/sqrt(ms + eps). Predicted
    // by harness/norm_eps_check.py (case cpu_twin_seeded_L0tm): the no-eps chain
    // (--no-norm-eps) misses the 1e-2 bar, the eps chain sits at the Newton floor.
    if (normSeedIters > 0) {
        const double msTest = -normSeedA / (normSeedB * 16.0);
        if (!std::isfinite(msTest) || msTest <= 0.0) {
            std::cout << "{\"cell\":\"rmsnorm_seeded\",\"skipped\":\"seed has no positive ms\"}" << std::endl;
        } else {
            double msv = 0; for (uint32_t i = 0; i < d; i++) msv += v[i] * v[i]; msv /= d;
            const double sc = std::sqrt(msTest / msv);
            std::vector<double> vs(d); for (uint32_t i = 0; i < d; i++) vs[i] = v[i] * sc;
            const double eps = 1e-5, inv = 1.0 / std::sqrt(msTest + eps);
            std::vector<double> g(d, 1.0);
            auto c = enc(packCh(vs));
            auto y = rmsnorm(c, packCh(g), normSeedA, normSeedB, normSeedIters, eps);
            auto o = dec(y);
            double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - vs[i] * inv));
            std::cout << "{\"cell\":\"rmsnorm_seeded\",\"a\":" << normSeedA << ",\"b\":" << normSeedB
                      << ",\"iters\":" << normSeedIters << ",\"msTest\":" << msTest << ",\"scale\":" << sc
                      << ",\"normEps\":" << normEpsUsed << ",\"maxAbsErr\":" << e << "}" << std::endl;
            verdict("rmsnorm_seeded", e < 1e-2, e, false);
        }
    }

    // ---- 5. learned degree-3 gate: g = z (.) (p0 + p1 z + p2 z^2) --------
    {
        std::vector<double> p0(d), p1(d), p2(d);
        for (uint32_t i = 0; i < d; i++) { p0[i] = 0.5 + 0.01 * i / d; p1[i] = 0.25; p2[i] = 0.1; }
        auto z = enc(packCh(v));
        TrackedCt z2{cc->EvalMult(z.ct, z.ct), z.used};
        cc->RescaleInPlace(z2.ct); z2.used = z.used + 1;
        TrackedCt t2 = z2; mulPt(t2, packCh(p2));
        TrackedCt t1 = z;  mulPt(t1, packCh(p1));
        alignTo(t1, t2.used);
        t1.ct = cc->EvalAdd(t1.ct, t2.ct);
        { auto p0pt = ptAt(packCh(p0), t1.used); t1.ct = cc->EvalAdd(t1.ct, p0pt); }
        TrackedCt zA = z; alignTo(zA, t1.used);
        TrackedCt gct{cc->EvalMult(zA.ct, t1.ct), zA.used};
        cc->RescaleInPlace(gct.ct); gct.used = zA.used + 1;
        auto o = dec(gct);
        double e = 0;
        for (uint32_t i = 0; i < d; i++) {
            double ref = v[i] * (p0[i] + p1[i] * v[i] + p2[i] * v[i] * v[i]);
            e = std::max(e, std::abs(o[i] - ref));
        }
        verdict("poly_gate_deg3", e < 1e-3, e, true);
    }

    // ---- 6. Lemma-2 diagonal scan: s_t = a (.) s_{t-1} + b (.) x_t -------
    {
        const uint32_t Tn = 8;
        std::vector<double> a(d), b(d);
        for (uint32_t i = 0; i < d; i++) { a[i] = 0.90 + 0.09 * (i % 7) / 7.0; b[i] = 1.0 - a[i]; }
        std::vector<std::vector<double>> xs(Tn, std::vector<double>(d));
        for (uint32_t t = 0; t < Tn; t++)
            for (uint32_t i = 0; i < d; i++) xs[t][i] = std::sin(0.3 * t + 0.11 * i) * 0.4;
        std::vector<double> zero(SLOTS, 0.0);
        TrackedCt s = enc(zero);
        for (uint32_t t = 0; t < Tn; t++) {
            TrackedCt x = enc(packCh(xs[t]));
            mulPt(s, packCh(a));
            TrackedCt bx = x; mulPt(bx, packCh(b));
            alignTo(s, bx.used); alignTo(bx, s.used);
            s.ct = cc->EvalAdd(s.ct, bx.ct);
        }
        auto o = dec(s);
        std::vector<double> ref(d, 0.0);
        for (uint32_t t = 0; t < Tn; t++)
            for (uint32_t i = 0; i < d; i++) ref[i] = a[i] * ref[i] + b[i] * xs[t][i];
        double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - ref[i]));
        verdict("lemma2_scan", e < 1e-3, e, true);
    }

    // ---- 7. optional: bootstrap a depleted ciphertext --------------------
    if (doBootstrap) {
        auto c = enc(packCh(v));
        std::vector<double> one(SLOTS, 1.0);
        for (int i = 0; i < 3; i++) { auto opt2 = ptAt(one, c.used); c.ct = cc->EvalMult(c.ct, opt2);
                                      cc->RescaleInPlace(c.ct); c.used++; }
        c.ct = cc->EvalBootstrap(c.ct); c.used = depth - levelsAfter;
        auto o = dec(c);
        double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - v[i]));
        verdict("bootstrap", e < 5e-3, e, true);
    }


    // =====================================================================
    // S3.3 validation cells (results/theory/SCHEDULE_DESIGN_20260903.md)
    // =====================================================================

    if (cells.count("level-probe")) {
        // FLEXIBLEAUTO semantics of the primitives the replay models (Lemma 7 of
        // TICK_SCALING_BOUNDS_20260903.md): GetLevel() and GetNoiseScaleDeg() after
        // ct x pt, ct x ct, rotate, add, RescaleInPlace, and a ct x pt whose plaintext
        // was encoded at a DIFFERENT level than the ciphertext.
        auto rep = [&](const std::string& what, const Ciphertext<DCRTPoly>& c) {
            std::cout << "{\"levelProbe\":\"" << what << "\",\"level\":" << c->GetLevel() << ",\"deg\":" << c->GetNoiseScaleDeg() << "}" << std::endl;
        };
        std::vector<double> one(SLOTS, 1.0), half(SLOTS, 0.5);
        auto c0 = enc(packCh(v)).ct; rep("fresh", c0);
        auto p0 = cc->MakeCKKSPackedPlaintext(half, 1, c0->GetLevel());
        auto c1 = cc->EvalMult(c0, p0); rep("ct*pt(level-matched)", c1);
        auto r1 = cc->EvalRotate(c1, 1); rep("rotate(deg2)", r1);
        auto a1 = cc->EvalAdd(c1, r1); rep("add(deg2,deg2)", a1);
        auto s1 = a1; cc->RescaleInPlace(s1); rep("RescaleInPlace(deg2)", s1);
        auto p1 = cc->MakeCKKSPackedPlaintext(half, 1, s1->GetLevel());
        auto c2 = cc->EvalMult(s1, p1); rep("ct*pt on deg2 input (level-matched pt)", c2);
        auto cc2 = cc->EvalMult(c2, c2); rep("ct*ct on deg2 inputs", cc2);
        auto p_lo = cc->MakeCKKSPackedPlaintext(half, 1, 0);
        auto c3 = cc->EvalMult(c2, p_lo); rep("ct*pt with pt encoded at level 0 (deeper ct)", c3);
        TrackedCt t3{c3, 0}, t2{c2, 0};
        auto d3 = dec(t3); auto d2 = dec(t2);
        double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(d3[i] - 0.5 * d2[i]));
        std::cout << "{\"levelProbe\":\"value check pt@0 vs pt@level\",\"maxAbsErr\":" << e << "}" << std::endl;
        auto r2 = cc->EvalRotate(c2, 1); rep("rotate(deg2, level>0)", r2);
        auto a2 = cc->EvalAdd(c2, r2); rep("add after rotate", a2);
        auto g2 = cc->EvalRotate(a2, 8); rep("giant rotate of the sum", g2);
        verdict("level_probe_ran", true, 0.0, false);
    }

    if (cells.count("shared-babies")) {
        // The d -> dff map as NCHUNK chunk calls on ONE input: (a) stock, every
        // call computes its own baby rotations; (b) shared, one baby set feeds
        // every call. Bit-identity = the output ciphertext polynomials are
        // equal element-wise (DCRTPoly::operator==) for every chunk.
        std::vector<double> W((size_t)dff * d);
        for (uint32_t i = 0; i < dff; i++)
            for (uint32_t j = 0; j < d; j++)
                W[(size_t)i * d + j] = std::sin(0.07 * i + 0.19 * j) * 0.2;
        auto c = enc(packCh(v));
        babyShared = nullptr;
        auto ysStock = matvecChunked(c, W, dff, d);
        std::vector<Ciphertext<DCRTPoly>> baby(BS);
        baby[0] = c.ct;
        for (int j = 1; j < BS; j++) baby[j] = cc->EvalRotate(c.ct, j);
        babyShared = &baby;
        auto ysShared = matvecChunked(c, W, dff, d);
        babyShared = nullptr;
        bool polysEqual = ysStock.size() == ysShared.size();
        double maxDiff = 0; int chunksEq = 0;
        for (size_t ch = 0; polysEqual && ch < ysStock.size(); ch++) {
            const auto& ea = ysStock[ch].ct->GetElements();
            const auto& eb = ysShared[ch].ct->GetElements();
            bool eq = ea.size() == eb.size();
            for (size_t k = 0; eq && k < ea.size(); k++) eq = (ea[k] == eb[k]);
            if (eq) chunksEq++; else polysEqual = false;
            auto oa = dec(ysStock[ch]); auto ob = dec(ysShared[ch]);
            for (uint32_t i = 0; i < D; i++) maxDiff = std::max(maxDiff, std::abs(oa[i] - ob[i]));
        }
        // CONTROL: decrypting the SAME ciphertext twice differs at the same
        // magnitude -- OpenFHE's CKKS decryption adds flooding noise (a decrypt-
        // side property, not a ciphertext difference). The bit-identity claim is
        // therefore the polynomial equality, and the decrypted diff is reported
        // beside its control, never used as the verdict.
        double sameCtTwice = 0;
        { auto o1 = dec(ysStock[0]); auto o2 = dec(ysStock[0]);
          for (uint32_t i = 0; i < D; i++) sameCtTwice = std::max(sameCtTwice, std::abs(o1[i] - o2[i])); }
        std::cout << "{\"cell\":\"shared-babies\",\"chunks\":" << ysStock.size() << ",\"chunksPolyEqual\":" << chunksEq
                  << ",\"polysEqual\":" << (polysEqual ? "true" : "false") << ",\"maxAbsDiffDecrypted\":" << maxDiff
                  << ",\"controlSameCtDecryptedTwice\":" << sameCtTwice
                  << ",\"babyRotationsStock\":" << (BS - 1) * (int)ysStock.size() << ",\"babyRotationsShared\":" << (BS - 1) << "}" << std::endl;
        verdict("shared_babies_bit_identical", polysEqual, maxDiff, true);
    }

    if (cells.count("newton-depth2")) {
        // Same seed (a0, b0), same iteration count, same input: stock chain vs the
        // depth-2 re-association. Reports relErr of each vs exact rsqrt, the
        // rms difference between the two, and the levels consumed by the loop.
        auto runNorm = [&](const std::vector<double>& xv, int iters, bool nd2, double& lvlLoop) -> std::vector<double> {
            double ms0 = 0; for (uint32_t i = 0; i < d; i++) ms0 += xv[i] * xv[i]; ms0 /= d;
            double a0 = 1.5 / std::sqrt(ms0), b0 = -0.5 / (ms0 * std::sqrt(ms0));
            TrackedCt x = enc(packCh(xv));
            TrackedCt sq{cc->EvalMult(x.ct, x.ct), 0}; cc->RescaleInPlace(sq.ct);
            Ciphertext<DCRTPoly> red = sq.ct;
            for (uint32_t p = 1; p < D; p <<= 1) red = cc->EvalAdd(red, cc->EvalRotate(red, p));
            std::vector<double> invd(SLOTS, 1.0 / (double)d);
            auto invdpt = cc->MakeCKKSPackedPlaintext(invd, 1, sq.ct->GetLevel());
            TrackedCt ms{cc->EvalMult(red, invdpt), 0}; cc->RescaleInPlace(ms.ct);
            std::vector<double> b0v(SLOTS, b0), a0v(SLOTS, a0);
            auto b0pt = cc->MakeCKKSPackedPlaintext(b0v, 1, ms.ct->GetLevel());
            TrackedCt y{cc->EvalMult(ms.ct, b0pt), 0}; cc->RescaleInPlace(y.ct);
            auto a0pt = cc->MakeCKKSPackedPlaintext(a0v, 1, y.ct->GetLevel());
            y.ct = cc->EvalAdd(y.ct, a0pt);
            const uint32_t lvl0 = y.ct->GetLevel();
            TrackedCt msh;
            if (nd2) { std::vector<double> mh(SLOTS, -0.5);
                       auto mhp = cc->MakeCKKSPackedPlaintext(mh, 1, ms.ct->GetLevel());
                       msh.ct = cc->EvalMult(ms.ct, mhp); cc->RescaleInPlace(msh.ct); }
            for (int it = 0; it < iters; it++) {
                if (nd2) {   // mirrors gpu_real_model.cu --newton-depth2
                    TrackedCt a{cc->EvalMult(msh.ct, y.ct), 0}; cc->RescaleInPlace(a.ct);
                    TrackedCt b{cc->EvalMult(y.ct, y.ct), 0};   cc->RescaleInPlace(b.ct);
                    TrackedCt tt{cc->EvalMult(a.ct, b.ct), 0};  cc->RescaleInPlace(tt.ct);
                    std::vector<double> o15(SLOTS, 1.5);
                    auto op15 = cc->MakeCKKSPackedPlaintext(o15, 1, y.ct->GetLevel());
                    TrackedCt y15{cc->EvalMult(y.ct, op15), 0}; cc->RescaleInPlace(y15.ct);
                    y15.ct = cc->EvalAdd(y15.ct, tt.ct);
                    y = y15;
                } else {     // mirrors the stock loop (:2973-2992)
                    TrackedCt y2{cc->EvalMult(y.ct, y.ct), 0}; cc->RescaleInPlace(y2.ct);
                    TrackedCt tms{cc->EvalMult(ms.ct, y2.ct), 0}; cc->RescaleInPlace(tms.ct);
                    std::vector<double> half(SLOTS, -0.5), oneP5(SLOTS, 1.5);
                    auto halfpt = cc->MakeCKKSPackedPlaintext(half, 1, tms.ct->GetLevel());
                    tms.ct = cc->EvalMult(tms.ct, halfpt); cc->RescaleInPlace(tms.ct);
                    auto op5 = cc->MakeCKKSPackedPlaintext(oneP5, 1, tms.ct->GetLevel());
                    tms.ct = cc->EvalAdd(tms.ct, op5);
                    y.ct = cc->EvalMult(y.ct, tms.ct); cc->RescaleInPlace(y.ct);
                }
            }
            lvlLoop = (double)y.ct->GetLevel() - (double)lvl0;
            TrackedCt out{cc->EvalMult(x.ct, y.ct), 0}; cc->RescaleInPlace(out.ct);
            return dec(out);
        };
        std::vector<std::vector<double>> inputs = {v};
        { std::vector<double> w(d); for (uint32_t i = 0; i < d; i++) w[i] = 0.7 * std::cos(1.3 * i + 0.4) + 0.05 * std::sin(7.1 * i); inputs.push_back(w); }
        int idx = 0;
        for (const auto& xv : inputs) {
            double ms0 = 0; for (uint32_t i = 0; i < d; i++) ms0 += xv[i] * xv[i]; ms0 /= d;
            const double inv = 1.0 / std::sqrt(ms0);
            double lvS = 0, lvN = 0;
            auto oS = runNorm(xv, cellIters, false, lvS);
            auto oN = runNorm(xv, cellIters, true, lvN);
            double eS = 0, eN = 0, dSN = 0, ref2 = 0;
            for (uint32_t i = 0; i < d; i++) {
                const double r = xv[i] * inv;
                eS += (oS[i] - r) * (oS[i] - r); eN += (oN[i] - r) * (oN[i] - r); dSN += (oS[i] - oN[i]) * (oS[i] - oN[i]); ref2 += r * r;
            }
            const double relS = std::sqrt(eS / ref2), relN = std::sqrt(eN / ref2), relD = std::sqrt(dSN / ref2);
            std::cout << "{\"cell\":\"newton-depth2\",\"input\":" << idx << ",\"iters\":" << cellIters
                      << ",\"relErrRmsStock\":" << relS << ",\"relErrRmsDepth2\":" << relN << ",\"relErrRmsStockVsDepth2\":" << relD
                      << ",\"levelsPerIterStock\":" << lvS / cellIters << ",\"levelsPerIterDepth2\":" << lvN / cellIters << "}" << std::endl;
            // the re-association must be within the stock chain's own error class (R3: rms), and consume 2 levels/iteration
            verdict("newton_depth2_input" + std::to_string(idx), relN <= 4.0 * std::max(relS, 1e-9) && std::abs(lvN / cellIters - 2.0) < 1e-9
                    && std::abs(lvS / cellIters - 4.0) < 1e-9, relN, true);
            idx++;
        }
    }

    if (cells.count("pf-trace")) {
        // ONE layer of the real-model circuit, T = 1, h = 1 (no lane masks), shift-mix
        // without a carry (u_{-1} = 0), the stock scan on a fresh zero state, the
        // learned gates, K = dff/D chunks -- the statement order of gpu_real_model.cu's
        // layer loop, with refresh()/boot() placed per --schedule {stock,parent-first}.
        // Every refresh/boot prints (site, GetLevel, remaining). Compare with:
        //   python3 results/theory/tick_level_trace.py --depth 30 --lam <observed> --boot-floor 3 --margin 4
        //       --dpad 64 --d 48 --dff 128 --layers 1 --iters 3 --ticks 1 --lazy --trace-levels [--schedule parent-first]
        const uint32_t F = (uint32_t)cellF, margin = (uint32_t)cellM;
        int boots = 0; uint32_t lamObserved = 0;
        // S3.7 V2: an INDEPENDENT tally of the norm-internal boots (set by rmsnormL
        // while its window is open) to check the .cu's snapshot subtraction: the same
        // dt goes into bootMs, into normBootDirectMs when inside a norm window, and
        // -- via the window's bootMs delta -- into normBootMs; the two must agree exactly.
        bool inNormWindow = false; double normBootDirectMs = 0; int bootsInNorm = 0;
        std::vector<std::string> traceOut;
        auto emit = [&](const std::string& site, const std::string& kind, uint32_t lvl) {
            std::ostringstream o; o << "{\"pfTrace\":\"" << kind << "\",\"site\":\"" << site << "\",\"level\":" << lvl
                                    << ",\"remaining\":" << (int)depth - (int)lvl << "}";
            traceOut.push_back(o.str()); std::cout << o.str() << std::endl;
        };
        auto boot = [&](TrackedCt& c, const std::string& site) {
            const uint32_t pre = c.ct->GetLevel();
            syncDev();
            auto tB = Clock::now();                             // S3.7 V2: boot()'s bootMs window
            c.ct = cc->EvalBootstrap(c.ct); boots++;
            const double dt = msSince(tB);
            bootMs += dt;
            if (inNormWindow) { normBootDirectMs += dt; bootsInNorm++; }
            const uint32_t post = c.ct->GetLevel();
            if (!lamObserved) lamObserved = depth - post;
            std::ostringstream o; o << "{\"pfTrace\":\"boot\",\"site\":\"" << site << "\",\"lvlPre\":" << pre << ",\"lvlPost\":" << post
                                    << ",\"remainingAfter\":" << (int)depth - (int)post << "}";
            traceOut.push_back(o.str()); std::cout << o.str() << std::endl;
        };
        auto refresh = [&](TrackedCt& c, uint32_t need, const std::string& site) {   // gpu_real_model.cu:2156-2182
            const uint32_t lvl = c.ct->GetLevel();
            emit(site, "refresh", lvl);
            if (lvl > depth) { boot(c, site); return; }
            if (depth - lvl < need + F) boot(c, site);
        };
        const bool pf = (cellSchedule == "parent-first");
        const uint32_t nb = 3, dN = cellNd2 ? 2u : 4u;
        auto needTail = [&](int kind) -> uint32_t { return kind == 1 ? 12u : kind == 2 ? 2u + 1u + 4u + margin : 4u; };
        auto needEntry = [&](int kind, int iters) -> uint32_t {
            const uint32_t loop = nb + dN * (uint32_t)std::max(iters - 1, 0) + 5u, branch = needTail(kind) + 2u;
            if (pfEntry == "loop") return loop; if (pfEntry == "branch") return branch;
            if (pfEntry == "both") return std::max(loop, branch); return depth + 1u; };
        auto ptL = [&](const std::vector<double>& packed, const Ciphertext<DCRTPoly>& ref) {
            return cc->MakeCKKSPackedPlaintext(packed, 1, ref->GetLevel()); };
        auto mulPtL = [&](TrackedCt& c, const std::vector<double>& packed) {
            c.ct = cc->EvalMult(c.ct, ptL(packed, c.ct)); cc->RescaleInPlace(c.ct); };
        auto alignToL = [&](TrackedCt& c, uint32_t target) {
            std::vector<double> one(SLOTS, 1.0);
            while (c.ct->GetLevel() < target) { c.ct = cc->EvalMult(c.ct, ptL(one, c.ct)); cc->RescaleInPlace(c.ct); } };
        auto addAlignedL = [&](TrackedCt& a, TrackedCt& b, const std::string& site) {
            refresh(a, 2, site + ".a"); refresh(b, 2, site + ".b"); a.ct = cc->EvalAdd(a.ct, b.ct); };
        // rmsnorm with the harness's refresh placements (kind: 1 tm, 2 cm, 0 out)
        auto rmsnormL = [&](TrackedCt x, const std::vector<double>& gpk, double a0, double b0, int iters, int kind, const std::string& site) -> TrackedCt {
            auto tN = Clock::now();                             // S3.7 V2: the rmsnorm window with its own boots inside
            const double normBoot0 = bootMs; inNormWindow = true;
            TrackedCt sq{cc->EvalMult(x.ct, x.ct), 0}; cc->RescaleInPlace(sq.ct);
            Ciphertext<DCRTPoly> red = sq.ct;
            for (uint32_t p = 1; p < D; p <<= 1) red = cc->EvalAdd(red, cc->EvalRotate(red, p));
            std::vector<double> invd(SLOTS, 1.0 / (double)d);
            TrackedCt ms{cc->EvalMult(red, ptL(invd, red)), 0}; cc->RescaleInPlace(ms.ct);
            std::vector<double> b0v(SLOTS, b0), a0v(SLOTS, a0);
            TrackedCt y{cc->EvalMult(ms.ct, ptL(b0v, ms.ct)), 0}; cc->RescaleInPlace(y.ct);
            { auto p_ = ptL(a0v, y.ct); y.ct = cc->EvalAdd(y.ct, p_); }
            emit(site + ".y0", "born", y.ct->GetLevel());
            TrackedCt msh;
            if (cellNd2) { std::vector<double> mh(SLOTS, -0.5); msh.ct = cc->EvalMult(ms.ct, ptL(mh, ms.ct)); cc->RescaleInPlace(msh.ct); }
            for (int it = 0; it < iters; it++) {
                if (cellNd2) {
                    refresh(y, 5, site + ".newton" + std::to_string(it) + ".y"); refresh(msh, 5, site + ".newton" + std::to_string(it) + ".ms");
                    TrackedCt a{cc->EvalMult(msh.ct, y.ct), 0}; cc->RescaleInPlace(a.ct);
                    TrackedCt b{cc->EvalMult(y.ct, y.ct), 0};   cc->RescaleInPlace(b.ct);
                    TrackedCt tt{cc->EvalMult(a.ct, b.ct), 0};  cc->RescaleInPlace(tt.ct);
                    std::vector<double> o15(SLOTS, 1.5);
                    TrackedCt y15{cc->EvalMult(y.ct, ptL(o15, y.ct)), 0}; cc->RescaleInPlace(y15.ct);
                    y15.ct = cc->EvalAdd(y15.ct, tt.ct); y = y15;
                } else {
                    refresh(y, 5, site + ".newton" + std::to_string(it) + ".y"); refresh(ms, 5, site + ".newton" + std::to_string(it) + ".ms");
                    TrackedCt y2{cc->EvalMult(y.ct, y.ct), 0}; cc->RescaleInPlace(y2.ct);
                    TrackedCt tms{cc->EvalMult(ms.ct, y2.ct), 0}; cc->RescaleInPlace(tms.ct);
                    std::vector<double> half(SLOTS, -0.5), oneP5(SLOTS, 1.5);
                    tms.ct = cc->EvalMult(tms.ct, ptL(half, tms.ct)); cc->RescaleInPlace(tms.ct);
                    { auto p_ = ptL(oneP5, tms.ct); tms.ct = cc->EvalAdd(tms.ct, p_); }
                    y.ct = cc->EvalMult(y.ct, tms.ct); cc->RescaleInPlace(y.ct);
                }
            }
            refresh(y, pf ? needTail(kind) : 4u, site + ".tail.y");
            TrackedCt out{cc->EvalMult(x.ct, y.ct), 0}; cc->RescaleInPlace(out.ct);
            mulPtL(out, gpk);
            evalMs += msSince(tN);                              // S3.7 V2: the .cu's :3370 exit ...
            normBootMs += bootMs - normBoot0; inNormWindow = false;   // ... and the snapshot subtraction
            return out;
        };
        auto polyGate3L = [&](TrackedCt z, const std::vector<double>& p0, const std::vector<double>& p1, const std::vector<double>& p2) -> TrackedCt {
            TrackedCt z2{cc->EvalMult(z.ct, z.ct), 0}; cc->RescaleInPlace(z2.ct);
            TrackedCt term2 = z2; mulPtL(term2, packCh(p2));
            TrackedCt term1 = z;  mulPtL(term1, packCh(p1));
            alignToL(term1, term2.ct->GetLevel());
            term1.ct = cc->EvalAdd(term1.ct, term2.ct);
            { auto p_ = ptL(packCh(p0), term1.ct); term1.ct = cc->EvalAdd(term1.ct, p_); }
            TrackedCt zA = z; alignToL(zA, term1.ct->GetLevel());
            TrackedCt g{cc->EvalMult(zA.ct, term1.ct), 0}; cc->RescaleInPlace(g.ct);
            return g;
        };
        // the layer's constants (deterministic, in-range)
        const int I = cellIters;
        std::vector<double> g1(d, 1.0), g2(d, 1.0), mix(d), imix(d), dec_(d), bco(d), cc_(d), dd(d), p0(d), p1(d), p2(d);
        for (uint32_t i = 0; i < d; i++) { mix[i] = 0.5 + 0.2 * std::sin(0.3 * i); imix[i] = 1.0 - mix[i]; dec_[i] = 0.90 + 0.09 * (i % 7) / 7.0; bco[i] = 1.0 - dec_[i];
                                          cc_[i] = 0.5 * std::cos(0.2 * i); dd[i] = 0.5 * std::sin(0.25 * i); p0[i] = 0.5; p1[i] = 0.25; p2[i] = 0.05 * std::cos(0.1 * i); }
        std::vector<double> q0(dff, 0.0), q1(dff, 1.0), q2(dff, 0.1), q3(dff, 0.02), r0(dff, 0.5), r1(dff, 0.25), r2(dff, 0.05);
        std::vector<double> Wsq((size_t)d * d), Wk((size_t)dff * d), Wr((size_t)dff * d), Wv((size_t)d * dff);
        for (uint32_t i = 0; i < d; i++) for (uint32_t j = 0; j < d; j++) Wsq[(size_t)i * d + j] = std::cos(0.11 * i + 0.23 * j) * 0.1;
        for (uint32_t i = 0; i < dff; i++) for (uint32_t j = 0; j < d; j++) { Wk[(size_t)i * d + j] = std::sin(0.07 * i + 0.19 * j) * 0.1; Wr[(size_t)i * d + j] = std::cos(0.05 * i + 0.13 * j) * 0.1; }
        for (uint32_t i = 0; i < d; i++) for (uint32_t j = 0; j < dff; j++) Wv[(size_t)i * dff + j] = std::sin(0.03 * i + 0.17 * j) * 0.05;
        double ms0 = 0; for (uint32_t i = 0; i < d; i++) ms0 += v[i] * v[i]; ms0 /= d;
        const double a0 = 1.5 / std::sqrt(ms0), b0 = -0.5 / (ms0 * std::sqrt(ms0));
        TrackedCt Hs = enc(packCh(v));                                   // fresh client ct, level 0
        std::vector<double> zero(SLOTS, 0.0);
        TrackedCt scan = enc(zero);                                       // cold scan state
        // S3.7 V2: the layer loop's clock and the five stage windows, placed exactly
        // where gpu_real_model.cu places them (shift-mix, scan, gate, residual, hidden,
        // ffnAccum -> residual, cm residual). Snapshot the accumulators first so the
        // timers object below reports this layer only.
        const double tEv0 = evalMs, tNb0 = normBootMs, tBo0 = bootMs, tEn0 = encMs, tDe0 = decMs;
        const double tSm0 = shiftMixMs, tSc0 = scanMs, tGa0 = gateMs, tHi0 = hiddenMs, tRe0 = residualMs;
        const long long tPc0 = hostPtEncodeCount, tPs0 = hostPtEncodeStageCount;
        if (stageTimers) syncDev();
        const auto tLoop0 = Clock::now();
        const double loopEv0 = evalMs, loopNb0 = normBootMs, loopBo0 = bootMs, loopEn0 = encMs, loopDe0 = decMs;
        // ---- time mix ----
        emit("Hs@tm.entry", "probe", Hs.ct->GetLevel());
        if (pf) refresh(Hs, needEntry(1, I), "L0.tm.rmsnorm.Hs"); else refresh(Hs, 3, "L0.tm.rmsnorm.Hs");
        TrackedCt u = rmsnormL(Hs, packCh(g1), a0, b0, I, 1, "L0.tm.rmsnorm");
        const auto stShift = stageOpen();                                 // S3.7 V2: shiftMixMs
        TrackedCt ua = u; mulPtL(ua, packCh(mix));                        // shift-mix, u_{-1} = 0
        u = ua;
        stageClose(stShift, shiftMixMs);
        u.used = u.ct->GetLevel();                                       // matvec encodes at .used (review: keep it = GetLevel)
        TrackedCt x = matvec(u, Wsq, d, d, 0);                            // win
        const auto stScan = stageOpen();                                  // S3.7 V2: scanMs
        refresh(scan, 3, "L0.tm.scan.state"); refresh(x, 3, "L0.tm.scan.x");
        mulPtL(scan, packCh(dec_));
        TrackedCt bx = x; mulPtL(bx, packCh(bco));
        addAlignedL(scan, bx, "L0.tm.scan");
        TrackedCt S = scan;
        stageClose(stScan, scanMs);
        const auto stGate = stageOpen();                                  // S3.7 V2: gateMs
        TrackedCt zc = S; mulPtL(zc, packCh(cc_));
        TrackedCt zd = x; mulPtL(zd, packCh(dd));
        addAlignedL(zc, zd, "L0.tm.gate.readout");
        refresh(zc, 5, "L0.tm.gate.zc");
        TrackedCt g = polyGate3L(zc, p0, p1, p2);
        stageClose(stGate, gateMs);
        g.used = g.ct->GetLevel();
        TrackedCt attn = matvec(g, Wsq, d, d, 0);                          // wout
        const auto stResTm = stageOpen();                                 // S3.7 V2: residualMs
        addAlignedL(Hs, attn, "L0.tm.residual");
        stageClose(stResTm, residualMs);
        // ---- channel mix ----
        emit("Hs@cm.entry", "probe", Hs.ct->GetLevel());
        if (pf) refresh(Hs, needEntry(2, I), "L0.cm.rmsnorm.Hs"); else refresh(Hs, 3, "L0.cm.rmsnorm.Hs");
        TrackedCt u2 = rmsnormL(Hs, packCh(g2), a0, b0, I, 2, "L0.cm.rmsnorm");
        if (pf) refresh(u2, needTail(2) - 2u + 1u, "L0.cm.u2");
        u2.used = u2.ct->GetLevel();
        auto kks = matvecChunked(u2, Wk, dff, d);
        auto rrs = matvecChunked(u2, Wr, dff, d);
        TrackedCt ffn; bool haveFfn = false;
        for (uint32_t ch = 0; ch < NCHUNK; ch++) {
            const uint32_t n = std::min(D, dff - ch * D);
            auto chunkOf = [&](const std::vector<double>& c) { std::vector<double> s_(n); for (uint32_t i = 0; i < n; i++) s_[i] = c[ch * D + i]; return packCh(s_); };
            const std::string sch = "L0.cm.hidden.ch" + std::to_string(ch);
            const auto stHid = stageOpen();                               // S3.7 V2: hiddenMs (per chunk)
            TrackedCt kk = kks[ch], rr = rrs[ch];
            emit("kk@birth", "probe", kk.ct->GetLevel());
            refresh(kk, 5, sch + ".kk"); refresh(rr, 4, sch + ".rr");
            TrackedCt k2{cc->EvalMult(kk.ct, kk.ct), 0}; cc->RescaleInPlace(k2.ct);
            TrackedCt actT = kk; mulPtL(actT, chunkOf(q1));
            TrackedCt t2 = k2; mulPtL(t2, chunkOf(q2));
            addAlignedL(actT, t2, sch + ".act12");
            TrackedCt kkA = kk; alignToL(kkA, k2.ct->GetLevel());
            TrackedCt k3{cc->EvalMult(k2.ct, kkA.ct), 0}; cc->RescaleInPlace(k3.ct); mulPtL(k3, chunkOf(q3));
            addAlignedL(actT, k3, sch + ".act3");
            { auto p_ = ptL(chunkOf(q0), actT.ct); actT.ct = cc->EvalAdd(actT.ct, p_); }
            TrackedCt r2c{cc->EvalMult(rr.ct, rr.ct), 0}; cc->RescaleInPlace(r2c.ct);
            TrackedCt gt = rr; mulPtL(gt, chunkOf(r1));
            TrackedCt g2t = r2c; mulPtL(g2t, chunkOf(r2));
            addAlignedL(gt, g2t, sch + ".gate");
            { auto p_ = ptL(chunkOf(r0), gt.ct); gt.ct = cc->EvalAdd(gt.ct, p_); }
            refresh(actT, 2, sch + ".actT"); refresh(gt, 2, sch + ".gt");
            const uint32_t uu = std::max(actT.ct->GetLevel(), gt.ct->GetLevel());
            alignToL(actT, uu); alignToL(gt, uu);
            TrackedCt hd{cc->EvalMult(actT.ct, gt.ct), 0}; cc->RescaleInPlace(hd.ct);
            refresh(hd, 2, sch + ".hd");
            refresh(hd, margin, "L0.cm.wv.ch" + std::to_string(ch) + ".margin");
            emit("hd@wv.in", "probe", hd.ct->GetLevel());
            std::vector<double> Wb((size_t)d * D, 0.0);
            for (uint32_t i = 0; i < d; i++) for (uint32_t j = 0; j < n; j++) Wb[(size_t)i * D + j] = Wv[(size_t)i * dff + ch * D + j];
            hd.used = hd.ct->GetLevel();
            stageClose(stHid, hiddenMs);                                  // S3.7 V2 (the Wb slice is inside, as in the .cu)
            TrackedCt part = matvec(hd, Wb, d, D, 0);
            const auto stFfn = stageOpen();                               // S3.7 V2: residualMs (ffnAccum)
            if (!haveFfn) { ffn = part; haveFfn = true; } else addAlignedL(ffn, part, "L0.cm.ffnAccum.ch" + std::to_string(ch));
            stageClose(stFfn, residualMs);
        }
        const auto stResCm = stageOpen();                                 // S3.7 V2: residualMs (cm residual)
        addAlignedL(Hs, ffn, "L0.cm.residual");
        stageClose(stResCm, residualMs);
        if (stageTimers) {                                                // S3.7 V2: close the layer's clock
            syncDev();
            layerLoopMs += msSince(tLoop0);
            layerLoopTimedMs += ((evalMs - loopEv0) - (normBootMs - loopNb0)) + (bootMs - loopBo0)
                              + (encMs - loopEn0) + (decMs - loopDe0);
        }
        emit("Hs@layer.out", "probe", Hs.ct->GetLevel());
        auto o = dec(Hs);
        bool finite = true; for (uint32_t i = 0; i < d; i++) finite = finite && std::isfinite(o[i]);
        std::cout << "{\"cell\":\"pf-trace\",\"schedule\":\"" << cellSchedule << "\",\"pfEntry\":\"" << pfEntry << "\",\"newtonDepth2\":" << (cellNd2 ? "true" : "false")
                  << ",\"depth\":" << depth << ",\"lamObserved\":" << lamObserved << ",\"F\":" << F << ",\"margin\":" << margin << ",\"iters\":" << I
                  << ",\"Dpad\":" << D << ",\"d\":" << d << ",\"dff\":" << dff << ",\"boots\":" << boots << ",\"finite\":" << (finite ? "true" : "false")
                  << ",\"HsLevelOut\":" << Hs.ct->GetLevel() << "}" << std::endl;
        verdict("pf_trace_layer_finite", finite, 0.0, true);
        // ---- S3.7 V2: the timers object for THIS layer (deltas since the snapshot) and
        // the two accounting verdicts. (1) the norm-boot subtraction: normBootMs (the
        // window's bootMs delta) must equal normBootDirectMs (the same dt tallied at the
        // boot site while a norm window is open) EXACTLY, and be <= bootMs; the ex-boot
        // eval must be >= 0. (2) disjointness: the loop's untimed part minus the five
        // stages (the residual) is >= 0 up to rounding, and so is each stage.
        {
            const double dEv = evalMs - tEv0, dNb = normBootMs - tNb0, dBo = bootMs - tBo0, dEn = encMs - tEn0, dDe = decMs - tDe0;
            const double sSm = shiftMixMs - tSm0, sSc = scanMs - tSc0, sGa = gateMs - tGa0, sHi = hiddenMs - tHi0, sRe = residualMs - tRe0;
            const double stages = sSm + sSc + sGa + sHi + sRe;
            const double loopUntimed = layerLoopMs - layerLoopTimedMs;
            const double residual = loopUntimed - stages;
            const bool subOk = (dNb == normBootDirectMs) && (dNb <= dBo + 1e-9) && (dEv - dNb >= -1e-9);
            const bool disjOk = !stageTimers || (residual >= -1e-6 && sSm >= -1e-6 && sSc >= -1e-6 && sGa >= -1e-6 && sHi >= -1e-6 && sRe >= -1e-6
                                                 && layerLoopMs >= layerLoopTimedMs - 1e-6);
            std::cout << "{\"cell\":\"pf-trace\",\"timers\":{\"stageTimersOn\":" << (stageTimers ? "true" : "false")
                      << ",\"boots\":" << boots << ",\"bootsInNorm\":" << bootsInNorm
                      << ",\"evalMs\":" << dEv << ",\"bootMs\":" << dBo << ",\"normBootMs\":" << dNb << ",\"normBootDirectMs\":" << normBootDirectMs
                      << ",\"gpuEvalMsExBoot\":" << (dEv - dNb) << ",\"msPerTokenNaive\":" << (dEv + dBo) << ",\"msPerTokenExBoot\":" << ((dEv - dNb) + dBo)
                      << ",\"encMs\":" << dEn << ",\"decMs\":" << dDe
                      << ",\"layerLoopMs\":" << layerLoopMs << ",\"layerLoopTimedMs\":" << layerLoopTimedMs << ",\"layerLoopUntimedMs\":" << loopUntimed
                      << ",\"stageMs\":{\"shiftMix\":" << sSm << ",\"scan\":" << sSc << ",\"gate\":" << sGa << ",\"hidden\":" << sHi << ",\"residual\":" << sRe << "}"
                      << ",\"layerLoopResidualMs\":" << residual
                      << ",\"hostPtEncodeCount\":" << (hostPtEncodeCount - tPc0) << ",\"hostPtEncodeStageCount\":" << (hostPtEncodeStageCount - tPs0)
                      << ",\"hostPtEncodeMs\":" << hostPtEncodeMs << ",\"hostPtEncodeStageMs\":" << hostPtEncodeStageMs << "}}" << std::endl;
            verdict("timers_norm_boot_subtraction", subOk, dNb - normBootDirectMs, true);
            verdict("timers_stage_disjoint", disjOk, residual, true);
        }
    }

    std::cout << "{\"summary\":true,\"hardFail\":" << g_hard << ",\"softFail\":" << g_soft
              << ",\"normEps\":" << normEpsUsed                 // S3.7 V1: eps applied in rmsnorm (0 = --no-norm-eps), additive
              // S3.7 V2 (additive): the process-wide accumulators, same names as the .cu summary
              << ",\"timers\":{\"gpuEvalMs\":" << evalMs << ",\"gpuBootstrapMs\":" << bootMs << ",\"normBootMs\":" << normBootMs
              << ",\"gpuEvalMsExBoot\":" << (evalMs - normBootMs) << ",\"clientEncryptMs\":" << encMs << ",\"decryptMs\":" << decMs
              << ",\"hostPtEncodeMs\":" << hostPtEncodeMs << ",\"hostPtEncodeCount\":" << hostPtEncodeCount
              << ",\"hostPtEncodeStageMs\":" << hostPtEncodeStageMs << ",\"hostPtEncodeStageCount\":" << hostPtEncodeStageCount
              << ",\"layerLoopMs\":" << layerLoopMs << ",\"stageMs\":{\"shiftMix\":" << shiftMixMs << ",\"scan\":" << scanMs
              << ",\"gate\":" << gateMs << ",\"hidden\":" << hiddenMs << ",\"residual\":" << residualMs << "}}"
              << fhe_ssm::runStamp(argc, argv) << "}" << std::endl;
    return g_hard ? 3 : 0;
}
