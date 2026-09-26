// c11_sparse_repro.cpp — standalone reduction of C11 (S2.75, 2026-08-16).
//
// CLAIM under reduction (results/openfhe-bootstrap.md:76-96, 2026-07-10,
// stock OpenFHE v1.5.1 CPU): with SPARSE packing (64 of 2048 slots), a
// ciphertext that has been through ~29 levels of ct×pt arithmetic bootstraps
// to O(1) garbage (~0.4 abs err) — while (a) a FRESH sparse ciphertext
// bootstraps at ~1e-14, and (b) the SAME deep chain fully packed bootstraps
// at ~1e-14. Hypothesized mechanism: sparse bootstrapping assumes the
// ciphertext encodes a replication-periodic (subring) element; plaintext
// encodings are only approximately periodic (independent per-coefficient
// rounding), so each ct×pt injects a small aperiodic component that the
// sparse transforms amplify into the message band.
//
// Four arms, one number each (max abs err after EvalBootstrap):
//   sparse_fresh   : numSlots=64, bootstrap immediately        (expect ~1e-14..1e-8)
//   sparse_deep    : numSlots=64, 29 ct×pt levels, bootstrap   (CLAIM: O(0.1-1))
//   full_deep      : full slots,  29 ct×pt levels, bootstrap   (expect small)
//   sparse_deep_ctct: numSlots=64, 29 ct×CT levels, bootstrap  (isolates ct×pt
//                     encoding aperiodicity vs depth itself)
//
// UPDATE 2026-08-16 (S2.75+, results/upstream-bugs/C11_ASSESSMENT_20260816.md):
// the hypothesis in the header above is REFUTED, and so is the mixed-encoding
// re-attribution that this file's arms were built to test. Two facts to know
// before reading any output of this tool:
//   * The DEFAULT level budget here is {3,3}. The 2026-07-10 demo that
//     produced the claim ran {4,4}. At {4,4} the SLOT-MATCHED sparse_deep arm
//     collapses (5.2e-01) and the claim REPRODUCES; the trigger is the
//     DECODING budget leaving SelectLayers rem != 0, and it is independent of
//     operand texture (--texture 0 collapses too), of ct*pt vs ct*ct, and of
//     level headroom. Use --budget to move it.
//   * The mixed arm's error is NOT caused by the bootstrap: it is fully
//     present before it, because the 64-slot read-out of a mixed product is
//     the coset mean. See harness/c11_mixed_slot_diag.cpp.
//
// Build: ninja c11_sparse_repro
// Run:   ./c11_sparse_repro [--levels 29] [--texture 0.001] [--budget 3 3]
//                           [--headroom 2]
// Output: one JSON line per arm + a verdict line. NOTE the verdict line only
// looks at sparse_deep/full_deep/sparse_fresh, so at {3,3} it prints "NOT
// REPRODUCED" while the mixed arm is loud — read the arms, not the verdict.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "openfhe.h"

using namespace lbcrypto;

static double TEXTURE = 0.001;
// level budget: default {3,3}; the ORIGINAL 2026-07-10 demo used {4,4}
// (results/openfhe-bootstrap.md:67-68 — "the demo's {4,4} floor (3e-5)"), so
// --budget lets the reduction be run at the original config. Additive flag.
static uint32_t BUDGET0 = 3, BUDGET1 = 3;
// spare levels above (chain + bootstrap depth). The original reduction hard-coded
// +2; --headroom separates "the sparse bootstrap is broken at this budget" from
// "the context was one level too tight and the library returned garbage instead
// of throwing". Additive flag, default reproduces the 2026-08-16 records.
static int HEADROOM = 2;
// sparse slot count. 64 (of 2048) is the C11 config; --slots 256 tests whether
// the rem != 0 rule generalises beyond logSlots = 6.
static uint32_t SPARSE_SLOTS = 64;

static double run_arm(const std::string &name, uint32_t numSlots, int levels,
                      bool ctct, uint32_t fullSlots, bool fullSlotPt = false) {
    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecretKeyDist(UNIFORM_TERNARY);
    p.SetSecurityLevel(HEStd_NotSet);          // small ring: mechanism probe, matches the original demo class
    p.SetRingDim(1 << 12);                     // 4096 => full packing = 2048 slots
    std::vector<uint32_t> levelBudget = {BUDGET0, BUDGET1};
    uint32_t approxDepth = FHECKKSRNS::GetBootstrapDepth(levelBudget, UNIFORM_TERNARY);
    p.SetMultiplicativeDepth(levels + approxDepth + HEADROOM);
    p.SetScalingModSize(59);
    p.SetFirstModSize(60);
    p.SetScalingTechnique(FLEXIBLEAUTO);
    p.SetBatchSize(numSlots);

    auto cc = GenCryptoContext(p);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE); cc->Enable(FHE);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalBootstrapSetup(levelBudget, {0, 0}, numSlots);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    // "recurrence state"-like message, amplitude ~0.3
    std::vector<double> msg(numSlots);
    for (uint32_t i = 0; i < numSlots; ++i) msg[i] = 0.3 * std::sin(0.37 * i);
    auto ct = cc->Encrypt(keys.publicKey,
                          cc->MakeCKKSPackedPlaintext(msg, 1, 0, nullptr, numSlots));

    // near-identity per-level multiplier: value 1.0 + tiny per-slot texture so
    // the product stays in range while every level does REAL ct×pt work
    std::vector<double> gain(numSlots);
    for (uint32_t i = 0; i < numSlots; ++i) gain[i] = 1.0 + TEXTURE * std::cos(0.11 * i);
    Ciphertext<DCRTPoly> ctGain;
    if (ctct)
        ctGain = cc->Encrypt(keys.publicKey,
                             cc->MakeCKKSPackedPlaintext(gain, 1, 0, nullptr, numSlots));

    std::vector<double> ref = msg;
    for (int l = 0; l < levels; ++l) {
        if (ctct) {
            // fresh-ish encrypted gain at matching level via bootstrap-free
            // path: re-encrypt each round (keeps the multiplier shallow)
            auto g = cc->Encrypt(keys.publicKey,
                                 cc->MakeCKKSPackedPlaintext(gain, 1, ct->GetLevel(),
                                                             nullptr, numSlots));
            ct = cc->EvalMult(ct, g);
        } else if (fullSlotPt) {
            // MIXED-ENCODING arm: operand plaintext encoded at FULL slots
            // against the sparse-slot state — the product is aperiodic w.r.t.
            // the 64-slot subring by O(1), not by rounding. Tests whether the
            // original collapse was this sharp edge rather than accumulated
            // rounding aperiodicity.
            std::vector<double> gainFull(fullSlots);
            for (uint32_t i = 0; i < fullSlots; ++i)
                gainFull[i] = 1.0 + TEXTURE * std::cos(0.11 * i);
            auto pt = cc->MakeCKKSPackedPlaintext(gainFull, 1, ct->GetLevel(),
                                                  nullptr, fullSlots);
            ct = cc->EvalMult(ct, pt);
        } else {
            // canonical discipline: encode the operand at the ciphertext's
            // exact level (the original finding ADOPTED this and still failed)
            auto pt = cc->MakeCKKSPackedPlaintext(gain, 1, ct->GetLevel(), nullptr, numSlots);
            ct = cc->EvalMult(ct, pt);
        }
        for (uint32_t i = 0; i < numSlots; ++i) ref[i] *= gain[i];
    }

    ct->SetSlots(numSlots);
    auto ctBoot = cc->EvalBootstrap(ct);

    Plaintext out;
    cc->Decrypt(keys.secretKey, ctBoot, &out);
    out->SetLength(numSlots);
    double maxErr = 0.0;
    auto vals = out->GetRealPackedValue();
    for (uint32_t i = 0; i < numSlots; ++i)
        maxErr = std::max(maxErr, std::fabs(vals[i] - ref[i]));

    std::printf("{\"arm\":\"%s\",\"numSlots\":%u,\"fullSlots\":%u,\"levels\":%d,"
                "\"ringDim\":%u,\"depthTotal\":%u,\"maxAbsErrAfterBoot\":%.6e}\n",
                name.c_str(), numSlots, fullSlots, levels,
                cc->GetRingDimension(), p.GetMultiplicativeDepth(), maxErr);
    std::fflush(stdout);
    return maxErr;
}

int main(int argc, char **argv) {
    int levels = 29;
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--levels") levels = std::atoi(argv[i + 1]);
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--texture") TEXTURE = std::atof(argv[i + 1]);
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--headroom") HEADROOM = std::atoi(argv[i + 1]);
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--slots") SPARSE_SLOTS = std::atoi(argv[i + 1]);
    for (int i = 1; i < argc - 2; ++i)
        if (std::string(argv[i]) == "--budget") {
            BUDGET0 = std::atoi(argv[i + 1]);
            BUDGET1 = std::atoi(argv[i + 2]);
        }
    std::printf("{\"texture\":%g,\"levelBudget\":[%u,%u],\"headroom\":%d,\"sparseSlots\":%u}\n",
                TEXTURE, BUDGET0, BUDGET1, HEADROOM, SPARSE_SLOTS);
    const uint32_t FULL = (1 << 12) / 2;   // 2048
    const uint32_t SPARSE = SPARSE_SLOTS;

    double eFresh  = run_arm("sparse_fresh", SPARSE, 0, false, FULL);
    double eDeep   = run_arm("sparse_deep", SPARSE, levels, false, FULL);
    double eFull   = run_arm("full_deep", FULL, levels, false, FULL);
    double eCtCt   = run_arm("sparse_deep_ctct", SPARSE, levels, true, FULL);
    double eMixed  = run_arm("sparse_deep_fullslot_pt", SPARSE, levels, false, FULL, true);
    std::printf("{\"mixed_encoding_arm\":%.3e}\n", eMixed);

    const char *verdict =
        (eDeep > 1e-2 && eFull < 1e-4 && eFresh < 1e-4)
            ? "REPRODUCED: sparse+deep-ct*pt collapses while fresh-sparse and full-deep stay clean"
        : (eDeep < 1e-4)
            ? "NOT REPRODUCED at this config: sparse_deep bootstraps clean"
            : "AMBIGUOUS: see arm values";
    std::printf("{\"c11_verdict\":\"%s\",\"sparse_fresh\":%.3e,\"sparse_deep\":%.3e,"
                "\"full_deep\":%.3e,\"sparse_deep_ctct\":%.3e}\n",
                verdict, eFresh, eDeep, eFull, eCtCt);
    return 0;
}
