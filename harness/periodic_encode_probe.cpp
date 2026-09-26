// periodic_encode_probe.cpp -- S3.5 (2026-09-04): the host-encode term of tick 0, dense vs periodic.
//
// QUESTION. The demo store holds 344,064 plaintexts per level trajectory (688,128 under
// --lanes-block), each produced by cc->MakeCKKSPackedPlaintext(diag, 1, lvl) on a slot vector
// that is Dpad-periodic (REP = N/(2 Dpad) replicas, gpu_real_model.cu buildComp). The dense
// encode is an inverse special FFT over N/2 slots plus one NTT of N coefficients per RNS limb;
// the pod's tick 0 spent 2,131,944 ms in it (demo2/server.log req 0). A Dpad-periodic slot
// vector encodes to a polynomial in X^REP, i.e. to the CKKS encoding of ONE period in the ring
// of degree N' = N/REP, so the same plaintext can be produced with an N'/2-point FFT and
// N'-point NTTs -- REP x (log N / log N') fewer operations. This probe (CPU, OpenFHE 1.5.1,
// the same library the pod's encode runs on) measures both on one machine and PROVES the
// identity bit for bit:
//   (1) the dense encode's coefficient vector is exactly zero off the residue class j = 0 mod REP
//       (OpenFHE's FFTSpecialInv computes v = (a-b)*xi with a == b bit-identical for the first
//       log2(REP) stages, dftransform.cpp:209-235, so the zeros are exact, not rounded);
//   (2) the ring-N' NTT (root psi^REP) of the subsampled coefficients q[j'] = p[REP j'] equals
//       the harness's compressed limb comp[j2] = limb[j2 << log2 REP] word for word (the index
//       map of Lemma 13 / fhe_ssm_expand_limbs_v);
//   (3) the ring-N' CKKS encode of one period (FFTSpecialInv at cyclotomic order 2N', the same
//       scale, the same rounding) reproduces q bit for bit, hence the whole store entry.
// Then it times: dense encode (MakeCKKSPackedPlaintext), the harness's compress/verify loop,
// and the periodic encode (small FFT + rounding + small NTTs), per level.
//
// Build (Mac, no cmake): see the clang++ recipe in the memory file / harness/CMakeLists.txt.
// Run:   ./harness/build/periodic_encode_probe [--log-ring 17] [--depth 41] [--levels 0,19,24,35,38] [--reps 5]
// Output: one JSON line per (vector, level) with timings (medians) and the three identity verdicts,
//         plus a header line with the machine/build identity. Exit 3 if any identity fails.
#include "openfhe.h"
#include "math/dftransform.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;
static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}
#ifndef PROBE_GIT_SHA
#define PROBE_GIT_SHA "unknown"
#endif
#ifndef PROBE_BUILD_UTC
#define PROBE_BUILD_UTC "unknown"
#endif

int main(int argc, char** argv) {
    uint32_t logRing = 17, depth = 41, dpad = 1024;
    int reps = 5;
    std::vector<uint32_t> levels = {0, 19, 24, 35, 38};
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--log-ring") logRing = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--depth") depth = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--dpad") dpad = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--reps") reps = std::stoi(argv[++i]);
        else if (a == "--levels") {
            levels.clear();
            std::string s = argv[++i];
            size_t p = 0;
            while (p <= s.size()) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                levels.push_back((uint32_t)std::stoi(s.substr(p, q - p)));
                p = q + 1;
            }
        }
    }
    // The demo chain: 2^17, depth 41 (10 + 19 + extra 12), Delta 2^59, first modulus 60, FLEXIBLEAUTO,
    // HEStd_128_classic (gpu_real_model.cu --secure; the prime chain is a function of (N, depth,
    // scale bits) alone, the security level only checks the ring).
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetRingDim(1u << logRing);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(depth);
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    const uint32_t N = cc->GetRingDimension(), SLOTS = N / 2, REP = SLOTS / dpad, Np = N / REP;
    uint32_t logRep = 0; while ((1u << logRep) < REP) logRep++;
    const uint32_t CW = N / REP;                       // COMP_CW words per limb (:1350)
    auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    const auto& prs = cp->GetElementParams()->GetParams();
    std::cout << "{\"probe\":\"periodic_encode\",\"N\":" << N << ",\"slots\":" << SLOTS << ",\"dpad\":" << dpad
              << ",\"rep\":" << REP << ",\"Nprime\":" << Np << ",\"depth\":" << depth << ",\"limbsTotal\":" << prs.size()
              << ",\"logQ\":" << cp->GetElementParams()->GetModulus().GetMSB()
              << ",\"scalingTech\":\"FLEXIBLEAUTO\",\"openfhe\":\"1.5.1 (vendor/install, WITH_OPENMP=OFF)\""
              << ",\"gitSha\":\"" << PROBE_GIT_SHA << "\",\"buildUtc\":\"" << PROBE_BUILD_UTC << "\"}" << std::endl;

    // small-ring FFT tables (the map is keyed by cyclotomic order; the 2N tables stay)
    DiscreteFourierTransform::Initialize(2 * Np, Np / 2);

    // ---- test vectors: one period each (length dpad), replicated REP times for the dense path
    struct TV { std::string name; std::vector<double> period; };
    std::vector<TV> tvs;
    {
        std::vector<double> w(dpad), ones(dpad, 1.0), lo(dpad, 0.0), zeros(dpad, 0.0);
        uint64_t s = 0x9E3779B97F4A7C15ULL;                        // fixed LCG: weight-like values ~0.02*N(0,1)
        for (uint32_t i = 0; i < dpad; i++) {
            double u = 0;
            for (int k = 0; k < 12; k++) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; u += (double)(s >> 11) / 9007199254740992.0; }
            w[i] = 0.02 * (u - 6.0);
        }
        for (uint32_t i = 0; i < dpad - 100; i++) lo[i] = w[i];    // a lanes-block lo half (k = 100)
        tvs = {{"weightlike", w}, {"ones", ones}, {"lohalf_k100", lo}, {"zeros_pooltemplate", zeros}};
    }
    int failures = 0;
    for (const auto& tv : tvs) {
        std::vector<double> dense(SLOTS, 0.0);
        for (uint32_t r = 0; r < REP; r++) for (uint32_t i = 0; i < dpad; i++) dense[r * dpad + i] = tv.period[i];
        for (uint32_t lvl : levels) {
            if (lvl > depth) continue;
            const uint32_t nLimbs = depth + 1 - lvl;
            // ---- dense encode timing (the harness's call, :2906) ----
            std::vector<double> tD;
            Plaintext pt;
            for (int r = 0; r < reps; r++) {
                auto t0 = Clock::now();
                pt = cc->MakeCKKSPackedPlaintext(dense, 1, lvl);
                tD.push_back(msSince(t0));
            }
            const double sf = pt->GetScalingFactor();
            const DCRTPoly& e = pt->GetElement<DCRTPoly>();
            if (e.GetNumOfElements() != nLimbs) { std::cerr << "limb count mismatch\n"; return 2; }
            // ---- the harness's compress/verify loop (buildComp :2914-2927), timed ----
            std::vector<std::vector<uint64_t>> compDense(nLimbs, std::vector<uint64_t>(CW));
            bool okmap = true;
            std::vector<double> tC;
            for (int r = 0; r < reps; r++) {
                auto t0 = Clock::now();
                okmap = true;
                for (uint32_t l = 0; okmap && l < nLimbs; l++) {
                    const auto& limbv = e.GetElementAtIndex(l).GetValues();
                    if (limbv.GetLength() != N) { okmap = false; break; }
                    for (uint32_t i2 = 0; i2 < N; i2++)
                        if (limbv[i2] != limbv[(i2 >> logRep) << logRep]) { okmap = false; break; }
                    if (okmap) for (uint32_t j2 = 0; j2 < CW; j2++) compDense[l][j2] = limbv[j2 << logRep].ConvertToInt();
                }
                tC.push_back(msSince(t0));
            }
            // ---- (1) coefficient domain: exact zeros off the residue class, and the subsample q ----
            DCRTPoly ec = e;
            ec.SetFormat(Format::COEFFICIENT);
            uint64_t offClassNonzero = 0;
            std::vector<NativeVector> q(nLimbs);
            for (uint32_t l = 0; l < nLimbs; l++) {
                const auto& cv = ec.GetElementAtIndex(l).GetValues();
                q[l] = NativeVector(Np, cv.GetModulus());
                for (uint32_t j = 0; j < N; j++) {
                    if (j % REP) { if (cv[j] != NativeInteger(0)) offClassNonzero++; }
                    else q[l][j / REP] = cv[j];
                }
            }
            // ---- (2) ring-N' NTT of q with root psi^REP == the compressed limb? ----
            uint64_t mapMismatch = 0;
            std::vector<NativeInteger> rootP(nLimbs);
            for (uint32_t l = 0; l < nLimbs; l++) {
                const NativeInteger mod = prs[l]->GetModulus();
                rootP[l] = prs[l]->GetRootOfUnity().ModExp(NativeInteger(REP), mod);
                NativeVector res(Np, mod);
                ChineseRemainderTransformFTT<NativeVector>().ForwardTransformToBitReverse(q[l], rootP[l], 2 * Np, &res);
                for (uint32_t j2 = 0; j2 < CW; j2++) if (res[j2].ConvertToInt() != compDense[l][j2]) mapMismatch++;
            }
            // ---- (3) periodic encode from the period alone, timed: small FFT + rounding + small NTTs ----
            std::vector<double> tP, tF, tR, tN;
            uint64_t periodicMismatch = 0; int64_t maxCoeffDiff = 0;
            for (int r = 0; r < reps; r++) {
                auto t0 = Clock::now();
                std::vector<std::complex<double>> inv(Np / 2);
                for (uint32_t i = 0; i < Np / 2; i++) inv[i] = std::complex<double>(tv.period[i], 0.0);
                DiscreteFourierTransform::FFTSpecialInv(inv, 2 * Np);             // the ring-N' special inverse FFT
                double tf = msSince(t0);
                auto t1 = Clock::now();
                // OpenFHE ckkspackedencoding.cpp:262-283 (NATIVEINT == 64): scale, overflow check, llround
                std::vector<int64_t> coef(Np);
                int32_t logc = std::numeric_limits<int32_t>::min();
                for (uint32_t i = 0; i < Np / 2; i++) {
                    inv[i] *= sf;
                    if (inv[i].real() != 0.) logc = std::max(logc, (int32_t)std::ceil(std::log2(std::abs(inv[i].real()))));
                    if (inv[i].imag() != 0.) logc = std::max(logc, (int32_t)std::ceil(std::log2(std::abs(inv[i].imag()))));
                }
                if (logc > 62) { std::cerr << "{\"fatal\":\"scaled value exceeds 62 bits (approxFactor path not replicated)\"}\n"; return 2; }
                for (uint32_t i = 0; i < Np / 2; i++) {
                    coef[i] = std::llround(inv[i].real());
                    coef[i + Np / 2] = std::llround(inv[i].imag());
                }
                double tr = msSince(t1);
                auto t2 = Clock::now();
                std::vector<std::vector<uint64_t>> compP(nLimbs, std::vector<uint64_t>(CW));
                for (uint32_t l = 0; l < nLimbs; l++) {
                    const NativeInteger mod = prs[l]->GetModulus();
                    const uint64_t qm = mod.ConvertToInt();
                    NativeVector nv(Np, mod);
                    for (uint32_t j = 0; j < Np; j++) {
                        int64_t c = coef[j];
                        uint64_t rres = c >= 0 ? (uint64_t)c % qm : (qm - ((uint64_t)(-c) % qm)) % qm;
                        nv[j] = NativeInteger(rres);
                    }
                    NativeVector res(Np, mod);
                    ChineseRemainderTransformFTT<NativeVector>().ForwardTransformToBitReverse(nv, rootP[l], 2 * Np, &res);
                    for (uint32_t j2 = 0; j2 < CW; j2++) compP[l][j2] = res[j2].ConvertToInt();
                }
                double tn = msSince(t2);
                tP.push_back(msSince(t0)); tF.push_back(tf); tR.push_back(tr); tN.push_back(tn);
                if (r == 0) {
                    for (uint32_t l = 0; l < nLimbs; l++)
                        for (uint32_t j2 = 0; j2 < CW; j2++) if (compP[l][j2] != compDense[l][j2]) periodicMismatch++;
                    // integer coefficient difference vs the dense subsample (limb 0 is a faithful signed view
                    // only for |c| < q/2, which holds at Delta = 2^59 < q_0 = 2^60 for |v| < 2)
                    const uint64_t q0 = prs[0]->GetModulus().ConvertToInt();
                    for (uint32_t j = 0; j < Np; j++) {
                        uint64_t dv = q[0][j].ConvertToInt();
                        int64_t dsigned = dv > q0 / 2 ? (int64_t)dv - (int64_t)q0 : (int64_t)dv;
                        maxCoeffDiff = std::max<int64_t>(maxCoeffDiff, std::llabs(dsigned - coef[j]));
                    }
                }
            }
            // ---- standalone building blocks of the dense path, for the cost split ----
            double tFFTbig = 0, tNTTbig = 0;
            {
                std::vector<double> a, b;
                for (int r = 0; r < std::max(1, reps / 2); r++) {
                    std::vector<std::complex<double>> big(SLOTS);
                    for (uint32_t i = 0; i < SLOTS; i++) big[i] = std::complex<double>(dense[i], 0.0);
                    auto t0 = Clock::now();
                    DiscreteFourierTransform::FFTSpecialInv(big, 2 * N);
                    a.push_back(msSince(t0));
                    NativeVector nv(N, prs[0]->GetModulus());
                    for (uint32_t j = 0; j < N; j++) nv[j] = NativeInteger((uint64_t)(j * 2654435761u) % prs[0]->GetModulus().ConvertToInt());
                    NativeVector res(N, prs[0]->GetModulus());
                    auto t1 = Clock::now();
                    ChineseRemainderTransformFTT<NativeVector>().ForwardTransformToBitReverse(nv, prs[0]->GetRootOfUnity(), 2 * N, &res);
                    b.push_back(msSince(t1));
                }
                tFFTbig = median(a); tNTTbig = median(b);
            }
            const bool ok = okmap && offClassNonzero == 0 && mapMismatch == 0 && periodicMismatch == 0;
            if (!ok) failures++;
            std::cout << "{\"vector\":\"" << tv.name << "\",\"level\":" << lvl << ",\"limbs\":" << nLimbs
                      << ",\"scalingFactorLog2\":" << std::log2(sf)
                      << ",\"denseEncodeMs\":" << median(tD) << ",\"compressVerifyMs\":" << median(tC)
                      << ",\"periodicEncodeMs\":" << median(tP) << ",\"periodicFftMs\":" << median(tF)
                      << ",\"periodicRoundMs\":" << median(tR) << ",\"periodicNttMs\":" << median(tN)
                      << ",\"denseFftBigMs\":" << tFFTbig << ",\"denseNttOneLimbMs\":" << tNTTbig
                      << ",\"ratioDenseOverPeriodic\":" << (median(tP) > 0 ? median(tD) / median(tP) : 0.0)
                      << ",\"okmap\":" << (okmap ? "true" : "false")
                      << ",\"offClassNonzeroCoeffs\":" << offClassNonzero
                      << ",\"nttIndexMapMismatchWords\":" << mapMismatch
                      << ",\"periodicVsDenseMismatchWords\":" << periodicMismatch
                      << ",\"maxIntCoeffDiff\":" << maxCoeffDiff
                      << ",\"wordsPerEntry\":" << (uint64_t)nLimbs * CW
                      << ",\"identity\":\"" << (ok ? "EXACT" : "FAIL") << "\"}" << std::endl;
        }
    }
    std::cout << "{\"done\":true,\"identityFailures\":" << failures << "}" << std::endl;
    return failures ? 3 : 0;
}
