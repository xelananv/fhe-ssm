// overflow_tail_probe2.cpp -- DIAGNOSTIC successor of overflow_tail_probe.cpp (which stays frozen: its records carry its sha256).
// The first run found, for the "evaluated" class only, 2 of 204.8 M overflows beyond 6 sigma (0.35 expected) and a maximum of 7.12 sigma.
// This version keeps every intermediate ciphertext of the pipeline and, whenever the FINAL ciphertext has max|I| >= 5.3 sigma, logs where
// along the pipeline the outlier appeared (exact I of the q0 tower of every stage, at the final argmax and stage-wide), how many large
// values that ciphertext has, and whether the ciphertext still decrypts to the right message.
//
// (original header follows)
// overflow_tail_probe.cpp -- is the far tail of the ModRaise overflow I = round((c0 + c1*s)/q0) the same for EVALUATED
// ciphertexts as for FRESH encryptions?  (A11 tick-42 investigation, addendum to probe C, 2026-09-18.)
//
// WHY.  Probe C measured I on 56 saved ring-2^17 ciphertexts: the 28 fresh requests follow the Gaussian model, the 28 pipeline
// responses show a small excess in the far tail (one value at 5.89 sigma, 36 vs 24.7 expected beyond 4.5 sigma).  Theory says the
// tail should not depend on how the ciphertext was produced as long as c1 is uniform given s.  This probe tests exactly that in
// the same library on the CPU, with enough samples to resolve an excess of that size: many ciphertexts of two kinds under ONE
// stock uniform-ternary key, no bootstrapping, the exact integer I of the q0 tower of each (negacyclic convolution in __int128).
//   fresh      Encrypt(pk, pt)                                  (what a request is)
//   evaluated  a short layer-like pipeline: ct*ct (relinearised), rotation, plaintext product, additions, a square, rescales
// It writes AGGREGATES only (histogram of I, tail counts, per-ciphertext sigma scatter); the key never leaves the process.
//
// Build: PATH=$PWD/.venv/bin:$PATH ninja -C harness/build overflow_tail_probe2
// Run:   tools/memguard.sh 4 harness/build/overflow_tail_probe2 --log-ring 12 --cts 50000 --classes evaluated

#include "openfhe.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "run_stamp.h"

using namespace lbcrypto;
using PolyT = DCRTPoly::PolyType;
typedef __int128 i128;

static std::vector<int64_t> centered(PolyT p) {   // the library's centering rule: v > q>>1 is negative (mubintvecnat.cpp:109-:121)
    p.SetFormat(Format::COEFFICIENT);
    const uint64_t q = p.GetModulus().ConvertToInt<uint64_t>(), half = q >> 1;
    const uint32_t n = p.GetLength();
    std::vector<int64_t> out(n);
    for (uint32_t k = 0; k < n; k++) { const uint64_t v = p[k].ConvertToInt<uint64_t>(); out[k] = (v > half) ? -(int64_t)(q - v) : (int64_t)v; }
    return out;
}

static double Qf(double x) { return 0.5 * std::erfc(x / std::sqrt(2.0)); }

struct Acc {
    std::map<int64_t, uint64_t> hist; uint64_t n = 0, cts = 0; long double sum = 0, sum2 = 0; int64_t maxAbs = 0;
    long double zz = 0;   // sum over ciphertexts of ((sigma_hat^2 - sigma^2) / se_indep)^2, se_indep = sigma^2 * sqrt(2/N)
};

int main(int argc, char** argv) {
    uint32_t logRing = 12, M = 2000, seed = 20260918, depth = 8;
    std::string classesArg = "fresh,evaluated"; double thrSigma = 5.3;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--log-ring") logRing = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--cts") M = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--seed") seed = (uint32_t)std::stoul(argv[++i]);
        else if (a == "--classes") classesArg = argv[++i];
        else if (a == "--thr-sigma") thrSigma = std::stod(argv[++i]);
        else { std::cerr << "unknown flag " << a << std::endl; return 2; }
    }
    if (logRing > 15) { std::cerr << "W5: local rings are 2^15 or smaller" << std::endl; return 2; }
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1u << logRing);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(depth);
    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    const uint32_t N = cc->GetRingDimension(), slots = N / 2;
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalAtIndexKeyGen(keys.secretKey, {1, 5, -3});
    const auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    const uint64_t q0 = cp->GetElementParams()->GetParams()[0]->GetModulus().ConvertToInt<uint64_t>();

    const std::vector<int64_t> s = centered(keys.secretKey->GetPrivateElement().GetElementAtIndex(0));
    long long h = 0; for (int64_t x : s) h += (x != 0);
    const double sig2 = (h + 1) / 12.0, sigma = std::sqrt(sig2);
    // R2 = sum over nonzero lags of (negacyclic autocorrelation of s / h)^2 -> sample-variance inflation 1 + R2 - 0.6 (h-1)/h
    double R2 = 0;
    for (uint32_t k = 1; k < N; k++) {
        long long a = 0;
        for (uint32_t i = 0; i + k < N; i++) a += s[i] * s[i + k];
        for (uint32_t i = N - k; i < N; i++) a -= s[i] * s[i + k - N];
        R2 += ((double)a / h) * ((double)a / h);
    }
    const double inflate = 1.0 + R2 - 0.6 * (h - 1) / (double)h;
    std::cout << "{\"rec\":\"header\",\"harness\":\"overflow_tail_probe2\",\"ringDim\":" << N << ",\"q0\":" << q0 << ",\"depth\":" << depth
              << ",\"towersQ\":" << cp->GetElementParams()->GetParams().size() << ",\"hamming\":" << h << ",\"sigmaPred\":" << sigma
              << ",\"R2\":" << R2 << ",\"sampleVarianceInflation\":" << inflate << ",\"ctsPerClass\":" << M << ",\"seed\":" << seed
              << fhe_ssm::runStamp(argc, argv) << "}" << std::endl;

    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> ud(-1.0, 1.0);
    auto randVec = [&]() { std::vector<double> v(slots); for (auto& x : v) x = ud(g); return v; };
    const Plaintext ptDiag = cc->MakeCKKSPackedPlaintext(randVec());

    // exact overflow vector of the q0 tower of any ciphertext
    auto overflow = [&](ConstCiphertext<DCRTPoly> ct, std::vector<double>* fracOut = nullptr) {
        const auto c0 = centered(ct->GetElements()[0].GetElementAtIndex(0));
        const auto c1 = centered(ct->GetElements()[1].GetElementAtIndex(0));
        std::vector<i128> acc(N, 0);
        for (uint32_t i = 0; i < N; i++) {
            const int64_t si = s[i];
            if (si == 0) continue;
            for (uint32_t k = 0; k < N - i; k++) acc[i + k] += (i128)si * (i128)c1[k];
            for (uint32_t k = N - i; k < N; k++) acc[i + k - N] -= (i128)si * (i128)c1[k];
        }
        const i128 Q = (i128)q0, H = Q / 2;
        std::vector<int64_t> I(N);
        if (fracOut) fracOut->resize(N);
        for (uint32_t j = 0; j < N; j++) {
            const i128 t = acc[j] + (i128)c0[j]; const i128 u = t + H;
            const i128 Ii = (u >= 0) ? (u / Q) : -((-u + Q - 1) / Q);
            I[j] = (int64_t)Ii;
            if (fracOut) (*fracOut)[j] = (double)(t - Ii * Q) / (double)q0;
        }
        return I;
    };
    auto measure = [&](ConstCiphertext<DCRTPoly> ct, Acc& A) {
        const std::vector<int64_t> I = overflow(ct);
        long double cs = 0, cs2 = 0;
        for (uint32_t j = 0; j < N; j++) {
            A.hist[I[j]]++; cs += I[j]; cs2 += (long double)I[j] * I[j];
            const int64_t aI = I[j] < 0 ? -I[j] : I[j]; if (aI > A.maxAbs) A.maxAbs = aI;
        }
        A.n += N; A.cts++; A.sum += cs; A.sum2 += cs2;
        const double var = (double)(cs2 / N - (cs / N) * (cs / N));
        const double z = (var - sig2) / (sig2 * std::sqrt(2.0 / N));
        A.zz += (long double)z * z;
        return I;
    };
    auto statsOf = [&](const std::vector<int64_t>& I, int64_t& mx, uint32_t& arg, double& sd, uint32_t& n4) {
        long double a = 0, b = 0; mx = 0; arg = 0; n4 = 0; const int64_t b4 = (int64_t)std::floor(4 * sigma);
        for (uint32_t j = 0; j < N; j++) { a += I[j]; b += (long double)I[j] * I[j]; const int64_t v = I[j] < 0 ? -I[j] : I[j]; if (v > mx) { mx = v; arg = j; } if (v > b4) n4++; }
        sd = std::sqrt((double)(b / N - (a / N) * (a / N)));
    };
    auto report = [&](const std::string& name, const Acc& A, double secs, uint32_t towers, uint32_t nsd, bool final) {
        const double mean = (double)(A.sum / A.n), var = (double)(A.sum2 / A.n) - mean * mean;
        std::cout << "{\"rec\":\"" << (final ? "class" : "progress") << "\",\"class\":\"" << name << "\",\"ciphertexts\":" << A.cts << ",\"samples\":" << A.n
                  << ",\"towers\":" << towers << ",\"noiseScaleDeg\":" << nsd << ",\"mean\":" << mean << ",\"sigma\":" << std::sqrt(var)
                  << ",\"sigmaOverPred\":" << std::sqrt(var) / sigma << ",\"maxAbsI\":" << A.maxAbs << ",\"maxOverSigma\":" << A.maxAbs / sigma
                  << ",\"perCtVarChi2_independentSE\":" << (double)A.zz << ",\"perCtVarChi2_correlatedSE\":" << (double)A.zz / inflate << ",\"dof\":" << A.cts
                  << ",\"tails\":[";
        bool first = true;
        for (double z : {3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0}) {
            const int64_t b = (int64_t)std::floor(z * sigma); uint64_t obs = 0;
            for (auto& kv : A.hist) if (kv.first > b || kv.first < -b) obs += kv.second;
            std::cout << (first ? "" : ",") << "{\"z\":" << z << ",\"bound\":" << b << ",\"observed\":" << obs << ",\"expectedGaussian\":" << A.n * 2 * Qf((b + 0.5) / sigma) << "}";
            first = false;
        }
        std::cout << "],\"seconds\":" << secs;
        if (final) { std::cout << ",\"hist\":{"; bool f2 = true; for (auto& kv : A.hist) { std::cout << (f2 ? "" : ",") << "\"" << kv.first << "\":" << kv.second; f2 = false; } std::cout << "}"; }
        std::cout << "}" << std::endl;
    };

    const int64_t thr = (int64_t)std::ceil(thrSigma * sigma);
    std::vector<std::string> classes; { std::stringstream ss(classesArg); std::string tok; while (std::getline(ss, tok, ',')) if (!tok.empty()) classes.push_back(tok); }
    const std::vector<double> pv = ptDiag->GetRealPackedValue();
    for (const std::string& name : classes) {
        Acc A; auto t0 = std::chrono::steady_clock::now(); uint32_t towers = 0, nsd = 0;
        for (uint32_t t = 0; t < M; t++) {
            Ciphertext<DCRTPoly> z;
            const std::vector<double> xv = randVec(), yv = randVec();
            auto x = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(xv));
            std::vector<std::pair<std::string, Ciphertext<DCRTPoly>>> stages;
            if (name == "fresh") z = x;
            else {
                auto y = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(yv));
                stages.push_back({"x_fresh", x}); stages.push_back({"y_fresh", y});
                z = cc->EvalMult(x, y);                       stages.push_back({"mult_xy", z});
                z = cc->EvalRotate(z, 1);                     stages.push_back({"rot1", z});
                z = cc->EvalMult(z, ptDiag);                  stages.push_back({"ptmult", z});
                auto xr = cc->EvalRotate(x, 5);               stages.push_back({"rot5_x", xr});
                z = cc->EvalAdd(z, xr);                       stages.push_back({"add", z});
                z = cc->EvalMult(z, z);                       stages.push_back({"square", z});
                auto yr = cc->EvalRotate(y, -3);              stages.push_back({"rotm3_y", yr});
                z = cc->EvalSub(z, yr);                       stages.push_back({"sub", z});
                z = cc->EvalMult(z, 0.5);                     stages.push_back({"scalar", z->Clone()});   // Clone: the next line works in place
                cc->GetScheme()->ModReduceInternalInPlace(z, z->GetNoiseScaleDeg() - 1);
            }
            towers = z->GetElements()[0].GetNumOfElements(); nsd = z->GetNoiseScaleDeg();
            const std::vector<int64_t> I = measure(z, A);
            int64_t mx; uint32_t arg, n4; double sd; statsOf(I, mx, arg, sd, n4);
            if (mx >= thr) {
                std::vector<double> frac; overflow(z, &frac);
                std::cout << "{\"rec\":\"outlier\",\"class\":\"" << name << "\",\"ct\":" << t << ",\"maxAbsI\":" << mx << ",\"maxOverSigma\":" << mx / sigma
                          << ",\"signedI\":" << I[arg] << ",\"argmax\":" << arg << ",\"sigmaHat\":" << sd << ",\"nOver4sigma\":" << n4
                          << ",\"fracAtArgmax\":" << frac[arg] << ",\"towers\":" << towers;
                // is the ciphertext healthy?  decrypt and compare with the clear computation
                if (name != "fresh") {
                    std::vector<double> ref(slots);
                    for (uint32_t i = 0; i < slots; i++) {
                        const double m1 = xv[(i + 1) % slots] * yv[(i + 1) % slots] * pv[i] + xv[(i + 5) % slots];
                        ref[i] = 0.5 * (m1 * m1 - yv[(i + slots - 3) % slots]);
                    }
                    Plaintext p; double sse = 0, sref = 0; bool thrown = false;
                    try { cc->Decrypt(keys.secretKey, z, &p); p->SetLength(slots); auto v = p->GetRealPackedValue();
                          for (uint32_t i = 0; i < slots; i++) { sse += (v[i] - ref[i]) * (v[i] - ref[i]); sref += ref[i] * ref[i]; } }
                    catch (const std::exception&) { thrown = true; }
                    std::cout << ",\"decryptThrew\":" << (thrown ? "true" : "false") << ",\"relErrRms\":" << (thrown ? -1.0 : std::sqrt(sse / sref));
                }
                std::cout << ",\"stages\":[";
                for (size_t k = 0; k < stages.size(); k++) {
                    const std::vector<int64_t> Is = overflow(stages[k].second);
                    int64_t smx; uint32_t sarg, sn4; double ssd; statsOf(Is, smx, sarg, ssd, sn4);
                    std::cout << (k ? "," : "") << "{\"stage\":\"" << stages[k].first << "\",\"towers\":" << stages[k].second->GetElements()[0].GetNumOfElements()
                              << ",\"nsd\":" << stages[k].second->GetNoiseScaleDeg() << ",\"elements\":" << stages[k].second->GetElements().size()
                              << ",\"IatFinalArgmax\":" << Is[arg] << ",\"maxAbsI\":" << smx << ",\"argmax\":" << sarg << ",\"sigmaHat\":" << ssd << ",\"nOver4sigma\":" << sn4 << "}";
                }
                std::cout << "]}" << std::endl;
            }
            if ((t + 1) % 5000 == 0 && t + 1 < M)
                report(name, A, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), towers, nsd, false);
        }
        report(name, A, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), towers, nsd, true);
    }
    return 0;
}
