// boot_floor_bench.cpp — measures the FIXED per-layer FHE cost floor at real
// CKKS (ring 2^16, depth 29, {4,4} bootstrap budget): one bootstrap, one ct*ct
// gate (relin), one Newton-rsqrt norm. Composed with ring_matmul_bench's per-op
// matmul numbers (dense BSGS vs circulant ring-mult), this gives the LAYER-level
// speedup of ring-native linear layers -- i.e. how much the ~4000x per-op matmul
// win actually buys once the bootstrap floor and the nonlinear ops are included.
//
// Build: cmake --build harness/build --target boot_floor_bench
// Run:   ./harness/build/boot_floor_bench --reps 3

#include "openfhe.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;
static double msOf(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
    uint32_t logRing = 16, reps = 3, D = 1024;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--log-ring") logRing = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--reps") reps   = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--d")    D      = (uint32_t)std::stoi(argv[++i]);
    }

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1u << logRing);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    std::vector<uint32_t> levelBudget = {4, 4};
    const uint32_t levelsAfter = 10, depth = levelsAfter + 19;   // 29
    parameters.SetMultiplicativeDepth(depth);
    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE); cc->Enable(FHE);

    const uint32_t SLOTS = cc->GetRingDimension() / 2;
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    std::vector<int32_t> rots;                                   // for a log2(D) reduction
    for (uint32_t p = 1; p < D; p <<= 1) rots.push_back((int32_t)p);
    cc->EvalRotateKeyGen(keys.secretKey, rots);
    cc->EvalBootstrapSetup(levelBudget, {0, 0}, SLOTS);
    cc->EvalBootstrapKeyGen(keys.secretKey, SLOTS);

    std::vector<double> xv(SLOTS); for (auto& v : xv) v = (double)rand() / RAND_MAX * 0.5;
    auto onePt = cc->MakeCKKSPackedPlaintext(std::vector<double>(SLOTS, 1.0));
    auto fresh = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(xv));

    // deplete a ciphertext to near the bottom of the chain, as a real layer does
    // before refreshing (bootstrapping a depleted ct, not a fresh one).
    auto deplete = [&](int nlev) {
        auto c = fresh->Clone();
        for (int i = 0; i < nlev; i++) { c = cc->EvalMult(c, onePt); cc->RescaleInPlace(c); }
        return c;
    };

    // ---- bootstrap ----
    { auto c = deplete((int)(depth - levelsAfter - 1)); auto b = cc->EvalBootstrap(c); (void)b; }  // warmup
    double boot_ms = 0;
    for (uint32_t r = 0; r < reps; r++) {
        auto c = deplete((int)(depth - levelsAfter - 1));
        auto t = Clock::now(); auto b = cc->EvalBootstrap(c); boot_ms += msOf(t, Clock::now()); (void)b;
    }
    boot_ms /= reps;

    // ---- ct*ct gate (bilinear): multiply (with relin) + rescale ----
    { auto g = cc->EvalMult(fresh, fresh); cc->RescaleInPlace(g); }                 // warmup
    double gate_ms = 0;
    for (uint32_t r = 0; r < reps; r++) {
        auto t = Clock::now(); auto g = cc->EvalMult(fresh, fresh); cc->RescaleInPlace(g);
        gate_ms += msOf(t, Clock::now());
    }
    gate_ms /= reps;

    // ---- Newton-rsqrt norm: a log2(D) rotate-sum reduction + 4 Newton iters ----
    auto rsqrtNorm = [&]() {
        auto sq = cc->EvalMult(fresh, fresh); cc->RescaleInPlace(sq);               // x^2
        auto ms = sq;                                                               // reduce over channels
        for (uint32_t p = 1; p < D; p <<= 1) ms = cc->EvalAdd(ms, cc->EvalRotate(ms, (int32_t)p));
        auto y = cc->EvalMult(ms, 0.3); cc->RescaleInPlace(y);                      // crude seed
        for (int it = 0; it < 4; it++) {                                            // y = y*(1.5 - 0.5*ms*y*y)
            auto y2 = cc->EvalMult(y, y); cc->RescaleInPlace(y2);
            auto t  = cc->EvalMult(ms, y2); cc->RescaleInPlace(t);
            t = cc->EvalMult(t, -0.5); cc->RescaleInPlace(t);
            t = cc->EvalAdd(t, 1.5);
            y = cc->EvalMult(y, t); cc->RescaleInPlace(y);
        }
        return cc->EvalMult(fresh, y);
    };
    rsqrtNorm();                                                                    // warmup
    double norm_ms = 0;
    for (uint32_t r = 0; r < reps; r++) { auto t = Clock::now(); auto o = rsqrtNorm();
        norm_ms += msOf(t, Clock::now()); (void)o; }
    norm_ms /= reps;

    std::cout.setf(std::ios::fixed); std::cout.precision(3);
    std::cout << "{\"harness\":\"boot_floor_bench\",\"ringDim\":" << cc->GetRingDimension()
              << ",\"depth\":" << depth << ",\"bootstrap_ms\":" << boot_ms
              << ",\"ctxct_gate_ms\":" << gate_ms << ",\"newton_rsqrt_norm_ms\":" << norm_ms
              << ",\"reps\":" << reps << "}" << std::endl;
    return 0;
}
