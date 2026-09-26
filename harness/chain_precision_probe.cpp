// chain_precision_probe.cpp — how many bits survive a BOOTSTRAP-FREE chain?
//
// The shallow-pass ("fast") designs all rest on running a long multiplicative
// chain at a SMALL ring and a LOW scaling factor, with no bootstrap anywhere.
// Nobody has measured whether such a chain delivers usable precision. This probe
// does exactly that and nothing else:
//
//   encrypt a known random vector at full slots
//   -> consume `--levels` levels, alternating ct x pt and ct x ct (+relin)
//   -> rescale normally, NO bootstrap
//   -> decrypt and compare against a plaintext reference carried in double
//   -> report delivered bits = -log2(relative error)
//
// It also prints the 128-bit legality arithmetic rather than asserting "it fits":
//   logQP ~= (q0 + depth*scaleBits) * (1 + 1/dnum)
// against the standard ternary HEStd_128_classic ceilings
//   881 (2^15) / 1747 (2^16) / 3523 (2^17).
// If OpenFHE REJECTS a parameter set at context creation, that rejection IS the
// result — it is caught and reported as JSON, not allowed to abort the process.
//
// Build: cmake --build harness/build-linux --target chain_precision_probe
// Run:   ./chain_precision_probe --log-ring 17 --scale-bits 59 --first-mod 60
//                                --depth 29 --levels 20
//        --dnum 0 means "leave OpenFHE's automatic choice alone" (production).

#include "openfhe.h"

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace lbcrypto;

int main(int argc, char** argv) {
    uint32_t logRing = 17, scaleBits = 59, firstMod = 60, depth = 29, levels = 20;
    uint32_t dnum = 0;                 // 0 = auto (production default)
    std::string tag = "P?";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--log-ring")   logRing   = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--scale-bits") scaleBits = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--first-mod")  firstMod  = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--depth")      depth     = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--levels")     levels    = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--dnum")       dnum      = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--tag")        tag       = argv[++i];
    }

    // ---- legality arithmetic, printed as a NUMBER (reported either way) ------
    const double dn      = (dnum == 0 ? 3.0 : (double)dnum);   // production auto == 3
    const double logQP   = (firstMod + (double)depth * scaleBits) * (1.0 + 1.0 / dn);
    const double ceiling = (logRing == 15 ? 881.0 : logRing == 16 ? 1747.0 :
                            logRing == 17 ? 3523.0 : -1.0);
    const bool   legal   = (ceiling > 0 && logQP <= ceiling);

    auto head = [&](const char* status, const std::string& extra) {
        std::cout.setf(std::ios::fixed); std::cout.precision(1);
        std::cout << "{\"harness\":\"chain_precision_probe\",\"tag\":\"" << tag
                  << "\",\"status\":\"" << status
                  << "\",\"logRing\":" << logRing
                  << ",\"scaleBits\":" << scaleBits
                  << ",\"firstMod\":" << firstMod
                  << ",\"depth\":" << depth
                  << ",\"levels\":" << levels
                  << ",\"dnum\":" << (dnum == 0 ? 3 : (int)dnum)
                  << ",\"dnumAuto\":" << (dnum == 0 ? "true" : "false")
                  << ",\"logQP\":" << logQP
                  << ",\"ceiling128\":" << ceiling
                  << ",\"legal128\":" << (legal ? "true" : "false")
                  << extra << "}" << std::endl;
    };

    CryptoContext<DCRTPoly> cc;
    uint32_t SLOTS = 0;
    KeyPair<DCRTPoly> keys;
    try {
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetSecretKeyDist(UNIFORM_TERNARY);
        parameters.SetSecurityLevel(HEStd_NotSet);      // ring forced; legality above
        parameters.SetRingDim(1u << logRing);
        parameters.SetScalingModSize(scaleBits);
        parameters.SetFirstModSize(firstMod);
        parameters.SetScalingTechnique(FLEXIBLEAUTO);
        parameters.SetMultiplicativeDepth(depth);
        if (dnum > 0) parameters.SetNumLargeDigits(dnum);
        cc = GenCryptoContext(parameters);
        cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);                        // no FHE: bootstrap-free by design
        SLOTS = cc->GetRingDimension() / 2;
        keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);
    } catch (const std::exception& e) {
        // A rejection here IS the result.
        head("context_rejected", std::string(",\"error\":\"") + e.what() + "\"");
        return 0;
    }

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> ud(-1.0, 1.0), uw(0.90, 1.10);

    std::vector<double> x(SLOTS), ref(SLOTS);
    for (uint32_t i = 0; i < SLOTS; i++) { x[i] = ud(rng); ref[i] = x[i]; }

    try {
        auto pt  = cc->MakeCKKSPackedPlaintext(x);
        auto ct  = cc->Encrypt(keys.publicKey, pt);

        // Alternate ct x pt and ct x ct(+relin). Multipliers sit near 1.0 so the
        // message magnitude neither overflows nor decays into the noise floor --
        // this isolates SCALE-CHAIN precision, not dynamic range.
        uint32_t ctct = 0;
        for (uint32_t l = 0; l < levels; l++) {
            std::vector<double> w(SLOTS);
            for (uint32_t i = 0; i < SLOTS; i++) w[i] = uw(rng);
            for (uint32_t i = 0; i < SLOTS; i++) ref[i] *= w[i];
            if (l % 2 == 0) {                       // ct x pt
                auto wpt = cc->MakeCKKSPackedPlaintext(w, 1, ct->GetLevel());
                ct = cc->EvalMult(ct, wpt);
            } else {                                // ct x ct (relin inside EvalMult)
                auto wct = cc->Encrypt(keys.publicKey,
                                       cc->MakeCKKSPackedPlaintext(w, 1, ct->GetLevel()));
                ct = cc->EvalMult(ct, wct);
                ctct++;
            }
            cc->RescaleInPlace(ct);
        }

        Plaintext out;
        cc->Decrypt(keys.secretKey, ct, &out);
        out->SetLength(SLOTS);
        auto v = out->GetRealPackedValue();

        double maxAbs = 0, maxRef = 0;
        for (uint32_t i = 0; i < SLOTS; i++) {
            maxAbs = std::max(maxAbs, std::fabs(v[i] - ref[i]));
            maxRef = std::max(maxRef, std::fabs(ref[i]));
        }
        const double rel  = maxAbs / std::max(maxRef, 1e-300);
        const double bits = -std::log2(std::max(rel, 1e-300));

        std::cout.setf(std::ios::fixed); std::cout.precision(1);
        std::cout << "{\"harness\":\"chain_precision_probe\",\"tag\":\"" << tag
                  << "\",\"status\":\"ok\""
                  << ",\"logRing\":" << logRing << ",\"ringDim\":" << cc->GetRingDimension()
                  << ",\"slots\":" << SLOTS
                  << ",\"scaleBits\":" << scaleBits << ",\"firstMod\":" << firstMod
                  << ",\"depth\":" << depth << ",\"levels\":" << levels
                  << ",\"ctxctMults\":" << ctct
                  << ",\"finalLevel\":" << ct->GetLevel()
                  << ",\"dnum\":" << (dnum == 0 ? 3 : (int)dnum)
                  << ",\"dnumAuto\":" << (dnum == 0 ? "true" : "false")
                  << ",\"logQP\":" << logQP << ",\"ceiling128\":" << ceiling
                  << ",\"legal128\":" << (legal ? "true" : "false");
        std::cout.precision(6);
        std::cout << ",\"maxAbsErr\":" << maxAbs
                  << ",\"maxAbsRef\":" << maxRef;
        std::cout.precision(3);
        std::cout << ",\"relErr\":" << rel
                  << ",\"deliveredBits\":" << bits << "}" << std::endl;
    } catch (const std::exception& e) {
        head("eval_failed", std::string(",\"error\":\"") + e.what() + "\"");
        return 0;
    }
    return 0;
}
