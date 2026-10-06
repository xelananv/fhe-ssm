// logqp_probe.cpp — measure, on the library, the ACTUAL logQP that
// gpu_real_model.cu's context policy produces, instead of deriving it.
//
// Mirrors hpc_gpu_port/gpu_real_model.cu:907-923 line for line:
//   SetSecretKeyDist(UNIFORM_TERNARY)                         (:907)
//   depth = levelsAfter(10) + 19 + extraDepth                  (:909-910)
//   --secure -> SetSecurityLevel(HEStd_128_classic), ring free (:911)
//   else      -> HEStd_NotSet + SetRingDim(1<<logRing)         (:912)
//   SetScalingModSize(scaleBits=59)                            (:919)
//   SetFirstModSize(min(60, scaleBits+5))                      (:920)
//   SetNumLargeDigits only if --num-digits > 0                 (:921)
//   SetScalingTechnique(FLEXIBLEAUTO)                          (:922)
//   SetMultiplicativeDepth(depth)                              (:923)
// Key switching is left at the CKKSRNS default (HYBRID,
// gen-cryptocontext-params-defaults.h:53) exactly as the harness does.
//
// Prints one JSON line: the ring OpenFHE chose, sizeQ/sizeP/dnum, the MSB
// of Q, P and QP (the same GetParamsQP()->GetModulus().GetMSB() the library
// itself checks against the HEStd table at
// ckksrns-parametergeneration.cpp:196-200), and the exact log2 of QP.
// If GenCryptoContext throws (HE-standard non-compliance, or a digit split
// that cannot be made), the exception text is printed instead.
//
// Build (Mac, no cmake needed — recipe from the estimator-tooling note):
//   clang++ -std=c++17 -O2 -DOPENFHE_VERSION=1.5.1 -DMATHBACKEND=4 \
//     -Wno-unknown-pragmas \
//     -I vendor/install/include/openfhe{,/third-party/include,/core,/pke,/binfhe} \
//     harness/logqp_probe.cpp -L vendor/install/lib \
//     -lOPENFHEcore -lOPENFHEpke -lOPENFHEbinfhe \
//     -Wl,-rpath,$PWD/vendor/install/lib -o harness/build/logqp_probe
#include "openfhe.h"
#include <cmath>
#include <iostream>
#include <string>

using namespace lbcrypto;

int main(int argc, char** argv) {
    int extraDepth = 12, numDigits = 0, scaleBits = 59, logRing = 17;
    bool secure = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--extra-depth") extraDepth = std::stoi(argv[++i]);
        else if (a == "--num-digits")  numDigits  = std::stoi(argv[++i]);
        else if (a == "--scale-bits")  scaleBits  = std::stoi(argv[++i]);
        else if (a == "--log-ring")    logRing    = std::stoi(argv[++i]);
        else if (a == "--no-secure")   secure     = false;
    }
    const uint32_t levelsAfter = 10;
    const uint32_t depth = levelsAfter + 19 + extraDepth;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    if (secure) parameters.SetSecurityLevel(HEStd_128_classic);
    else { parameters.SetSecurityLevel(HEStd_NotSet); parameters.SetRingDim(1u << logRing); }
    parameters.SetScalingModSize((uint32_t)scaleBits);
    parameters.SetFirstModSize((uint32_t)std::min(60, scaleBits + 5));
    if (numDigits > 0) parameters.SetNumLargeDigits((uint32_t)numDigits);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(depth);

    std::cout << "{\"probe\":\"logqp_probe\",\"secure\":" << (secure ? "true" : "false")
              << ",\"extraDepth\":" << extraDepth << ",\"depth\":" << depth
              << ",\"scaleBits\":" << scaleBits << ",\"firstMod\":" << std::min(60, scaleBits + 5)
              << ",\"numDigitsArg\":" << numDigits;
    try {
        CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
        auto cp = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
        auto pQ  = cp->GetElementParams();
        auto pP  = cp->GetParamsP();
        auto pQP = cp->GetParamsQP();
        double log2Q = 0, log2P = 0;
        for (auto& q : pQ->GetParams()) log2Q += std::log2(q->GetModulus().ConvertToDouble());
        for (auto& p : pP->GetParams()) log2P += std::log2(p->GetModulus().ConvertToDouble());
        std::cout << ",\"threw\":false"
                  << ",\"ringDim\":" << cc->GetRingDimension()
                  << ",\"ksTech\":" << (cp->GetKeySwitchTechnique() == HYBRID ? "\"HYBRID\"" : "\"other\"")
                  << ",\"numPartQ\":" << cp->GetNumPartQ()
                  << ",\"numPerPartQ\":" << cp->GetNumPerPartQ()
                  << ",\"auxBits\":" << cp->GetAuxBits()
                  << ",\"sizeQ\":" << pQ->GetParams().size()
                  << ",\"sizeP\":" << pP->GetParams().size()
                  << ",\"msbQ\":" << pQ->GetModulus().GetMSB()
                  << ",\"msbP\":" << pP->GetModulus().GetMSB()
                  << ",\"msbQP\":" << pQP->GetModulus().GetMSB()
                  << ",\"log2Q\":" << log2Q << ",\"log2P\":" << log2P
                  << ",\"log2QP\":" << (log2Q + log2P)
                  << ",\"q0bits\":" << pQ->GetParams()[0]->GetModulus().GetMSB()
                  << ",\"qLastBits\":" << pQ->GetParams().back()->GetModulus().GetMSB()
                  << ",\"p0bits\":" << pP->GetParams()[0]->GetModulus().GetMSB()
                  << "}" << std::endl;
    } catch (const std::exception& e) {
        std::string m = e.what();
        for (auto& ch : m) if (ch == '"') ch = '\'';
        std::cout << ",\"threw\":true,\"what\":\"" << m << "\"}" << std::endl;
    }
    return 0;
}
