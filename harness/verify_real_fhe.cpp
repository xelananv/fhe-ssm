// verify_real_fhe.cpp — decisive test that this pipeline is REAL FHE, not
// simulated/fake math. Same method as the SEAL version: encrypt a known
// vector under keypair A; decrypt with the correct secret key (must match);
// decrypt the SAME ciphertext with an independently-generated wrong secret
// key B (must fail catastrophically). This property cannot be faked by any
// non-cryptographic stand-in. This is the SAME OpenFHE CryptoContext API
// (just CPU, not FIDESlib-GPU-accelerated) that the paper's headline GPU
// results are built on — a fake pipeline would not reproduce this.
#include "openfhe.h"
#include <iostream>
#include <cmath>
using namespace lbcrypto;

int main() {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetRingDim(1 << 14);
    parameters.SetScalingModSize(50);
    parameters.SetFirstModSize(60);
    parameters.SetMultiplicativeDepth(3);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);

    auto keysA = cc->KeyGen();          // the key holder
    auto keysB = cc->KeyGen();          // independently-generated, WRONG key

    uint32_t slots = cc->GetRingDimension() / 2;
    std::vector<double> original(slots);
    for (uint32_t i = 0; i < slots; i++) original[i] = std::sin(i * 0.017) * 0.5;

    Plaintext pt = cc->MakeCKKSPackedPlaintext(original);
    auto ct = cc->Encrypt(keysA.publicKey, pt);

    Plaintext ptCorrect;
    cc->Decrypt(keysA.secretKey, ct, &ptCorrect);
    ptCorrect->SetLength(slots);
    auto vCorrect = ptCorrect->GetRealPackedValue();

    double errCorrect = 0;
    for (uint32_t i = 0; i < slots; i++)
        errCorrect = std::max(errCorrect, std::abs(vCorrect[i] - original[i]));

    // WRONG key on purpose: real CKKS either (a) throws — OpenFHE's own
    // internal approximation-error sanity check rejects the decode — or
    // (b) returns astronomically-large garbage. Either is a genuine crypto
    // failure; a non-cryptographic stand-in could not reproduce either.
    bool wrongKeyThrew = false;
    std::string wrongKeyException;
    double errWrong = 0, meanAbsWrong = 0;
    try {
        Plaintext ptWrong;
        cc->Decrypt(keysB.secretKey, ct, &ptWrong);
        ptWrong->SetLength(slots);
        auto vWrong = ptWrong->GetRealPackedValue();
        double sumAbsWrong = 0;
        for (uint32_t i = 0; i < slots; i++) {
            errWrong = std::max(errWrong, std::abs(vWrong[i] - original[i]));
            sumAbsWrong += std::abs(vWrong[i]);
        }
        meanAbsWrong = sumAbsWrong / slots;
    } catch (const std::exception& e) {
        wrongKeyThrew = true;
        wrongKeyException = e.what();
    }

    std::cout << "{\"test\":\"wrong-key-decrypt (OpenFHE, same lib family as the GPU harness)\""
              << ",\"slots\":" << slots
              << ",\"correctKey_maxAbsErr\":" << errCorrect
              << ",\"verdict_correctKeyWorks\":" << (errCorrect < 1e-4 ? "true" : "false")
              << ",\"wrongKey_threwException\":" << (wrongKeyThrew ? "true" : "false");
    if (wrongKeyThrew) {
        std::cout << ",\"wrongKey_exceptionMsg\":\"" << wrongKeyException << "\""
                   << ",\"verdict_wrongKeyFails\":true";
    } else {
        std::cout << ",\"wrongKey_maxAbsErr\":" << errWrong
                   << ",\"wrongKey_meanAbsDecoded\":" << meanAbsWrong
                   << ",\"verdict_wrongKeyFails\":" << (errWrong > 0.1 ? "true" : "false");
    }
    std::cout << "}" << std::endl;
    return 0;
}
