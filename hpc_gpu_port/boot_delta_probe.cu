// boot_delta_probe.cu — ONE GPU bootstrap at a reduced scaling factor, next to
// the CPU control under the SAME keys, with the returned metadata printed and
// re-stamped (2026-09-03, demo-pod prep; MGPU_FIT_MODEL_20260903.md §4).
//
// Why. Every FIDESlib GPU bootstrap below Δ = 2^59 has failed decode in this
// project (A100 campaign Δ35/40/45/50; the 2026-08-23 C1 ladder at Δ40, ring
// 2^16: DECODE_FAIL already at ONE bootstrap on both v2.1.2 and main), while
// CPU OpenFHE at Δ40 is clean (2.07e-12 after one bootstrap). FIDESlib's own
// test table says `scaleModSize = 59 /*35 fails*/` for exactly the ring-2^16
// depth-29 case (test/ParametrizedTest.cuh:280). That failure is what forbids
// a 128-bit ring-2^16 chain (Δ=59 needs 1771 > 1747 bits).
//
// The one-line hypothesis this probe tests: the bug is in the METADATA the
// GPU result carries back to OpenFHE, not in the values. GetOpenFHECipherText
// stamps SetScalingFactor(raw.Noise) / SetNoiseScaleDeg(raw.NoiseLevel)
// (RawCiphertext.cu:125-126) from the bootstrap's NoiseFactor bookkeeping
// (Bootstrap.cu:440-448, 526, 601). At Δ=59 with a 60-bit first modulus every
// candidate value is within 2x of the truth, so a wrong stamp is invisible;
// at Δ=40 with a 45-bit first modulus it is 2^5 off -- a "5-bit decode guard"
// failure at the first bootstrap, which is what the ladders show.
//
// What it does (all API calls are ones gpu_real_model.cu already makes):
//   1. context per the harness policy (UNIFORM_TERNARY, depth = 10+19+xd,
//      scale S, first min(60,S+5), FLEXIBLEAUTO), keys, bootstrap keys,
//      LoadContext; the secret key is serialized to --out so plain OpenFHE
//      can decrypt GPU results under the same key.
//   2. encrypt a Dpad-periodic vector, burn 3 levels, GPU bootstrap, decrypt
//      through the wrapper -> gpuErr (the number the ladders reported).
//   3. sync the GPU result to its OpenFHE form; print scalingFactor, level,
//      noiseScaleDeg, limb count.
//   4. CPU EvalBootstrap of the SAME input (its OpenFHE form) with the same
//      keys -> cpuErr and its metadata.
//   5. re-stamp the GPU result with the CPU result's metadata and decrypt with
//      plain OpenFHE -> gpuErrRestamped. If it collapses to the cpuErr class,
//      the defect is the stamp (and its fix is one assignment); if not, the
//      values are wrong and the next step is a stage bisect (CtS / EvalMod /
//      StC), which is not a one-liner.
//
// Build: the harness CMake (target boot_delta_probe). Run on the box:
//   boot_delta_probe --log-ring 16 --scale-bits 40 --extra-depth 0 --device 0 --out /root/demo/bdp40
//   boot_delta_probe --log-ring 16 --scale-bits 59 --extra-depth 0 --device 0 --out /root/demo/bdp59   (control)
// Verdict line: {"bootDeltaProbe":true,...,"gpuErr":..,"cpuErr":..,"gpuErrRestamped":..,"reading":"..."}
// Exit 0 = GPU bootstrap fine at this Δ; 3 = metadata defect confirmed
// (restamped decrypt clean); 4 = numeric defect (restamped still bad).
#include <fideslib.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <any>
#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

using namespace fideslib;

static std::string argS(int argc, char** argv, const std::string& k, const std::string& d) {
    for (int i = 1; i + 1 < argc; i++) if (k == argv[i]) return argv[i + 1];
    if (argc > 1 && k == argv[argc - 1]) { fprintf(stderr, "{\"fatal\":\"%s without a value\"}\n", k.c_str()); exit(2); }
    return d;
}

int main(int argc, char** argv) {
    const int logRing = std::stoi(argS(argc, argv, "--log-ring", "16"));
    const int scaleBits = std::stoi(argS(argc, argv, "--scale-bits", "40"));
    const int extraDepth = std::stoi(argS(argc, argv, "--extra-depth", "0"));
    const int device = std::stoi(argS(argc, argv, "--device", "0"));
    const int Dpad = std::stoi(argS(argc, argv, "--dpad", "1024"));
    const std::string out = argS(argc, argv, "--out", "");
    if (out.empty()) { fprintf(stderr, "{\"fatal\":\"--out DIR required\"}\n"); return 2; }
    cudaSetDevice(device);

    // ---- 1. context, exactly the harness policy (gpu_real_model.cu ~:950-975)
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    std::vector<uint32_t> levelBudget = {3, 3};
    const uint32_t depth = 10 + 19 + (uint32_t)extraDepth;
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1u << logRing);
    parameters.SetScalingModSize((uint32_t)scaleBits);
    parameters.SetFirstModSize((uint32_t)std::min(60, scaleBits + 5));
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(depth);
    parameters.SetDevices(std::vector<int>{device});
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE); cc->Enable(FHE);
    const uint32_t SLOTS = cc->GetRingDimension() / 2;
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, std::vector<int32_t>{1});
    cc->EvalBootstrapSetup(levelBudget, {0, 0}, SLOTS);
    cc->EvalBootstrapKeyGen(keys.secretKey, SLOTS);
    if (!fideslib::Serial::SerializeToFile(out + "/secret.key", keys.secretKey, fideslib::SerType::BINARY)) {
        fprintf(stderr, "{\"fatal\":\"cannot write %s/secret.key (does --out exist?)\"}\n", out.c_str()); return 2;
    }
    cc->LoadContext(keys.publicKey);
    std::cout << "{\"bootDeltaProbe\":\"context\",\"ringDim\":" << cc->GetRingDimension() << ",\"slots\":" << SLOTS
              << ",\"depth\":" << depth << ",\"scaleBits\":" << scaleBits
              << ",\"firstMod\":" << std::min(60, scaleBits + 5) << "}" << std::endl;

    // plain-OpenFHE views of the same context and key (the harness's own any_casts)
    auto& lbCc = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
    lbcrypto::PrivateKey<lbcrypto::DCRTPoly> lbSk;
    if (!lbcrypto::Serial::DeserializeFromFile(out + "/secret.key", lbSk, lbcrypto::SerType::BINARY)) {
        fprintf(stderr, "{\"fatal\":\"secret.key re-read failed\"}\n"); return 2;
    }

    // ---- 2. input, 3 burnt levels, GPU bootstrap
    std::vector<double> v(SLOTS, 0.0);
    for (uint32_t r = 0; r * Dpad < SLOTS; r++)
        for (int i = 0; i < Dpad && r * Dpad + i < SLOTS; i++) v[(size_t)r * Dpad + i] = 0.01 * (double)((i * 37) % 97);
    auto pt = cc->MakeCKKSPackedPlaintext(v, 1, 0);
    auto ct = cc->Encrypt(keys.publicKey, pt);
    std::vector<double> ones(SLOTS, 1.0);
    // 2026-09-03 (pod run bdp59/40/45): with only 3 burnt levels OpenFHE's CPU
    // EvalBootstrap is the silent no-op of MEASUREMENTS.md 1d (cpuLevel stayed 3,
    // cpuErr 2e-12 = roundoff), so the restamp control was vacuous. Burn to
    // 4 remaining levels so BOTH sides genuinely bootstrap (--burn N overrides).
    int burn = (int)depth - 4;
    for (int i = 1; i + 1 < argc; i++) if (std::string(argv[i]) == "--burn") burn = std::atoi(argv[i + 1]);
    for (int i = 0; i < burn; i++) {
        auto op = cc->MakeCKKSPackedPlaintext(ones, 1, ct->GetLevel());
        ct = cc->EvalMult(ct, op); cc->RescaleInPlace(ct);
    }
    // the INPUT's OpenFHE form, for the CPU control (sync GPU -> host first)
    auto hostOf = [&](Ciphertext<DCRTPoly>& c) -> lbcrypto::Ciphertext<lbcrypto::DCRTPoly> {
        auto& host = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(c->cpu);
        if (c->loaded) {
            auto gpuCt = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(cc->GetDeviceCiphertext(c->gpu));
            FIDESlib::CKKS::RawCipherText raw;
            gpuCt->store(raw);
            FIDESlib::CKKS::GetOpenFHECipherText(host, raw);
        }
        return host->Clone();
    };
    auto inHost = hostOf(ct);
    const uint32_t lvlPre = ct->GetLevel();
    auto maxAbs = [&](const std::vector<double>& o) { double e = 0; for (uint32_t i = 0; i < SLOTS; i++) e = std::max(e, std::abs(o[i] - v[i])); return e; };

    cudaDeviceSynchronize();
    ct = cc->EvalBootstrap(ct);
    cudaDeviceSynchronize();
    const uint32_t lvlPost = ct->GetLevel();
    double gpuErr = -1; std::string gpuThrow;
    try { Plaintext p; cc->Decrypt(keys.secretKey, ct, &p); p->SetLength(SLOTS); gpuErr = maxAbs(p->GetRealPackedValue()); }
    catch (const std::exception& e) { gpuThrow = e.what(); }

    // ---- 3. the GPU result's OpenFHE form and its metadata
    auto gpuHost = hostOf(ct);
    const double gpuSF = gpuHost->GetScalingFactor();
    const uint32_t gpuLvl = gpuHost->GetLevel();
    const uint32_t gpuNsd = gpuHost->GetNoiseScaleDeg();
    const size_t gpuLimbs = gpuHost->GetElements()[0].GetNumOfElements();

    // ---- 4. CPU control on the same input under the same keys
    double cpuErr = -1; std::string cpuThrow; double cpuSF = 0; uint32_t cpuLvl = 0, cpuNsd = 0; size_t cpuLimbs = 0;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cpuOut;
    try {
        cpuOut = lbCc->EvalBootstrap(inHost);
        lbcrypto::Plaintext p; lbCc->Decrypt(lbSk, cpuOut, &p); p->SetLength(SLOTS); cpuErr = maxAbs(p->GetRealPackedValue());
        cpuSF = cpuOut->GetScalingFactor(); cpuLvl = cpuOut->GetLevel(); cpuNsd = cpuOut->GetNoiseScaleDeg();
        cpuLimbs = cpuOut->GetElements()[0].GetNumOfElements();
    } catch (const std::exception& e) { cpuThrow = e.what(); }

    // ---- 5. re-stamp the GPU values with the CPU metadata, decrypt with plain OpenFHE
    double gpuErrRestamped = -1, gpuErrPlain = -1; std::string reThrow;
    try { lbcrypto::Plaintext p; lbCc->Decrypt(lbSk, gpuHost, &p); p->SetLength(SLOTS); gpuErrPlain = maxAbs(p->GetRealPackedValue()); }
    catch (const std::exception& e) { reThrow = "plain:" + std::string(e.what()); }
    if (cpuOut && cpuLimbs == gpuLimbs) {
        try {
            auto re = gpuHost->Clone();
            re->SetScalingFactor(cpuSF); re->SetNoiseScaleDeg(cpuNsd); re->SetLevel(cpuLvl);
            lbcrypto::Plaintext p; lbCc->Decrypt(lbSk, re, &p); p->SetLength(SLOTS); gpuErrRestamped = maxAbs(p->GetRealPackedValue());
        } catch (const std::exception& e) { reThrow += " restamped:" + std::string(e.what()); }
    }

    const bool gpuOk = gpuErr >= 0 && gpuErr < 5e-3;
    const bool reOk = gpuErrRestamped >= 0 && gpuErrRestamped < 5e-3;
    std::string reading = gpuOk ? "GPU bootstrap fine at this scale"
                        : reOk ? "METADATA defect: GPU values decrypt cleanly under the CPU result's scalingFactor/noiseScaleDeg/level -- fix the stamp in GetOpenFHECipherText / Bootstrap NoiseFactor"
                        : "NUMERIC defect: values wrong regardless of metadata -- bisect CtS / EvalMod / StC";
    std::cout << "{\"bootDeltaProbe\":true,\"scaleBits\":" << scaleBits << ",\"ringDim\":" << cc->GetRingDimension()
              << ",\"lvlPre\":" << lvlPre << ",\"lvlPost\":" << lvlPost
              << ",\"gpuErr\":" << gpuErr << ",\"gpuThrow\":\"" << gpuThrow << "\""
              << ",\"gpuSF\":" << gpuSF << ",\"gpuLevel\":" << gpuLvl << ",\"gpuNoiseScaleDeg\":" << gpuNsd << ",\"gpuLimbs\":" << gpuLimbs
              << ",\"cpuErr\":" << cpuErr << ",\"cpuThrow\":\"" << cpuThrow << "\""
              << ",\"cpuSF\":" << cpuSF << ",\"cpuLevel\":" << cpuLvl << ",\"cpuNoiseScaleDeg\":" << cpuNsd << ",\"cpuLimbs\":" << cpuLimbs
              << ",\"gpuErrPlainDecrypt\":" << gpuErrPlain << ",\"gpuErrRestamped\":" << gpuErrRestamped
              << ",\"reThrow\":\"" << reThrow << "\",\"reading\":\"" << reading << "\"}" << std::endl;
    return gpuOk ? 0 : (reOk ? 3 : 4);
}
