// overflow_dist_dump.cpp -- dump what is needed to measure the ModRaise overflow I = round((c0 + c1*s)/q0)
// on SAVED ciphertexts of the recorded 2^17 session (A11 tick-42 investigation, probe C, 2026-09-18).
//
// LOAD-ONLY: deserialises a crypto context, a secret key and ciphertexts; generates and loads NO evaluation
// key and performs NO homomorphic operation.  For every ciphertext it writes, for TOWER 0 ONLY, the
// COEFFICIENT-format centered residues of c0 and c1 (int64 little-endian, N each), with the library's own
// centering rule (v > q0>>1 is negative: core/lib/math/hal/intnat/mubintvecnat.cpp:109-:121, the rule ModRaise
// applies at ckksrns-fhe.cpp:592-:600), and, once, the secret's coefficient vector (int8, values in {-1,0,1}).
//
// SECRET HYGIENE.  The secret's coefficients are key material.  This program refuses any --out-dir that is
// not under /tmp/ (so it can never write them into the repository); the caller creates that
// directory with mode 700 and deletes it when the statistics are done.  The input kit is opened read-only.
//
// Build: PATH=$PWD/.venv/bin:$PATH ninja -C harness/build overflow_dist_dump
// Run:   tools/memguard.sh 8 harness/build/overflow_dist_dump --ctx <kit>/keys/cryptocontext.bin
//            --sec <kit>/keys/secret.key --out-dir /tmp/.../overflow_dist --cts <file> <file> ...

#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

#include <sys/stat.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "run_stamp.h"

using namespace lbcrypto;
using PolyT = DCRTPoly::PolyType;

static std::vector<int64_t> centered(PolyT p) {
    p.SetFormat(Format::COEFFICIENT);
    const uint64_t q = p.GetModulus().ConvertToInt<uint64_t>(), half = q >> 1;
    const uint32_t n = p.GetLength();
    std::vector<int64_t> out(n);
    for (uint32_t k = 0; k < n; k++) {
        const uint64_t v = p[k].ConvertToInt<uint64_t>();
        out[k] = (v > half) ? -(int64_t)(q - v) : (int64_t)v;
    }
    return out;
}

static bool writeRaw(const std::string& path, const void* data, size_t bytes) {
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os.good()) return false;
    os.write(reinterpret_cast<const char*>(data), (std::streamsize)bytes);
    os.close();
    chmod(path.c_str(), 0600);
    return os.good();
}

int main(int argc, char** argv) {
    std::string ctxPath, secPath, outDir; std::vector<std::string> cts;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--ctx") ctxPath = argv[++i];
        else if (a == "--sec") secPath = argv[++i];
        else if (a == "--out-dir") outDir = argv[++i];
        else if (a == "--cts") { while (i + 1 < argc) cts.push_back(argv[++i]); }
        else { std::cerr << "unknown flag " << a << std::endl; return 2; }
    }
    if (ctxPath.empty() || secPath.empty() || outDir.empty() || cts.empty()) {
        std::cerr << "usage: overflow_dist_dump --ctx F --sec F --out-dir /tmp/... --cts F [F ...]" << std::endl; return 2;
    }
    if (outDir.rfind("/tmp/", 0) != 0) {
        std::cerr << "refusing --out-dir outside /tmp/: secret-derived files must never land in the repository" << std::endl; return 2;
    }
    umask(077);

    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(ctxPath, cc, SerType::BINARY)) { std::cerr << "context load failed" << std::endl; return 3; }
    PrivateKey<DCRTPoly> sk;
    if (!Serial::DeserializeFromFile(secPath, sk, SerType::BINARY)) { std::cerr << "secret key load failed" << std::endl; return 3; }
    const auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    const uint32_t N = cc->GetRingDimension();
    const auto& qp = cp->GetElementParams()->GetParams();
    const uint64_t q0 = qp[0]->GetModulus().ConvertToInt<uint64_t>();

    // the secret, read from tower 0 and cross-checked against tower 1 (a small signed integer reads the same in both)
    const DCRTPoly& sEl = sk->GetPrivateElement();
    const std::vector<int64_t> s = centered(sEl.GetElementAtIndex(0));
    bool ternary = true, towersAgree = true; long long h = 0;
    for (int64_t x : s) { if (x < -1 || x > 1) ternary = false; h += (x != 0); }
    if (sEl.GetNumOfElements() > 1) { const auto s1 = centered(sEl.GetElementAtIndex(1)); for (uint32_t k = 0; k < N; k++) if (s1[k] != s[k]) towersAgree = false; }
    std::vector<int8_t> s8(N); for (uint32_t k = 0; k < N; k++) s8[k] = (int8_t)s[k];
    if (!writeRaw(outDir + "/secret.i8", s8.data(), s8.size())) { std::cerr << "cannot write into " << outDir << std::endl; return 4; }

    // the bootstrap correction factor the serialized scheme carries (FHECKKSRNS::save/load, ckksrns-fhe.h:266-:279); -1 if absent
    long corFactor = -1;
    try { corFactor = (long)cc->GetScheme()->GetCKKSBootCorrectionFactor(); } catch (const std::exception&) { corFactor = -1; }

    std::ofstream idx(outDir + "/index.jsonl", std::ios::trunc);
    idx << "{\"rec\":\"header\",\"harness\":\"overflow_dist_dump\",\"ringDim\":" << N << ",\"q0\":" << q0 << ",\"towersQ\":" << qp.size()
        << ",\"secretKeyDist\":" << (int)cp->GetSecretKeyDist() << ",\"scalingTechnique\":" << (int)cp->GetScalingTechnique()
        << ",\"secretIsTernary\":" << (ternary ? "true" : "false") << ",\"secretTowers01Agree\":" << (towersAgree ? "true" : "false")
        << ",\"hamming\":" << h << ",\"serializedCorrectionFactor\":" << corFactor << fhe_ssm::runStamp(argc, argv) << "}" << std::endl;
    std::cout << "{\"ringDim\":" << N << ",\"q0\":" << q0 << ",\"towersQ\":" << qp.size() << ",\"secretIsTernary\":" << (ternary ? "true" : "false")
              << ",\"secretTowers01Agree\":" << (towersAgree ? "true" : "false") << ",\"hamming\":" << h
              << ",\"serializedCorrectionFactor\":" << corFactor << "}" << std::endl;

    for (size_t k = 0; k < cts.size(); k++) {
        Ciphertext<DCRTPoly> ct;
        if (!Serial::DeserializeFromFile(cts[k], ct, SerType::BINARY)) { std::cerr << "ct load failed: " << cts[k] << std::endl; return 3; }
        const auto& cv = ct->GetElements();
        const uint64_t q0ct = cv[0].GetElementAtIndex(0).GetModulus().ConvertToInt<uint64_t>();
        const auto c0 = centered(cv[0].GetElementAtIndex(0));
        const auto c1 = centered(cv[1].GetElementAtIndex(0));
        char name[64];
        snprintf(name, sizeof name, "/ct_%03zu", k);
        const bool ok = writeRaw(outDir + name + ".c0.i64", c0.data(), c0.size() * 8) && writeRaw(outDir + name + ".c1.i64", c1.data(), c1.size() * 8);
        if (!ok) { std::cerr << "write failed for " << cts[k] << std::endl; return 4; }
        std::ostringstream line;
        line << "{\"rec\":\"ct\",\"k\":" << k << ",\"path\":\"" << fhe_ssm::jsonEscape(cts[k]) << "\",\"elements\":" << cv.size()
             << ",\"towers\":" << cv[0].GetNumOfElements() << ",\"q0\":" << q0ct << ",\"level\":" << ct->GetLevel()
             << ",\"noiseScaleDeg\":" << ct->GetNoiseScaleDeg() << ",\"log2ScalingFactor\":" << std::log2(ct->GetScalingFactor())
             << ",\"slots\":" << ct->GetSlots() << "}";
        idx << line.str() << std::endl;
        std::cout << line.str() << std::endl;
    }
    return 0;
}
