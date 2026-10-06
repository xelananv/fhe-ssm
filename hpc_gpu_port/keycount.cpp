// keycount.cpp -- how many automorphism keys, and at which indices, are in an
// evalrot.bin? The Mac-generated set is 12.43 GB vs the pod's 8.34 GB for the
// SAME ring, SAME 73 rotation indices in the .dev sidecar, and byte-identical
// evalmult/public/secret/context. So per-key size matches and the COUNT
// differs. This prints the actual index set so the two can be diffed instead
// of theorised about (the "patched OpenFHE" theory was already falsified: the
// FIDESlib patch is visibility-only plus a defaulted parameter).
#include <fstream>
#include <iostream>
#include <set>
#include "openfhe.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
using namespace lbcrypto;
int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: keycount CTX EVALROT\n"; return 2; }
    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
        std::cerr << "ctx load failed\n"; return 2; }
    std::ifstream f(argv[2], std::ios::binary);
    if (!f.good()) { std::cerr << "evalrot open failed\n"; return 2; }
    if (!cc->DeserializeEvalAutomorphismKey(f, SerType::BINARY)) {
        std::cerr << "evalrot deserialize failed\n"; return 2; }
    const uint32_t M = cc->GetCyclotomicOrder();
    std::cout << "{\"ring\":" << cc->GetRingDimension() << ",\"M\":" << M << "}\n";
    // enumerate every tag's map
    for (const auto& tag : {std::string("")}) { (void)tag; }
    auto& ids = CryptoContextImpl<DCRTPoly>::GetAllEvalAutomorphismKeys();
    for (auto& kv : ids) {
        std::cout << "{\"tag\":\"" << kv.first << "\",\"count\":" << kv.second->size() << "}\n";
        std::set<uint32_t> idx;
        for (auto& e : *kv.second) idx.insert(e.first);
        std::cout << "indices:";
        for (auto i : idx) std::cout << " " << i;
        std::cout << "\n";
    }
    return 0;
}
