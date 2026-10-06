// patchkeys.cpp -- close the six-index gap that makes Mac-generated keys abort
// the harness with std::out_of_range: map::at.
//
// Diagnosis (2026-09-01): all 73 ROTATION keys match between the Mac set and the
// pod's; the BOOTSTRAP sets differ (pod 29, Mac 79, 23 shared) because OpenFHE's
// automatic giant-step selection differs between builds. The Mac set is not a
// superset -- it lacks exactly 7233, 11713, 15681, 24129, 36033, 52929, which is
// what LoadContext throws on.
//
// Fix: mint keys for precisely those automorphism indices and merge them into
// the Mac's automorphism map. The 56 surplus Mac keys are harmless.
//
// NOTE ON SOVEREIGNTY. This test mints them on the pod from the secret key that
// was uploaded for the earlier selftest, so the merge can be validated with ZERO
// transfer. That is a TEST shortcut, not the production path: for the real run
// the Mac mints its own set plus these six and uploads once, and the pod never
// sees the secret key. The index list is public information about the circuit.
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include "openfhe.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
using namespace lbcrypto;

int main(int argc, char** argv) {
    if (argc < 5) { std::cerr << "usage: patchkeys CTX SECKEY EVALROT_IN EVALROT_OUT idx...\n"; return 2; }
    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) { std::cerr << "ctx fail\n"; return 2; }
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE); cc->Enable(FHE);
    PrivateKey<DCRTPoly> sk;
    if (!Serial::DeserializeFromFile(argv[2], sk, SerType::BINARY)) { std::cerr << "sk fail\n"; return 2; }
    { std::ifstream in(argv[3], std::ios::binary);
      if (!in.good() || !cc->DeserializeEvalAutomorphismKey(in, SerType::BINARY)) {
          std::cerr << "evalrot in fail\n"; return 2; } }
    auto& all = CryptoContextImpl<DCRTPoly>::GetAllEvalAutomorphismKeys();
    std::string tag; size_t before = 0;
    for (auto& kv : all) { tag = kv.first; before = kv.second->size(); }
    // AUDIT FIX 2026-09-02: a FOREIGN secret key would mint keys under its
    // own tag, leaving the loaded set untouched, and the naive counters
    // would still print the expected numbers. Refuse unless the tags match.
    if (sk->GetKeyTag() != tag) {
        std::cerr << "{\"fatal\":\"secret key tag " << sk->GetKeyTag()
                  << " != loaded evalrot tag " << tag << " -- wrong key\"}\n";
        return 2;
    }
    std::vector<uint32_t> want;
    for (int i = 5; i < argc; i++) want.push_back((uint32_t)std::stoul(argv[i]));
    std::cout << "{\"tag\":\"" << tag << "\",\"before\":" << before
              << ",\"minting\":" << want.size() << "}\n";
    auto minted = cc->EvalAutomorphismKeyGen(sk, want);
    size_t added = 0;
    for (auto& kv : *minted) {
        if (all[tag]->find(kv.first) == all[tag]->end()) { (*all[tag])[kv.first] = kv.second; added++; }
    }
    std::cout << "{\"added\":" << added << ",\"after\":" << all[tag]->size() << "}\n";
    // KeyGen inserts into the static map itself (that is why the real run
    // printed added:0). A nonzero manual add means the minted keys landed
    // under a DIFFERENT tag than the one we just verified -- treat as error.
    if (added != 0) { std::cerr << "{\"fatal\":\"manual insert added " << added
                                << " keys: tag mismatch between KeyGen and loaded set\"}\n"; return 2; }
    for (auto i : want)
        if (all[tag]->find(i) == all[tag]->end())
            std::cout << "{\"STILL_MISSING\":" << i << "}\n";
    { std::ofstream out(argv[4], std::ios::binary);
      if (!out.good() || !cc->SerializeEvalAutomorphismKey(out, SerType::BINARY)) {
          std::cerr << "evalrot out fail\n"; return 2; } }
    std::cout << "{\"wrote\":\"" << argv[4] << "\"}\n";
    return 0;
}
