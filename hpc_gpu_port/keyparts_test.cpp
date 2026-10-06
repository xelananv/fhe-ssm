// keyparts_test.cpp — end-to-end proof of the part-file key pipeline
// (2026-09-02) at ring 2^12, HEStd_NotSet, depth 10+19+3, levelBudget {3,3},
// scale 59 / first 60, FLEXIBLEAUTO, slots = ring/2.
//
//   A. build the CONTRACT the way the pod harness will: block-layout matvec
//      rotation amounts (Dpad 1024) -> automorphism indices via
//      cc->FindAutomorphismIndex, plus the bootstrap index set read off the
//      static key map after EvalBootstrapSetup + EvalBootstrapKeyGen with a
//      THROWAWAY key pair (then cleared); write it in the --indices format;
//   B. run the REAL client path: do_keygen(... --indices F --batch 5), i.e.
//      the same code mac_fhe_client runs (this file #includes it);
//   C. in a FRESH context deserialized from cryptocontext.bin, load
//      public/secret/evalmult and every evalrot.part*.bin through
//      DeserializeEvalAutomorphismKey(istream), assert the loaded index set
//      equals the contract, check the .dev sidecar, EvalRotate every contract
//      amount against the expected cyclic shift, conjugate through the minted
//      M-1 key (EvalAutomorphism), and EvalBootstrap a low-level ciphertext.
//
// One JSON verdict line on stdout: {"keypartsTest":...,"pass":true|false,...}.
// Build: bash hpc_gpu_port/mac_build_keyparts_test.sh
#define MAC_FHE_CLIENT_NO_MAIN
#include "mac_fhe_client.cpp"

#include <complex>
#include <filesystem>
#include <set>
#include <sstream>

namespace fs = std::filesystem;

static double cyc(const std::vector<double>& v, long long i) {
    const long long n = (long long)v.size();
    return v[(size_t)(((i % n) + n) % n)];
}

static std::vector<long long> devLineInts(const std::string& line) {
    // FIDESlib api/Serialize.cpp reader: "<label>: { a b c }" -> ints
    std::istringstream iss(line);
    std::string label; char brace; long long x;
    iss >> label >> brace;
    std::vector<long long> out;
    while (iss >> x) out.push_back(x);
    return out;
}

int main(int argc, char** argv) {
    const std::string outDir = arg(argc, argv, "--out");
    const int batch = std::stoi(arg(argc, argv, "--batch", "5"));
    const int Dpad = std::stoi(arg(argc, argv, "--dpad", "1024"));
    const int lanes = std::stoi(arg(argc, argv, "--lanes-block", "1"));
    ClientCtxParams P;
    P.logRing = std::stoi(arg(argc, argv, "--log-ring", "12"));
    P.extraDepth = std::stoi(arg(argc, argv, "--extra-depth", "3"));
    P.numDigits = std::stoi(arg(argc, argv, "--num-digits", "0"));
    P.lbA = 3; P.lbB = 3; P.scaleBits = 59; P.secure = 0;
    const std::vector<uint32_t> levelBudget = {3, 3};

    fs::create_directories(outDir);
    for (auto& e : fs::directory_iterator(outDir)) {      // stale parts from an earlier run
        const std::string n = e.path().filename().string();
        if (n.rfind("evalrot.", 0) == 0) fs::remove(e.path());
    }
    const std::string contractPath = outDir + "/indices.json";

    uint32_t depth = 0, N = 0, SLOTS = 0, M = 0;
    std::vector<int32_t> rots;
    std::vector<uint32_t> contract;
    std::set<uint32_t> matvecSet, bootSet;
    size_t overlap = 0;
    // ---- Phase A ----------------------------------------------------------
    {
        CryptoContext<DCRTPoly> cc = makeClientContext(P, depth);
        N = cc->GetRingDimension(); SLOTS = N / 2; M = cc->GetCyclotomicOrder();
        rots = blockLayoutRots(Dpad, lanes);
        for (auto r : rots) matvecSet.insert(cc->FindAutomorphismIndex((uint32_t)r));
        auto kpThrow = cc->KeyGen();
        cc->EvalBootstrapSetup(levelBudget, {0, 0}, SLOTS);
        cc->EvalBootstrapKeyGen(kpThrow.secretKey, SLOTS);
        const std::string tTag = kpThrow.secretKey->GetKeyTag();
        for (auto& kv : CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(tTag)) bootSet.insert(kv.first);
        CryptoContextImpl<DCRTPoly>::ClearEvalAutomorphismKeys(tTag);
        std::set<uint32_t> all(matvecSet);
        for (auto i : bootSet) { if (matvecSet.count(i)) overlap++; all.insert(i); }
        contract.assign(all.begin(), all.end());
        std::ofstream jf(contractPath);
        jf << "{\"ring\":" << N << ",\"slots\":" << SLOTS << ",\"depth\":" << depth << ",\n";
        jf << " \"rotationAmounts\":[";
        for (size_t i = 0; i < rots.size(); i++) jf << (i ? "," : "") << rots[i];
        jf << "],\n \"autoIndices\":[";
        for (size_t i = 0; i < contract.size(); i++) jf << (i ? "," : "") << contract[i];
        jf << "],\n \"bootstrapSlots\":" << SLOTS << "}\n";
        jf.close();
        printf("{\"phase\":\"A\",\"ring\":%u,\"slots\":%u,\"depth\":%u,\"M\":%u,\"rotationAmounts\":%zu,"
               "\"matvecAutos\":%zu,\"bootAutos\":%zu,\"overlap\":%zu,\"contractKeys\":%zu,"
               "\"conjIndexInBoot\":%s,\"bootstrapDepth\":%u,\"contract\":\"%s\"}\n",
               N, SLOTS, depth, M, rots.size(), matvecSet.size(), bootSet.size(), overlap, contract.size(),
               bootSet.count(M - 1) ? "true" : "false",
               FHECKKSRNS::GetBootstrapDepth(levelBudget, UNIFORM_TERNARY), contractPath.c_str());
        fflush(stdout);
    }
    // ---- Phase B: the real client path ------------------------------------
    int rc = -1;
    {
        std::vector<std::string> a = {"mac_fhe_client", "keygen", "--out", outDir,
            "--log-ring", std::to_string(P.logRing), "--extra-depth", std::to_string(P.extraDepth),
            "--lb-a", "3", "--lb-b", "3", "--scale-bits", "59", "--dpad", std::to_string(Dpad),
            "--lanes-block", std::to_string(lanes), "--indices", contractPath, "--batch", std::to_string(batch)};
        if (P.numDigits > 0) { a.push_back("--num-digits"); a.push_back(std::to_string(P.numDigits)); }
        std::vector<char*> av;
        for (auto& s : a) av.push_back(&s[0]);
        rc = do_keygen((int)av.size(), av.data());
        printf("{\"phase\":\"B\",\"doKeygenRc\":%d,\"batch\":%d}\n", rc, batch);
        fflush(stdout);
        if (rc != 0) {
            printf("{\"keypartsTest\":\"2026-09-02\",\"pass\":false,\"why\":\"do_keygen rc %d\"}\n", rc);
            return 1;
        }
    }
    // ---- Phase C: fresh context, everything from disk ----------------------
    CryptoContextImpl<DCRTPoly>::ClearEvalAutomorphismKeys();
    CryptoContextImpl<DCRTPoly>::ClearEvalMultKeys();
    CryptoContextFactory<DCRTPoly>::ReleaseAllContexts();
    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(outDir + "/cryptocontext.bin", cc, SerType::BINARY)) {
        printf("{\"keypartsTest\":\"2026-09-02\",\"pass\":false,\"why\":\"context load\"}\n"); return 1;
    }
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE); cc->Enable(FHE);
    PublicKey<DCRTPoly> pk; PrivateKey<DCRTPoly> sk;
    if (!Serial::DeserializeFromFile(outDir + "/public.key", pk, SerType::BINARY) ||
        !Serial::DeserializeFromFile(outDir + "/secret.key", sk, SerType::BINARY)) {
        printf("{\"keypartsTest\":\"2026-09-02\",\"pass\":false,\"why\":\"key load\"}\n"); return 1;
    }
    {
        std::ifstream em(outDir + "/evalmult.bin", std::ios::binary);
        if (!em.good() || !cc->DeserializeEvalMultKey(em, SerType::BINARY)) {
            printf("{\"keypartsTest\":\"2026-09-02\",\"pass\":false,\"why\":\"evalmult load\"}\n"); return 1;
        }
    }
    std::vector<std::string> parts;
    for (auto& e : fs::directory_iterator(outDir)) {
        const std::string n = e.path().filename().string();
        if (n.rfind("evalrot.part", 0) == 0 && n.size() > 4 && n.substr(n.size() - 4) == ".bin")
            parts.push_back(e.path().string());
    }
    std::sort(parts.begin(), parts.end());
    long long partBytes = 0;
    for (auto& p : parts) {
        std::ifstream is(p, std::ios::binary | std::ios::ate);
        partBytes += (long long)is.tellg();
        is.seekg(0);
        if (!is.good() || !cc->DeserializeEvalAutomorphismKey(is, SerType::BINARY)) {
            printf("{\"keypartsTest\":\"2026-09-02\",\"pass\":false,\"why\":\"part load %s\"}\n", p.c_str());
            return 1;
        }
    }
    const bool evalrotBinAbsent = !fs::exists(outDir + "/evalrot.bin");
    const std::string tag = pk->GetKeyTag();
    const bool tagMatch = (tag == sk->GetKeyTag());
    const auto& loaded = CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(tag);
    std::set<uint32_t> loadedSet;
    for (auto& kv : loaded) loadedSet.insert(kv.first);
    const bool indexSetMatch = (loadedSet == std::set<uint32_t>(contract.begin(), contract.end()));
    // .dev sidecar: RotationIndexes must be the rotation AMOUNTS, BootstrapSlots {SLOTS}
    bool devOk = false;
    {
        std::ifstream dv(outDir + "/cryptocontext.bin.dev");
        std::string l1, l2, l3, l4, l5, l6;
        std::getline(dv, l1); std::getline(dv, l2); std::getline(dv, l3);
        std::getline(dv, l4); std::getline(dv, l5); std::getline(dv, l6);
        const auto dr = devLineInts(l4), db = devLineInts(l6);
        devOk = (l4.rfind("RotationIndexes:", 0) == 0) && (l6.rfind("BootstrapSlots:", 0) == 0) &&
                dr.size() == rots.size() && db.size() == 1 && db[0] == (long long)SLOTS;
        for (size_t i = 0; devOk && i < rots.size(); i++) devOk = (dr[i] == (long long)rots[i]);
    }
    // rotations
    std::vector<double> v(SLOTS);
    for (uint32_t i = 0; i < SLOTS; i++) v[i] = 0.5 * std::sin(0.37 * i + 0.1);
    double rotMax = 0, rotSq = 0; long long rotN = 0;
    {
        auto ct = cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(v));
        for (auto r : rots) {
            auto cr = cc->EvalRotate(ct, r);
            Plaintext d; cc->Decrypt(sk, cr, &d); d->SetLength(SLOTS);
            const auto w = d->GetRealPackedValue();
            for (uint32_t i = 0; i < SLOTS; i++) {
                const double e = std::fabs(w[i] - cyc(v, (long long)i + r));
                rotMax = std::max(rotMax, e); rotSq += e * e; rotN++;
            }
        }
    }
    const double rotRms = std::sqrt(rotSq / (double)rotN);
    // The minted M-1 key as a KEY-SWITCH key: EvalAutomorphism(ct, M-1) on a
    // REAL vector must be the identity to ~1e-12. A garbage key or a key for
    // another index leaves O(1) noise; a rotation key shifts the slots. The
    // SEMANTIC conjugation cannot be probed standalone here: the context is
    // REAL-typed like the harness's, and OpenFHE drops imaginary parts at
    // encode (ckkspackedencoding.h:97) and decode (ckkspackedencoding.cpp:495,
    // the MEASUREMENTS.md 1a coeff_conv trap), so ct - conj(ct) is 0 by
    // construction. Conjugation is exercised where it matters: inside
    // EvalBootstrap below (FHECKKSRNS::Conjugate on this same M-1 key).
    double conjMax = 0;
    {
        auto cct = cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(v));
        auto cj = cc->EvalAutomorphism(cct, M - 1, loaded);
        Plaintext d; cc->Decrypt(sk, cj, &d); d->SetLength(SLOTS);
        const auto w = d->GetRealPackedValue();
        for (uint32_t i = 0; i < SLOTS; i++) conjMax = std::max(conjMax, std::fabs(w[i] - v[i]));
    }
    // bootstrap at low level (the precompute is not serialized: re-run Setup on the fresh context)
    cc->EvalBootstrapSetup(levelBudget, {0, 0}, SLOTS);
    uint32_t levelBefore = 0, levelAfter = 0;
    double bootMax = 0, bootSq = 0;
    {
        auto ctL = cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(v, 1, depth - 1));
        levelBefore = ctL->GetLevel();
        auto ctB = cc->EvalBootstrap(ctL);
        levelAfter = ctB->GetLevel();
        Plaintext d; cc->Decrypt(sk, ctB, &d); d->SetLength(SLOTS);
        const auto w = d->GetRealPackedValue();
        for (uint32_t i = 0; i < SLOTS; i++) {
            const double e = std::fabs(w[i] - v[i]);
            bootMax = std::max(bootMax, e); bootSq += e * e;
        }
    }
    const double bootRms = std::sqrt(bootSq / (double)SLOTS);
    const bool bootRestored = levelAfter < levelBefore;
    const bool pass = (rc == 0) && tagMatch && indexSetMatch && loadedSet.size() == contract.size() &&
                      parts.size() > 1 && evalrotBinAbsent && devOk && rotMax < 1e-6 && conjMax < 1e-6 &&
                      bootRestored && bootMax < 1e-2 && std::isfinite(bootMax) && std::isfinite(rotMax);
    printf("{\"keypartsTest\":\"2026-09-02\",\"pass\":%s,\"ring\":%u,\"slots\":%u,\"depth\":%u,"
           "\"contractKeys\":%zu,\"matvecAutos\":%zu,\"bootAutos\":%zu,\"overlap\":%zu,"
           "\"parts\":%zu,\"batch\":%d,\"partBytes\":%lld,\"loadedKeys\":%zu,\"indexSetMatch\":%s,"
           "\"evalrotBinAbsent\":%s,\"devSidecarOk\":%s,\"rotChecked\":%zu,\"rotMaxAbsErr\":%.3e,"
           "\"rotRmsErr\":%.3e,\"conjKeyIdentityMaxAbsErr\":%.3e,\"bootLevelBefore\":%u,\"bootLevelAfter\":%u,"
           "\"bootRestored\":%s,\"bootMaxAbsErr\":%.3e,\"bootRmsErr\":%.3e}\n",
           pass ? "true" : "false", N, SLOTS, depth, contract.size(), matvecSet.size(), bootSet.size(), overlap,
           parts.size(), batch, partBytes, loadedSet.size(), indexSetMatch ? "true" : "false",
           evalrotBinAbsent ? "true" : "false", devOk ? "true" : "false", rots.size(), rotMax, rotRms, conjMax,
           levelBefore, levelAfter, bootRestored ? "true" : "false", bootMax, bootRms);
    return pass ? 0 : 1;
}
