// periodic_encode_test.cpp -- S3.7 V5 (2026-09-10): hpc_gpu_port/periodic_encode.hpp against the dense path
// on REAL bundle entries at the demo ring (2^17, depth 41, Delta 2^59, FLEXIBLEAUTO), plus expandCompToPlaintext.
//
// Derived from results/theory/s36_20260905/t01_t02_t15_tick0/store_entry_probe.cpp (S3.6 A6), whose INLINE
// builder was checked word for word on 480/480 real entries. Here the builder under test is the header's
// PeriodicEncoder -- the object gpu_real_model.cu --periodic-encode uses -- and three things are asserted per
// entry: (1) encode() == the dense path's compressed words (the exact algebra of buildComp :2896-:2927:
// MakeCKKSPackedPlaintext(diag, 1, lvl), then comp[l*CW + j2] = limb_l[j2 << logRep] once every limb is REP-
// periodic); (2) expandCompToPlaintext() reproduces the dense plaintext's element limb for limb, and its
// level / scaling factor; (3) on a sample, EvalMult(ct, expanded) and EvalMult(ct, dense) give bit-identical
// ciphertexts (ct x pt is deterministic), so the expanded plaintext is usable where the dense one is.
//
// Modes: --mode identity (default). Correctness only: no timing lock; the ms fields are indicative.
// Exit 0 iff every checked entry is EXACT; 3 otherwise. One JSON line per entry, one summary line.
#include "periodic_encode.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;
static double msSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; }

// bundle index reader (gpu_real_model.cu Bundle format: "<name> <offset_bytes> <nElems> <dims...>")
struct BundleIdx {
    std::string bin;
    std::map<std::string, std::pair<size_t, std::vector<int>>> idx;
    void load(const std::string& binPath, const std::string& idxPath) {
        bin = binPath;
        std::ifstream xf(idxPath);
        if (!xf) { std::cerr << "{\"fatal\":\"no bundle index " << idxPath << "\"}\n"; std::exit(2); }
        std::string line;
        while (std::getline(xf, line)) {
            std::istringstream ss(line);
            std::string name; size_t offBytes, nEl;
            if (!(ss >> name >> offBytes >> nEl)) continue;
            std::vector<int> shape; int d;
            while (ss >> d) shape.push_back(d);
            idx[name] = {offBytes, shape};
        }
    }
    std::vector<double> mat(const std::string& n, int& rows, int& cols) const {
        auto it = idx.find(n);
        if (it == idx.end()) { std::cerr << "{\"fatal\":\"missing tensor " << n << "\"}\n"; std::exit(2); }
        const auto& s = it->second.second; rows = s[0]; cols = s.size() > 1 ? s[1] : 1;
        std::vector<double> v((size_t)rows * cols);
        std::ifstream bf(bin, std::ios::binary);
        if (!bf) { std::cerr << "{\"fatal\":\"no bundle bin " << bin << "\"}\n"; std::exit(2); }
        bf.seekg((std::streamoff)it->second.first);
        bf.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(v.size() * sizeof(double)));
        if (!bf) { std::cerr << "{\"fatal\":\"short read " << n << "\"}\n"; std::exit(2); }
        return v;
    }
};

// the record's operand levels (demo2/server.log req 1 poolBuild.lvl; op: 0 win, 1 wout, 2 wk, 3 wr, 4 wv)
static uint32_t recordLevel(int layer, int op) {
    if (layer == 0) { switch (op) { case 0: return 26; case 1: return 32; case 2: case 3: return 35; default: return 24; } }
    switch (op) { case 0: return 36; case 1: return 25; case 2: case 3: return 35; default: return 24; }
}
struct Job { int layer, op, chunk, rowOff, diag, half; uint32_t lvl; int rows, inCols; const std::vector<double>* W; };

int main(int argc, char** argv) {
    uint32_t logRing = 17, depth = 41, dpad = 1024;
    std::string bundleDir = "$MAC_ART", tag = "pbd430a", mode = "identity";
    int lanesBlock = 0, perOp = 8, nLayers = 4, seed = 20260910, mulEvery = 8;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--log-ring") logRing = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--depth") depth = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--dpad") dpad = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--bundle-dir") bundleDir = argv[++i];
        else if (a == "--tag") tag = argv[++i];
        else if (a == "--mode") mode = argv[++i];
        else if (a == "--lanes-block") lanesBlock = std::stoi(argv[++i]);
        else if (a == "--per-op") perOp = std::stoi(argv[++i]);
        else if (a == "--layers") nLayers = std::stoi(argv[++i]);
        else if (a == "--seed") seed = std::stoi(argv[++i]);
        else if (a == "--mul-every") mulEvery = std::stoi(argv[++i]);   // ct x pt functional check on every k-th entry (0 = never)
    }
    // the demo chain (gpu_real_model.cu policy): 2^17, depth 41, Delta 2^59, first modulus 60, FLEXIBLEAUTO
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetSecurityLevel(logRing >= 17 ? HEStd_128_classic : HEStd_NotSet);
    parameters.SetRingDim(1u << logRing);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(depth);
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    const uint32_t N = cc->GetRingDimension(), SLOTS = N / 2, REP = SLOTS / dpad, CW = N / REP;
    uint32_t logRep = 0; while ((1u << logRep) < REP) logRep++;
    auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    const uint32_t nqTotal = (uint32_t)cp->GetElementParams()->GetParams().size();

    // ---- the object under test ----
    auto tC = Clock::now();
    fhe_ssm::PeriodicEncoder enc(cc, logRep);
    const double ctorMs = msSince(tC);
    fhe_ssm::PeriodicEncoder::Scratch scratch;
    const bool geomOk = (enc.CW == CW) && (enc.period == dpad) && (enc.nqTotal == nqTotal) && (enc.REP == REP);

    KeyPair<DCRTPoly> keys;
    if (mulEvery > 0) keys = cc->KeyGen();   // the ct x pt check needs an encryption; no eval keys

    std::cout << "{\"test\":\"periodic_encode_test\",\"mode\":\"" << mode << "\",\"N\":" << N << ",\"slots\":" << SLOTS << ",\"dpad\":" << dpad
              << ",\"rep\":" << REP << ",\"Nprime\":" << enc.Np << ",\"depth\":" << depth << ",\"limbsTotal\":" << nqTotal
              << ",\"lanesBlock\":" << lanesBlock << ",\"geometryOk\":" << (geomOk ? "true" : "false") << ",\"encoderCtorMs\":" << ctorMs
              << ",\"bundleDir\":\"" << bundleDir << "\",\"tag\":\"" << tag << "\",\"openfhe\":\"1.5.1 (vendor/install, WITH_OPENMP=OFF)\"}" << std::endl;
    if (!geomOk) { std::cout << "{\"summary\":true,\"pass\":false,\"why\":\"geometry\"}" << std::endl; return 3; }

    BundleIdx B; B.load(bundleDir + "/bundle_" + tag + ".bin", bundleDir + "/bundle_" + tag + ".index.txt");

    // buildComp's slot vector (block layout), verbatim
    auto buildDiag = [&](const Job& J, std::vector<double>& diag, bool& nonzero) {
        std::fill(diag.begin(), diag.end(), 0.0); nonzero = false;
        const int g = J.diag;
        for (int row = 0; row < J.rows; row++) {
            int col = (row + g) % (int)dpad;
            if (col >= J.inCols) continue;
            if (lanesBlock && ((row + g >= (int)dpad) ? 1 : 0) != J.half) continue;
            double wv = (*J.W)[(size_t)(row + J.rowOff) * J.inCols + col];
            uint32_t slot = (uint32_t)((row + g) % (int)dpad);
            for (uint32_t r = 0; r < REP; r++) diag[r * dpad + slot] = wv;
            if (std::abs(wv) > 0.0) nonzero = true;
        }
    };
    // the dense reference: the harness's compress/verify loop on the pure-OpenFHE plaintext element
    auto denseComp = [&](const std::vector<double>& diag, uint32_t lvl, Plaintext& pt, std::vector<uint64_t>& comp, double& encMs) -> bool {
        auto t0 = Clock::now();
        pt = cc->MakeCKKSPackedPlaintext(diag, 1, lvl);
        encMs = msSince(t0);
        const DCRTPoly& e = pt->GetElement<DCRTPoly>();
        const uint32_t nLimbs = (uint32_t)e.GetNumOfElements();
        comp.assign((size_t)nLimbs * CW, 0);
        bool okmap = (nLimbs == depth + 1 - lvl);
        for (uint32_t l = 0; okmap && l < nLimbs; l++) {
            const auto& limbv = e.GetElementAtIndex(l).GetValues();
            if (limbv.GetLength() != N) { okmap = false; break; }
            for (uint32_t i2 = 0; i2 < N; i2++) if (limbv[i2] != limbv[(i2 >> logRep) << logRep]) { okmap = false; break; }
            if (okmap) for (uint32_t j2 = 0; j2 < CW; j2++) comp[(size_t)l * CW + j2] = limbv[j2 << logRep].ConvertToInt();
        }
        return okmap;
    };
    // expanded vs dense: element limb for limb, then metadata
    auto elemIdentical = [&](const Plaintext& a, const Plaintext& b, uint64_t& mism) -> bool {
        const DCRTPoly& ea = a->GetElement<DCRTPoly>(); const DCRTPoly& eb = b->GetElement<DCRTPoly>();
        mism = 0;
        if (ea.GetNumOfElements() != eb.GetNumOfElements() || ea.GetFormat() != eb.GetFormat()) return false;
        for (uint32_t l = 0; l < ea.GetNumOfElements(); l++) {
            const auto& va = ea.GetElementAtIndex(l).GetValues(); const auto& vb = eb.GetElementAtIndex(l).GetValues();
            if (va.GetLength() != vb.GetLength() || va.GetModulus() != vb.GetModulus()) return false;
            for (uint32_t i = 0; i < va.GetLength(); i++) if (va[i] != vb[i]) mism++;
        }
        return mism == 0;
    };
    auto ctIdentical = [&](const Ciphertext<DCRTPoly>& a, const Ciphertext<DCRTPoly>& b) -> bool {
        const auto& ea = a->GetElements(); const auto& eb = b->GetElements();
        if (ea.size() != eb.size()) return false;
        for (size_t k = 0; k < ea.size(); k++) {
            if (ea[k].GetNumOfElements() != eb[k].GetNumOfElements()) return false;
            for (uint32_t l = 0; l < ea[k].GetNumOfElements(); l++)
                if (ea[k].GetElementAtIndex(l).GetValues() != eb[k].GetElementAtIndex(l).GetValues()) return false;
        }
        return a->GetLevel() == b->GetLevel() && a->GetScalingFactor() == b->GetScalingFactor() && a->GetNoiseScaleDeg() == b->GetNoiseScaleDeg();
    };

    std::mt19937_64 rng((uint64_t)seed);
    struct Mats { std::vector<double> win, wout, wk, wr, wv; int winR, winC, woutR, woutC, wkR, wkC, wrR, wrC, wvR, wvC; std::vector<std::vector<double>> wvBlocks; };
    std::map<int, Mats> mats;
    auto loadLayer = [&](int l) -> Mats& {
        auto it = mats.find(l); if (it != mats.end()) return it->second;
        Mats M; std::string p = "L" + std::to_string(l) + ".";
        M.win = B.mat(p + "tm.win", M.winR, M.winC); M.wout = B.mat(p + "tm.wout", M.woutR, M.woutC);
        M.wk = B.mat(p + "cm.wk", M.wkR, M.wkC); M.wr = B.mat(p + "cm.wr", M.wrR, M.wrC); M.wv = B.mat(p + "cm.wv", M.wvR, M.wvC);
        const int K = M.wvC / (int)dpad; M.wvBlocks.resize(K);
        for (int ch = 0; ch < K; ch++) { M.wvBlocks[ch].assign((size_t)M.wvR * dpad, 0.0);
            for (int i = 0; i < M.wvR; i++) for (uint32_t j = 0; j < dpad; j++) M.wvBlocks[ch][(size_t)i * dpad + j] = M.wv[(size_t)i * M.wvC + ch * dpad + j]; }
        return mats.emplace(l, std::move(M)).first->second;
    };
    auto makeJob = [&](int l, int op, int chunk, int diag, int half) -> Job {
        Mats& M = loadLayer(l);
        Job J; J.layer = l; J.op = op; J.diag = diag; J.half = half; J.lvl = recordLevel(l, op);
        switch (op) {
            case 0: J.chunk = 0; J.rowOff = 0; J.rows = std::min((int)dpad, M.winR); J.inCols = M.winC; J.W = &M.win; break;
            case 1: J.chunk = 0; J.rowOff = 0; J.rows = std::min((int)dpad, M.woutR); J.inCols = M.woutC; J.W = &M.wout; break;
            case 2: J.chunk = 0; J.rowOff = chunk * (int)dpad; J.rows = std::min((int)dpad, M.wkR - J.rowOff); J.inCols = M.wkC; J.W = &M.wk; break;
            case 3: J.chunk = 0; J.rowOff = chunk * (int)dpad; J.rows = std::min((int)dpad, M.wrR - J.rowOff); J.inCols = M.wrC; J.W = &M.wr; break;
            default: J.chunk = chunk; J.rowOff = 0; J.rows = M.wvR; J.inCols = (int)dpad; J.W = &M.wvBlocks[chunk]; break;
        }
        return J;
    };
    const int HALVES = lanesBlock ? 2 : 1;
    std::vector<int> layerSet;
    { std::vector<int> all(24); for (int i = 0; i < 24; i++) all[i] = i; layerSet.push_back(0); std::shuffle(all.begin() + 1, all.end(), rng);
      for (int i = 0; i < nLayers - 1 && i < 23; i++) layerSet.push_back(all[1 + i]); }

    int failures = 0, entries = 0, dead = 0, mulChecked = 0, mulFail = 0, expandFail = 0, encodeFail = 0; uint64_t wordsChecked = 0;
    std::vector<double> diag(SLOTS, 0.0), vtest(SLOTS, 0.0); std::vector<uint64_t> cd, cpv;
    for (uint32_t i = 0; i < SLOTS; i++) vtest[i] = 0.25 * std::sin(0.013 * i + 0.7);
    std::map<uint32_t, std::vector<double>> tDense, tPer, tExp;
    int n = 0;
    for (int l : layerSet) for (int op = 0; op < 5; op++) for (int k = 0; k < perOp; k++) {
        const int chunk = (op >= 2) ? (int)(rng() % 4) : 0;
        const int dg = (int)(rng() % dpad);
        for (int half = 0; half < HALVES; half++) {
            Job J = makeJob(l, op, chunk, dg, half);
            bool nz = false; buildDiag(J, diag, nz);
            entries++;
            std::ostringstream key; key << "{\"layer\":" << l << ",\"op\":" << op << ",\"chunk\":" << J.chunk << ",\"rowOff\":" << J.rowOff << ",\"diag\":" << dg << ",\"half\":" << half << ",\"lvl\":" << J.lvl << "}";
            if (!nz) { dead++; std::cout << "{\"entry\":" << key.str() << ",\"dead\":true}" << std::endl; continue; }
            double em = 0; Plaintext ptD;
            const bool okmap = denseComp(diag, J.lvl, ptD, cd, em);
            // (1) the header's encoder on the period (replica 0)
            auto t0 = Clock::now();
            const int rc = enc.encode(diag.data(), J.lvl, cpv, scratch);
            const double pm = msSince(t0);
            uint64_t mism = 0;
            if (rc != fhe_ssm::PE_OK || cd.size() != cpv.size()) mism = (uint64_t)-1;
            else for (size_t i = 0; i < cd.size(); i++) if (cd[i] != cpv[i]) mism++;
            wordsChecked += cd.size();
            const bool encOk = okmap && rc == fhe_ssm::PE_OK && mism == 0;
            if (!encOk) encodeFail++;
            // (2) expandCompToPlaintext vs the dense plaintext
            bool expOk = false; uint64_t emism = 0; double xm = 0; std::string expWhy;
            if (encOk) {
                try {
                    auto t1 = Clock::now();
                    Plaintext ptX = fhe_ssm::expandCompToPlaintext(cc, enc, cpv, J.lvl);
                    xm = msSince(t1);
                    const bool lvlOk = ptX->GetLevel() == ptD->GetLevel(), sfOk = ptX->GetScalingFactor() == ptD->GetScalingFactor(),
                               degOk = ptX->GetNoiseScaleDeg() == ptD->GetNoiseScaleDeg(), lenOk = ptX->GetLength() == ptD->GetLength();
                    // GetLength() is the DECODED value count; the expanded plaintext deliberately carries no decoded values
                    // (only what EvalMult/EvalAdd read), so length is reported, not required.
                    const bool metaOk = lvlOk && sfOk && degOk;
                    expOk = elemIdentical(ptX, ptD, emism) && metaOk;
                    if (!metaOk) { std::ostringstream w; w << "metadata level " << ptX->GetLevel() << "/" << ptD->GetLevel() << " sf " << ptX->GetScalingFactor() << "/" << ptD->GetScalingFactor()
                                                      << " deg " << ptX->GetNoiseScaleDeg() << "/" << ptD->GetNoiseScaleDeg() << " len " << ptX->GetLength() << "/" << ptD->GetLength(); expWhy = w.str(); }
                    else if (!lenOk) expWhy = "len " + std::to_string(ptX->GetLength()) + "/" + std::to_string(ptD->GetLength()) + " (informational)";
                    // (3) ct x pt with each plaintext, on every mulEvery-th entry
                    if (expOk && mulEvery > 0 && (n % mulEvery) == 0) {
                        mulChecked++;
                        auto ct = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(vtest, 1, J.lvl));
                        auto cA = cc->EvalMult(ct, ptD), cB = cc->EvalMult(ct, ptX);
                        if (!ctIdentical(cA, cB)) { mulFail++; expOk = false; expWhy = "ct x pt differs"; }
                    }
                } catch (const std::exception& e) { expWhy = e.what(); expOk = false; }
                if (!expOk) expandFail++;
            }
            n++;
            const bool ok = encOk && expOk;
            if (!ok) failures++;
            tDense[J.lvl].push_back(em); tPer[J.lvl].push_back(pm); tExp[J.lvl].push_back(xm);
            std::cout << "{\"entry\":" << key.str() << ",\"limbs\":" << (depth + 1 - J.lvl) << ",\"words\":" << cd.size()
                      << ",\"okmap\":" << (okmap ? "true" : "false") << ",\"encodeStatus\":" << rc << ",\"mismatchWords\":" << (long long)mism
                      << ",\"expandMismatchWords\":" << (long long)emism << ",\"expandWhy\":\"" << expWhy << "\""
                      << ",\"denseEncodeMs\":" << em << ",\"periodicEncodeMs\":" << pm << ",\"expandMs\":" << xm
                      << ",\"identity\":\"" << (ok ? "EXACT" : "FAIL") << "\"}" << std::endl;
        }
    }
    std::cout << "{\"summary\":true,\"pass\":" << (failures == 0 && entries > dead ? "true" : "false") << ",\"entries\":" << entries << ",\"dead\":" << dead
              << ",\"failures\":" << failures << ",\"encodeFailures\":" << encodeFail << ",\"expandFailures\":" << expandFail
              << ",\"ctPtChecked\":" << mulChecked << ",\"ctPtFailures\":" << mulFail << ",\"wordsChecked\":" << wordsChecked << ",\"layers\":[";
    for (size_t i = 0; i < layerSet.size(); i++) std::cout << (i ? "," : "") << layerSet[i];
    std::cout << "],\"perLevelMedianMs\":{"; bool first = true;
    for (auto& kv : tDense) { std::cout << (first ? "" : ",") << "\"" << kv.first << "\":{\"limbs\":" << (depth + 1 - kv.first) << ",\"dense\":" << median(kv.second)
                                        << ",\"periodic\":" << median(tPer[kv.first]) << ",\"expand\":" << median(tExp[kv.first]) << ",\"n\":" << kv.second.size() << "}"; first = false; }
    std::cout << "},\"timingLock\":\"NOT held (identity mode; ms indicative only)\"}" << std::endl;
    return failures ? 3 : 0;
}
