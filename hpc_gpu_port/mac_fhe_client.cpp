// mac_fhe_client.cpp — Mac-side (CPU, OpenFHE-only) client for the FHE-SSM
// demo (S3.1, 2026-09-01, decision: encrypt/decrypt on the Mac, not the
// pod). Interop premise: the pod harness serializes ciphertexts with pure
// lbcrypto::Serial (gpu_real_model.cu saveCt/loadCt) and keys with the
// FIDESlib api wrapper AROUND lbcrypto Serial — so a same-version (1.5.1)
// OpenFHE CPU build reads/writes compatible bytes. The .dev key sidecar is
// FIDESlib-only and is NOT needed here.
//
//   enc: rows.f64 (NL*T x d, lane-major) -> prefix.<t> ct files + prefix.meta
//        (block-lane packing: slot rep*Dpad+i = lane (rep mod NL), cyclic —
//        exactly --client-lanes in the harness)
//   dec: result cts -> hidden.f64 ((T*NL) x d, tok-major lane-minor).
//        The head matvec stays in Python (numpy memmap of the bundle).
//
// keygen modes (2026-09-02, see do_keygen): default (evalrot.bin, full set),
//   --indices FILE --batch N (pod-dumped index set minted into RAM-bounded
//   evalrot.partNNN.bin files, no superset, no evalrot.bin), --minimal 1
//   (context + key pair + evalmult only). --num-digits N mirrors the harness.
//   --seeded 1 (S3.7 V4, 2026-09-05) with --indices: evalrot.partNNN.seeded.bin
//   (32-byte seed + b limbs per key, 2.000x smaller; seeded_keys.hpp).
//   keyparts_test.cpp #includes this file (MAC_FHE_CLIENT_NO_MAIN) to prove
//   the part pipeline end to end at ring 2^12; seeded_keys_test.cpp does the
//   same for the seeded parts against stock keys of the same secret.
// Build (see mac_build_client.sh): clang++ against vendor/install, which
// vendor/build/CMakeCache.txt records as built from vendor/openfhe-development
// (NOT vendor/openfhe-fideslib, whose header differs; the serialized bytes are
// the same 1.5.1 format either way).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include "seeded_keys.hpp"   // S3.7 V4 (2026-09-05): --seeded keygen, S3.6 T12 §2.3

using namespace lbcrypto;

static std::string arg(int argc, char** argv, const std::string& k,
                       const std::string& dflt = "") {
    for (int i = 1; i + 1 < argc; i++)
        if (k == argv[i]) return argv[i + 1];
    // AUDIT FIX 2026-09-02: a flag given as the LAST argv token has no value
    // and used to fall through to the default -- e.g. a dangling
    // --require-meta silently became "not required". Loud instead.
    if (argc > 1 && k == argv[argc - 1]) {
        fprintf(stderr, "{\"fatal\":\"%s given without a value\"}\n", k.c_str()); exit(2);
    }
    if (dflt.empty()) { fprintf(stderr, "missing %s\n", k.c_str()); exit(2); }
    return dflt;
}

// F63 (2026-09-01): the harness stamps a .meta sidecar beside every ciphertext
// set it writes (count/packTokens/tokensReal/dpad/interleave/lanes/ringDim).
// `dec` used to take lanes/count/dpad from its OWN command line and never open
// that file — which meant the fully-secure Mac path was still exposed to
// exactly the bug F62 fixed on the pod-side decryptor: a response carrying
// lanes:1 sliced as if it carried lanes:4 yields four plausible hidden vectors,
// three of which are other lanes' slots read at the wrong stride. Same integer
// string-find parser as gpu_real_model.cu's metaInt, so the two agree by
// construction.
static long long metaInt(const std::string& text, const std::string& key,
                         long long dflt) {
    auto pos = text.find("\"" + key + "\":");
    if (pos == std::string::npos) return dflt;
    return std::atoll(text.c_str() + pos + key.size() + 3);
}

// ---------------------------------------------------------------------------
// keygen helpers. keyparts_test.cpp #includes THIS FILE with
// MAC_FHE_CLIENT_NO_MAIN defined, so the test and the client run one copy of
// the context policy, the rotation set, the contract reader, the batched
// minter and the .dev writer -- there is no second implementation to drift.
// ---------------------------------------------------------------------------

// Minimal readers for the pod-written index contract (--indices FILE). The
// file is a flat JSON object of ints and int arrays, e.g.
//   {"ring":131072,"slots":65536,"depth":38,"rotationAmounts":[1,2,...],
//    "autoIndices":[3,5,...],"bootstrapSlots":65536}
// No JSON library is linked on the Mac build. Keys are matched in their quoted
// form followed by ':' so "slots" cannot match inside "bootstrapSlots".
static bool jsonSeek(const std::string& t, const std::string& key, size_t& pos) {
    const std::string q = "\"" + key + "\"";
    size_t p = t.find(q);
    if (p == std::string::npos) return false;
    p += q.size();
    while (p < t.size() && isspace((unsigned char)t[p])) p++;
    if (p >= t.size() || t[p] != ':') return false;
    p++;
    while (p < t.size() && isspace((unsigned char)t[p])) p++;
    pos = p;
    return true;
}
static bool jsonInt(const std::string& t, const std::string& key, long long& out) {
    size_t p;
    if (!jsonSeek(t, key, p)) return false;
    char* end = nullptr;
    out = std::strtoll(t.c_str() + p, &end, 10);
    return end != t.c_str() + p;
}
static bool jsonIntList(const std::string& t, const std::string& key, std::vector<long long>& out) {
    size_t p;
    if (!jsonSeek(t, key, p)) return false;
    if (p >= t.size() || t[p] != '[') return false;
    p++;
    out.clear();
    for (;;) {
        while (p < t.size() && isspace((unsigned char)t[p])) p++;
        if (p >= t.size()) return false;
        if (t[p] == ']') return true;
        char* end = nullptr;
        const long long v = std::strtoll(t.c_str() + p, &end, 10);
        if (end == t.c_str() + p) return false;
        out.push_back(v);
        p = (size_t)(end - t.c_str());
        while (p < t.size() && isspace((unsigned char)t[p])) p++;
        if (p < t.size() && t[p] == ',') { p++; continue; }
        if (p < t.size() && t[p] == ']') return true;
        return false;
    }
}

struct ClientCtxParams {
    int logRing = 15, secure = 0, extraDepth = 9, lbA = 3, lbB = 3, scaleBits = 59, numDigits = 0;
};

// Replicates gpu_real_model.cu's context policy exactly (UNIFORM_TERNARY,
// depth = 10 + 19 + extraDepth, scale 59 / first 60, optional
// SetNumLargeDigits, FLEXIBLEAUTO). --num-digits N mirrors the harness's
// `if (numDigits > 0) parameters.SetNumLargeDigits(numDigits)`: the hybrid
// key-switch digit count fixes the key-switch key layout, so a client/pod
// disagreement here makes every key unloadable, not merely slow.
static CryptoContext<DCRTPoly> makeClientContext(const ClientCtxParams& P, uint32_t& depthOut) {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    const uint32_t depth = 10 + 19 + (uint32_t)P.extraDepth;
    if (P.secure) parameters.SetSecurityLevel(HEStd_128_classic);
    else { parameters.SetSecurityLevel(HEStd_NotSet); parameters.SetRingDim(1u << P.logRing); }
    parameters.SetScalingModSize((uint32_t)P.scaleBits);
    parameters.SetFirstModSize((uint32_t)std::min(60, P.scaleBits + 5));
    if (P.numDigits > 0) parameters.SetNumLargeDigits((uint32_t)P.numDigits);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(depth);
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE); cc->Enable(FHE);
    depthOut = depth;
    return cc;
}

// Block-layout (RM = 1) matvec rotation AMOUNTS as gpu_real_model.cu builds
// them: babies 1..BS-1, giants BS..Dpad-BS step BS, the power-of-two
// reduction tree, and (lanes) the T1c keys -Dpad and -2^k. Sorted, unique.
static std::vector<int32_t> blockLayoutRots(int Dpad, int lanes) {
    const int BS = (int)std::ceil(std::sqrt((double)Dpad));
    std::vector<int32_t> rots;
    for (int j = 1; j < BS; j++) rots.push_back(j);
    for (int g = 0; g < Dpad; g += BS) if (g) rots.push_back(g);
    for (int pw = 1; pw < Dpad; pw <<= 1) rots.push_back(pw);
    if (lanes) { rots.push_back(-Dpad);
                 for (int pw = 1; pw < Dpad; pw <<= 1) rots.push_back(-pw); }
    std::sort(rots.begin(), rots.end());
    rots.erase(std::unique(rots.begin(), rots.end()), rots.end());
    return rots;
}

// The FIDESlib .dev sidecar (api/Serialize.cpp text format). RotationIndexes
// are rotation AMOUNTS: FIDESlib's LoadContext calls
// GetRotationKeySwitchKey(pk, step) for each listed step and looks the step's
// automorphism index up in the static key map; BootstrapSlots drives
// AddBootstrapPrecomputation per listed slot count.
static bool writeDevSidecar(const std::string& outDir, const std::vector<int32_t>& rots,
                            const std::vector<uint32_t>& bootSlots) {
    std::ofstream dv(outDir + "/cryptocontext.bin.dev", std::ios::binary);
    dv << "1 { 0 }\n" << "AutoLoadCiphertexts: 1\n" << "AutoLoadPlaintexts: 0\n";
    dv << "RotationIndexes: { ";
    for (auto r : rots) dv << r << " ";
    dv << "}\n" << "KeyDist: 1\n";
    dv << "BootstrapSlots: { ";
    for (auto s : bootSlots) dv << s << " ";
    dv << "}\n";
    return dv.good();
}

// cryptocontext.bin, public.key, secret.key, evalmult.bin -- everything that
// is not an automorphism key.
static bool writeBaseArtifacts(const std::string& outDir, CryptoContext<DCRTPoly> cc,
                               const KeyPair<DCRTPoly>& keys) {
    bool ok = true;
    ok = ok && Serial::SerializeToFile(outDir + "/cryptocontext.bin", cc, SerType::BINARY);
    ok = ok && Serial::SerializeToFile(outDir + "/public.key", keys.publicKey, SerType::BINARY);
    ok = ok && Serial::SerializeToFile(outDir + "/secret.key", keys.secretKey, SerType::BINARY);
    { std::ofstream em(outDir + "/evalmult.bin", std::ios::binary);
      ok = ok && em.good() && cc->SerializeEvalMultKey(em, SerType::BINARY); }
    return ok;
}

// Batched, RAM-bounded minting of an EXPLICIT automorphism-index set into
// OUTDIR/evalrot.partNNN.bin (NNN zero-padded from 000).
//
// Why (2026-09-02): at ring 2^17 one automorphism key is ~336 MB and the pod
// loads every key it finds, so the Mac must mint exactly the pod's index set
// and no superset; and the full 2^17 set (~48 GB serialized) does not fit in
// the Mac's 17 GB, so keys are minted `batch` at a time, written, and freed.
//
// Per batch: cc->EvalAutomorphismKeyGen(sk, chunk) -- which inserts the keys
// into CryptoContextImpl's static s_evalAutomorphismKeyMap under sk's tag
// (cryptocontext.h EvalAutomorphismKeyGen -> InsertEvalAutomorphismKey);
// SerializeEvalAutomorphismKey(os, BINARY, keyTag, chunk) -- the subset form,
// which pulls exactly `chunk` (AUTOMORPHISM indices, uint32) out of that map
// via GetPartialEvalAutomorphismKeyMapPtr and throws if one is missing; then
// ClearEvalAutomorphismKeys(keyTag) so the batch's RAM is released. The pod
// merges the parts because DeserializeEvalAutomorphismKey(istream) ->
// InsertEvalAutomorphismKey adds only indices the tag's map does not yet
// hold (cryptocontext.cpp InsertEvalAutomorphismKey, both vendored sources).
//
// The conjugation key (index M-1) needs no special path: ConjugateKeyGen and
// EvalAutomorphismKeyGen({M-1}) both build PrecomputeAutoMap(N, 2N-1),
// AutomorphismTransform(2N-1) of the secret and KeySwitchGen on it (M-1 is
// its own inverse mod M); they differ only in fresh key-switch randomness.
// keyparts_test.cpp proves the minted M-1 key conjugates and bootstraps.
//
// Even indices are rejected: the only even entries EvalBootstrapKeyGen ever
// makes are the SPARSE_ENCAPSULATED switch keys M-2/M-4, which are not
// automorphism keys and cannot be minted here (this client is UNIFORM_TERNARY).
struct MintResult { size_t parts = 0, keys = 0; long long bytes = 0; bool ok = false; std::string err; };
static MintResult mintKeyParts(CryptoContext<DCRTPoly> cc, const PrivateKey<DCRTPoly>& sk,
                               std::vector<uint32_t> autoIdx, int batch, const std::string& outDir) {
    MintResult R;
    if (batch < 1) { R.err = "--batch must be >= 1"; return R; }
    std::sort(autoIdx.begin(), autoIdx.end());
    autoIdx.erase(std::unique(autoIdx.begin(), autoIdx.end()), autoIdx.end());
    if (autoIdx.empty()) { R.err = "empty automorphism index set"; return R; }
    const uint32_t M = cc->GetCyclotomicOrder();
    for (auto i : autoIdx)
        if (i == 0 || i >= M || (i & 1u) == 0) {
            R.err = "automorphism index " + std::to_string(i) + " is not an odd integer in [1, M-1] (M=" +
                    std::to_string(M) + ")";
            return R;
        }
    const std::string keyTag = sk->GetKeyTag();
    // The tag's map must hold nothing but the current batch: the part files
    // are the only output, and a stale map would keep freed batches alive.
    CryptoContextImpl<DCRTPoly>::ClearEvalAutomorphismKeys(keyTag);
    for (size_t b = 0; b * (size_t)batch < autoIdx.size(); b++) {
        const size_t lo = b * (size_t)batch, hi = std::min(autoIdx.size(), lo + (size_t)batch);
        const std::vector<uint32_t> chunk(autoIdx.begin() + lo, autoIdx.begin() + hi);
        cc->EvalAutomorphismKeyGen(sk, chunk);   // return value discarded on purpose: the map owns them
        char name[64];
        snprintf(name, sizeof name, "/evalrot.part%03zu.bin", b);
        const std::string path = outDir + name;
        long long bytes = -1;
        {
            std::ofstream os(path, std::ios::binary);
            if (!os.good() || !cc->SerializeEvalAutomorphismKey(os, SerType::BINARY, keyTag, chunk)) {
                R.err = "part write failed: " + path; return R;
            }
            os.flush();
            bytes = (long long)os.tellp();
            if (!os.good()) { R.err = "part flush failed: " + path; return R; }
        }
        CryptoContextImpl<DCRTPoly>::ClearEvalAutomorphismKeys(keyTag);
        R.parts++; R.keys += chunk.size(); R.bytes += bytes;
        fprintf(stderr, "{\"keygen\":\"part\",\"file\":\"%s\",\"keys\":%zu,\"bytes\":%lld,"
                "\"firstIndex\":%u,\"lastIndex\":%u}\n",
                path.c_str(), chunk.size(), bytes, chunk.front(), chunk.back());
    }
    R.ok = true;
    return R;
}

// S3.7 V4 (2026-09-05, S3.6 T12 §2.3): the seeded twin of mintKeyParts. Same
// validation, same batching, same part numbering, but every key is built by
// seeded_keys.hpp::makeSeededAutomorphismKey -- a_j expanded from a fresh
// 32-byte /dev/urandom seed, b_j computed with the library's own formula and
// noise -- and written to OUTDIR/evalrot.partNNN.seeded.bin (the raw b limbs +
// the seed; no a limbs, no cereal). At 2^17 that is 176,160,804 B per key
// instead of 352,332,820 B: the 50.74 GB set becomes 25.4 GB on disk and on
// the wire. The pod loader (gpu_real_model.cu --keys-load) re-expands a_j on
// the host before FIDESlib's LoadContext; VRAM is unchanged. Never touches the
// lbcrypto static key map, so nothing lingers between batches.
// seeded_keys_test.cpp proves at 2^12 that a seeded key rotates, conjugates and
// bootstraps like a stock key for the same secret and index (same noise class).
static MintResult mintSeededKeyParts(CryptoContext<DCRTPoly> cc, const PrivateKey<DCRTPoly>& sk,
                                     std::vector<uint32_t> autoIdx, int batch, const std::string& outDir) {
    using namespace fhe_ssm::seeded;
    MintResult R;
    if (batch < 1) { R.err = "--batch must be >= 1"; return R; }
    std::sort(autoIdx.begin(), autoIdx.end());
    autoIdx.erase(std::unique(autoIdx.begin(), autoIdx.end()), autoIdx.end());
    if (autoIdx.empty()) { R.err = "empty automorphism index set"; return R; }
    const uint32_t M = cc->GetCyclotomicOrder();
    for (auto i : autoIdx)
        if (i == 0 || i >= M || (i & 1u) == 0) {
            R.err = "automorphism index " + std::to_string(i) + " is not an odd integer in [1, M-1] (M=" +
                    std::to_string(M) + ")";
            return R;
        }
    const std::string keyTag = sk->GetKeyTag();
    for (size_t b = 0; b * (size_t)batch < autoIdx.size(); b++) {
        const size_t lo = b * (size_t)batch, hi = std::min(autoIdx.size(), lo + (size_t)batch);
        std::vector<SeededKeyRecord> recs;
        recs.reserve(hi - lo);
        for (size_t k = lo; k < hi; k++) {
            SeededKeyRecord r;
            r.index = autoIdx[k];
            if (!osRandomSeed(r.seed)) { R.err = "/dev/urandom unavailable"; return R; }
            r.key = makeSeededAutomorphismKey(cc, sk, r.index, r.seed);
            recs.push_back(std::move(r));
        }
        char name[64];
        snprintf(name, sizeof name, "/evalrot.part%03zu.seeded.bin", b);
        const std::string path = outDir + name;
        std::string err;
        if (!writeSeededPart(path, cc, keyTag, recs, &err)) { R.err = "seeded part write failed: " + err; return R; }
        long long bytes = -1;
        { std::ifstream f(path, std::ios::binary | std::ios::ate); bytes = (long long)f.tellg(); }
        R.parts++; R.keys += recs.size(); R.bytes += bytes;
        fprintf(stderr, "{\"keygen\":\"part\",\"seeded\":true,\"file\":\"%s\",\"keys\":%zu,\"bytes\":%lld,"
                "\"firstIndex\":%u,\"lastIndex\":%u}\n",
                path.c_str(), recs.size(), bytes, recs.front().index, recs.back().index);
    }
    R.ok = true;
    return R;
}

// keygen: FULL client-side key generation (decision 2026-09-01) — the
// server NEVER sees the secret key. Three modes:
//   default        rotation set + EvalBootstrapKeyGen (+ --extra-autos) into
//                  evalrot.bin, exactly as before 2026-09-02;
//   --indices F    mint EXACTLY the automorphism indices the pod dumped into
//                  F, --batch N (default 16) keys per evalrot.partNNN.bin; no
//                  evalrot.bin is written; .dev lists F's rotationAmounts and
//                  bootstrapSlots. Closes the 2^15 six-index gap by
//                  construction and bounds RAM at 2^17;
//   --minimal 1    cryptocontext.bin(+.dev, empty lists), public.key,
//                  secret.key, evalmult.bin only -- fast local dry runs.
static int do_keygen(int argc, char** argv) {
    const std::string outDir = arg(argc, argv, "--out");
    ClientCtxParams P;
    P.logRing = std::stoi(arg(argc, argv, "--log-ring", "15"));
    P.secure = std::stoi(arg(argc, argv, "--secure", "0"));
    P.extraDepth = std::stoi(arg(argc, argv, "--extra-depth", "9"));
    P.lbA = std::stoi(arg(argc, argv, "--lb-a", "3"));
    P.lbB = std::stoi(arg(argc, argv, "--lb-b", "3"));
    P.scaleBits = std::stoi(arg(argc, argv, "--scale-bits", "59"));
    P.numDigits = std::stoi(arg(argc, argv, "--num-digits", "0"));
    const int Dpad = std::stoi(arg(argc, argv, "--dpad", "1024"));
    const int lanes = std::stoi(arg(argc, argv, "--lanes-block", "1"));
    const std::string indicesPath = arg(argc, argv, "--indices", "-");
    const int batch = std::stoi(arg(argc, argv, "--batch", "16"));
    const int minimal = std::stoi(arg(argc, argv, "--minimal", "0"));
    // S3.7 V4 (2026-09-05): --seeded 1 mints evalrot.partNNN.seeded.bin
    // (seed + b limbs, half the bytes) instead of evalrot.partNNN.bin. Default 0
    // = today's stock parts, untouched. Only the --indices route has parts, so
    // the flag is refused elsewhere rather than silently ignored.
    const int seeded = std::stoi(arg(argc, argv, "--seeded", "0"));
    if (batch < 1) { fprintf(stderr, "{\"fatal\":\"--batch must be >= 1\"}\n"); return 2; }
    if (minimal && indicesPath != "-") {
        fprintf(stderr, "{\"fatal\":\"--minimal 1 and --indices are exclusive\"}\n"); return 2;
    }
    if (seeded && indicesPath == "-") {
        fprintf(stderr, "{\"fatal\":\"--seeded 1 needs --indices FILE (the part-file route)\"}\n"); return 2;
    }
    std::vector<uint32_t> levelBudget = {(uint32_t)P.lbA, (uint32_t)P.lbB};
    uint32_t depth = 0;
    CryptoContext<DCRTPoly> cc = makeClientContext(P, depth);
    const uint32_t SLOTS = cc->GetRingDimension() / 2;
    fprintf(stderr, "{\"keygen\":\"context\",\"ringDim\":%u,\"depth\":%u,\"numDigits\":%d}\n",
            cc->GetRingDimension(), depth, P.numDigits);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    if (minimal) {
        const bool ok = writeBaseArtifacts(outDir, cc, keys) && writeDevSidecar(outDir, {}, {});
        printf("{\"macKeygen\":%s,\"mode\":\"minimal\",\"outDir\":\"%s\",\"ringDim\":%u,\"depth\":%u,"
               "\"rots\":0,\"autoKeys\":0,\"slots\":%u}\n",
               ok ? "true" : "false", outDir.c_str(), cc->GetRingDimension(), depth, SLOTS);
        return ok ? 0 : 2;
    }

    if (indicesPath != "-") {
        std::ifstream jf(indicesPath);
        if (!jf) { fprintf(stderr, "{\"fatal\":\"--indices %s unreadable\"}\n", indicesPath.c_str()); return 2; }
        const std::string text((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
        long long ring = -1, slots = -1, jdepth = -1, bootSlots = -1;
        std::vector<long long> rotAmt, autoIdx;
        const char* missing = nullptr;
        if (!jsonInt(text, "ring", ring)) missing = "ring";
        else if (!jsonInt(text, "slots", slots)) missing = "slots";
        else if (!jsonInt(text, "depth", jdepth)) missing = "depth";
        else if (!jsonInt(text, "bootstrapSlots", bootSlots)) missing = "bootstrapSlots";
        else if (!jsonIntList(text, "rotationAmounts", rotAmt)) missing = "rotationAmounts";
        else if (!jsonIntList(text, "autoIndices", autoIdx)) missing = "autoIndices";
        if (missing) {
            fprintf(stderr, "{\"fatal\":\"--indices %s: field %s missing or malformed\"}\n",
                    indicesPath.c_str(), missing); return 2;
        }
        // The contract must describe THIS context, or the keys are for a
        // different ring/modulus chain and the pod would load garbage.
        if (ring != (long long)cc->GetRingDimension() || slots != (long long)SLOTS || jdepth != (long long)depth) {
            fprintf(stderr, "{\"fatal\":\"--indices contract mismatch\",\"expected\":{\"ring\":%u,\"slots\":%u,"
                    "\"depth\":%u},\"got\":{\"ring\":%lld,\"slots\":%lld,\"depth\":%lld}}\n",
                    cc->GetRingDimension(), SLOTS, depth, ring, slots, jdepth); return 2;
        }
        if (autoIdx.empty()) { fprintf(stderr, "{\"fatal\":\"--indices: autoIndices is empty\"}\n"); return 2; }
        const uint32_t M = cc->GetCyclotomicOrder();
        std::vector<uint32_t> autos; autos.reserve(autoIdx.size());
        for (auto v : autoIdx) {
            if (v < 1 || v >= (long long)M) {
                fprintf(stderr, "{\"fatal\":\"--indices: autoIndex %lld outside [1, M-1]\"}\n", v); return 2;
            }
            autos.push_back((uint32_t)v);
        }
        std::sort(autos.begin(), autos.end());
        autos.erase(std::unique(autos.begin(), autos.end()), autos.end());
        std::vector<int32_t> rots; rots.reserve(rotAmt.size());
        for (auto v : rotAmt) {
            if (v == 0 || v <= -(long long)SLOTS || v >= (long long)SLOTS) {
                fprintf(stderr, "{\"fatal\":\"--indices: rotation amount %lld outside (-slots, slots)\"}\n", v); return 2;
            }
            rots.push_back((int32_t)v);
            // A listed rotation amount whose automorphism index is not in the
            // contract is exactly the map::at failure the --extra-autos note
            // describes; refuse to mint an inconsistent set.
            const uint32_t ai = cc->FindAutomorphismIndex((uint32_t)(int32_t)v);
            if (!std::binary_search(autos.begin(), autos.end(), ai)) {
                fprintf(stderr, "{\"fatal\":\"--indices: rotation amount %lld needs automorphism index %u, "
                        "which autoIndices does not list\"}\n", v, ai); return 2;
            }
        }
        std::sort(rots.begin(), rots.end());
        rots.erase(std::unique(rots.begin(), rots.end()), rots.end());
        fprintf(stderr, "{\"keygen\":\"indices\",\"file\":\"%s\",\"rots\":%zu,\"autoKeys\":%zu,\"batch\":%d,"
                "\"bootstrapSlots\":%lld}\n", indicesPath.c_str(), rots.size(), autos.size(), batch, bootSlots);
        // Small artifacts first: a multi-hour 2^17 run that dies mid-mint
        // still leaves context/keys/evalmult on disk.
        bool ok = writeBaseArtifacts(outDir, cc, keys);
        if (!ok) { fprintf(stderr, "{\"fatal\":\"base artifact write failed in %s\"}\n", outDir.c_str()); return 2; }
        // REVIEW 2026-09-03: stale evalrot.bin / evalrot.partNNN.bin from an
        // earlier mint (more parts, or a default-mode keygen) were left in
        // outDir, uploaded, and deserialized on the pod under a dead key tag.
        // The part files are the ONLY rotation-key output of this mode, so
        // anything matching is removed first, loudly.
        {
            size_t removed = 0;
            std::error_code ec;
            for (const auto& de : std::filesystem::directory_iterator(outDir, ec)) {
                const std::string fn = de.path().filename().string();
                const bool part = fn.rfind("evalrot.part", 0) == 0 && fn.size() > 4 && fn.substr(fn.size() - 4) == ".bin";
                if (fn == "evalrot.bin" || part) { std::filesystem::remove(de.path(), ec); removed++; }
            }
            fprintf(stderr, "{\"keygen\":\"cleanedStaleRotFiles\",\"removed\":%zu}\n", removed);
        }
        MintResult R;
        try {
            R = seeded ? mintSeededKeyParts(cc, keys.secretKey, autos, batch, outDir)
                       : mintKeyParts(cc, keys.secretKey, autos, batch, outDir);
        } catch (const std::exception& e) {
            fprintf(stderr, "{\"fatal\":\"mint threw\",\"what\":\"%s\"}\n", e.what()); return 2;
        }
        if (!R.ok) { fprintf(stderr, "{\"fatal\":\"mint failed\",\"why\":\"%s\"}\n", R.err.c_str()); return 2; }
        ok = ok && writeDevSidecar(outDir, rots, {(uint32_t)bootSlots});
        // "seeded"/"partSuffix" are ADDITIVE (S3.7 V4); every older field keeps its meaning.
        printf("{\"macKeygen\":%s,\"mode\":\"indices\",\"outDir\":\"%s\",\"ringDim\":%u,\"depth\":%u,"
               "\"slots\":%u,\"rots\":%zu,\"autoKeys\":%zu,\"parts\":%zu,\"batch\":%d,\"partBytes\":%lld,"
               "\"bootstrapSlots\":%lld,\"indices\":\"%s\",\"evalrotBin\":false,\"seeded\":%s,\"partSuffix\":\"%s\"}\n",
               ok ? "true" : "false", outDir.c_str(), cc->GetRingDimension(), depth, SLOTS, rots.size(),
               R.keys, R.parts, batch, R.bytes, bootSlots, indicesPath.c_str(), seeded ? "true" : "false",
               seeded ? ".seeded.bin" : ".bin");
        return ok ? 0 : 2;
    }

    // ---- default path: unchanged behaviour (evalrot.bin with the full set) --
    std::vector<int32_t> rots = blockLayoutRots(Dpad, lanes);   // block layout: RM = 1
    cc->EvalRotateKeyGen(keys.secretKey, rots);
    fprintf(stderr, "{\"keygen\":\"rots\",\"count\":%zu}\n", rots.size());
    cc->EvalBootstrapSetup(levelBudget, {0, 0}, SLOTS);
    cc->EvalBootstrapKeyGen(keys.secretKey, SLOTS);
    fprintf(stderr, "{\"keygen\":\"bootstrapKeys\":true}\n");
    // --extra-autos a,b,c : mint keys for additional AUTOMORPHISM indices.
    //
    // Why this exists (2026-09-01). A Mac-generated set aborted the CUDA
    // harness with std::out_of_range: map::at. All 73 ROTATION keys matched;
    // the BOOTSTRAP sets differed (pod 29, Mac 79, 23 shared) because
    // OpenFHE's automatic giant-step selection differs between builds, and the
    // Mac set was NOT a superset -- it lacked six indices the server required.
    // Minting exactly those and merging fixed it: 152 -> 158 keys, and the
    // harness selftest then returned hardFail 0 with the same profile as the
    // server's own keys (encdec 9.26e-13, rotate 3.28e-11, matvec 3.25e-12,
    // bootstrap 2.74e-4). The 56 surplus Mac keys are harmless.
    //
    // The list is NOT hardcoded because it is ring-specific: 7233 11713 15681
    // 24129 36033 52929 are the gap at 2^15 and say nothing about 2^17.
    // Discover it per ring by enumerating both key sets and diffing (the
    // keycount probe (keycount.cpp) prints the index set), then
    // pass the difference here. Key content for an index is fixed by
    // (secret key, index, parameters), not by which build mints it.
    //
    // 2026-09-02: --indices FILE supersedes this for the pod route -- the pod
    // dumps its exact set and the gap cannot exist. --extra-autos stays for
    // the default path.
    {
        const std::string extra = arg(argc, argv, "--extra-autos", "-");
        if (extra != "-") {
            std::vector<uint32_t> idx;
            size_t p = 0;
            while (p < extra.size()) {
                size_t c = extra.find(',', p);
                if (c == std::string::npos) c = extra.size();
                const std::string tok = extra.substr(p, c - p);
                if (!tok.empty()) idx.push_back((uint32_t)std::stoul(tok));
                p = c + 1;
            }
            if (!idx.empty()) {
                cc->EvalAutomorphismKeyGen(keys.secretKey, idx);
                fprintf(stderr, "{\"keygen\":\"extraAutos\",\"count\":%zu}\n", idx.size());
            }
        }
    }
    bool ok = writeBaseArtifacts(outDir, cc, keys);
    { std::ofstream ea(outDir + "/evalrot.bin", std::ios::binary);
      ok = ok && ea.good() && cc->SerializeEvalAutomorphismKey(ea, SerType::BINARY); }
    ok = ok && writeDevSidecar(outDir, rots, {SLOTS});
    printf("{\"macKeygen\":%s,\"outDir\":\"%s\",\"ringDim\":%u,\"rots\":%zu,\"slots\":%u}\n",
           ok ? "true" : "false", outDir.c_str(), cc->GetRingDimension(), rots.size(), SLOTS);
    return ok ? 0 : 2;
}

#ifndef MAC_FHE_CLIENT_NO_MAIN
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: mac_fhe_client keygen|enc|dec ...\n"); return 2; }
    const std::string mode = argv[1];
    if (mode == "keygen") return do_keygen(argc, argv);
    const std::string ctxPath = arg(argc, argv, "--ctx");

    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(ctxPath, cc, SerType::BINARY)) {
        fprintf(stderr, "{\"fatal\":\"context load failed: %s\"}\n", ctxPath.c_str());
        return 2;
    }
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE); cc->Enable(FHE);
    const uint32_t ringDim = cc->GetRingDimension();
    const uint32_t SLOTS = ringDim / 2;
    const int Dpad = std::stoi(arg(argc, argv, "--dpad", "1024"));
    const uint32_t REP = SLOTS / (uint32_t)Dpad;
    const int d = std::stoi(arg(argc, argv, "--d", "1024"));
    const int lanes = std::stoi(arg(argc, argv, "--lanes", "1"));
    // AUDIT FIX 2026-09-02: lanes ride replicas (lane = rep % lanes), so more
    // lanes than replicas cannot be packed. enc used to drop lanes >= REP
    // silently while the meta still stamped lanes:N -- dec then read past the
    // decrypted vector. Mirrors gpu_real_model.cu's dec-out guard.
    if (REP == 0 || lanes < 1 || (uint32_t)lanes > REP || (size_t)lanes * Dpad > SLOTS) {
        fprintf(stderr, "{\"fatal\":\"--lanes %d does not fit: REP %u replicas of Dpad %d in %u slots\"}\n",
                lanes, REP, Dpad, SLOTS); return 2;
    }

    if (mode == "enc") {
        const std::string rowsPath = arg(argc, argv, "--rows");
        const int T = std::stoi(arg(argc, argv, "--tokens"));
        const std::string out = arg(argc, argv, "--out");
        PublicKey<DCRTPoly> pk;
        if (!Serial::DeserializeFromFile(arg(argc, argv, "--pub"), pk, SerType::BINARY)) {
            fprintf(stderr, "{\"fatal\":\"public key load failed\"}\n"); return 2;
        }
        std::ifstream rf(rowsPath, std::ios::binary | std::ios::ate);
        const long long need = (long long)lanes * T * d * 8;
        if (!rf || (long long)rf.tellg() < need) {
            fprintf(stderr, "{\"fatal\":\"rows short: need %lld bytes\"}\n", need); return 2;
        }
        rf.seekg(0);
        std::vector<double> rows((size_t)lanes * T * d);
        rf.read(reinterpret_cast<char*>(rows.data()), need);
        long long bytes = 0;
        for (int t = 0; t < T; t++) {
            std::vector<double> sv(SLOTS, 0.0);
            for (uint32_t rep = 0; rep < REP; rep++) {
                const int lane = (int)(rep % (uint32_t)lanes);
                const double* rw = rows.data() + ((size_t)lane * T + t) * d;
                for (int i = 0; i < d; i++) sv[(size_t)rep * Dpad + i] = rw[i];
            }
            auto pt = cc->MakeCKKSPackedPlaintext(sv);
            auto ct = cc->Encrypt(pk, pt);
            const std::string p = out + "." + std::to_string(t);
            if (!Serial::SerializeToFile(p, ct, SerType::BINARY)) {
                fprintf(stderr, "{\"fatal\":\"ct write failed: %s\"}\n", p.c_str()); return 2;
            }
            std::ifstream f(p, std::ios::binary | std::ios::ate); bytes += (long long)f.tellg();
        }
        std::ofstream mf(out + ".meta");
        mf << "{\"count\":" << T << ",\"packTokens\":1,\"tokensReal\":" << T
           << ",\"dpad\":" << Dpad << ",\"interleave\":0,\"lanes\":" << lanes
           << ",\"ringDim\":" << ringDim << ",\"level\":0}\n";
        printf("{\"macEnc\":true,\"tokens\":%d,\"lanes\":%d,\"totalBytes\":%lld,\"ringDim\":%u}\n",
               T, lanes, bytes, ringDim);
        return 0;
    }
    if (mode == "dec") {
        const std::string in = arg(argc, argv, "--in");
        const int n = std::stoi(arg(argc, argv, "--count"));
        const std::string out = arg(argc, argv, "--out");
        // F63: validate against the producer's own meta before touching a
        // single ciphertext. --require-meta PATH makes this mandatory; without
        // it we still check {in}.meta when one happens to be there.
        std::string metaPath = arg(argc, argv, "--require-meta", "-");
        // AUDIT FIX 2026-09-02: the sidecar is ALWAYS required. The previous
        // opt-in (warn-only when --require-meta was absent) left the exact
        // F62/F63 layout hole open for any caller that forgot the flag.
        if (metaPath == "-") metaPath = in + ".meta";
        const bool metaRequired = true;
        {
            std::ifstream mf(metaPath);
            if (!mf) {
                if (metaRequired) {
                    fprintf(stderr, "{\"fatal\":\"--require-meta given but %s is "
                            "missing; refusing to guess the layout of a reply\"}\n",
                            metaPath.c_str());
                    return 2;
                }
                fprintf(stderr, "{\"warn\":\"no meta beside %s; layout unverified\"}\n",
                        in.c_str());
            } else {
                std::string m((std::istreambuf_iterator<char>(mf)),
                              std::istreambuf_iterator<char>());
                struct { const char* k; long long want; } req[] = {
                    {"lanes", lanes}, {"count", n}, {"dpad", Dpad},
                    {"ringDim", (long long)ringDim}, {"packTokens", 1},
                    {"interleave", 0},
                };
                bool bad = false;
                for (auto& c : req) {
                    const long long got = metaInt(m, c.k, -1);
                    if (got != c.want) {
                        fprintf(stderr, "{\"fatal\":\"meta mismatch\",\"field\":\"%s\","
                                "\"expected\":%lld,\"got\":%lld,\"meta\":\"%s\"}\n",
                                c.k, c.want, got, metaPath.c_str());
                        bad = true;
                    }
                }
                if (bad) {
                    fprintf(stderr, "{\"fatal\":\"refusing to decrypt: the reply's "
                            "layout is not the one this client packed. Slicing it "
                            "anyway would produce plausible hidden vectors read at "
                            "the wrong stride (F62/F63).\"}\n");
                    return 2;
                }
            }
        }
        PrivateKey<DCRTPoly> sk;
        if (!Serial::DeserializeFromFile(arg(argc, argv, "--sec"), sk, SerType::BINARY)) {
            fprintf(stderr, "{\"fatal\":\"secret key load failed\"}\n"); return 2;
        }
        std::ofstream of(out, std::ios::binary);
        for (int t = 0; t < n; t++) {
            Ciphertext<DCRTPoly> ct;
            if (!Serial::DeserializeFromFile(in + "." + std::to_string(t), ct, SerType::BINARY)) {
                fprintf(stderr, "{\"fatal\":\"ct load failed: %s.%d\"}\n", in.c_str(), t); return 2;
            }
            Plaintext pt;
            cc->Decrypt(sk, ct, &pt);
            pt->SetLength(SLOTS);
            auto v = pt->GetRealPackedValue();
            for (int lane = 0; lane < lanes; lane++) {
                of.write(reinterpret_cast<const char*>(v.data() + (size_t)lane * Dpad),
                         (std::streamsize)(d * 8));
            }
            printf("{\"macDec\":true,\"token\":%d,\"lanes\":%d}\n", t, lanes);
        }
        printf("{\"macDecDone\":true,\"rows\":%d,\"d\":%d,\"out\":\"%s\"}\n",
               n * lanes, d, out.c_str());
        return 0;
    }
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
#endif  // MAC_FHE_CLIENT_NO_MAIN
