// seeded_keys.hpp — seed-expanded hybrid key-switching keys (S3.7 V4, 2026-09-05;
// finding S3.6 T12 §2.3). Host-only: plain C++17 + the OpenFHE 1.5.1 PUBLIC API,
// no CUDA, no library patch. Included by mac_fhe_client.cpp (the Mac keygen),
// seeded_keys_test.cpp (the 2^12 proof) and, under FHE_SSM_DEMO_SER, by
// gpu_real_model.cu (the pod loader, --keys-load).
//
// WHY (T12 §2.3). A hybrid key-switching key for automorphism k is numPartQ
// digits of (a_j, b_j) over Q·P. The a_j are UNIFORM in R_QP and carry no secret
// (keyswitch-hybrid.cpp:96-:125 in both vendored 1.5.1 trees: `DugType dug` at
// :96, `DCRTPoly(dug, paramsQP, Format::EVALUATION)` at :100), so a 32-byte
// seed can stand in for them on the wire and on disk. At N = 2^17 that is
// 3 x 56 x 2^17 x 8 = 176,160,768 B -> 32 B per key: the 50.74 GB key set
// becomes 25.4 GB (2.000x minus the header). VRAM is unchanged: the card holds
// expanded keys; the expansion happens on the host BEFORE FIDESlib's
// LoadContext reads the lbcrypto static key map.
//
// WHY NOT the library's own PRNG. OpenFHE's Blake2Engine is publicly
// constructible with a chosen seed (utils/prng/blake2engine.h), but the
// uniform sampler that KeySwitchGen uses (DiscreteUniformGeneratorImpl::
// GenerateInteger, discreteuniformgenerator-impl.h:73-:75) draws 32-bit chunks
// from the PROCESS-GLOBAL PseudoRandomNumberGenerator::GetPRNG() (a
// threadprivate static, distributiongenerator.h:68/:75) and the public API has
// no way to point it at a per-key engine. So the expander below is
// self-contained (RFC 8439 ChaCha20 in counter mode + rejection sampling),
// which also removes the risk T12 §5 names: "a PRG mismatch between Mac and pod
// breaks every rotation" -- the bytes depend on nothing but this header.
//
// EXPANDER (normative; any reimplementation must match bit for bit):
//   key      = the 32-byte seed as 8 little-endian uint32 words;
//   nonce    = {digit j, limb i, 0x59454b53} (three uint32; the third is the
//              ASCII tag "SKEY" read little-endian);
//   counter  = 0, 1, 2, ... (one 64-byte ChaCha20 block each, 20 rounds);
//   stream   = the blocks' 16 words serialized little-endian, read as
//              little-endian uint64 words w_0, w_1, ...;
//   sampling = for coefficient k = 0..N-1 in order: take the next w; if
//              w < lim := 2^64 - (2^64 mod q) accept a[k] = w mod q, else take
//              the next w. lim is a multiple of q, so w mod q conditioned on
//              acceptance is EXACTLY uniform on Z_q; for the 59/60-bit NTT
//              primes the rejection probability is (2^64 mod q)/2^64 < 2^-4.
//   a_j is filled directly in EVALUATION format, as the library does
//   (dcrtpoly-impl.h:153-:160 sets the uniform values with the requested
//   format and performs no NTT): distribution-identical by construction.
//
// PART FILE  evalrot.partNNN.seeded.bin  (all integers little-endian):
//   char[8] magic "FSSMSKY1"; u32 version = 1; u32 expander = 1 (the scheme
//   above); u32 N; u32 sizeQ; u32 sizeP; u32 numPartQ; u32 numPerPartQ;
//   u32 count; u32 keyTagLen; keyTag bytes; u64 moduli[sizeQ + sizeP] (the
//   context's QP chain in order -- the loader refuses a file whose chain is not
//   the loaded context's); then per key: u32 automorphism index; u8 seed[32];
//   u64 b[numPartQ][sizeQ + sizeP][N] in EVALUATION format.
//   Per key at 2^17: 4 + 32 + 176,160,768 = 176,160,804 B.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "openfhe.h"

namespace fhe_ssm {
namespace seeded {

using lbcrypto::DCRTPoly;
using PolyT = DCRTPoly::PolyType;   // PolyImpl<NativeVector>
using Seed32 = std::array<uint8_t, 32>;

static constexpr char kMagic[8] = {'F', 'S', 'S', 'M', 'S', 'K', 'Y', '1'};
static constexpr uint32_t kVersion = 1;
static constexpr uint32_t kExpander = 1;
static constexpr uint32_t kNonceTag = 0x59454b53u;   // "SKEY" little-endian

// ---- RFC 8439 ChaCha20 block function -------------------------------------
struct ChaCha20 {
    static inline uint32_t rotl(uint32_t v, int c) { return (v << c) | (v >> (32 - c)); }
    static inline void qr(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
        a += b; d ^= a; d = rotl(d, 16);
        c += d; b ^= c; b = rotl(b, 12);
        a += b; d ^= a; d = rotl(d, 8);
        c += d; b ^= c; b = rotl(b, 7);
    }
    static void block(const uint32_t key[8], uint32_t counter, const uint32_t nonce[3], uint32_t out[16]) {
        uint32_t s[16] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u,
                          key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
                          counter, nonce[0], nonce[1], nonce[2]};
        uint32_t x[16];
        for (int i = 0; i < 16; i++) x[i] = s[i];
        for (int r = 0; r < 10; r++) {
            qr(x[0], x[4], x[8],  x[12]); qr(x[1], x[5], x[9],  x[13]);
            qr(x[2], x[6], x[10], x[14]); qr(x[3], x[7], x[11], x[15]);
            qr(x[0], x[5], x[10], x[15]); qr(x[1], x[6], x[11], x[12]);
            qr(x[2], x[7], x[8],  x[13]); qr(x[3], x[4], x[9],  x[14]);
        }
        for (int i = 0; i < 16; i++) out[i] = x[i] + s[i];
    }
};

// One keystream per (seed, digit, limb): 64-bit words in the order defined above.
class SeedStream {
public:
    SeedStream(const Seed32& seed, uint32_t digit, uint32_t limb) : m_pos(8), m_counter(0) {
        for (int k = 0; k < 8; k++)
            m_key[k] = (uint32_t)seed[4 * k] | ((uint32_t)seed[4 * k + 1] << 8) |
                       ((uint32_t)seed[4 * k + 2] << 16) | ((uint32_t)seed[4 * k + 3] << 24);
        m_nonce[0] = digit; m_nonce[1] = limb; m_nonce[2] = kNonceTag;
    }
    inline uint64_t next64() {
        if (m_pos == 8) { ChaCha20::block(m_key, m_counter++, m_nonce, m_w); m_pos = 0; }
        const uint64_t v = (uint64_t)m_w[2 * m_pos] | ((uint64_t)m_w[2 * m_pos + 1] << 32);
        m_pos++;
        return v;
    }
private:
    uint32_t m_key[8]; uint32_t m_nonce[3]; uint32_t m_w[16]; int m_pos; uint32_t m_counter;
};

// Exactly uniform on Z_q by rejection (see the header comment). q odd (an NTT
// prime); r == 0 (q | 2^64) would mean "accept everything" and cannot occur here.
static inline void fillUniformMod(SeedStream& ss, uint64_t q, uint64_t* out, size_t n) {
    const uint64_t r = (0 - q) % q;      // 2^64 mod q  (0 - q wraps to 2^64 - q)
    const uint64_t lim = 0 - r;          // q * floor(2^64 / q); wraps to 0 iff r == 0
    for (size_t k = 0; k < n; k++) {
        uint64_t w = ss.next64();
        while (r != 0 && w >= lim) w = ss.next64();
        out[k] = w % q;
    }
}

// a_j for digit `digit` over the QP chain, EVALUATION format, from the seed alone.
// The limb loop is embarrassingly parallel (one stream per limb); the pragma is
// a no-op in the single-threaded Mac build and active in the OpenMP pod build.
inline DCRTPoly expandA(const Seed32& seed, const std::shared_ptr<DCRTPoly::Params>& paramsQP, uint32_t digit) {
    DCRTPoly a(paramsQP, Format::EVALUATION, true);
    const auto& prs = paramsQP->GetParams();
    const int sizeQP = (int)prs.size();
#pragma omp parallel for
    for (int i = 0; i < sizeQP; i++) {
        const uint32_t N = prs[i]->GetRingDimension();
        const uint64_t q = prs[i]->GetModulus().template ConvertToInt<uint64_t>();
        std::vector<uint64_t> w(N);
        SeedStream ss(seed, digit, (uint32_t)i);
        fillUniformMod(ss, q, w.data(), N);
        lbcrypto::NativeVector v(N, prs[i]->GetModulus());
        for (uint32_t k = 0; k < N; k++) v[k] = lbcrypto::NativeInteger(w[k]);
        PolyT p(prs[i], Format::EVALUATION, true);
        p.SetValues(std::move(v), Format::EVALUATION);
        a.SetElementAtIndex(i, std::move(p));
    }
    return a;
}

// tau_{k^-1}(s): the "new" key of KeySwitchGen(sk, permuted) exactly as
// LeveledSHEBase::EvalAutomorphismKeyGen builds it (base-leveledshe.cpp:369-:375
// in both vendored trees: index = k^-1 mod M, PrecomputeAutoMap, AutomorphismTransform(index, vec)).
inline DCRTPoly permutedSecret(const DCRTPoly& s, uint32_t autoIndex) {
    const uint32_t N = s.GetRingDimension(), M = s.GetCyclotomicOrder();
    const uint32_t inv = lbcrypto::NativeInteger(autoIndex).ModInverse(M).template ConvertToInt<uint32_t>();
    std::vector<uint32_t> vec(N);
    lbcrypto::PrecomputeAutoMap(N, inv, &vec);
    return s.AutomorphismTransform(inv, vec);
}

// Basis Q -> Q·P extension of a secret, as KeySwitchHYBRID::KeySwitchGenInternal
// does it (keyswitch-hybrid.cpp:59-:84): Q limbs copied, P limbs from limb 0 in
// COEFFICIENT form through SwitchModulus, back to EVALUATION.
inline DCRTPoly extendToQP(const DCRTPoly& sNew, const std::shared_ptr<DCRTPoly::Params>& paramsQP, uint32_t sizeQ) {
    const auto& pparamsQP = paramsQP->GetParams();
    const uint32_t sizeQP = (uint32_t)pparamsQP.size();
    DCRTPoly sNewExt(paramsQP, Format::EVALUATION, true);
    auto sNew0 = sNew.GetElementAtIndex(0);
    sNew0.SetFormat(Format::COEFFICIENT);
    for (uint32_t i = 0; i < sizeQP; ++i) {
        if (i < sizeQ) {
            auto tmp = sNew.GetElementAtIndex(i);
            tmp.SetFormat(Format::EVALUATION);
            sNewExt.SetElementAtIndex(i, std::move(tmp));
        } else {
            auto tmp = sNew0;
            tmp.SwitchModulus(pparamsQP[i]->GetModulus(), pparamsQP[i]->GetRootOfUnity(),
                              lbcrypto::NativeInteger(0), lbcrypto::NativeInteger(0));
            tmp.SetFormat(Format::EVALUATION);
            sNewExt.SetElementAtIndex(i, std::move(tmp));
        }
    }
    return sNewExt;
}

struct Layout { uint32_t N = 0, sizeQ = 0, sizeP = 0, numPartQ = 0, numPerPartQ = 0; std::vector<uint64_t> moduli; };

inline std::shared_ptr<lbcrypto::CryptoParametersRNS> rnsParams(const lbcrypto::CryptoContext<DCRTPoly>& cc) {
    auto p = std::dynamic_pointer_cast<lbcrypto::CryptoParametersRNS>(cc->GetCryptoParameters());
    if (!p) throw std::runtime_error("seeded_keys: context is not an RNS scheme");
    return p;
}

inline Layout layoutOf(const lbcrypto::CryptoContext<DCRTPoly>& cc) {
    auto cp = rnsParams(cc);
    Layout L;
    const auto& prs = cp->GetParamsQP()->GetParams();
    L.sizeQ = (uint32_t)cp->GetElementParams()->GetParams().size();
    L.sizeP = (uint32_t)prs.size() - L.sizeQ;
    L.numPartQ = cp->GetNumPartQ();
    L.numPerPartQ = cp->GetNumPerPartQ();
    L.N = cc->GetRingDimension();
    for (auto& p : prs) L.moduli.push_back(p->GetModulus().template ConvertToInt<uint64_t>());
    return L;
}

// The keygen half. b_j = -a_j * tau_{k^-1}(s)_ext + ns * e_j (+ PModq_i * s_i on
// digit j's own Q limbs) -- keyswitch-hybrid.cpp:101-:117 verbatim, with a_j
// from the seed instead of the library's DUG and e_j from the SAME discrete
// Gaussian the library uses (GetDiscreteGaussianGenerator()).
inline lbcrypto::EvalKey<DCRTPoly> makeSeededAutomorphismKey(const lbcrypto::CryptoContext<DCRTPoly>& cc,
                                                             const lbcrypto::PrivateKey<DCRTPoly>& sk,
                                                             uint32_t autoIndex, const Seed32& seed) {
    auto cp = rnsParams(cc);
    const auto& paramsQP = cp->GetParamsQP();
    const uint32_t sizeQ = (uint32_t)cp->GetElementParams()->GetParams().size();
    const uint32_t sizeQP = (uint32_t)paramsQP->GetParams().size();
    const uint32_t numPerPartQ = cp->GetNumPerPartQ(), numPartQ = cp->GetNumPartQ();
    const auto& s = sk->GetPrivateElement();                       // oldKey = sk
    const DCRTPoly sNewExt = extendToQP(permutedSecret(s, autoIndex), paramsQP, sizeQ);
    const lbcrypto::NativeInteger ns(cp->GetNoiseScale());
    auto dgg = cp->GetDiscreteGaussianGenerator();
    const auto& PModq = cp->GetPModq();
    std::vector<DCRTPoly> av(numPartQ), bv(numPartQ);
    for (uint32_t part = 0; part < numPartQ; ++part) {
        DCRTPoly a = expandA(seed, paramsQP, part);
        DCRTPoly e(dgg, paramsQP, Format::EVALUATION);
        DCRTPoly b(paramsQP, Format::EVALUATION, true);
        const uint32_t startPartIdx = numPerPartQ * part;
        const uint32_t endPartIdx = (sizeQ > startPartIdx + numPerPartQ) ? (startPartIdx + numPerPartQ) : sizeQ;
        for (uint32_t i = 0; i < sizeQP; ++i) {
            const auto& ai = a.GetElementAtIndex(i);
            const auto& ei = e.GetElementAtIndex(i);
            const auto& sni = sNewExt.GetElementAtIndex(i);
            PolyT bi = ai.Negate().Times(sni).Plus(ei.Times(ns));
            if (i >= startPartIdx && i < endPartIdx) bi = bi.Plus(s.GetElementAtIndex(i).Times(PModq[i]));
            b.SetElementAtIndex(i, std::move(bi));
        }
        av[part] = std::move(a);
        bv[part] = std::move(b);
    }
    auto ek = std::make_shared<lbcrypto::EvalKeyRelinImpl<DCRTPoly>>(cc);
    ek->SetAVector(std::move(av));
    ek->SetBVector(std::move(bv));
    ek->SetKeyTag(sk->GetKeyTag());
    return ek;
}

// The loader half: a full EvalKey from (seed, b words) -- the a_j re-expanded.
inline lbcrypto::EvalKey<DCRTPoly> keyFromSeedAndB(const lbcrypto::CryptoContext<DCRTPoly>& cc, const std::string& keyTag,
                                                   const Seed32& seed, const std::vector<uint64_t>& bw) {
    auto cp = rnsParams(cc);
    const auto& paramsQP = cp->GetParamsQP();
    const auto& prs = paramsQP->GetParams();
    const uint32_t sizeQP = (uint32_t)prs.size(), numPartQ = cp->GetNumPartQ();
    const uint32_t N = cc->GetRingDimension();
    if (bw.size() != (size_t)numPartQ * sizeQP * N) throw std::runtime_error("seeded_keys: b word count mismatch");
    std::vector<DCRTPoly> av(numPartQ), bv(numPartQ);
    for (uint32_t part = 0; part < numPartQ; ++part) {
        av[part] = expandA(seed, paramsQP, part);
        DCRTPoly b(paramsQP, Format::EVALUATION, true);
        for (uint32_t i = 0; i < sizeQP; ++i) {
            const uint64_t* src = bw.data() + ((size_t)part * sizeQP + i) * N;
            lbcrypto::NativeVector v(N, prs[i]->GetModulus());
            for (uint32_t k = 0; k < N; k++) v[k] = lbcrypto::NativeInteger(src[k]);
            PolyT p(prs[i], Format::EVALUATION, true);
            p.SetValues(std::move(v), Format::EVALUATION);
            b.SetElementAtIndex(i, std::move(p));
        }
        bv[part] = std::move(b);
    }
    auto ek = std::make_shared<lbcrypto::EvalKeyRelinImpl<DCRTPoly>>(cc);
    ek->SetAVector(std::move(av));
    ek->SetBVector(std::move(bv));
    ek->SetKeyTag(keyTag);
    return ek;
}

// b_j of a key as the flat word block the part file stores.
inline void bWords(const lbcrypto::EvalKey<DCRTPoly>& ek, std::vector<uint64_t>& out) {
    const auto& bv = ek->GetBVector();
    out.clear();
    for (const auto& b : bv) {
        if (b.GetFormat() != Format::EVALUATION) throw std::runtime_error("seeded_keys: b not in EVALUATION");
        for (uint32_t i = 0; i < b.GetNumOfElements(); i++) {
            const auto& vals = b.GetElementAtIndex(i).GetValues();
            for (uint32_t k = 0; k < vals.GetLength(); k++) out.push_back(vals[k].template ConvertToInt<uint64_t>());
        }
    }
}

// ---- part file I/O ---------------------------------------------------------
namespace io {
inline void put32(std::ostream& os, uint32_t v) { uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)}; os.write((const char*)b, 4); }
inline void put64(std::ostream& os, uint64_t v) { uint8_t b[8]; for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (8 * i)); os.write((const char*)b, 8); }
inline bool get32(std::istream& is, uint32_t& v) { uint8_t b[4]; if (!is.read((char*)b, 4)) return false; v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24); return true; }
inline bool get64(std::istream& is, uint64_t& v) { uint8_t b[8]; if (!is.read((char*)b, 8)) return false; v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i); return true; }
// Bulk word I/O: the in-memory representation is written/read as-is on
// little-endian hosts (every machine this project runs on: x86-64, arm64).
inline void putWords(std::ostream& os, const std::vector<uint64_t>& w) {
    static_assert(sizeof(uint64_t) == 8, "");
    os.write(reinterpret_cast<const char*>(w.data()), (std::streamsize)(w.size() * 8));
}
inline bool getWords(std::istream& is, std::vector<uint64_t>& w, size_t n) {
    w.resize(n);
    return (bool)is.read(reinterpret_cast<char*>(w.data()), (std::streamsize)(n * 8));
}
}  // namespace io

struct SeededKeyRecord { uint32_t index = 0; Seed32 seed{}; lbcrypto::EvalKey<DCRTPoly> key; };

inline size_t bytesPerKey(const Layout& L) { return 4 + 32 + (size_t)L.numPartQ * (L.sizeQ + L.sizeP) * L.N * 8; }

inline size_t headerBytes(const Layout& L, const std::string& keyTag) {
    return 8 + 4 * 9 + keyTag.size() + 8 * L.moduli.size();
}

inline bool writeSeededPart(const std::string& path, const lbcrypto::CryptoContext<DCRTPoly>& cc, const std::string& keyTag,
                            const std::vector<SeededKeyRecord>& keys, std::string* err = nullptr) {
    const Layout L = layoutOf(cc);
    std::ofstream os(path, std::ios::binary);
    if (!os.good()) { if (err) *err = "cannot open " + path; return false; }
    os.write(kMagic, 8);
    io::put32(os, kVersion); io::put32(os, kExpander);
    io::put32(os, L.N); io::put32(os, L.sizeQ); io::put32(os, L.sizeP);
    io::put32(os, L.numPartQ); io::put32(os, L.numPerPartQ);
    io::put32(os, (uint32_t)keys.size());
    io::put32(os, (uint32_t)keyTag.size()); os.write(keyTag.data(), (std::streamsize)keyTag.size());
    for (auto m : L.moduli) io::put64(os, m);
    std::vector<uint64_t> bw;
    for (const auto& r : keys) {
        io::put32(os, r.index);
        os.write((const char*)r.seed.data(), 32);
        bWords(r.key, bw);
        if (bw.size() != (size_t)L.numPartQ * (L.sizeQ + L.sizeP) * L.N) { if (err) *err = "b word count"; return false; }
        io::putWords(os, bw);
    }
    os.flush();
    if (!os.good()) { if (err) *err = "write failed: " + path; return false; }
    return true;
}

struct PartHeader { uint32_t version = 0, expander = 0, count = 0; Layout L; std::string keyTag; };

inline bool readSeededPartHeader(std::istream& is, PartHeader& H, std::string* err = nullptr) {
    char magic[8];
    if (!is.read(magic, 8) || std::memcmp(magic, kMagic, 8) != 0) { if (err) *err = "bad magic"; return false; }
    uint32_t sizeQP = 0, tagLen = 0;
    bool ok = io::get32(is, H.version) && io::get32(is, H.expander) && io::get32(is, H.L.N) && io::get32(is, H.L.sizeQ) &&
              io::get32(is, H.L.sizeP) && io::get32(is, H.L.numPartQ) && io::get32(is, H.L.numPerPartQ) &&
              io::get32(is, H.count) && io::get32(is, tagLen);
    if (!ok) { if (err) *err = "short header"; return false; }
    if (H.version != kVersion || H.expander != kExpander) { if (err) *err = "unsupported version/expander"; return false; }
    if (tagLen > 4096) { if (err) *err = "keyTag too long"; return false; }
    H.keyTag.assign(tagLen, '\0');
    if (tagLen && !is.read(&H.keyTag[0], tagLen)) { if (err) *err = "short keyTag"; return false; }
    sizeQP = H.L.sizeQ + H.L.sizeP;
    H.L.moduli.resize(sizeQP);
    for (uint32_t i = 0; i < sizeQP; i++) if (!io::get64(is, H.L.moduli[i])) { if (err) *err = "short moduli"; return false; }
    return true;
}

inline bool layoutMatches(const Layout& a, const Layout& b) {
    return a.N == b.N && a.sizeQ == b.sizeQ && a.sizeP == b.sizeP && a.numPartQ == b.numPartQ &&
           a.numPerPartQ == b.numPerPartQ && a.moduli == b.moduli;
}

struct ExpandStats { size_t keys = 0; size_t inserted = 0; std::string keyTag; long long bytes = 0; };

// Re-expand every key of one part file and insert the full EvalKeys into the
// lbcrypto static automorphism-key map under the file's key tag -- the same
// map DeserializeEvalAutomorphismKey(istream) fills for stock parts and the
// map FIDESlib's LoadContext reads (gpu_real_model.cu --keys-load). Like the
// stock path, InsertEvalAutomorphismKey adds only indices the tag does not
// hold yet. Keys are read one at a time (176 MiB each at 2^17).
inline bool expandSeededKeys(const lbcrypto::CryptoContext<DCRTPoly>& cc, const std::string& path, ExpandStats& st,
                             std::string* err = nullptr) {
    std::ifstream is(path, std::ios::binary);
    if (!is.good()) { if (err) *err = "cannot open " + path; return false; }
    PartHeader H;
    if (!readSeededPartHeader(is, H, err)) return false;
    const Layout L = layoutOf(cc);
    if (!layoutMatches(H.L, L)) { if (err) *err = "part file " + path + " was minted for another context (N/chain/digits differ)"; return false; }
    if (H.keyTag.empty()) { if (err) *err = "empty keyTag"; return false; }
    const size_t nWords = (size_t)L.numPartQ * (L.sizeQ + L.sizeP) * L.N;
    auto mp = std::make_shared<std::map<uint32_t, lbcrypto::EvalKey<DCRTPoly>>>();
    std::vector<uint64_t> bw;
    for (uint32_t n = 0; n < H.count; n++) {
        uint32_t index = 0; Seed32 seed{};
        if (!io::get32(is, index) || !is.read((char*)seed.data(), 32) || !io::getWords(is, bw, nWords)) {
            if (err) *err = "short key record " + std::to_string(n) + " in " + path; return false;
        }
        (*mp)[index] = keyFromSeedAndB(cc, H.keyTag, seed, bw);
        st.keys++;
    }
    st.bytes = (long long)is.tellg();
    st.keyTag = H.keyTag;
    // 2026-09-10: GetEvalAutomorphismKeyMap THROWS ("EvalAutomorphismKeys are not generated for ID") when the tag
    // has no map yet (cryptocontext.cpp:211) -- which is exactly the state of a fresh loader, or of a test that has
    // just cleared the tag. Count from zero in that case; after the insert the map exists.
    size_t before = 0;
    try { before = lbcrypto::CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(H.keyTag).size(); } catch (const std::exception&) { before = 0; }
    lbcrypto::CryptoContextImpl<DCRTPoly>::InsertEvalAutomorphismKey(mp, H.keyTag);
    st.inserted += lbcrypto::CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(H.keyTag).size() - before;
    return true;
}

// Test/audit helper (needs the secret): the key's noise polynomial
// e_j = b_j + a_j * tau_{k^-1}(s)_ext - [digit] PModq * s, centered, max |.| over
// all digits and limbs. A valid key of either mint has max |e| of a few sigma of
// the library's discrete Gaussian (sigma 3.19); a wrong convention gives ~q/2.
inline double keyNoiseMaxAbs(const lbcrypto::CryptoContext<DCRTPoly>& cc, const lbcrypto::PrivateKey<DCRTPoly>& sk,
                             uint32_t autoIndex, const lbcrypto::EvalKey<DCRTPoly>& ek) {
    auto cp = rnsParams(cc);
    const auto& paramsQP = cp->GetParamsQP();
    const uint32_t sizeQ = (uint32_t)cp->GetElementParams()->GetParams().size();
    const uint32_t numPerPartQ = cp->GetNumPerPartQ();
    const auto& s = sk->GetPrivateElement();
    const DCRTPoly sNewExt = extendToQP(permutedSecret(s, autoIndex), paramsQP, sizeQ);
    const auto& PModq = cp->GetPModq();
    const auto& av = ek->GetAVector(); const auto& bv = ek->GetBVector();
    double mx = 0;
    for (uint32_t part = 0; part < av.size(); ++part) {
        const uint32_t startPartIdx = numPerPartQ * part;
        const uint32_t endPartIdx = (sizeQ > startPartIdx + numPerPartQ) ? (startPartIdx + numPerPartQ) : sizeQ;
        for (uint32_t i = 0; i < av[part].GetNumOfElements(); ++i) {
            PolyT e = bv[part].GetElementAtIndex(i).Plus(av[part].GetElementAtIndex(i).Times(sNewExt.GetElementAtIndex(i)));
            if (i >= startPartIdx && i < endPartIdx) e = e.Minus(s.GetElementAtIndex(i).Times(PModq[i]));
            e.SetFormat(Format::COEFFICIENT);
            const auto& vals = e.GetValues();
            const uint64_t q = vals.GetModulus().template ConvertToInt<uint64_t>();
            for (uint32_t k = 0; k < vals.GetLength(); k++) {
                const uint64_t x = vals[k].template ConvertToInt<uint64_t>();
                const double c = (x > q / 2) ? -(double)(q - x) : (double)x;
                if (std::fabs(c) > mx) mx = std::fabs(c);
            }
        }
    }
    return mx;
}

// 32 bytes from the OS CSPRNG (/dev/urandom on macOS and Linux). Loud on failure.
inline bool osRandomSeed(Seed32& out) {
    std::ifstream ur("/dev/urandom", std::ios::binary);
    return ur.good() && (bool)ur.read((char*)out.data(), 32);
}

}  // namespace seeded
}  // namespace fhe_ssm
