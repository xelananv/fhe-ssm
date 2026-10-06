// periodic_encode.hpp -- S3.7 V5 (2026-09-05): the periodic store-entry encode, host-only.
//
// WHY THIS EXISTS
// ---------------
// Every compressed-store entry (`buildComp`, gpu_real_model.cu :2888-:2925) is the CKKS encoding of a
// Dpad-PERIODIC slot vector: REP = N/(2*Dpad) replicas of one period.  Such a vector encodes to a
// polynomial in X^REP, i.e. to the CKKS encoding of ONE period in the ring of degree N' = N/REP, and its
// store entry (the N/REP distinct eval-domain words per limb, `comp[j2] = limb[j2 << log2 REP]`) is the
// ring-N' NTT of that small polynomial with root psi^REP in OpenFHE's own bit-reversed order.  Proved
// (index map + exactness of OpenFHE's FFTSpecialInv on a periodic input) in S3.5 section 3.1
// (results/theory/TICK_COMPLEXITY_20260904.md) and checked bit for bit on 480/480 REAL bundle entries
// (S3.6 T01 section 3.1, results/theory/s36_20260905/t01_t02_t15_tick0/identity_h{1,2}.jsonl).  The
// dense path costs a 65,536-point FFT + l NTTs of 2^17 points + a 2^17-word raw copy + the okmap
// verify loop per entry (6.196 ms on the pod at mean l 10.79, demo2/server.log:429 / 344,064); this path
// costs a 1,024-point FFT + l NTTs of 2,048 points (0.16 ms single-thread on the Mac, T01 section 3.2)
// and produces the entry IN PLACE -- no raw copy, no verify loop, no dense plaintext.
//
// WHAT IT REPLICATES, LINE FOR LINE (OpenFHE 1.5.1, the pod's CPU library, vendor/openfhe-fideslib/openfhe-src)
//   * scale       : CryptoParametersRNS::GetScalingFactorReal(level), exactly what
//                   CryptoContextImpl::MakeCKKSPackedPlaintextInternal passes for FLEXIBLEAUTO
//                   (pke/include/cryptocontext.h :401; noiseScaleDeg 1 -> no Times(currPowP)).
//   * inverse FFT : DiscreteFourierTransform::FFTSpecialInv(vals, 2N') -- the LIBRARY's own routine and
//                   tables at the small order (core/lib/math/dftransform.cpp :209-:235).  The twiddles of
//                   the two orders coincide as doubles (S3.5 section 3.1), so the surviving block of the
//                   dense FFT IS this small FFT, bit for bit.
//   * rounding    : ckkspackedencoding.cpp :199-:283 (NATIVEINT 64): logc scan after *= scalingFactor,
//                   MAX_BITS_IN_WORD 61 (pke/include/constants-defs.h :113), approxFactor, llround,
//                   negative -> Max64BitValue() + re, then FitToNativeVector (:516-:533) == re mod q.
//   * NTT         : NumberTheoreticTransformNat::ForwardTransformToBitReverse (core/include/math/hal/intnat/
//                   transformnat-impl.h :200-:242) with the root table Table[brev(i)] = psi'^i
//                   (PreCompute :714-:745), psi' = psi^REP -- OWN tables (OpenFHE keys its NTT tables by
//                   modulus only and REGENERATES on a length mismatch, :718, so calling the library's
//                   transform at the second order inside a process that also encodes at ring N would thrash
//                   the tables -- and mutate them under any concurrent dense encode).  Shoup precomputation
//                   (w' = floor(w * 2^64 / q)) instead of the library's Barrett mu: the residues are the
//                   same integers, only the reduction differs.
//   * approxFactor: Times(crtApprox) with crtApprox = 2^logApprox mod q, applied AFTER the NTT (a scalar
//                   commutes with the NTT; same residues).  Never taken for a weight entry at Delta 2^59
//                   (|coefficient| <= max|w| * 2^59 < 2^61 for |w| < 4) but replicated so the two paths agree
//                   on every input the library accepts.
//
// WHAT IT DOES NOT DO
//   * interleaved layout (--interleave): not Dpad-periodic in the slot index; the caller must refuse.
//   * NATIVEINT 128 builds: the rounding path differs (ckkspackedencoding.cpp :140-:197); static_assert.
//   * thread the entries: `encode` is re-entrant (own Scratch per caller; the library's FFT tables are
//     read-only once both orders are Initialize()d in the constructor), so a caller MAY run it over
//     entries in parallel; this header does not spawn threads (batch 2).
//
// Include-able from gpu_real_model.cu inside its FHE_SSM_DEMO_SER block (OpenFHE headers on the path)
// and from any host program built with the Mac recipe (see V5_periodic_encode/BUILD_IDENTITY.txt).
#pragma once

#include "openfhe.h"
#include "math/dftransform.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(NATIVEINT) && NATIVEINT != 64
#error "periodic_encode.hpp replicates the NATIVEINT == 64 rounding path of ckkspackedencoding.cpp only"
#endif

namespace fhe_ssm {

// Error codes of PeriodicEncoder::encode. Every nonzero code corresponds to an input on which the
// library's dense path THROWS (or would produce a different limb count); the caller falls back to the
// dense path so that behaviour is unchanged on such inputs.
enum PeriodicEncodeStatus : int {
    PE_OK = 0,
    PE_LEVEL_OUT_OF_RANGE = 1,   // level >= number of towers (cryptocontext.h :380 throws)
    PE_SCALE_TOO_SMALL = 2,      // logc < 0 (ckkspackedencoding.cpp :207 throws "Scaling factor too small")
    PE_OVERFLOW = 3,             // is64BitOverflow after approxFactor (ckkspackedencoding.cpp :225 throws)
    PE_BAD_PERIOD = 4            // caller passed a null period
};

struct PeriodicEncoder {
    uint32_t N = 0;         // ring dimension of the context
    uint32_t logRep = 0;    // log2 REP
    uint32_t REP = 0;       // replicas of the period across the slots (N/2 / period)
    uint32_t Np = 0;        // N' = N / REP, the small ring
    uint32_t period = 0;    // N'/2 = one period of the slot vector (== Dpad in the harness)
    uint32_t CW = 0;        // N / REP words per limb == Np (the harness's COMP_CW)
    uint32_t nqTotal = 0;   // towers of the Q chain (sizeQ = multiplicativeDepth + 1 under FLEXIBLEAUTO)
    std::vector<uint64_t> q;                        // moduli, tower order
    std::vector<uint64_t> psiRep;                   // psi^REP per tower (the ring-N' root), for the record
    std::vector<std::vector<uint64_t>> rootTab;     // per tower: Np entries, rootTab[brev(i)] = psi'^i
    std::vector<std::vector<uint64_t>> rootShoup;   // per tower: floor(rootTab * 2^64 / q)
    std::vector<double> scaleAtLevel;               // GetScalingFactorReal(l), l in [0, nqTotal)

    struct Scratch {
        std::vector<std::complex<double>> inv;
        std::vector<int64_t> coef;
        std::vector<uint64_t> a;
    };

    static uint32_t bitReverse(uint32_t v, uint32_t bits) {
        uint32_t r = 0;
        for (uint32_t b = 0; b < bits; b++) { r = (r << 1) | (v & 1u); v >>= 1; }
        return r;
    }
    static uint64_t mulmod128(uint64_t a, uint64_t b, uint64_t m) {
        return (uint64_t)(((unsigned __int128)a * b) % m);
    }
    static uint64_t powmod(uint64_t a, uint64_t e, uint64_t m) {
        uint64_t r = 1 % m; a %= m;
        while (e) { if (e & 1) r = mulmod128(r, a, m); a = mulmod128(a, a, m); e >>= 1; }
        return r;
    }
    // Shoup: x * w mod q with w' = floor(w * 2^64 / q); q < 2^63.
    static inline uint64_t mulShoup(uint64_t x, uint64_t w, uint64_t wp, uint64_t q) {
        const uint64_t hi = (uint64_t)(((unsigned __int128)x * wp) >> 64);
        uint64_t r = x * w - hi * q;   // wraps mod 2^64; the true value is in [0, 2q)
        return r >= q ? r - q : r;
    }

    // cc: the OpenFHE context the dense path encodes with (the harness's `cc->cpu`).  log2Rep: log2 of
    // the replica count (the harness's compLogRep).  Throws std::runtime_error on a context this header
    // cannot serve (non-power-of-two geometry, a root that is not a primitive 2N-th root, ...).
    PeriodicEncoder(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, uint32_t log2Rep) {
        using namespace lbcrypto;
        auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
        if (!cp) throw std::runtime_error("periodic_encode: not a CKKSRNS context");
        N = cc->GetRingDimension();
        logRep = log2Rep;
        REP = 1u << logRep;
        if (N == 0 || (N & (N - 1)) != 0 || REP == 0 || REP > N / 2)
            throw std::runtime_error("periodic_encode: geometry (N, REP) must be powers of two with REP <= N/2");
        Np = N >> logRep;
        period = Np / 2;
        CW = Np;
        const auto& prs = cp->GetElementParams()->GetParams();
        nqTotal = (uint32_t)prs.size();
        q.resize(nqTotal); psiRep.resize(nqTotal); rootTab.resize(nqTotal); rootShoup.resize(nqTotal);
        uint32_t logNp = 0; while ((1u << logNp) < Np) logNp++;
        for (uint32_t l = 0; l < nqTotal; l++) {
            const uint64_t ql = prs[l]->GetModulus().ConvertToInt();
            const uint64_t psi = prs[l]->GetRootOfUnity().ConvertToInt();
            if (ql >= (1ull << 63)) throw std::runtime_error("periodic_encode: modulus >= 2^63 (Shoup path needs q < 2^63)");
            // The dense path's NTT uses the params' own root (NativePoly::SwitchFormat passes
            // m_params->GetRootOfUnity()); it must be a primitive 2N-th root of unity mod q, or the
            // deserialized context is not the one the dense path would encode with.
            if (psi == 0 || powmod(psi, N, ql) != ql - 1)
                throw std::runtime_error("periodic_encode: tower " + std::to_string(l) + " root of unity is not a primitive 2N-th root");
            q[l] = ql;
            const uint64_t psiP = powmod(psi, REP, ql);   // primitive 2N'-th root
            psiRep[l] = psiP;
            rootTab[l].assign(Np, 0); rootShoup[l].assign(Np, 0);
            uint64_t x = 1;
            for (uint32_t i = 0; i < Np; i++) {
                const uint32_t ir = bitReverse(i, logNp);
                rootTab[l][ir] = x;
                rootShoup[l][ir] = (uint64_t)((((unsigned __int128)x) << 64) / ql);
                x = mulmod128(x, psiP, ql);
            }
        }
        scaleAtLevel.resize(nqTotal);
        for (uint32_t l = 0; l < nqTotal; l++) scaleAtLevel[l] = cp->GetScalingFactorReal(l);
        // Both cyclotomic orders must be present in the library's FFT table map BEFORE any threaded use:
        // FFTSpecialInv throws on an absent order (dftransform.cpp :211), and Initialize() inserts only
        // when absent (:78), so after this constructor the map is never mutated again by this path.
        DiscreteFourierTransform::Initialize(2 * Np, Np / 2);
        DiscreteFourierTransform::Initialize(2 * N, N / 2);
    }

    uint32_t limbsAt(uint32_t level) const { return level < nqTotal ? nqTotal - level : 0; }
    size_t wordsAt(uint32_t level) const { return (size_t)limbsAt(level) * CW; }

    // period: `period` doubles (one period of the slot vector, slot s of replica 0).  level: the
    // plaintext level of the dense call MakeCKKSPackedPlaintext(diag, 1, level).  comp: on PE_OK, exactly
    // the words buildComp stores -- limb-major, limbsAt(level) limbs of CW words, comp[l*CW + j2] ==
    // rawLimb_l[j2 << logRep].  logApproxOut (optional): the approxFactor exponent the library would have
    // used (0 for every ordinary entry).
    int encode(const double* per, uint32_t level, std::vector<uint64_t>& comp, Scratch& s, int32_t* logApproxOut = nullptr) const {
        using namespace lbcrypto;
        if (!per) return PE_BAD_PERIOD;
        if (level >= nqTotal) return PE_LEVEL_OUT_OF_RANGE;
        const uint32_t nLimbs = nqTotal - level;
        const uint32_t slots = period;               // N'/2 -- full packing at ring N'
        const double scalingFactor = scaleAtLevel[level];
        // ---- inverse special FFT at order 2N' (the library's routine and tables) ----
        s.inv.resize(slots);
        for (uint32_t i = 0; i < slots; i++) s.inv[i] = std::complex<double>(per[i], 0.0);
        DiscreteFourierTransform::FFTSpecialInv(s.inv, 2 * Np);
        // ---- ckkspackedencoding.cpp :199-:236 ----
        int32_t logc = std::numeric_limits<int32_t>::min();
        for (uint32_t i = 0; i < slots; ++i) {
            s.inv[i] *= scalingFactor;
            if (s.inv[i].real() != 0.) {
                auto logci = static_cast<int32_t>(std::ceil(std::log2(std::abs(s.inv[i].real()))));
                if (logc < logci) logc = logci;
            }
            if (s.inv[i].imag() != 0.) {
                auto logci = static_cast<int32_t>(std::ceil(std::log2(std::abs(s.inv[i].imag()))));
                if (logc < logci) logc = logci;
            }
        }
        logc = (logc == std::numeric_limits<int32_t>::min()) ? 0 : logc;
        if (logc < 0) return PE_SCALE_TOO_SMALL;
        constexpr int32_t MAX_BITS_IN_WORD = 61;      // LargeScalingFactorConstants::MAX_BITS_IN_WORD
        const int32_t logValid = (logc <= MAX_BITS_IN_WORD) ? logc : MAX_BITS_IN_WORD;
        const int32_t logApprox = logc - logValid;
        const double approxFactor = std::pow(2, logApprox);
        constexpr int64_t MaxBitValue = static_cast<int64_t>((uint64_t(1) << 63) - (uint64_t(1) << 9) - 1);   // Max64BitValue()
        s.coef.resize(2 * slots);
        for (uint32_t i = 0; i < slots; ++i) {
            const double dre = s.inv[i].real() / approxFactor;
            const double dim = s.inv[i].imag() / approxFactor;
            if (std::abs(dre) > static_cast<double>(MaxBitValue) || std::abs(dim) > static_cast<double>(MaxBitValue)) return PE_OVERFLOW;
            s.coef[i] = std::llround(dre);
            s.coef[i + slots] = std::llround(dim);
        }
        if (logApproxOut) *logApproxOut = logApprox;
        // ---- per tower: FitToNativeVector (== c mod q), NTT with psi^REP, approxFactor scale-back ----
        comp.resize((size_t)nLimbs * CW);
        s.a.resize(Np);
        for (uint32_t l = 0; l < nLimbs; l++) {
            const uint64_t ql = q[l];
            for (uint32_t j = 0; j < Np; j++) {
                const int64_t c = s.coef[j];
                s.a[j] = c >= 0 ? (uint64_t)c % ql : (ql - ((uint64_t)(-c) % ql)) % ql;
            }
            ntt(s.a.data(), l);
            uint64_t* out = comp.data() + (size_t)l * CW;
            if (logApprox > 0) {
                const uint64_t f = powmod(2, (uint64_t)logApprox, ql);
                for (uint32_t j = 0; j < Np; j++) out[j] = mulmod128(s.a[j], f, ql);
            } else {
                for (uint32_t j = 0; j < Np; j++) out[j] = s.a[j];
            }
        }
        return PE_OK;
    }

    // transformnat-impl.h :200-:242, natural-order input, bit-reversed output, Table[brev(i)] = psi'^i.
    void ntt(uint64_t* a, uint32_t l) const {
        const uint64_t ql = q[l];
        const uint64_t* tab = rootTab[l].data();
        const uint64_t* tabp = rootShoup[l].data();
        const uint32_t n = Np;
        for (uint32_t m = 1, t = n >> 1; m < n; m <<= 1, t >>= 1) {
            for (uint32_t i = 0; i < m; ++i) {
                const uint32_t j1 = i * (t << 1), j2 = j1 + t;
                const uint64_t w = tab[m + i], wp = tabp[m + i];
                for (uint32_t lo = j1; lo < j2; ++lo) {
                    const uint32_t hi = lo + t;
                    const uint64_t x = mulShoup(a[hi], w, wp, ql);
                    const uint64_t lv = a[lo];
                    uint64_t hv = lv + x; if (hv >= ql) hv -= ql;
                    uint64_t lw = lv >= x ? lv - x : lv + ql - x;
                    a[lo] = hv; a[hi] = lw;
                }
            }
        }
    }
};

// periodicEncodeComp: the brief's signature.  Returns exactly the words buildComp stores (see
// PeriodicEncoder::encode); on a nonzero status returns an EMPTY vector and sets *status (the caller
// must then take the dense path, which throws on the same inputs).
inline std::vector<uint64_t> periodicEncodeComp(const PeriodicEncoder& enc, const std::vector<double>& periodVector,
                                                uint32_t level, int* status = nullptr) {
    std::vector<uint64_t> comp;
    PeriodicEncoder::Scratch s;
    if (periodVector.size() < enc.period) { if (status) *status = PE_BAD_PERIOD; return comp; }
    const int rc = enc.encode(periodVector.data(), level, comp, s);
    if (status) *status = rc;
    if (rc != PE_OK) comp.clear();
    return comp;
}
// Convenience form (builds the tables per call; ~1 ms at 42 towers -- for one-offs and tests, not for
// the miss loop, which keeps ONE encoder per context).
inline std::vector<uint64_t> periodicEncodeComp(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                                const std::vector<double>& periodVector, uint32_t level,
                                                uint32_t log2Rep, int* status = nullptr) {
    PeriodicEncoder enc(cc, log2Rep);
    return periodicEncodeComp(enc, periodVector, level, status);
}

// expandCompToPlaintext: a full OpenFHE Plaintext from a store entry (for the elementwise plaintext
// cache of batch 2).  Every limb l of the element is comp[l*CW + (i >> logRep)] at index i (the
// index map of fhe_ssm_expand_limbs_v), in EVALUATION format; metadata (level, scalingFactor =
// GetScalingFactorReal(level), noiseScaleDeg 1, slots N/2, CKKS data type) as
// MakeCKKSPackedPlaintextInternal sets it.  The `value` (decoded slots) vector is NOT populated: only
// what EvalMult/EvalAdd read (element + metadata).  Bit-identity of the element against the dense
// plaintext's GetElement<DCRTPoly>() is a Mac test (periodic_encode_test.cpp --mode small/identity).
namespace detail {
struct ExpandedCKKSPackedEncoding : public lbcrypto::CKKSPackedEncoding {
    using lbcrypto::CKKSPackedEncoding::CKKSPackedEncoding;
    void markEncoded() { isEncoded = true; }   // isEncoded is a protected member of PlaintextImpl
};
}  // namespace detail

inline lbcrypto::Plaintext expandCompToPlaintext(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                                 const PeriodicEncoder& enc, const std::vector<uint64_t>& comp,
                                                 uint32_t level) {
    using namespace lbcrypto;
    auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    if (!cp) throw std::runtime_error("expandCompToPlaintext: not a CKKSRNS context");
    const uint32_t nLimbs = enc.limbsAt(level);
    if (nLimbs == 0 || comp.size() != (size_t)nLimbs * enc.CW)
        throw std::runtime_error("expandCompToPlaintext: entry size does not match the level");
    // element params with `level` towers popped (cryptocontext.h :404-:410)
    std::shared_ptr<ILDCRTParams<DCRTPoly::Integer>> elemParamsPtr;
    if (level != 0) {
        ILDCRTParams<DCRTPoly::Integer> elemParams = *(cp->GetElementParams());
        for (uint32_t i = 0; i < level; i++) elemParams.PopLastParam();
        elemParamsPtr = std::make_shared<ILDCRTParams<DCRTPoly::Integer>>(elemParams);
    } else {
        elemParamsPtr = cp->GetElementParams();
    }
    const double scFact = cp->GetScalingFactorReal(level);
    const uint32_t slots = enc.N / 2;
    auto p = std::make_shared<detail::ExpandedCKKSPackedEncoding>(elemParamsPtr, cc->GetEncodingParams(),
                                                                  std::vector<std::complex<double>>(), (size_t)1,
                                                                  level, scFact, slots, cc->GetCKKSDataType());
    DCRTPoly poly(elemParamsPtr, Format::EVALUATION);
    const auto& towers = elemParamsPtr->GetParams();
    for (uint32_t l = 0; l < nLimbs; l++) {
        NativeVector nv(enc.N, towers[l]->GetModulus());
        const uint64_t* src = comp.data() + (size_t)l * enc.CW;
        for (uint32_t i = 0; i < enc.N; i++) nv[i] = NativeInteger(src[i >> enc.logRep]);
        NativePoly np(towers[l], Format::EVALUATION);
        np.SetValues(std::move(nv), Format::EVALUATION);
        poly.SetElementAtIndex(l, std::move(np));
    }
    p->GetElement<DCRTPoly>() = std::move(poly);
    p->markEncoded();
    return Plaintext(p);
}

}  // namespace fhe_ssm
