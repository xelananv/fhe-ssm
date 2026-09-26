// boot_overflow_probe.cpp -- does OpenFHE's CKKS bootstrap fail exactly when the ModRaise overflow
// max_j |I_j| leaves the range the approximate modular reduction was fitted for (K_UNIFORM = 512),
// and at what overshoot?  (2026-09-18, A11 tick-42 investigation; brief: results/dense-demo-s31/
// pod_2xbw_20260917/a11_tick42/README.md)
//
// MECHANISM UNDER TEST.  EvalBootstrap (vendor/openfhe-development/src/pke/lib/scheme/ckksrns/
// ckksrns-fhe.cpp, v1.5.1) takes the q0 tower of the adjusted input ciphertext, reads it as CENTERED
// residues and re-embeds it in the full chain (:592-:600; the centered read is NativeVector::
// SwitchModulus, core/lib/math/hal/intnat/mubintvecnat.cpp:109).  The raised ciphertext therefore
// decrypts to  t = [c0]_q0 + [c1]_q0 * s  over the integers  =  m' + q0 * I ,  I_j = round(t_j / q0).
// The approximate modular reduction (Chebyshev g_coefficientsUniform on x = t/(q0*K), then R_UNIFORM = 6
// double-angle steps) is fitted on |x| <= 1 only, i.e. |I_j| <= K_UNIFORM = 512.
//
// WHAT THIS PROBE DOES.  It runs the library's own EvalBootstrap and, for EVERY trial, computes the exact
// integer vector I of the ciphertext that enters ModRaise (negacyclic convolution over the integers in
// __int128, no floating point), then records the decrypted error next to max_j |I_j|.
//   --mode stock   library KeyGen (uniform ternary): the control.
//   --mode hand    secret built BY HAND with coefficients uniform in {-a..a} (a = 1 is the ternary
//                  distribution); sigma_I = sqrt((sum s_i^2 + 1)/12) is then a dial, while the library's
//                  SecretKeyDist stays UNIFORM_TERNARY so it still selects K = 512 / R = 6 / the uniform
//                  table.  Public key built by hand exactly as PKEBase::KeyGenInternal does
//                  (schemebase/base-pke.cpp:47-:96: b = ns*e - a*s, elements {b, a}); every evaluation
//                  key comes from the library's own EvalMultKeyGen / EvalBootstrapKeyGen on that secret.
//   --plant lo:hi:step   (either mode) ONE coefficient is driven to a chosen overflow I* = T while the
//                  ciphertext stays a valid encryption of the same message under the same key: the
//                  ModRaise input is rebuilt as c1'' = U + d (U uniform in a reduced range so nothing
//                  wraps, d_k = A*sgn of the secret coefficient that multiplies c1_k in (c1*s)_{j*}, so
//                  (d*s)_{j*} = A*sum|s_i|; j* = N-1 unless --plant-index), c0'' = c0' + (c1'-c1'')*s
//                  mod q0, and pulled back through the library's adjustment step (which is linear mod q0
//                  when the dropped tower is zero: c' = c * mu, mu MEASURED from the library, not assumed).
//                  The probe then runs the library's adjustment on the crafted ciphertext and requires
//                  bit-equality with the intended (c0'', c1'') before it bootstraps.
//
// WHY THE RECORDED I IS EXACT (not a model).  EvalBootstrap's path from its argument to ModRaise is
// (ckksrns-fhe.cpp:558-:562)  Clone -> ModReduceInternalInPlace(noiseScaleDeg-1) -> AdjustCiphertext(2^-corr,
// lvl 0), all deterministic.  The probe calls the SAME library functions on a clone of the SAME ciphertext
// (AdjustCiphertext is private; it is reached through an explicit template instantiation, which the
// language exempts from access checking -- no header is modified, no #define tricks), and ALSO re-derives
// the step from the public API line for line; the two must agree bit for bit on every trial
// ("adjustMatchesLibrary").  The q0 tower is then read with the library's own centering rule (v > q0>>1
// -> v - q0).
//
// ERROR READOUTS.  (1) the client's view: cc->Decrypt, whose CKKS decode throws "approximation error is too
// high" when the imaginary-part noise estimate exceeds p - 5 bits (encoding/ckkspackedencoding.cpp:449-:455);
// (2) a decode-free view: b + a*s over the output towers, CRT-interpolated, compared COEFFICIENT BY
// COEFFICIENT with the encoded reference -- this localises the damage, so the record can say whether the
// worst coefficient IS the overflowing one.
//
// Build: PATH=$PWD/.venv/bin:$PATH ninja -C harness/build boot_overflow_probe
// Run:   tools/memguard.sh 6 harness/build/boot_overflow_probe --log-ring 13 --mode hand --widths 1,8 --trials 50

#include "openfhe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "run_stamp.h"

using namespace lbcrypto;
using PolyT = DCRTPoly::PolyType;
typedef __int128 i128;

// ---- access to the private FHECKKSRNS::AdjustCiphertext without touching the library ---------------
// [temp.spec]/6: the usual access checking rules do not apply to names in an explicit instantiation.
namespace access {
using AdjustFn = void (FHECKKSRNS::*)(Ciphertext<DCRTPoly>&, double, uint32_t, bool) const;
struct AdjustTag { friend AdjustFn adjustPtr(AdjustTag); };
template <typename Tag, AdjustFn P> struct Expose { friend AdjustFn adjustPtr(Tag) { return P; } };
template struct Expose<AdjustTag, &FHECKKSRNS::AdjustCiphertext>;
}  // namespace access

struct Ctx {
    CryptoContext<DCRTPoly> cc;
    std::shared_ptr<CryptoParametersCKKSRNS> cp;
    uint32_t N = 0, slots = 0, depth = 0, F = 0, corr = 0, compositeDegree = 1;
    int32_t deg = 0;
    uint64_t q0 = 0;
};

static double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// centered residues of one tower, with the library's own rule (mubintvecnat.cpp:109-:121: v > q>>1 is negative)
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

// one tower from signed coefficients, returned in EVALUATION format
static PolyT polyFromSigned(const std::vector<int64_t>& c, const std::shared_ptr<PolyT::Params>& par) {
    const uint64_t q = par->GetModulus().ConvertToInt<uint64_t>();
    NativeVector v((uint32_t)c.size(), par->GetModulus());
    for (uint32_t k = 0; k < c.size(); k++) {
        const int64_t x = c[k];
        const uint64_t m = (uint64_t)(x >= 0 ? x : -x) % q;
        v[k] = NativeInteger((x >= 0 || m == 0) ? m : q - m);
    }
    PolyT p(par, Format::COEFFICIENT, true);
    p.SetValues(std::move(v), Format::COEFFICIENT);
    p.SetFormat(Format::EVALUATION);
    return p;
}

static DCRTPoly dcrtFromSigned(const std::vector<int64_t>& c, const std::shared_ptr<DCRTPoly::Params>& params) {
    DCRTPoly out(params, Format::EVALUATION, true);
    const auto& prs = params->GetParams();
    for (uint32_t i = 0; i < prs.size(); i++) out.SetElementAtIndex(i, polyFromSigned(c, prs[i]));
    return out;
}

// the hand-built key pair: PKEBase::KeyGenInternal (base-pke.cpp:47-:96) with s supplied instead of sampled
static KeyPair<DCRTPoly> handKeys(const CryptoContext<DCRTPoly>& cc, const std::vector<int64_t>& sCoef) {
    const auto cp = std::dynamic_pointer_cast<CryptoParametersRLWE<DCRTPoly>>(cc->GetCryptoParameters());
    const auto elementParams = cp->GetElementParams();
    const auto paramsPK = cp->GetParamsPK();
    if (!paramsPK) throw std::runtime_error("no paramsPK");
    DCRTPoly s = dcrtFromSigned(sCoef, paramsPK);
    DCRTPoly::DugType dug;
    DCRTPoly a(dug, paramsPK, Format::EVALUATION);
    DCRTPoly e(cp->GetDiscreteGaussianGenerator(), paramsPK, Format::EVALUATION);
    NativeInteger ns = cp->GetNoiseScale();
    DCRTPoly b(std::move((e *= ns) -= (a * s)));   // b = ns*e - a*s
    const auto sizeQ = elementParams->GetParams().size();
    const auto sizePK = paramsPK->GetParams().size();
    if (sizePK > sizeQ) s.DropLastElements(sizePK - sizeQ);
    KeyPair<DCRTPoly> kp(std::make_shared<PublicKeyImpl<DCRTPoly>>(cc), std::make_shared<PrivateKeyImpl<DCRTPoly>>(cc));
    kp.secretKey->SetPrivateElement(std::move(s));
    kp.publicKey->SetPublicElements(std::vector<DCRTPoly>{std::move(b), std::move(a)});
    kp.publicKey->SetKeyTag(kp.secretKey->GetKeyTag());
    return kp;
}

// the ciphertext that enters ModRaise, by the LIBRARY's own functions (ckksrns-fhe.cpp:558-:562)
static Ciphertext<DCRTPoly> preModRaiseLib(const Ctx& C, ConstCiphertext<DCRTPoly> ct) {
    auto raised = ct->Clone();
    C.cc->GetScheme()->ModReduceInternalInPlace(raised, C.compositeDegree * (raised->GetNoiseScaleDeg() - 1));
    FHECKKSRNS fhe;   // AdjustCiphertext reads nothing but its arguments (ckksrns-fhe.cpp:2228-:2274)
    (fhe.*adjustPtr(access::AdjustTag{}))(raised, std::pow(2, -static_cast<int32_t>(C.corr)), 0, true);
    return raised;
}

// the same step re-derived from the PUBLIC API, line for line (FLEXIBLEAUTO branch, 64-bit)
static Ciphertext<DCRTPoly> preModRaiseReplica(const Ctx& C, ConstCiphertext<DCRTPoly> ct) {
    auto raised = ct->Clone();
    auto algo = C.cc->GetScheme();
    algo->ModReduceInternalInPlace(raised, C.compositeDegree * (raised->GetNoiseScaleDeg() - 1));
    const double targetSF = C.cp->GetScalingFactorReal(0);
    const double sourceSF = raised->GetScalingFactor();
    const uint32_t numTowers = raised->GetElements()[0].GetNumOfElements();
    const double modToDrop = C.cp->GetElementParams()->GetParams()[numTowers - 1]->GetModulus().ConvertToDouble();
    const double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF) * std::pow(2, -static_cast<int32_t>(C.corr));
    C.cc->EvalMultInPlace(raised, adjustmentFactor);
    algo->ModReduceInternalInPlace(raised, C.compositeDegree);
    raised->SetScalingFactor(targetSF);
    return raised;
}

static bool sameElements(ConstCiphertext<DCRTPoly> a, ConstCiphertext<DCRTPoly> b) {
    const auto& x = a->GetElements(); const auto& y = b->GetElements();
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); i++) if (!(x[i] == y[i])) return false;
    return true;
}

struct IStats {
    int64_t maxAbs = 0, signedAtMax = 0; uint32_t argmax = 0, nOverK = 0;
    std::vector<int64_t> top;   // the 5 largest |I|, signed
    double sigmaHat = 0, meanHat = 0, maxFracOverQ0 = 0;
    std::vector<int64_t> I;
};

// EXACT: t_j = c0_j + sum_i s_i * c1_{j-i} (negacyclic) over the integers in __int128; I_j = round(t_j/q0)
static IStats overflowExact(const std::vector<int64_t>& c0, const std::vector<int64_t>& c1,
                            const std::vector<int64_t>& s, uint64_t q0, uint32_t K) {
    const uint32_t N = (uint32_t)s.size();
    std::vector<i128> acc(N, 0);
    for (uint32_t i = 0; i < N; i++) {
        const int64_t si = s[i];
        if (si == 0) continue;
        // X^i * c1: coefficient k lands on j = i + k, with a sign flip when it wraps past N
        for (uint32_t k = 0; k < N - i; k++) acc[i + k] += (i128)si * (i128)c1[k];
        for (uint32_t k = N - i; k < N; k++) acc[i + k - N] -= (i128)si * (i128)c1[k];
    }
    IStats st; st.I.resize(N);
    const i128 Q = (i128)q0, H = Q / 2;
    long double sum = 0, sum2 = 0;
    std::vector<std::pair<int64_t, int64_t>> mags(N);
    for (uint32_t j = 0; j < N; j++) {
        const i128 t = acc[j] + (i128)c0[j];
        const i128 u = t + H;
        const i128 I = (u >= 0) ? (u / Q) : -((-u + Q - 1) / Q);   // floor((t + q0/2)/q0)
        const i128 frac = t - I * Q;                               // m' + e, in [-q0/2, q0/2)
        const int64_t Ii = (int64_t)I;
        st.I[j] = Ii;
        const int64_t aI = Ii < 0 ? -Ii : Ii;
        if (aI > st.maxAbs) { st.maxAbs = aI; st.argmax = j; st.signedAtMax = Ii; }
        if (aI > (int64_t)K) st.nOverK++;
        sum += (long double)Ii; sum2 += (long double)Ii * (long double)Ii;
        const double f = std::fabs((double)frac) / (double)q0;
        if (f > st.maxFracOverQ0) st.maxFracOverQ0 = f;
        mags[j] = {aI, Ii};
    }
    st.meanHat = (double)(sum / N);
    st.sigmaHat = std::sqrt((double)(sum2 / N) - st.meanHat * st.meanHat);
    std::partial_sort(mags.begin(), mags.begin() + 5, mags.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (int k = 0; k < 5; k++) st.top.push_back(mags[k].second);
    return st;
}

// decode-free decryption: b + a*s over the ciphertext's towers (DecryptCore, ckksrns-pke / base-pke), CRT-interpolated,
// centered, divided by the ciphertext's scaling factor -> the plaintext POLYNOMIAL's coefficients in message units.
static std::vector<double> rawCoeffs(ConstCiphertext<DCRTPoly> ct, const DCRTPoly& sFull) {
    const auto& cv = ct->GetElements();
    const size_t sizeQl = cv[0].GetNumOfElements();
    DCRTPoly s(sFull);
    s.DropLastElements(sFull.GetNumOfElements() - sizeQl);
    DCRTPoly b(cv[0]); b.SetFormat(Format::EVALUATION);
    DCRTPoly sPow(s);
    for (size_t i = 1; i < cv.size(); i++) {
        DCRTPoly ci(cv[i]); ci.SetFormat(Format::EVALUATION);
        b += sPow * ci;
        if (i + 1 < cv.size()) sPow *= s;
    }
    b.SetFormat(Format::COEFFICIENT);
    auto big = b.CRTInterpolate();
    const BigInteger Q = big.GetModulus(); const BigInteger half = Q >> 1;
    const double sf = ct->GetScalingFactor();
    const uint32_t n = big.GetLength();
    std::vector<double> out(n);
    for (uint32_t j = 0; j < n; j++) {
        const BigInteger v = big[j];
        out[j] = ((v > half) ? -((Q - v).ConvertToDouble()) : v.ConvertToDouble()) / sf;
    }
    return out;
}

static std::vector<double> plainCoeffs(const Plaintext& pt) {
    DCRTPoly e(pt->GetElement<DCRTPoly>());
    e.SetFormat(Format::COEFFICIENT);
    auto big = e.CRTInterpolate();
    const BigInteger Q = big.GetModulus(); const BigInteger half = Q >> 1;
    const double sf = pt->GetScalingFactor();
    const uint32_t n = big.GetLength();
    std::vector<double> out(n);
    for (uint32_t j = 0; j < n; j++) {
        const BigInteger v = big[j];
        out[j] = ((v > half) ? -((Q - v).ConvertToDouble()) : v.ConvertToDouble()) / sf;
    }
    return out;
}

static std::string jnum(double v) {   // JSON has no NaN/Inf
    if (!std::isfinite(v)) return "null";
    std::ostringstream o; o << std::setprecision(9) << std::scientific << v; return o.str();
}

struct Eval {
    bool decodeFail = false; std::string what;
    uint32_t nonFinite = 0; double relErrRms = 0, relErrMax = 0, maxAbsErr = 0, medAbsErr = 0;
    double coefErrRms = 0, coefErrMax = 0, coefErrAtArgmaxI = 0, coefErrAtPartner = 0, coefErrMedAbs = 0, coefRefRms = 0;
    uint32_t argmaxCoefErr = 0, coefNonFinite = 0;
    std::vector<std::pair<uint32_t, double>> topCoefErr;   // the 6 largest |coefficient error|, (index, signed error)
    std::vector<double> coefErrVec;                        // the decode-free error of every coefficient (for --err-by-absI; diagnostic only)
};

static Eval evaluate(const Ctx& C, const PrivateKey<DCRTPoly>& sk, ConstCiphertext<DCRTPoly> out,
                     const std::vector<double>& ref, const std::vector<double>& refCoef, uint32_t argmaxI) {
    Eval E;
    // (1) the client's view
    Plaintext p;
    try {
        C.cc->Decrypt(sk, out, &p);
        p->SetLength(C.slots);
        auto v = p->GetRealPackedValue();
        double sse = 0, sref = 0; std::vector<double> ae; ae.reserve(C.slots);
        for (uint32_t i = 0; i < C.slots; i++) {
            if (!std::isfinite(v[i])) { E.nonFinite++; continue; }
            const double d = v[i] - ref[i];
            sse += d * d; sref += ref[i] * ref[i];
            if (std::fabs(d) > E.maxAbsErr) E.maxAbsErr = std::fabs(d);
            ae.push_back(std::fabs(d));
        }
        E.relErrRms = (sref > 0) ? std::sqrt(sse / sref) : 0;
        double refMax = 0; for (double r : ref) refMax = std::max(refMax, std::fabs(r));
        E.relErrMax = E.maxAbsErr / refMax;
        if (!ae.empty()) { std::nth_element(ae.begin(), ae.begin() + ae.size() / 2, ae.end()); E.medAbsErr = ae[ae.size() / 2]; }
    } catch (const std::exception& ex) {
        E.decodeFail = true; E.what = ex.what();
        if (E.what.size() > 160) E.what.resize(160);
    }
    // (2) the decode-free view, coefficient by coefficient
    try {
        auto oc = rawCoeffs(out, sk->GetPrivateElement());
        double sse = 0, sref = 0;
        for (uint32_t j = 0; j < oc.size(); j++) {
            const double d = oc[j] - refCoef[j];
            if (!std::isfinite(d)) { E.coefNonFinite++; continue; }
            sse += d * d; sref += refCoef[j] * refCoef[j];
            if (std::fabs(d) > E.coefErrMax) { E.coefErrMax = std::fabs(d); E.argmaxCoefErr = j; }
        }
        E.coefErrRms = std::sqrt(sse / oc.size());
        E.coefRefRms = std::sqrt(sref / oc.size());
        E.coefErrAtArgmaxI = oc[argmaxI] - refCoef[argmaxI];
        const uint32_t partner = (argmaxI + (uint32_t)oc.size() / 2) % (uint32_t)oc.size();   // the other half of the same slot
        E.coefErrAtPartner = oc[partner] - refCoef[partner];
        std::vector<std::pair<double, uint32_t>> mags(oc.size());
        for (uint32_t j = 0; j < oc.size(); j++) { const double d = oc[j] - refCoef[j]; mags[j] = {std::isfinite(d) ? std::fabs(d) : 0.0, j}; }
        std::partial_sort(mags.begin(), mags.begin() + 6, mags.end(), [](auto& a, auto& b) { return a.first > b.first; });
        for (int k = 0; k < 6; k++) E.topCoefErr.push_back({mags[k].second, oc[mags[k].second] - refCoef[mags[k].second]});
        E.coefErrVec.resize(oc.size());
        for (uint32_t j = 0; j < oc.size(); j++) E.coefErrVec[j] = oc[j] - refCoef[j];
        std::nth_element(mags.begin(), mags.begin() + mags.size() / 2, mags.end());
        E.coefErrMedAbs = mags[mags.size() / 2].first;
    } catch (const std::exception& ex) {
        E.coefNonFinite = 0xffffffffu;
    }
    return E;
}

static std::vector<int64_t> parseList(const std::string& s) {
    std::vector<int64_t> v; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stoll(tok));
    return v;
}

int main(int argc, char** argv) {
    uint32_t logRing = 13, levelsAfter = 3, trials = 20, corrArg = 0, seed = 20260918;
    int scaleBits = 59, firstModBits = 60;
    std::string mode = "hand";
    std::vector<int64_t> widths = {1}, trialsList;
    std::vector<uint32_t> levelBudget = {3, 3};
    double amp = 1.0;
    bool plant = false; int64_t plantLo = 0, plantHi = 0, plantStep = 1; int64_t plantIndexArg = -1;   // -1 => N-1
    uint32_t errByAbsI = 0;     // --err-by-absI W: extra "errByAbsI" record per trial (additive; changes no computation)
    uint32_t kBoundArg = 512;   // --k-bound: FHECKKSRNS::K_UNIFORM unless the run is the Ext arm (K_UNIFORMEXT = 768); enters NO computation
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--log-ring") logRing = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--levels-after") levelsAfter = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--trials") trials = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--corr") corrArg = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--seed") seed = (uint32_t)std::stoul(argv[++i]);
        else if (a == "--scale-bits") scaleBits = std::stoi(argv[++i]);
        else if (a == "--first-mod-bits") firstModBits = std::stoi(argv[++i]);
        else if (a == "--mode") mode = argv[++i];
        else if (a == "--widths") widths = parseList(argv[++i]);
        else if (a == "--trials-list") trialsList = parseList(argv[++i]);   // one trial count per width (overrides --trials)
        else if (a == "--amp") amp = std::stod(argv[++i]);
        else if (a == "--level-budget") { auto v = parseList(argv[++i]); levelBudget = {(uint32_t)v.at(0), (uint32_t)v.at(1)}; }
        else if (a == "--plant-index") plantIndexArg = std::stoll(argv[++i]);   // which coefficient carries the planted overflow
        else if (a == "--err-by-absI") errByAbsI = (uint32_t)std::stoul(argv[++i]);   // DIAGNOSTIC record per trial: decode-free coefficient error aggregated in |I| bins of this width (0 = off)
        else if (a == "--k-bound") kBoundArg = (uint32_t)std::stoul(argv[++i]);   // REPORTING ONLY (header K, KoverSigma, nOverK): 512 stock, 768 for the Ext table (k768 build + OPENFHE_BOOT_UNIFORM_EXT=1)
        else if (a == "--plant") {   // lo:hi:step, the target overflow of the planted coefficient (negative allowed)
            plant = true; std::string s = argv[++i]; std::replace(s.begin(), s.end(), ':', ',');
            auto v = parseList(s); plantLo = v.at(0); plantHi = v.at(1); plantStep = v.size() > 2 ? v[2] : 1;
            if (plantStep <= 0 || plantHi < plantLo) { std::cerr << "--plant lo:hi:step needs lo <= hi, step > 0\n"; return 2; }
        }
        else { std::cerr << "unknown flag " << a << std::endl; return 2; }
    }
    if (logRing > 15) { std::cerr << "W5: local rings are 2^15 or smaller" << std::endl; return 2; }
    if (mode != "hand" && mode != "stock") { std::cerr << "--mode hand|stock" << std::endl; return 2; }
    if (mode == "stock") widths = {1};
    if (!trialsList.empty() && trialsList.size() != widths.size()) { std::cerr << "--trials-list needs one count per width" << std::endl; return 2; }

    const uint32_t bootDepth = FHECKKSRNS::GetBootstrapDepth(levelBudget, UNIFORM_TERNARY);
    Ctx C;
    C.depth = levelsAfter + bootDepth;
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);   // selects K_UNIFORM / R_UNIFORM / g_coefficientsUniform (ckksrns-fhe.cpp:626-:631, :700-:701)
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1u << logRing);
    parameters.SetScalingModSize((uint32_t)scaleBits);
    parameters.SetFirstModSize((uint32_t)firstModBits);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(C.depth);
    C.cc = GenCryptoContext(parameters);
    C.cc->Enable(PKE); C.cc->Enable(KEYSWITCH); C.cc->Enable(LEVELEDSHE); C.cc->Enable(ADVANCEDSHE); C.cc->Enable(FHE);
    C.cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(C.cc->GetCryptoParameters());
    C.N = C.cc->GetRingDimension(); C.slots = C.N / 2;
    C.compositeDegree = C.cp->GetCompositeDegree();
    C.q0 = C.cp->GetElementParams()->GetParams()[0]->GetModulus().ConvertToInt<uint64_t>();
    const uint32_t K = kBoundArg;   // default 512 = FHECKKSRNS::K_UNIFORM (ckksrns-fhe.h:424; private, so restated here and checked by the report); --k-bound 768 = K_UNIFORMEXT (:426)
    // The k768 build of the library (vendor/install-k768, results/.../overflow_probe/k768/openfhe_k768_env.patch) reads this
    // variable at run time; the stock library ignores it. Recorded so that every file says which arm it is; a label that
    // contradicts the variable is refused rather than written.
    const char* envExtRaw = std::getenv("OPENFHE_BOOT_UNIFORM_EXT");
    const bool envExt = envExtRaw && envExtRaw[0] == '1' && envExtRaw[1] == '\0';
    if ((envExt && K != 768) || (!envExt && K != 512)) {
        std::cerr << "--k-bound " << K << " contradicts OPENFHE_BOOT_UNIFORM_EXT=" << (envExtRaw ? envExtRaw : "(unset)")
                  << " (512 when unset, 768 when 1)" << std::endl;
        return 2;
    }
    const uint32_t plantIndex = (plantIndexArg < 0 || plantIndexArg >= (int64_t)C.N) ? C.N - 1 : (uint32_t)plantIndexArg;

    auto tSetup = std::chrono::steady_clock::now();
    C.cc->EvalBootstrapSetup(levelBudget, {0, 0}, C.slots, corrArg);
    const double setupMs = msSince(tSetup);
    C.F = C.cc->GetScheme()->GetCKKSBootCorrectionFactor();
    {   // ckksrns-fhe.cpp:532-:541
        const double qDouble = (double)C.q0;
        const double powP = std::pow(2, C.cp->GetPlaintextModulus());
        C.deg = (int32_t)std::round(std::log2(qDouble / powP));
        C.corr = C.F - C.deg;
    }

    std::cout << "{\"rec\":\"header\",\"harness\":\"boot_overflow_probe\",\"mode\":\"" << mode << "\",\"ringDim\":" << C.N
              << ",\"slots\":" << C.slots << ",\"depth\":" << C.depth << ",\"bootDepth\":" << bootDepth
              << ",\"levelsAfter\":" << levelsAfter << ",\"levelBudget\":[" << levelBudget[0] << "," << levelBudget[1] << "]"
              << ",\"scaleBits\":" << scaleBits << ",\"firstModBits\":" << firstModBits << ",\"scalingTechnique\":\"FLEXIBLEAUTO\""
              << ",\"secretKeyDistParam\":\"UNIFORM_TERNARY\",\"K\":" << K
              << ",\"envUniformExt\":" << (envExtRaw ? "\"" + fhe_ssm::jsonEscape(envExtRaw) + "\"" : std::string("null")) << ",\"q0\":" << C.q0
              << ",\"towersQ\":" << C.cp->GetElementParams()->GetParams().size()
              << ",\"correctionFactor\":" << C.F << ",\"deg\":" << C.deg << ",\"corr\":" << C.corr
              << ",\"compositeDegree\":" << C.compositeDegree << ",\"amp\":" << amp << ",\"seed\":" << seed
              << ",\"trials\":" << trials << ",\"plant\":" << (plant ? "true" : "false") << ",\"plantIndex\":" << plantIndex
              << ",\"bootSetupMs\":" << (long)setupMs << fhe_ssm::runStamp(argc, argv) << "}" << std::endl;

    // the fixed known message, values in [-amp, amp]
    std::vector<double> ref(C.slots);
    { std::mt19937_64 g(seed ^ 0x9e3779b97f4a7c15ull); std::uniform_real_distribution<double> u(-amp, amp); for (auto& x : ref) x = u(g); }
    const Plaintext ptRef = C.cc->MakeCKKSPackedPlaintext(ref, 1, 0, nullptr, C.slots);
    const std::vector<double> refCoef = plainCoeffs(ptRef);

    for (size_t wi = 0; wi < widths.size(); wi++) {
        const int64_t a = widths[wi];
        if (!trialsList.empty()) trials = (uint32_t)trialsList[wi];
        // ---- key set -------------------------------------------------------------------------------
        auto tKey = std::chrono::steady_clock::now();
        KeyPair<DCRTPoly> keys;
        if (mode == "stock") keys = C.cc->KeyGen();
        else {
            std::mt19937_64 g(((uint64_t)seed << 8) ^ (uint64_t)a);
            std::uniform_int_distribution<int64_t> u(-a, a);
            std::vector<int64_t> sc(C.N); for (auto& x : sc) x = u(g);
            keys = handKeys(C.cc, sc);
        }
        C.cc->EvalMultKeyGen(keys.secretKey);
        C.cc->EvalBootstrapKeyGen(keys.secretKey, C.slots);
        const double keyMs = msSince(tKey);
        // the secret as the library holds it (read back from the key, both modes)
        const std::vector<int64_t> s = centered(keys.secretKey->GetPrivateElement().GetElementAtIndex(0));
        long long sumS2 = 0, sumAbs = 0, h = 0, sMax = 0;
        for (int64_t x : s) { sumS2 += x * x; sumAbs += std::llabs(x); h += (x != 0); sMax = std::max<long long>(sMax, std::llabs(x)); }
        const double sigmaPred = std::sqrt((sumS2 + 1) / 12.0);
        std::cout << "{\"rec\":\"keyset\",\"mode\":\"" << mode << "\",\"a\":" << a << ",\"N\":" << C.N << ",\"sumS2\":" << sumS2
                  << ",\"sumAbsS\":" << sumAbs << ",\"hamming\":" << h << ",\"maxAbsS\":" << sMax
                  << ",\"sigmaIpred\":" << jnum(sigmaPred) << ",\"KoverSigma\":" << jnum(K / sigmaPred)
                  << ",\"keygenMs\":" << (long)keyMs << "}" << std::endl;

        // ---- the multiplier of the library's adjustment step on the q0 tower (planting only) ------------
        // With the dropped tower zero the step is c -> c * mu mod q0; mu is MEASURED on a unit polynomial.
        NativeInteger muInv(0); bool muOk = false; uint64_t muVal = 0;
        const Plaintext ptIn = C.cc->MakeCKKSPackedPlaintext(ref, 1, C.depth - 1, nullptr, C.slots);
        if (plant) {
            auto g0 = C.cc->Encrypt(keys.publicKey, ptIn);
            auto unit = g0->Clone();
            const auto par2 = g0->GetElements()[0].GetParams();
            std::vector<int64_t> one(C.N, 0); one[0] = 1;
            std::vector<int64_t> zero(C.N, 0);
            DCRTPoly u(par2, Format::EVALUATION, true);
            u.SetElementAtIndex(0, polyFromSigned(one, par2->GetParams()[0]));
            for (uint32_t i = 1; i < par2->GetParams().size(); i++) u.SetElementAtIndex(i, polyFromSigned(zero, par2->GetParams()[i]));
            unit->SetElements(std::vector<DCRTPoly>{u, u});
            auto adj = preModRaiseLib(C, unit);
            PolyT m0 = adj->GetElements()[0].GetElementAtIndex(0); m0.SetFormat(Format::COEFFICIENT);
            muOk = true; for (uint32_t k = 1; k < C.N; k++) if (m0[k] != NativeInteger(0)) muOk = false;
            muVal = m0[0].ConvertToInt<uint64_t>();
            if (muVal == 0) muOk = false;
            if (muOk) muInv = m0[0].ModInverse(NativeInteger(C.q0));
            std::cout << "{\"rec\":\"adjustMultiplier\",\"a\":" << a << ",\"mu\":" << muVal << ",\"isScalar\":" << (muOk ? "true" : "false")
                      << ",\"towersIn\":" << par2->GetParams().size() << "}" << std::endl;
            if (!muOk) { std::cerr << "adjustment step is not a scalar on the q0 tower; planting impossible" << std::endl; return 3; }
        }

        std::mt19937_64 gPlant(((uint64_t)seed << 20) ^ (uint64_t)a ^ 0x5bd1e995ull);
        const PolyT s0 = keys.secretKey->GetPrivateElement().GetElementAtIndex(0);   // EVALUATION, mod q0
        std::vector<int64_t> targets;
        if (plant) for (int64_t T = plantLo; T <= plantHi; T += plantStep) targets.push_back(T);
        const uint32_t nTrials = plant ? (uint32_t)targets.size() * trials : trials;

        for (uint32_t t = 0; t < nTrials; t++) {
            auto ct = C.cc->Encrypt(keys.publicKey, ptIn);
            const uint32_t towersIn = ct->GetElements()[0].GetNumOfElements();
            int64_t T = 0; bool plantVerified = false; int64_t plantAchieved = 0;
            Ciphertext<DCRTPoly> pre = preModRaiseLib(C, ct);
            if (plant) {
                T = targets[t / trials];
                const uint32_t js = plantIndex;
                // negacyclic: (x*s)_js = sum_{k<=js} x_k s_{js-k} - sum_{k>js} x_k s_{js-k+N}
                auto sAt = [&](uint32_t k) -> int64_t { return (k <= js) ? s[js - k] : -s[js + C.N - k]; };
                const PolyT c0p = pre->GetElements()[0].GetElementAtIndex(0), c1p = pre->GetElements()[1].GetElementAtIndex(0);
                // U uniform in a reduced range so that U + d never wraps: |A| <= (|T| + 400) * q0 / sumAbs
                const long double Amax = ((long double)std::llabs(T) + 400.0L) * (long double)C.q0 / (long double)sumAbs;
                if (Amax >= 0.45L * (long double)C.q0) { std::cerr << "plant target too large for this secret" << std::endl; return 3; }
                const int64_t Ur = (int64_t)((long double)(C.q0 / 2) - Amax) - 2;
                std::uniform_int_distribution<int64_t> ud(-Ur, Ur);
                std::vector<int64_t> U(C.N); for (auto& x : U) x = ud(gPlant);
                i128 Us = 0;   // (U*s)_{N-1} = sum_k U_k s_{N-1-k}: no wrap at j = N-1
                for (uint32_t k = 0; k < C.N; k++) Us += (i128)U[k] * (i128)sAt(k);
                const long double Aexact = ((long double)T * (long double)C.q0 - (long double)Us) / (long double)sumAbs;
                const int64_t A = (int64_t)std::llround(Aexact);
                std::vector<int64_t> c1pp(C.N);
                for (uint32_t k = 0; k < C.N; k++) { const int64_t v = sAt(k); const int64_t sg = (v > 0) - (v < 0); c1pp[k] = U[k] + A * sg; }
                const PolyT c1ppP = polyFromSigned(c1pp, c1p.GetParams());
                const PolyT c0ppP = c0p.Plus(c1p.Minus(c1ppP).Times(s0));   // same message, same noise, mod q0
                // pull back through the adjustment: tower 0 = c'' * mu^-1, every dropped tower zero
                const auto par2 = ct->GetElements()[0].GetParams();
                std::vector<int64_t> zero(C.N, 0);
                auto mk = [&](const PolyT& x) {
                    DCRTPoly d(par2, Format::EVALUATION, true);
                    d.SetElementAtIndex(0, x.Times(muInv));
                    for (uint32_t i = 1; i < par2->GetParams().size(); i++) d.SetElementAtIndex(i, polyFromSigned(zero, par2->GetParams()[i]));
                    return d;
                };
                auto crafted = ct->Clone();
                crafted->SetElements(std::vector<DCRTPoly>{mk(c0ppP), mk(c1ppP)});
                auto preC = preModRaiseLib(C, crafted);
                plantVerified = (preC->GetElements()[0].GetElementAtIndex(0) == c0ppP) && (preC->GetElements()[1].GetElementAtIndex(0) == c1ppP);
                ct = crafted; pre = preC;
            }
            const bool adjOk = sameElements(pre, preModRaiseReplica(C, ct));
            const uint32_t towersPre = pre->GetElements()[0].GetNumOfElements();
            const IStats st = overflowExact(centered(pre->GetElements()[0].GetElementAtIndex(0)),
                                            centered(pre->GetElements()[1].GetElementAtIndex(0)), s, C.q0, K);
            if (plant) plantAchieved = st.I[plantIndex];

            Ciphertext<DCRTPoly> out; double bootMs = 0; std::string threw;
            auto t0 = std::chrono::steady_clock::now();
            try { out = C.cc->EvalBootstrap(ct); } catch (const std::exception& ex) { threw = ex.what(); if (threw.size() > 160) threw.resize(160); }
            bootMs = msSince(t0);

            std::cout << "{\"rec\":\"trial\",\"mode\":\"" << mode << "\",\"a\":" << a << ",\"trial\":" << t;
            if (plant) std::cout << ",\"plantTarget\":" << T << ",\"plantAchieved\":" << plantAchieved << ",\"plantVerified\":" << (plantVerified ? "true" : "false");
            std::cout << ",\"maxAbsI\":" << st.maxAbs << ",\"signedImax\":" << st.signedAtMax << ",\"argmaxI\":" << st.argmax
                      << ",\"IatPartner\":" << st.I[(st.argmax + C.N / 2) % C.N]
                      << ",\"nOverK\":" << st.nOverK << ",\"top5I\":[";
            for (size_t k = 0; k < st.top.size(); k++) std::cout << (k ? "," : "") << st.top[k];
            std::cout << "],\"sigmaIhat\":" << jnum(st.sigmaHat) << ",\"meanIhat\":" << jnum(st.meanHat)
                      << ",\"maxFracOverQ0\":" << jnum(st.maxFracOverQ0) << ",\"adjustMatchesLibrary\":" << (adjOk ? "true" : "false")
                      << ",\"towersIn\":" << towersIn << ",\"towersPreModRaise\":" << towersPre << ",\"bootMs\":" << (long)bootMs;
            if (!threw.empty()) { std::cout << ",\"bootThrew\":\"" << fhe_ssm::jsonEscape(threw) << "\"}" << std::endl; continue; }
            const uint32_t towersOut = out->GetElements()[0].GetNumOfElements();
            const Eval E = evaluate(C, keys.secretKey, out, ref, refCoef, st.argmax);
            std::cout << ",\"towersOut\":" << towersOut << ",\"bootRestored\":" << (towersOut > towersIn ? "true" : "false")
                      << ",\"outLevel\":" << out->GetLevel() << ",\"outNoiseScaleDeg\":" << out->GetNoiseScaleDeg()
                      << ",\"decodeFail\":" << (E.decodeFail ? "true" : "false");
            if (E.decodeFail) std::cout << ",\"what\":\"" << fhe_ssm::jsonEscape(E.what) << "\"";
            else std::cout << ",\"nonFinite\":" << E.nonFinite << ",\"relErrRms\":" << jnum(E.relErrRms) << ",\"relErrMax\":" << jnum(E.relErrMax)
                           << ",\"maxAbsErr\":" << jnum(E.maxAbsErr) << ",\"medAbsErr\":" << jnum(E.medAbsErr);
            std::cout << ",\"coefNonFinite\":" << E.coefNonFinite << ",\"coefErrRms\":" << jnum(E.coefErrRms) << ",\"coefErrMax\":" << jnum(E.coefErrMax)
                      << ",\"argmaxCoefErr\":" << E.argmaxCoefErr << ",\"coefErrAtArgmaxI\":" << jnum(E.coefErrAtArgmaxI)
                      << ",\"coefErrAtPartner\":" << jnum(E.coefErrAtPartner) << ",\"coefErrMedAbs\":" << jnum(E.coefErrMedAbs)
                      << ",\"coefRefRms\":" << jnum(E.coefRefRms) << ",\"topCoefErr\":[";
            for (size_t k = 0; k < E.topCoefErr.size(); k++) std::cout << (k ? "," : "") << "[" << E.topCoefErr[k].first << "," << jnum(E.topCoefErr[k].second) << "]";
            std::cout << "],\"topCoefErrI\":[";   // the exact overflow I_j at each of those coefficients (same order)
            for (size_t k = 0; k < E.topCoefErr.size(); k++) std::cout << (k ? "," : "") << st.I[E.topCoefErr[k].first];
            std::cout << "]}" << std::endl;
            if (errByAbsI > 0 && !E.coefErrVec.empty()) {
                // DIAGNOSTIC: is the decode-free coefficient error a function of the coefficient's own overflow I_j?
                // Bin b holds the coefficients with b*W <= |I_j| < (b+1)*W; per bin: count, sum of squared error, max |error|.
                const uint32_t nb = (uint32_t)(st.maxAbs / errByAbsI) + 1;
                std::vector<uint32_t> cnt(nb, 0); std::vector<double> ssq(nb, 0.0), mx(nb, 0.0);
                for (uint32_t j = 0; j < C.N; j++) {
                    const double d = E.coefErrVec[j]; if (!std::isfinite(d)) continue;
                    const int64_t aI = st.I[j] < 0 ? -st.I[j] : st.I[j];
                    const uint32_t b = (uint32_t)(aI / errByAbsI);
                    cnt[b]++; ssq[b] += d * d; if (std::fabs(d) > mx[b]) mx[b] = std::fabs(d);
                }
                std::cout << "{\"rec\":\"errByAbsI\",\"a\":" << a << ",\"trial\":" << t << ",\"binWidth\":" << errByAbsI << ",\"bins\":[";
                bool firstBin = true;
                for (uint32_t b = 0; b < nb; b++) {
                    if (!cnt[b]) continue;
                    std::cout << (firstBin ? "" : ",") << "[" << b << "," << cnt[b] << "," << jnum(ssq[b]) << "," << jnum(mx[b]) << "]";
                    firstBin = false;
                }
                std::cout << "]}" << std::endl;
            }
        }
        C.cc->ClearEvalMultKeys();
        C.cc->ClearEvalAutomorphismKeys();
    }
    return 0;
}
