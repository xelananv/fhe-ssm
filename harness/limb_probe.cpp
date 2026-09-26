// limb_probe.cpp -- D-5: does an encoded diagonal's RNS limb actually contain
// ~N/REP distinct machine words (Cerium, arXiv:2512.11269), for OUR layouts?
//
// Builds a native430-shaped diagonal exactly as gpu_real_model.cu:1450-1462
// does, in BOTH slot layouts:
//     block:       diag[r * Dpad + slot] = w[slot]   (Dpad-periodic)
//     interleave:  diag[slot * REP + tk] = w[slot]   (run-length REP)
// then MakeCKKSPackedPlaintext at the requested level and inspects, per RNS
// limb, in BOTH representations:
//   EVALUATION (Cerium's claim: values repeat -> distinct ~ N/REP)
//   COEFFICIENT (mechanism: block layout should be stride-REP sparse -- but
//     OpenFHE's double-precision FFT encode + rounding at Delta=2^59 may leave
//     small nonzero off-stride coefficients, which would BREAK exact-equality
//     compression; that is measured here, not assumed)
// A constant-diagonal control row validates the instrument (must compress
// maximally in every layout/representation).
//
// Per the plaintext-compression notes, section 5 (not included). Output: JSON lines to stdout.
// Build: cmake --build harness/build --target limb_probe
// Run:   ./harness/build/limb_probe --log-ring 15 --dpad 1024 --levels 0,10,18

#include "openfhe.h"
#include <cstdint>
#include <cstdio>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace lbcrypto;

struct LimbStats {
    size_t distinct, runs, maxRun;
    double hitREP, hitDpad, hitNdiv;   // fraction v[i]==v[i+P]
};

static LimbStats limbStats(const std::vector<uint64_t>& v, uint32_t REP,
                           uint32_t Dpad) {
    LimbStats s{};
    std::unordered_set<uint64_t> u(v.begin(), v.end());
    s.distinct = u.size();
    s.runs = v.empty() ? 0 : 1;
    size_t run = 1; s.maxRun = 1;
    for (size_t i = 1; i < v.size(); i++) {
        if (v[i] == v[i - 1]) { run++; if (run > s.maxRun) s.maxRun = run; }
        else { s.runs++; run = 1; }
    }
    auto hit = [&](size_t P) {
        if (P == 0 || P >= v.size()) return 0.0;
        size_t h = 0, n = v.size() - P;
        for (size_t i = 0; i < n; i++) if (v[i] == v[i + P]) h++;
        return (double)h / (double)n;
    };
    s.hitREP  = hit(REP);
    s.hitDpad = hit(Dpad);
    s.hitNdiv = hit(v.size() / (REP ? REP : 1));
    return s;
}

int main(int argc, char** argv) {
    uint32_t logRing = 15, Dpad = 1024, depth = 20, scaleBits = 59;
    std::string levels = "0,10,18";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--log-ring") logRing = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--dpad") Dpad = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--depth") depth = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--scale-bits") scaleBits = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--levels") levels = argv[++i];
    }
    const uint32_t N = 1u << logRing, SLOTS = N / 2, REP = SLOTS / Dpad;

    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecurityLevel(HEStd_NotSet);
    p.SetRingDim(N);
    p.SetScalingModSize(scaleBits);
    p.SetFirstModSize(60);
    p.SetScalingTechnique(FLEXIBLEAUTO);          // production technique
    p.SetMultiplicativeDepth(depth);
    p.SetSecretKeyDist(UNIFORM_TERNARY);
    auto cc = GenCryptoContext(p);
    cc->Enable(PKE); cc->Enable(LEVELEDSHE);

    printf("{\"probe\":\"limb_duplication\",\"ring\":%u,\"slots\":%u,"
           "\"Dpad\":%u,\"REP\":%u,\"depth\":%u,\"scaleBits\":%u}\n",
           N, SLOTS, Dpad, REP, depth, scaleBits);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> U(-0.1, 0.1);
    std::vector<double> w(Dpad);
    for (auto& x : w) x = U(rng);

    std::vector<uint32_t> lvls;
    { std::stringstream ss(levels); std::string t;
      while (std::getline(ss, t, ',')) lvls.push_back((uint32_t)std::stoi(t)); }

    for (int control = 0; control <= 1; control++)
    for (int inter = 0; inter <= 1; inter++) {
        std::vector<double> diag(SLOTS, 0.0);
        for (uint32_t slot = 0; slot < Dpad; slot++) {
            double wv = control ? 0.0625 : w[slot];
            if (inter)
                for (uint32_t tk = 0; tk < REP; tk++)
                    diag[(size_t)slot * REP + tk] = wv;
            else
                for (uint32_t r = 0; r < REP; r++)
                    diag[(size_t)r * Dpad + slot] = wv;
        }
        for (uint32_t lvl : lvls) {
            if (lvl >= depth) continue;
            auto pt = cc->MakeCKKSPackedPlaintext(diag, 1, lvl);
            DCRTPoly el = pt->GetElement<DCRTPoly>();
            const auto& towers = el.GetAllElements();
            uint32_t limbs = (uint32_t)towers.size();
            int fmt = (int)el.GetFormat();   // 0=EVALUATION, 1=COEFFICIENT (enum order per openfhe)

            // -------- evaluation representation (Cerium's object) ----------
            for (uint32_t t = 0; t < limbs; t++) {
                DCRTPoly ev = el;
                ev.SetFormat(Format::EVALUATION);
                const auto& tv = ev.GetAllElements()[t].GetValues();
                std::vector<uint64_t> v(tv.GetLength());
                for (size_t i = 0; i < v.size(); i++)
                    v[i] = tv[i].ConvertToInt<uint64_t>();
                LimbStats s = limbStats(v, REP, Dpad);
                printf("{\"layout\":\"%s\",\"control\":%d,\"level\":%u,"
                       "\"limbs\":%u,\"encFormat\":%d,\"rep\":\"eval\","
                       "\"limb\":%u,\"N\":%zu,\"distinct\":%zu,"
                       "\"ratio\":%.2f,\"runs\":%zu,\"maxRun\":%zu,"
                       "\"hitREP\":%.4f,\"hitDpad\":%.4f,\"hitNdivREP\":%.4f}\n",
                       inter ? "interleave" : "block", control, lvl, limbs,
                       fmt, t, v.size(), s.distinct,
                       (double)v.size() / (double)s.distinct, s.runs, s.maxRun,
                       s.hitREP, s.hitDpad, s.hitNdiv);
                if (t >= 2 && t + 1 < limbs) { t = limbs - 2; }  // first 3 + last
            }

            // -------- coefficient representation (the mechanism) -----------
            DCRTPoly co = el;
            co.SetFormat(Format::COEFFICIENT);
            for (uint32_t t = 0; t < 1; t++) {   // one limb suffices for support
                const auto& tw = co.GetAllElements()[t];
                const auto& tv = tw.GetValues();
                uint64_t q = tw.GetModulus().ConvertToInt<uint64_t>();
                size_t zero = 0, onStride = 0, offStride = 0, offBig = 0;
                uint64_t offMax = 0;
                for (size_t i = 0; i < tv.GetLength(); i++) {
                    uint64_t x = tv[i].ConvertToInt<uint64_t>();
                    uint64_t mag = x > q / 2 ? q - x : x;   // centered magnitude
                    if (mag == 0) { zero++; continue; }
                    if (i % REP == 0) onStride++;
                    else {
                        offStride++;
                        if (mag > offMax) offMax = mag;
                        if (mag > (1ull << 20)) offBig++;
                    }
                }
                printf("{\"layout\":\"%s\",\"control\":%d,\"level\":%u,"
                       "\"rep\":\"coeff\",\"limb\":%u,\"q\":%llu,"
                       "\"zeros\":%zu,\"nzOnStrideREP\":%zu,"
                       "\"nzOffStride\":%zu,\"offStrideMax\":%llu,"
                       "\"offStrideAbove2p20\":%zu}\n",
                       inter ? "interleave" : "block", control, lvl, t,
                       (unsigned long long)q, zero, onStride, offStride,
                       (unsigned long long)offMax, offBig);
            }
            fflush(stdout);
        }
    }
    return 0;
}
