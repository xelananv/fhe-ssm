// boot_envelope_probe.cpp — what sets the UPPER branch of the §1b stability envelope?
//
// FHE_ERROR_LAWS §1b measures a two-sided envelope in message amplitude: relative
// bootstrap error falls as ~1/amp (the absolute noise floor), reaches a minimum
// near amp≈20, then degrades sharply (13.9% at amp=100). §9.1 lists the upper
// branch as unexplained ("presumed bootstrap-approximation basin exit; never
// measured"). This probe measures it.
//
// Reading OpenFHE's ckksrns-fhe.cpp gives two candidate mechanisms, which make
// DIFFERENT predictions and are separated by the two knobs swept here:
//
//   H_wrap  The message must satisfy |m| < q0/(2*Delta) for ModRaise to recover
//           it; beyond that the level-0 representation has already wrapped.
//           => hard cliff at a_max = 2^(firstModBits - scaleBits - 1),
//              INDEPENDENT of the correction factor F.
//
//   H_sine  Bootstrapping approximates the identity by sin(2*pi*u)/(2*pi) with
//           u = Delta*m/(q0 * 2^corr) and corr = F - deg. The leading error is
//           the cubic term of the sine:
//                 relErr(a) = (2*pi)^2/6 * u^2 ,  absErr(a) = (2*pi)^2/6 * a*u^2
//           => SMOOTH cubic growth in absolute error, whose coefficient moves
//              by 4x per unit of F, and 4x per bit of (firstModBits-scaleBits).
//
// H_sine's cubic coefficient is fully determined -- there is no free parameter
// in it -- so the only fitted quantity is the additive floor eps0. The probe
// prints the predicted cubic term alongside the measurement.
//
// Sweeping F at fixed (q0/Delta) separates them: H_wrap says the cliff does not
// move, H_sine says the whole upper branch shifts by 4x in error per unit F.
//
// Build: cmake --build harness/build -j   (target boot_envelope_probe)
// Run:   ./harness/build/boot_envelope_probe --log-ring 13 --corr 0

#include "openfhe.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Provenance stamp (harness file + sha256, git commit, build time, exact argv).
// This probe is F3/P1b's subject — its records will be compared against a
// GPU port, so "which build produced this line" has to be in the line.
#include "run_stamp.h"

using namespace lbcrypto;

int main(int argc, char** argv) {
    uint32_t logRing = 13, levelsAfter = 10, consume = 5, boots = 1;
    int scaleBits = 59, firstModBits = 60;
    uint32_t corr = 0;  // 0 => let OpenFHE pick its fitted default
    std::vector<double> amps = {0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0};
    // E-F4 (2026-08-18): was hardcoded {4,4} below the parameter block, which
    // pinned every record in results/boot-envelope/ to one budget and blocked
    // the c2 = 0.1031 vs derived (2*pi)^2/6 = 6.58 question (a 64x gap). The
    // default is unchanged, so existing runs stay reproducible verbatim.
    std::vector<uint32_t> levelBudget = {4, 4};
    // E-F4: depth override. The DERIVED depth is correct but is NOT what the
    // pre-2026-08-18 corpus used (see the note at the derivation below), so a
    // budget sweep that wants to stay comparable to those records must be able
    // to pin depth by hand. -1 => derive.
    int depthOverride = -1;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--log-ring") logRing = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--consume") consume = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--boots") boots = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--scale-bits") scaleBits = std::stoi(argv[++i]);
        else if (a == "--first-mod-bits") firstModBits = std::stoi(argv[++i]);
        else if (a == "--corr") corr = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--levels-after") levelsAfter = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--amps") {
            amps.clear();
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) amps.push_back(std::stod(tok));
        }
        else if (a == "--depth") depthOverride = std::stoi(argv[++i]);
        else if (a == "--level-budget") {   // E-F4: "A,B" (encode,decode)
            levelBudget.clear();
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ','))
                levelBudget.push_back((uint32_t)std::stoi(tok));
        }
    }
    if (levelBudget.size() != 2) {
        std::cerr << "--level-budget needs exactly two comma-separated values"
                  << std::endl;
        return 2;
    }
    // Was `levelsAfter + 19`, hardcoded. Sweeping the budget against a fixed
    // 19 would silently mis-size the chain, so derive it as
    // harness/fhe_eval.cpp:66 and harness/striding_bootstrap.cpp:83 do.
    //
    // MEASURED 2026-08-18, AND IT IS NOT WHAT THE OLD CONSTANT ASSUMED:
    //   GetBootstrapDepth({4,4}, UNIFORM_TERNARY) = 22   (approxModDepth 14)
    //   GetBootstrapDepth({3,3}, UNIFORM_TERNARY) = 20
    //   GetBootstrapDepth({2,2}, UNIFORM_TERNARY) = 18
    // The probe set levelBudget {4,4} but provisioned depth as if it were 19 —
    // the {3,3}-shaped value. So `--levels-after 10` did NOT deliver 10 usable
    // levels after the bootstrap; at depth 29 it delivered 29-22 = 7.
    // CODESIGN_V3_POD_PLAN.md:232-237 rules the 19-vs-21 question "not a bug"
    // by noting the GPU harness runs `--level-budget 3 3` — true of THAT
    // harness, but this probe is a different binary and hardcodes {4,4}.
    //
    // CONSEQUENCE FOR COMPARABILITY: deriving changes the default depth from
    // 29 to 32, so records produced by this build are NOT chain-identical to
    // results/boot-envelope/**. Pass `--depth 29` to reproduce the old chain.
    // The header records levelBudget, the derived bootstrap depth, and which
    // source the depth came from, so no future reader has to guess.
    const uint32_t skDist = UNIFORM_TERNARY;   // matches SetSecretKeyDist below
    const uint32_t bootDepth =
        FHECKKSRNS::GetBootstrapDepth(levelBudget, (SecretKeyDist)skDist);
    const uint32_t depth = (depthOverride > 0) ? (uint32_t)depthOverride
                                               : levelsAfter + bootDepth;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetSecurityLevel(HEStd_NotSet);   // structural probe: ring set by hand
    parameters.SetRingDim(1u << logRing);
    parameters.SetScalingModSize((uint32_t)scaleBits);
    parameters.SetFirstModSize((uint32_t)firstModBits);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(depth);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE); cc->Enable(FHE);
    const uint32_t SLOTS = cc->GetRingDimension() / 2;
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalBootstrapSetup(levelBudget, {0, 0}, SLOTS, corr);
    cc->EvalBootstrapKeyGen(keys.secretKey, SLOTS);

    const uint32_t F = cc->GetScheme()->GetCKKSBootCorrectionFactor();
    // deg = round(log2(q0/Delta)); the message is scaled by 2^-(F-deg) before ModRaise,
    // so the sine argument is u = a * 2^-F  (the q0/Delta factors cancel).
    const int deg      = firstModBits - scaleBits;
    const double uPerA = std::pow(2.0, -(double)F);
    const double cubic = (4.0 * M_PI * M_PI) / 6.0;   // (2*pi)^2/6

    std::cout << "{\"harness\":\"boot_envelope_probe\",\"ringDim\":" << cc->GetRingDimension()
              << ",\"slots\":" << SLOTS << ",\"depth\":" << depth
              << ",\"scaleBits\":" << scaleBits << ",\"firstModBits\":" << firstModBits
              << ",\"deg\":" << deg << ",\"correctionFactor\":" << F
              // E-F4: emit the budget so records are self-describing. Every
              // pre-2026-08-18 boot-envelope record lacks this field and is
              // therefore implicitly {4,4} (the old hardcode).
              << ",\"levelBudget\":[" << levelBudget[0] << ","
              << levelBudget[1] << "]"
              << ",\"bootDepthDerived\":" << bootDepth
              << ",\"depthSource\":\""
              << (depthOverride > 0 ? "override" : "derived") << "\""
              << ",\"levelsAfterActual\":" << (int)(depth - bootDepth)
              << ",\"boots\":" << boots << ",\"consume\":" << consume
              << ",\"H_wrap_cliff\":" << std::pow(2.0, deg - 1)
              << ",\"H_sine_u_per_amp\":" << uPerA
              << fhe_ssm::runStamp(argc, argv) << "}" << std::endl;

    std::vector<double> ONES(SLOTS, 1.0);

    for (double amp : amps) {
        // Reference spans the full [-amp, amp] range, so the worst-slot error probes
        // the cubic term at its largest argument.
        std::vector<double> ref(SLOTS);
        for (uint32_t i = 0; i < SLOTS; i++) ref[i] = std::sin(0.37 * (i % 512)) * amp;

        auto ct = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(ref));

        auto report = [&](uint32_t n, uint32_t lvlPreBoot, double bootMs) {
            std::cout << "{\"amp\":" << amp << ",\"boots\":" << n
                      << ",\"lvlPreBoot\":" << lvlPreBoot << ",\"bootMs\":" << (long)bootMs;
            Plaintext p;
            try {
                cc->Decrypt(keys.secretKey, ct, &p);
            } catch (const std::exception&) {
                std::cout << ",\"DECODE_FAIL\":true}" << std::endl; return;
            }
            p->SetLength(SLOTS);
            auto v = p->GetRealPackedValue();

            // NaN is not zero (FHE_ERROR_LAWS §0.1): count non-finite slots explicitly
            // rather than letting std::max silently swallow them.
            uint32_t nonFinite = 0;
            double maxAbs = 0, sse = 0, sref = 0;
            for (uint32_t i = 0; i < SLOTS; i++) {
                if (!std::isfinite(v[i])) { nonFinite++; continue; }
                double d = v[i] - ref[i];
                maxAbs = std::max(maxAbs, std::abs(d));
                sse += d * d;
                sref += ref[i] * ref[i];
            }
            const double rmsRel  = (sref > 0) ? std::sqrt(sse / sref) : 0.0;
            const double u       = amp * uPerA;
            // VALIDITY GATE. A real bootstrap raises the modulus, so GetLevel() must
            // DROP across the call. If it does not, EvalBootstrap took a degenerate
            // path and the error reading below measures nothing. Never read the error
            // column without reading this one.
            std::cout << std::setprecision(6) << std::scientific
                      << ",\"bootRestored\":" << ((ct->GetLevel() < lvlPreBoot) ? "true" : "false")
                      << ",\"level\":" << ct->GetLevel()
                      << ",\"nonFinite\":" << nonFinite
                      << ",\"maxAbsErr\":" << maxAbs
                      << ",\"relErrMax\":" << maxAbs / amp
                      << ",\"relErrRms\":" << rmsRel
                      << ",\"u\":" << u
                      << ",\"predCubicAbs\":" << cubic * amp * u * u
                      << ",\"predCubicRel\":" << cubic * u * u
                      << "}" << std::defaultfloat << std::endl;
        };

        // Deplete exactly as cpu_boot_ladder does (fixed consume per cycle), so this
        // probe is directly comparable to the validated ladder arm.
        for (uint32_t n = 1; n <= boots; n++) {
            uint32_t lvlPreBoot = 0;
            double bootMs = 0;
            try {
                for (uint32_t k = 0; k < consume; k++) {
                    auto one = cc->MakeCKKSPackedPlaintext(ONES, 1, ct->GetLevel());
                    ct = cc->EvalMult(ct, one);
                    cc->RescaleInPlace(ct);
                }
                lvlPreBoot = ct->GetLevel();
                auto t0 = std::chrono::steady_clock::now();
                ct = cc->EvalBootstrap(ct);
                bootMs = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0).count();
            } catch (const std::exception& e) {
                std::cout << "{\"amp\":" << amp << ",\"boots\":" << n
                          << ",\"THREW\":\"" << e.what() << "\"}" << std::endl;
                break;
            }
            if (n == 1 || n == 2 || n == 5 || n == 10 || n == 20 || n == 35 || n == 50)
                report(n, lvlPreBoot, bootMs);
        }
    }
    (void)consume;
    return 0;
}
