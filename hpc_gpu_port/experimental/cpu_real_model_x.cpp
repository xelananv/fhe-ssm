// cpu_real_model.cpp — CPU/OpenFHE VALIDATOR for the real-model circuit.
//
// WHY THIS EXISTS (2026-07-18): hpc_gpu_port/gpu_real_model.cu was authored
// without GPU access. Its risk splits in two:
//   (a) CIRCUIT LOGIC — slot packing, rotation conventions, BSGS matvec
//       algebra, RMSNorm reduction + Newton rsqrt, level accounting;
//   (b) FIDESlib API specifics — names/signatures on the CUDA backend.
// FIDESlib's high-level API is deliberately OpenFHE-compatible, so (a) is
// fully testable here on CPU OpenFHE — which runs natively on the author's
// arm64 Mac — at a small ring, in seconds, for free. Any bug found here is
// the SAME bug in the CUDA harness. Only (b) needs NVIDIA hardware.
//
// It already earned its keep: writing it exposed two real defects in the
// first gpu_real_model.cu draft, both fixed here and ported back:
//   1. BSGS giant-step: the plaintext diagonal must be pre-rotated by -g,
//      because the giant rotation is applied AFTER the multiply. Deriving
//      it: rot(inner_g, g)[i] = sum_j d'_j[i+g] u[i+g+j], and we need
//      sum_j diag_{g+j}[i] u[i+g+j], hence d'_j = rot(diag_{g+j}, -g).
//      The draft used diag_k directly -> every matvec silently wrong.
//   2. PERIOD vs MODULUS (the deep one, caught by matvec_square FAIL with
//      err 1.17). The diagonal identity
//          y[i] = sum_k diag_k[i] * u[(i+k) mod cols],  diag_k[i]=W[i][(i+k)%cols]
//      needs rotation modulo **cols**. A ciphertext rotation is modulo the
//      packing period **D**. The draft had D=1024 with cols=768, so for
//      i+k >= 768 the rotation read PADDING ZEROS instead of wrapping —
//      silently wrong output, no error raised. Fix: the packing period must
//      EQUAL the padded input dimension (D == cols), and the input must be
//      replicated with that period.
//   3. Consequently rectangular matvecs must be CHUNKED: a d->dff map
//      (rows > D) is ceil(rows/D) separate D-column matvecs, each producing
//      one output ciphertext; dff->d is the transpose, ceil(cols/D) matvecs
//      summed. This also removes the draft's mixed-period defect (period
//      1024 in and 4096 out inside one ciphertext). Elementwise ops (poly
//      gates) act on each chunk independently — no cross-channel coupling —
//      so chunking is exact, and the FLOP count is inherently the same.
//
// Build:  cmake --build harness/build -j   (target cpu_real_model)
// Run:    ./harness/build/cpu_real_model            # all logic tests
//         ./harness/build/cpu_real_model --bootstrap  # + a bootstrap test
// Output: one JSON verdict per test; exit 0 = all pass, 3 = a hard failure.

#include "openfhe.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <set>
#include <string>
#include <vector>

// Provenance stamp (harness file + sha256, git commit, build time, exact argv).
#include "run_stamp.h"
// S3.7 X2 (2026-09-10): the elementwise plaintext cache -- the periodic encoder + expander of
// hpc_gpu_port/periodic_encode.hpp (S3.7 V5, proven bit-identical to the dense plaintext at 2^17)
// and the hash map the cache lives in.
#include "periodic_encode.hpp"
#include <cstring>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <algorithm>            // S3.7 X8 (2026-09-10): std::find in the pool-LRU census
#include <fstream>              // S3.7 X13: the sidecar file

using namespace lbcrypto;

// S3.7 V2 (2026-09-05): the timer shapes of gpu_real_model.cu (normBootMs, the
// layer-loop stage windows, the host-encode counters), mirrored here with the
// SAME names so the accounting compiles and runs on the Mac -- the .cu cannot be
// compiled here. Pure instrumentation: no test's arithmetic changes.
using Clock = std::chrono::steady_clock;
static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static int g_hard = 0, g_soft = 0;
static void verdict(const std::string& name, bool ok, double err, bool hard) {
    std::cout << "{\"test\":\"" << name << "\",\"result\":\"" << (ok ? "ok" : "FAIL")
              << "\",\"err\":" << err << "}" << std::endl;
    if (!ok) (hard ? g_hard : g_soft)++;
}

int main(int argc, char** argv) {
    bool doBootstrap = false;
    uint32_t logRing = 14;      // 2^14 is fine for the algebra tests; small-ring
                                // bootstrapping trips an OpenFHE correction-factor
                                // constraint, so --bootstrap wants >= 2^16.
    // ---- S3.3 (2026-09-03) validation cells, each OFF unless named by --cell:
    //   shared-babies : the BSGS baby set computed once and reused across the
    //                   chunk calls gives BIT-IDENTICAL ciphertexts (polynomials
    //                   compared element-wise, not just decrypted values).
    //   newton-depth2 : y' = 1.5y + ((-0.5 ms) y)(y y) against the stock chain
    //                   at the same seed/iterations: relErr of both vs exact, the
    //                   difference between them, and the levels each consumes.
    //   pf-trace      : ONE full layer (T = 1, h = 1) under the parent-first
    //                   placements with REAL OpenFHE bootstraps (needs --bootstrap,
    //                   2^16, level budget {3,3}); prints (site, GetLevel,
    //                   remaining) at every refresh/boot so the sequence can be
    //                   compared with results/theory/tick_level_trace.py
    //                   --trace-levels at the same (depth, lam, F, m, I, Dpad, dff).
    // The pre-existing tests above/below run exactly as before (their output is
    // unchanged when no --cell is given).
    std::set<std::string> cells;
    std::string pfEntry = "loop"; bool cellNd2 = false; int cellIters = 3; int cellF = 3, cellM = 4;
    std::string cellSchedule = "parent-first";
    // S3.7 V1 (2026-09-05, S3.6 T18 §3): the norm's eps. The GPU harness (and this twin)
    // accepted an `eps` argument and never read it, so the circuit normalised by
    // 1/sqrt(mean(x^2)) where the model uses mean + 1e-5. DEFAULT ON (the model's
    // frame); --no-norm-eps restores the recorded no-eps chain. Mirrors
    // gpu_real_model.cu's normEpsOn / normEpsUsed / the summary field normEps.
    bool normEpsOn = true; double normEpsUsed = 0.0;
    // S3.7 V1: --norm-seed A B ITERS (default OFF) runs the norm once more with a REAL
    // bundle seed, the CPU twin of gpu_real_model.cu's selftest 4b ("rmsnormSeeded").
    double normSeedA = 0.0, normSeedB = 0.0; int normSeedIters = 0;
    // S3.7 V2 (2026-09-05): --no-stage-timers mirrors the .cu flag (default ON; OFF =
    // no stage windows, the stage fields print 0). --allow-small-ring-boot lets
    // --bootstrap run below 2^16 (a Mac RAM aid for the pf-trace cell: at 2^16 the
    // cell peaked at 7,414,693,888 B RSS = 6.9 GiB, above the 6 GB cap -- record
    // results/theory/s36_20260905/t03_bootcount/cpu_pftrace_pf_nd2_i1_control.time;
    // at 2^15 the level trace is the same circuit (depth 30, budget {3,3}, full
    // packing) and the caller owns the small-ring bootstrap precision).
    bool stageTimers = true, allowSmallRingBoot = false;
    // ==== S3.7 X1 boot schedule (2026-09-10; S3.6 T03 §1.3-§1.4, §2.2, §5): state ====
    // The CPU mirror of gpu_real_model_x.cu's three levers, every one OFF by default so the twin's
    // selftests and cells are unchanged unless a flag names them:
    //   --deferred-scan M   the scan carry is only READ for M-1 token-steps and folded every M-th
    //                       (canonical_carry_trace.py SimDeferredScan; boots on the carry 24/M per tick);
    //   --u-drop-level C    the u carry is LevelReduceInternalInPlace'd to C instead of bootstrapped
    //                       (T03 `mixed` rule; OpenFHE's public LevelReduce is a NO-OP under FLEXIBLEAUTO,
    //                       rns-leveledshe.cpp:373-380, hence the Internal call);
    //   --branch-out-boot   boot the branch OUTPUT before the residual add iff it is >= 1 level deeper
    //                       than Hs and deeper than the post-boot level (T03 Prop.-1 variant A);
    //   --branch-out-post N the post-boot level that rule uses (-1 = learn: canonLevel, else the first
    //                       boot; a site reached before either is known is skipped and counted);
    //   --ticks N           the multi-tick carry cell inside --cell pf-trace (N > 1), boots per tick;
    //   --canonical-carry   the twin's mirror of the .cu's `canon` hand-off (HEAD 8afcfd7 rule);
    //   --u-drop-exit L     the algebra cell's exit level of the carry before the drop (default 7).
    // Cells: --cell deferred-scan (eager vs deferred, M in {2,4,8,16}), --cell u-drop (value exactness
    // and the FLEXIBLEAUTO scale-ratio bound), --cell pf-trace --ticks N (counts with real bootstraps).
    int x1DeferredScan = 0, x1UDropLevel = 0, x1BranchOutPost = -1, x1Ticks = 1, x1UDropExit = 7;
    bool x1BranchOutBoot = false, x1CanonicalCarry = false;
    // ==== S3.7 X2 pt cache (2026-09-10; S3.6 T05/T06 §3.2, §5.1-§5.3, N5): state ====
    // The CPU mirror of gpu_real_model_x.cu's two levers, both OFF by default (with both off every
    // plaintext is built exactly as at HEAD: cc->MakeCKKSPackedPlaintext at the operand's level):
    //   --pt-cache          every elementwise plaintext request (ptAt / the cells' ptL = the twin's mkPt)
    //                       is keyed by (call-site id, the vector's D-period = its value class, level);
    //                       a hit returns the plaintext built at the first use, a miss builds it with
    //                       PeriodicEncoder::encode + expandCompToPlaintext (the V5 route, bit-identical
    //                       to the dense encode) and keeps it; a vector that is not D-periodic in the
    //                       block layout takes the dense path and is not cached (ptCacheDenseFallback).
    //                       T05 §3.2 counts 1,688 such plaintexts per warm demo tick, every one
    //                       D-periodic (§5.1) and repeating tick to tick (§5.2: the level trajectory
    //                       has period 1 from tick 0 under the canonical carry, so every key recurs).
    //   --pt-cache-verify   on every miss also build the dense plaintext and compare the element limb
    //                       for limb + level/scale/degree (ptCacheVerified / ptCacheVerifyFail);
    //   --pt-cache-max-gb G stop inserting when the resident limbs would exceed G GiB (N x 8 B per
    //                       limb; default 16); later keys take the dense path (ptCacheSkipped);
    //   --pt-scalar         a CONSTANT vector (every slot bitwise equal) multiplied into / added to a
    //                       ciphertext of noiseScaleDeg 1 uses EvalMult(ct, double) / EvalAdd(ct, double)
    //                       instead of a plaintext. THE RULE, from the pod's CPU library source
    //                       (vendor/openfhe-fideslib/openfhe-src, OpenFHE 1.5.1): on a deg-1 operand the
    //                       scalar route forms round(c * sf_real(l)) per tower (ckksrns-leveledshe.cpp
    //                       GetElementForEvalMult / GetElementForEvalAddOrSub) and the plaintext route's
    //                       AdjustFor{Mult,AddOrSub}IsNoOp (rns-leveledshe.cpp :47-:70: same level, same
    //                       towers, both deg 1) is true, so both multiply the SAME words and record the
    //                       same scale -- word-identical (T05 §5.3: 0 differing words in 84 cells at
    //                       2^15/16/17). On a deg-2 operand (a pending FLEXIBLEAUTO rescale, X1 REPORT
    //                       §0) the plaintext route morphs the plaintext into a ciphertext and re-scales
    //                       it (AdjustLevelsAndDepthToOneInPlace, ckksrns-leveledshe.cpp :736) -- one
    //                       extra rounding, 2e-12..1.06e-11 relRms -- so the rule DECLINES the scalar
    //                       route there (ptScalarDeg2Fallback) and the plaintext route runs. FOUND
    //                       2026-09-10 (X2 REPORT §3, x2_mul_probe): even on a deg-1 operand the library's
    //                       scalar multiply can round c*sf to a DIFFERENT integer than the encoder --
    //                       GetElementForEvalMult forms `operand / approxFactor * scFactor + 0.5`, which
    //                       the compiler may contract into an FMA (this Mac's arm64 build does), so its
    //                       integer is round(exact(c*sf) + 0.5) while the encoder's is llround(double(c*sf)):
    //                       one double ulp apart whenever the exact product sits in the boundary zone
    //                       (1/48 and 0.0208 -> K+2 at 2^53.4; six other constants identical). So the rule
    //                       is SELF-CHECKING: the first use of each (constant, level, mul|add) class runs
    //                       BOTH routes and compares the ciphertexts word for word; the class is admitted
    //                       to the scalar route only if identical (ptScalarClasses / ptScalarClassesRejected),
    //                       otherwise it stays on the plaintext route for the session (ptScalarRejectedOps).
    //                       The result handed back by the check is the plaintext route's. Hence the flag
    //                       never changes a word of any output, on any library build. (On the pod the GPU
    //                       rescale is eager, T05 §5.3, so nearly every constant site is deg 1 there; on
    //                       this lazy CPU library many are deg 2 and fall back -- the counters say how many.)
    // Cells: --cell pt-scalar (the deg-1 word identity, and the deg-2 relaxation's error, per constant);
    // with --pt-cache on, --cell pf-trace [--ticks N] prints one ptCacheTick census line per tick.
    bool x2PtCache = false, x2PtScalar = false, x2PtCacheVerify = false; double x2PtCacheMaxGb = 16.0;
    // ==== S3.7 X8 drops + X13 sidecar (2026-09-10; S3.6 T07 §1.2, T18 §5): state ====
    // The CPU mirror of gpu_real_model_x.cu's two levers, every flag OFF by default (the selftests and cells are
    // unchanged unless a flag names them):
    //   --drop-wv/--drop-wkr/--drop-wout/--drop-win R   the EXACT DROP of that matvec class's INPUT to R remaining
    //                       levels (target level depth - R, R + 1 limbs) before the diagonals are encoded, in the
    //                       multi-tick cell (--cell pf-trace --ticks N) at the four class sites; 0 = off. The
    //                       discipline of X1's uDrop (X1 REPORT §0, §3.3): under FLEXIBLEAUTO the operand leaving a
    //                       multiply carries a PENDING rescale (noiseScaleDeg 2, RescaleInPlace is a no-op), so its
    //                       settled level is GetLevel() + 1; settle first (ModReduceInternalInPlace(1)), then
    //                       LevelReduceInternalInPlace to the target; an input already at or below the target is
    //                       untouched. Refused together with --schedule parent-first (T07 fact 4, as the .cu does).
    //   --pool-levels P     the .cu's compPoolGet LRU capacity (stock 2), mirrored here as a pure COUNTER: every
    //                       class-site matvec input level not among the P most recently used is one pool (re)build
    //                       (the term T07 fact 3 says vanishes at the code floor); per tick in the x8Tick line.
    //   --x8-census         print the per-class lines (level/degree/limbs in and out at every class site) and the
    //                       per-tick x8Tick / cell-end x8-ticks lines in the multi-tick cell; implied by any drop.
    //                       With every X8 flag off the ticks cell prints exactly the X1 lines (flag-off identity).
    //   --rsqrt-iters-sidecar FILE   per-site Newton counts (flat {site: N} or the seeds form {site: {"iters": N}},
    //                       "_"-prefixed keys are metadata): the twin's one layer uses L0.tm / L0.cm in both cells;
    //                       rsqrtItersSum in the summary is the sum over every site in the file (what the .cu would
    //                       use over the bundle's 49 sites), checked against the file's own "_sumIters" when present.
    // Cells: --cell x8-drop (value exactness of the settled drop vs the un-dropped path, the naive drop's kept level,
    // the limb count and the FLEXIBLEAUTO scale ratio, at R in {7, 4} and several exit levels, no bootstrap).
    int x8DropWv = 0, x8DropWkr = 0, x8DropWout = 0, x8DropWin = 0, x8PoolLevels = 2;
    bool x8CensusFlag = false;
    long long x8DropsApplied = 0, x8DropsSettled = 0, x8DropsSkippedDeeper = 0, x8DropsAtTarget = 0;
    std::string x13SidecarPath; std::map<std::string, int> x13Iters; long long x13Sum = 0, x13SumFile = -1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--bootstrap") doBootstrap = true;
        else if (a == "--no-norm-eps") normEpsOn = false;
        else if (a == "--norm-seed") { normSeedA = std::stod(argv[++i]); normSeedB = std::stod(argv[++i]); normSeedIters = std::stoi(argv[++i]); }
        else if (a == "--no-stage-timers") stageTimers = false;
        else if (a == "--stage-timers") stageTimers = true;
        else if (a == "--allow-small-ring-boot") allowSmallRingBoot = true;
        else if (a == "--log-ring") logRing = (uint32_t)std::stoi(argv[++i]);
        else if (a == "--cell") cells.insert(argv[++i]);
        else if (a == "--pf-entry") pfEntry = argv[++i];
        else if (a == "--newton-depth2") cellNd2 = true;
        else if (a == "--schedule") cellSchedule = argv[++i];
        else if (a == "--iters") cellIters = std::stoi(argv[++i]);
        else if (a == "--boot-floor") cellF = std::stoi(argv[++i]);
        else if (a == "--margin") cellM = std::stoi(argv[++i]);
        // ==== S3.7 X1 boot schedule ====
        else if (a == "--deferred-scan") x1DeferredScan = std::stoi(argv[++i]);
        else if (a == "--u-drop-level") x1UDropLevel = std::stoi(argv[++i]);
        else if (a == "--branch-out-boot") x1BranchOutBoot = true;
        else if (a == "--branch-out-post") x1BranchOutPost = std::stoi(argv[++i]);
        else if (a == "--ticks") x1Ticks = std::stoi(argv[++i]);
        else if (a == "--canonical-carry") x1CanonicalCarry = true;
        else if (a == "--u-drop-exit") x1UDropExit = std::stoi(argv[++i]);
        // ==== S3.7 X2 pt cache ====
        else if (a == "--pt-cache") x2PtCache = true;
        else if (a == "--pt-scalar") x2PtScalar = true;
        else if (a == "--pt-cache-verify") x2PtCacheVerify = true;
        else if (a == "--pt-cache-max-gb") x2PtCacheMaxGb = std::stod(argv[++i]);
        // ==== S3.7 X8 drops + X13 sidecar ====
        else if (a == "--drop-wv") x8DropWv = std::stoi(argv[++i]);
        else if (a == "--drop-wkr") x8DropWkr = std::stoi(argv[++i]);
        else if (a == "--drop-wout") x8DropWout = std::stoi(argv[++i]);
        else if (a == "--drop-win") x8DropWin = std::stoi(argv[++i]);
        else if (a == "--pool-levels") x8PoolLevels = std::stoi(argv[++i]);
        else if (a == "--x8-census") x8CensusFlag = true;
        else if (a == "--rsqrt-iters-sidecar") x13SidecarPath = argv[++i];
    }
    if (x1DeferredScan < 0 || x1UDropLevel < 0 || x1Ticks < 1 || x1UDropExit < 0) {
        std::cerr << "{\"fatal\":\"X1: --deferred-scan/--u-drop-level/--u-drop-exit must be >= 0 and --ticks >= 1\"}\n"; return 2;
    }
    // ==== S3.7 X8 drops + X13 sidecar: guards (the .cu's, refuse loudly, never degrade) ====
    if (x8DropWv < 0 || x8DropWkr < 0 || x8DropWout < 0 || x8DropWin < 0 || x8PoolLevels < 1) {
        std::cerr << "{\"fatal\":\"X8: --drop-* must be >= 0 (remaining levels; 0 = off) and --pool-levels >= 1\"}\n"; return 2;
    }
    const bool x8AnyDrop = (x8DropWv > 0 || x8DropWkr > 0 || x8DropWout > 0 || x8DropWin > 0);
    const bool x8Census = x8CensusFlag || x8AnyDrop;
    if (x8AnyDrop && cellSchedule == "parent-first") {
        std::cerr << "{\"fatal\":\"--drop-* excludes --schedule parent-first (T07 section 1.2 fact 4: levers on operand depth do not stack); pass --schedule stock and compare >= 4 ticks, mean and range\"}\n"; return 2;
    }
    if (!x13SidecarPath.empty()) {
        std::ifstream sf(x13SidecarPath);
        if (!sf) { std::cerr << "{\"fatal\":\"--rsqrt-iters-sidecar: cannot open\",\"path\":\"" << x13SidecarPath << "\"}\n"; return 2; }
        std::stringstream ss; ss << sf.rdbuf(); const std::string js = ss.str();
        auto ws = [](char ch) { return ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t'; };   // the .cu's reader, verbatim
        size_t pos = 0;
        while ((pos = js.find('"', pos)) != std::string::npos) {
            const size_t ke = js.find('"', pos + 1); if (ke == std::string::npos) break;
            const std::string key = js.substr(pos + 1, ke - pos - 1);
            size_t p = ke + 1; while (p < js.size() && ws(js[p])) p++;
            if (p >= js.size() || js[p] != ':') { pos = ke + 1; continue; }
            p++; while (p < js.size() && ws(js[p])) p++;
            if (key.empty() || key[0] == '_') {
                if (key == "_sumIters" && p < js.size() && js[p] != '"') x13SumFile = std::stoll(js.substr(p));
                if (p < js.size() && js[p] == '"') { const size_t ve = js.find('"', p + 1); pos = (ve == std::string::npos) ? js.size() : ve + 1; }
                else pos = p;
                continue;
            }
            if (p < js.size() && js[p] == '{') {
                int braces = 0; size_t q = p;
                for (; q < js.size(); q++) { if (js[q] == '{') braces++; else if (js[q] == '}') { braces--; if (braces == 0) break; } }
                const std::string obj = js.substr(p, q - p + 1);
                const size_t ip = obj.find("\"iters\"");
                if (ip == std::string::npos) { std::cerr << "{\"fatal\":\"--rsqrt-iters-sidecar: site without iters\",\"site\":\"" << key << "\"}\n"; return 2; }
                x13Iters[key] = std::stoi(obj.substr(obj.find(':', ip) + 1));
                pos = q + 1;
            } else {
                x13Iters[key] = std::stoi(js.substr(p));
                pos = p;
            }
        }
        if (x13Iters.empty()) { std::cerr << "{\"fatal\":\"--rsqrt-iters-sidecar: no sites read\",\"path\":\"" << x13SidecarPath << "\"}\n"; return 2; }
        for (const auto& kv : x13Iters) {
            if (kv.second < 0) { std::cerr << "{\"fatal\":\"--rsqrt-iters-sidecar: negative iters\",\"site\":\"" << kv.first << "\"}\n"; return 2; }
            x13Sum += kv.second;
        }
        if (x13SumFile >= 0 && x13SumFile != x13Sum) {
            std::cerr << "{\"fatal\":\"--rsqrt-iters-sidecar: _sumIters disagrees with the sites read\",\"sumFile\":" << x13SumFile << ",\"sumRead\":" << x13Sum << "}\n"; return 2;
        }
        if (!x13Iters.count("L0.tm") || !x13Iters.count("L0.cm")) { std::cerr << "{\"fatal\":\"--rsqrt-iters-sidecar: the twin needs L0.tm and L0.cm\"}\n"; return 2; }
        std::cout << "{\"rsqrtSidecar\":\"" << x13SidecarPath << "\",\"sites\":" << x13Iters.size() << ",\"sumIters\":" << x13Sum << ",\"sumItersFile\":" << x13SumFile << "}" << std::endl;
    }
    if (x8AnyDrop || !x13SidecarPath.empty())
        std::cout << "{\"x8Levers\":true,\"dropWv\":" << x8DropWv << ",\"dropWkr\":" << x8DropWkr << ",\"dropWout\":" << x8DropWout << ",\"dropWin\":" << x8DropWin
                  << ",\"poolLevels\":" << x8PoolLevels << ",\"rsqrtSidecar\":\"" << x13SidecarPath << "\",\"baseline\":\"" << cellSchedule << "\"}" << std::endl;
    if (cells.count("pf-trace") && !doBootstrap) { doBootstrap = true; }   // the trace needs real boots
    if (doBootstrap && logRing < 16 && !allowSmallRingBoot) logRing = 16;

    // ---- small parameters: this validates ALGEBRA, not performance -------
    const uint32_t d = 48;         // model channels (deliberately NOT a power
                                   // of 2, to exercise the padding path)
    const uint32_t dff = 128;      // ffn channels -> chunked into dff/D maps
    uint32_t D = 1;                // packing period == PADDED INPUT DIM.
    while (D < d) D <<= 1;         // = 64 here; rotation mod D == mod cols
    const uint32_t NCHUNK = (dff + D - 1) / D;   // ffn output chunks (=2)

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1u << logRing);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    std::vector<uint32_t> levelBudget = {4, 4};
    if (cells.count("pf-trace")) levelBudget = {3, 3};     // the demo budget (HARNESS_BUGS B4: {4,4} consumes 22)
    const uint32_t levelsAfter = 10;
    const uint32_t depth = doBootstrap ? (cells.count("pf-trace") ? 30u : levelsAfter + 19) : 30;
    parameters.SetMultiplicativeDepth(depth);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE); if (doBootstrap) cc->Enable(FHE);
    const uint32_t SLOTS = cc->GetRingDimension() / 2;
    const uint32_t REP = SLOTS / D;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    const int BS = (int)std::ceil(std::sqrt((double)D));
    {
        std::vector<int32_t> rots;
        for (int j = 1; j < BS; j++) rots.push_back(j);
        for (int g = 0; g < (int)D; g += BS) if (g) rots.push_back(g);
        for (int p = 1; p < (int)D; p <<= 1) rots.push_back(p);
        std::sort(rots.begin(), rots.end());
        rots.erase(std::unique(rots.begin(), rots.end()), rots.end());
        cc->EvalRotateKeyGen(keys.secretKey, rots);
    }
    if (doBootstrap) { cc->EvalBootstrapSetup(levelBudget, {0, 0}, SLOTS);
                       cc->EvalBootstrapKeyGen(keys.secretKey, SLOTS); }

    std::cout << "{\"harness\":\"cpu_real_model\",\"ringDim\":" << cc->GetRingDimension()
              << ",\"slots\":" << SLOTS << ",\"D\":" << D << ",\"REP\":" << REP
              << ",\"d\":" << d << ",\"dff\":" << dff << ",\"nChunk\":" << NCHUNK
              << ",\"BS\":" << BS << ",\"depth\":" << depth << "}" << std::endl;

    struct TrackedCt { Ciphertext<DCRTPoly> ct; uint32_t used = 0; };

    // ---- S3.7 V2 (2026-09-05): the .cu's timer accumulators, same names, same
    // arithmetic (gpu_real_model.cu, the block after syncDev). evalMs/bootMs/encMs/
    // decMs are the historical windows (matvec + rmsnorm; boot; enc; dec);
    // normBootMs = the rmsnorm windows' own boots (in evalMs AND bootMs, moved
    // out at the window exits); the stage windows bill wall minus the timed delta
    // inside them; hostPtEncode* counts every ptAt (the twin's mkPt). syncDev is a
    // no-op here: the CPU twin has no device to drain.
    double evalMs = 0, bootMs = 0, encMs = 0, decMs = 0, normBootMs = 0;
    double hostPtEncodeMs = 0, hostPtEncodeStageMs = 0;
    long long hostPtEncodeCount = 0, hostPtEncodeStageCount = 0;
    int stageOpenDepth = 0;
    double layerLoopMs = 0, layerLoopTimedMs = 0;
    double shiftMixMs = 0, scanMs = 0, gateMs = 0, hiddenMs = 0, residualMs = 0;
    auto syncDev = [&] { /* CPU twin: no device */ };
    struct StageMark { Clock::time_point t; double ev, nb, bo, en, de; };
    auto stageOpen = [&]() -> StageMark {
        if (!stageTimers) return StageMark{};
        syncDev();
        stageOpenDepth++;
        return StageMark{Clock::now(), evalMs, normBootMs, bootMs, encMs, decMs};
    };
    auto stageClose = [&](const StageMark& m, double& acc) {
        if (!stageTimers) return;
        syncDev();
        stageOpenDepth--;
        const double wall = msSince(m.t);
        const double timed = ((evalMs - m.ev) - (normBootMs - m.nb)) + (bootMs - m.bo)
                           + (encMs - m.en) + (decMs - m.de);
        acc += wall - timed;
    };

    // pack n channel values, replicated with period D across all slots
    auto packCh = [&](const std::vector<double>& v) {
        std::vector<double> s(SLOTS, 0.0);
        for (uint32_t r = 0; r < REP; r++)
            for (size_t i = 0; i < v.size() && i < D; i++) s[r * D + i] = v[i];
        return s;
    };
    auto enc = [&](const std::vector<double>& packed) {
        auto t = Clock::now();                                  // S3.7 V2: encCh's window
        auto pt = cc->MakeCKKSPackedPlaintext(packed);
        TrackedCt r{cc->Encrypt(keys.publicKey, pt), 0};
        encMs += msSince(t); return r;
    };
    auto dec = [&](TrackedCt& c) {
        auto t = Clock::now();                                  // S3.7 V2: decCh's window
        Plaintext p; cc->Decrypt(keys.secretKey, c.ct, &p);
        p->SetLength(SLOTS); auto v = p->GetRealPackedValue();
        decMs += msSince(t); return v;
    };
    // ==== S3.7 X2 pt cache (2026-09-10): the cache, the periodicity test, the scalar rule ====
    // Key = (call-site id, level, FNV-1a of the D-period); the entry keeps the period itself and a hit
    // requires it to match bit for bit, so a hash collision can never return a wrong plaintext (it takes
    // the dense path instead). The cached object is the OpenFHE Plaintext handle, shared by every user.
    struct X2Key { int site; uint32_t lvl; uint64_t hash;
        bool operator==(const X2Key& o) const { return site == o.site && lvl == o.lvl && hash == o.hash; } };
    struct X2KeyHash { size_t operator()(const X2Key& k) const {
        return (size_t)k.hash ^ ((size_t)k.lvl * 0x9E3779B97F4A7C15ull) ^ ((size_t)(k.site + 1) * 0xC2B2AE3D27D4EB4Full); } };
    struct X2Entry { Plaintext pt; std::vector<double> period; long long hits = 0; uint32_t limbs = 0; int lastTick = -1; };
    std::unordered_map<X2Key, X2Entry, X2KeyHash> x2Cache;
    long long ptCalls = 0, ptCacheEntries = 0, ptCacheHits = 0, ptCacheMisses = 0, ptCacheDenseFallback = 0,
              ptCacheSkipped = 0, ptCacheExcluded = 0, ptCacheVerified = 0, ptCacheVerifyFail = 0;
    double ptCacheMissMs = 0; size_t ptCacheLimbs = 0;
    long long ptScalarOps = 0, ptScalarMul = 0, ptScalarAdd = 0, ptScalarDeg2Fallback = 0;
    int x2TickNo = 0; long long x2DistinctThisTick = 0;   // the per-tick census (pf-trace / x1-ticks)
    std::unique_ptr<fhe_ssm::PeriodicEncoder> x2Enc; fhe_ssm::PeriodicEncoder::Scratch x2Scratch;
    uint32_t x2LogRep = 0; while ((1u << x2LogRep) < REP) x2LogRep++;
    // D-periodic in the block layout: every replica of the slot vector equals replica 0, bit for bit
    // (T05 §5.1: packCh / packChunk / the constants are periodic by construction; the interleaved and
    // lane-masked layouts are not and take the dense path).
    auto x2Period = [&](const std::vector<double>& v) -> const double* {
        if (v.size() != SLOTS) return nullptr;
        for (uint32_t r = 1; r < REP; r++)
            if (std::memcmp(v.data() + (size_t)r * D, v.data(), (size_t)D * sizeof(double)) != 0) return nullptr;
        return v.data();
    };
    auto x2Hash = [](const double* p, size_t n) -> uint64_t {          // FNV-1a over the period's bytes
        uint64_t h = 1469598103934665603ull; const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
        for (size_t i = 0; i < n * sizeof(double); i++) { h ^= b[i]; h *= 1099511628211ull; }
        return h;
    };
    auto x2IsConst = [&](const std::vector<double>& v, double& val) -> bool {   // every slot bitwise equal (a NaN never is)
        if (v.empty()) return false; val = v[0];
        for (size_t i = 1; i < v.size(); i++) if (!(v[i] == val)) return false;
        return true;
    };
    auto x2Identical = [&](const Plaintext& a, const Plaintext& b, uint64_t& mism) -> bool {   // V5's elemIdentical + metadata
        const DCRTPoly& ea = a->GetElement<DCRTPoly>(); const DCRTPoly& eb = b->GetElement<DCRTPoly>();
        mism = 0;
        if (ea.GetNumOfElements() != eb.GetNumOfElements() || ea.GetFormat() != eb.GetFormat()) return false;
        for (uint32_t l = 0; l < ea.GetNumOfElements(); l++) {
            const auto& va = ea.GetElementAtIndex(l).GetValues(); const auto& vb = eb.GetElementAtIndex(l).GetValues();
            if (va.GetLength() != vb.GetLength() || va.GetModulus() != vb.GetModulus()) return false;
            for (uint32_t i = 0; i < va.GetLength(); i++) if (va[i] != vb[i]) mism++;
        }
        return mism == 0 && a->GetLevel() == b->GetLevel() && a->GetScalingFactor() == b->GetScalingFactor()
            && a->GetNoiseScaleDeg() == b->GetNoiseScaleDeg();
    };
    auto x2CtSame = [&](const Ciphertext<DCRTPoly>& a, const Ciphertext<DCRTPoly>& b, uint64_t& mism) -> bool {   // words + metadata
        const auto& ea = a->GetElements(); const auto& eb = b->GetElements(); mism = 0;
        if (ea.size() != eb.size()) return false;
        for (size_t k = 0; k < ea.size(); k++) {
            if (ea[k].GetNumOfElements() != eb[k].GetNumOfElements()) return false;
            for (uint32_t l = 0; l < ea[k].GetNumOfElements(); l++) {
                const auto& va = ea[k].GetElementAtIndex(l).GetValues(); const auto& vb = eb[k].GetElementAtIndex(l).GetValues();
                if (va.GetLength() != vb.GetLength()) return false;
                for (uint32_t i = 0; i < va.GetLength(); i++) if (va[i] != vb[i]) mism++;
            }
        }
        return mism == 0 && a->GetLevel() == b->GetLevel() && a->GetScalingFactor() == b->GetScalingFactor()
            && a->GetNoiseScaleDeg() == b->GetNoiseScaleDeg();
    };
    // The scalar route's class table (flag block above): (constant bits, level, isAdd) -> admitted. The first use
    // of a class performs the plaintext op AND the scalar op and admits the class only if the two ciphertexts are
    // word-identical; the caller receives the plaintext route's result. `mkpt` builds the class's plaintext by the
    // caller's own route (ptAtS / ptL), so the check exercises exactly what the cache would serve.
    std::map<std::tuple<uint64_t, uint32_t, bool>, bool> x2ScalarClass;
    long long ptScalarClasses = 0, ptScalarClassesRejected = 0, ptScalarRejectedOps = 0;
    auto x2ScalarOp = [&](Ciphertext<DCRTPoly>& ct, double cval, bool isAdd, const std::function<Plaintext()>& mkpt) -> bool {
        if (ct->GetNoiseScaleDeg() != 1) { ptScalarDeg2Fallback++; return false; }
        uint64_t bits = 0; std::memcpy(&bits, &cval, sizeof bits);
        const auto key = std::make_tuple(bits, (uint32_t)ct->GetLevel(), isAdd);
        auto it = x2ScalarClass.find(key);
        if (it == x2ScalarClass.end()) {
            Plaintext pt = mkpt();
            Ciphertext<DCRTPoly> rp = isAdd ? cc->EvalAdd(ct, pt) : cc->EvalMult(ct, pt);
            Ciphertext<DCRTPoly> rs = isAdd ? cc->EvalAdd(ct, cval) : cc->EvalMult(ct, cval);
            uint64_t mism = 0; const bool same = x2CtSame(rp, rs, mism);
            x2ScalarClass[key] = same; ptScalarClasses++; if (!same) ptScalarClassesRejected++;
            { std::ostringstream o; o << "{\"ptScalarClass\":" << (same ? "\"admitted\"" : "\"rejected\"") << ",\"const\":" << std::setprecision(17) << cval
                                     << ",\"level\":" << ct->GetLevel() << ",\"op\":\"" << (isAdd ? "add" : "mul") << "\",\"mismatchWords\":" << mism << "}";
              std::cout << o.str() << std::endl; }   // own stream: std::cout's precision is left as it was
            ct = rp; return true;
        }
        if (!it->second) { ptScalarRejectedOps++; return false; }
        ct = isAdd ? cc->EvalAdd(ct, cval) : cc->EvalMult(ct, cval);
        ptScalarOps++; if (isAdd) ptScalarAdd++; else ptScalarMul++;
        return true;
    };
    // The lookup. site < 0 = never cached (the BSGS weight diagonals: on the demo they come from the
    // store, T05 §3.2 excludes them). Returns a null Plaintext when the request must take the dense path.
    auto x2Lookup = [&](const std::vector<double>& v, uint32_t lvl, int site) -> Plaintext {
        if (!x2PtCache) return Plaintext();
        if (site < 0) { ptCacheExcluded++; return Plaintext(); }
        const double* per = x2Period(v);
        if (!per) { ptCacheDenseFallback++; return Plaintext(); }
        X2Key key{site, lvl, x2Hash(per, D)};
        auto it = x2Cache.find(key);
        if (it != x2Cache.end()) {
            if (std::memcmp(it->second.period.data(), per, (size_t)D * sizeof(double)) != 0) { ptCacheDenseFallback++; return Plaintext(); }
            it->second.hits++; ptCacheHits++;
            if (it->second.lastTick != x2TickNo) { it->second.lastTick = x2TickNo; x2DistinctThisTick++; }
            return it->second.pt;
        }
        auto t0 = Clock::now();
        if (!x2Enc) x2Enc = std::make_unique<fhe_ssm::PeriodicEncoder>(cc, x2LogRep);
        const uint32_t limbs = x2Enc->limbsAt(lvl);
        if ((double)(ptCacheLimbs + limbs) * (double)cc->GetRingDimension() * 8.0 > x2PtCacheMaxGb * 1073741824.0) {
            ptCacheSkipped++; return Plaintext();
        }
        std::vector<uint64_t> comp;
        const int rc = x2Enc->encode(per, lvl, comp, x2Scratch);
        if (rc != fhe_ssm::PE_OK) { ptCacheDenseFallback++; return Plaintext(); }   // the dense path throws on the same inputs
        Plaintext pt = fhe_ssm::expandCompToPlaintext(cc, *x2Enc, comp, lvl);
        ptCacheMissMs += msSince(t0); ptCacheMisses++;
        if (x2PtCacheVerify) {
            Plaintext dense = cc->MakeCKKSPackedPlaintext(v, 1, lvl);
            uint64_t mism = 0; const bool same = x2Identical(pt, dense, mism);
            ptCacheVerified++;
            if (!same) { ptCacheVerifyFail++;
                std::cout << "{\"ptCacheVerifyFail\":true,\"site\":" << site << ",\"level\":" << lvl << ",\"mismatchWords\":" << mism << "}" << std::endl; }
        }
        X2Entry e; e.pt = pt; e.period.assign(per, per + D); e.limbs = limbs; e.lastTick = x2TickNo;
        ptCacheLimbs += limbs; ptCacheEntries++; x2DistinctThisTick++;
        x2Cache.emplace(key, std::move(e));
        return pt;
    };
    auto x2TickLine = [&](const std::string& label) {   // the census of one tick, deltas since the previous line
        static long long c0 = 0, h0 = 0, m0 = 0, f0 = 0, s0 = 0, x0 = 0, o0 = 0, g0 = 0;
        std::cout << "{\"ptCacheTick\":" << x2TickNo << ",\"label\":\"" << label << "\",\"ptCalls\":" << (ptCalls - c0)
                  << ",\"ptCacheHits\":" << (ptCacheHits - h0) << ",\"ptCacheMisses\":" << (ptCacheMisses - m0)
                  << ",\"ptCacheDenseFallback\":" << (ptCacheDenseFallback - f0) << ",\"ptCacheSkipped\":" << (ptCacheSkipped - s0)
                  << ",\"ptCacheExcluded\":" << (ptCacheExcluded - x0) << ",\"distinctKeys\":" << x2DistinctThisTick
                  << ",\"ptCacheEntries\":" << ptCacheEntries << ",\"ptCacheLimbs\":" << ptCacheLimbs
                  << ",\"ptScalarOps\":" << (ptScalarOps - o0) << ",\"ptScalarDeg2Fallback\":" << (ptScalarDeg2Fallback - g0)
                  << ",\"ptScalarClasses\":" << ptScalarClasses << ",\"ptScalarClassesRejected\":" << ptScalarClassesRejected
                  << ",\"ptScalarRejectedOps\":" << ptScalarRejectedOps << "}" << std::endl;
        c0 = ptCalls; h0 = ptCacheHits; m0 = ptCacheMisses; f0 = ptCacheDenseFallback; s0 = ptCacheSkipped; x0 = ptCacheExcluded;
        o0 = ptScalarOps; g0 = ptScalarDeg2Fallback; x2TickNo++; x2DistinctThisTick = 0;
    };
    auto ptAtS = [&](const std::vector<double>& packed, uint32_t used, int site) {   // S3.7 X2: the site-tagged funnel
        ptCalls++;
        if (x2PtCache) { Plaintext hit = x2Lookup(packed, used, site); if (hit) return hit; }
        auto tPt = Clock::now();                                // S3.7 V2: mkPt's host-encode counters
        Plaintext pt = cc->MakeCKKSPackedPlaintext(packed, 1, used);
        const double ptMs = msSince(tPt);
        hostPtEncodeMs += ptMs; hostPtEncodeCount++;
        if (stageOpenDepth > 0) { hostPtEncodeStageMs += ptMs; hostPtEncodeStageCount++; }
        return pt;
    };
    auto ptAt = [&](const std::vector<double>& packed, uint32_t used) { return ptAtS(packed, used, __LINE__); };
    // the scalar rule (flag block above): scalar iff --pt-scalar, the vector is constant and the operand
    // is deg 1; otherwise the plaintext route (cache or dense), the statements of HEAD's mulPt/alignTo.
    auto x2Mul = [&](Ciphertext<DCRTPoly>& ct, const std::vector<double>& v, uint32_t lvl, int site) {
        double cval = 0.0;
        if (x2PtScalar && x2IsConst(v, cval) && x2ScalarOp(ct, cval, false, [&]() { return ptAtS(v, lvl, site); })) return;
        auto pt = ptAtS(v, lvl, site); ct = cc->EvalMult(ct, pt);
    };
    auto x2Add = [&](Ciphertext<DCRTPoly>& ct, const std::vector<double>& v, uint32_t lvl, int site) {
        double cval = 0.0;
        if (x2PtScalar && x2IsConst(v, cval) && x2ScalarOp(ct, cval, true, [&]() { return ptAtS(v, lvl, site); })) return;
        auto pt = ptAtS(v, lvl, site); ct = cc->EvalAdd(ct, pt);
    };
    auto mulPt = [&](TrackedCt& c, const std::vector<double>& packed) {
        x2Mul(c.ct, packed, c.used, __LINE__);                  // S3.7 X2: was ptAt + EvalMult (same statements with the flags off)
        cc->RescaleInPlace(c.ct); c.used++;
    };
    auto alignTo = [&](TrackedCt& c, uint32_t target) {
        std::vector<double> one(SLOTS, 1.0);
        while (c.used < target) { x2Mul(c.ct, one, c.used, __LINE__);   // S3.7 X2: was ptAt + EvalMult
                                  cc->RescaleInPlace(c.ct); c.used++; }
    };

    // deterministic test vector
    std::vector<double> v(d);
    for (uint32_t i = 0; i < d; i++) v[i] = std::sin(0.37 * i) * 0.5;

    // ---- 1. pack / encrypt / decrypt roundtrip ---------------------------
    {
        auto c = enc(packCh(v)); auto o = dec(c);
        double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - v[i]));
        verdict("encdec", e < 1e-6, e, true);
    }

    // ---- 2. rotation convention: EvalRotate(ct,k)[i] == x[i+k] -----------
    {
        auto c = enc(packCh(v));
        TrackedCt r{cc->EvalRotate(c.ct, 1), 0}; auto o = dec(r);
        double e = 0; for (uint32_t i = 0; i + 1 < d; i++) e = std::max(e, std::abs(o[i] - v[i + 1]));
        verdict("rotate_left_convention", e < 1e-4, e, true);
    }

    // ---- 3. BSGS dense matvec vs plaintext -------------------------------
    // y[row] = sum_col W[row][col] * u[col], via y = sum_k diag_k (.) rot(u,k)
    // with diag_k[row] = W[row][(row+k) mod cols], BSGS over k = g + j and
    // the plaintext diagonal PRE-ROTATED by -g (the defect this file found).
    // CONTRACT: the input ciphertext is packed with period D and the matrix
    // is treated as (rows x D) with columns >= inCols zero-padded, so the
    // rotation modulus (D) EQUALS the diagonal modulus. rows must be <= D;
    // taller maps are chunked by matvecChunked below.
    // W is given row-major (rows x inCols); inCols <= D.
    const std::vector<Ciphertext<DCRTPoly>>* babyShared = nullptr;   // S3.3 shared-babies cell: set by the caller
    auto matvec = [&](TrackedCt u, const std::vector<double>& W,
                      uint32_t rows, uint32_t inCols, uint32_t rowOff) -> TrackedCt {
        auto t = Clock::now();                                  // S3.7 V2: the matvec evalMs window (no boot inside)
        std::vector<Ciphertext<DCRTPoly>> baby(BS);
        if (babyShared) baby = *babyShared;                       // reuse (S3.3); else compute as before
        else {
        baby[0] = u.ct;
        for (int j = 1; j < BS; j++) baby[j] = cc->EvalRotate(u.ct, j);
        }
        TrackedCt acc; bool have = false;
        for (uint32_t g = 0; g < D; g += BS) {
            Ciphertext<DCRTPoly> inner; bool hi = false;
            for (int j = 0; j < BS && g + (uint32_t)j < D; j++) {
                uint32_t k = g + j;
                std::vector<double> diag(SLOTS, 0.0);
                bool nonzero = false;
                for (uint32_t row = 0; row < rows; row++) {
                    uint32_t col = (row + k) % D;              // modulus == D
                    if (col >= inCols) continue;               // zero-padded column
                    double wv = W[(size_t)(row + rowOff) * inCols + col];
                    // pre-rotate by -g: the value for output `row` sits at
                    // slot (row+g) mod D so the giant rot(.,g) lands it back.
                    uint32_t slot = (row + g) % D;
                    for (uint32_t r = 0; r < REP; r++) diag[r * D + slot] = wv;
                    nonzero = true;
                }
                if (!nonzero) continue;                        // skip empty diagonal
                auto dpt = ptAtS(diag, u.used, -1);            // S3.7 X2: a weight diagonal, never cached
                auto term = cc->EvalMult(baby[j], dpt);
                if (!hi) { inner = term; hi = true; } else inner = cc->EvalAdd(inner, term);
            }
            if (!hi) continue;
            if (g > 0) inner = cc->EvalRotate(inner, g);
            if (!have) { acc.ct = inner; have = true; } else acc.ct = cc->EvalAdd(acc.ct, inner);
        }
        cc->RescaleInPlace(acc.ct); acc.used = u.used + 1;
        evalMs += msSince(t);                                   // S3.7 V2
        return acc;
    };
    // rows > D: ceil(rows/D) chunks, one output ciphertext each.
    auto matvecChunked = [&](TrackedCt u, const std::vector<double>& W,
                             uint32_t rows, uint32_t inCols) {
        std::vector<TrackedCt> out;
        for (uint32_t off = 0; off < rows; off += D)
            out.push_back(matvec(u, W, std::min(D, rows - off), inCols, off));
        return out;
    };
    {
        std::vector<double> W((size_t)d * d);
        for (uint32_t i = 0; i < d; i++)
            for (uint32_t j = 0; j < d; j++)
                W[(size_t)i * d + j] = std::cos(0.11 * i + 0.23 * j) * 0.3;
        auto c = enc(packCh(v));
        auto y = matvec(c, W, d, d, 0);
        auto o = dec(y);
        std::vector<double> ref(d, 0.0);
        for (uint32_t i = 0; i < d; i++)
            for (uint32_t j = 0; j < d; j++) ref[i] += W[(size_t)i * d + j] * v[j];
        double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - ref[i]));
        verdict("matvec_square", e < 1e-3, e, true);
    }
    // ---- 3b. rectangular matvec d -> dff (the mixed-period case) ---------
    {
        std::vector<double> W((size_t)dff * d);
        for (uint32_t i = 0; i < dff; i++)
            for (uint32_t j = 0; j < d; j++)
                W[(size_t)i * d + j] = std::sin(0.07 * i + 0.19 * j) * 0.2;
        auto c = enc(packCh(v));
        auto ys = matvecChunked(c, W, dff, d);          // NCHUNK ciphertexts
        std::vector<double> ref(dff, 0.0);
        for (uint32_t i = 0; i < dff; i++)
            for (uint32_t j = 0; j < d; j++) ref[i] += W[(size_t)i * d + j] * v[j];
        double e = 0;
        for (uint32_t ch = 0; ch < ys.size(); ch++) {
            auto o = dec(ys[ch]);
            uint32_t n = std::min(D, dff - ch * D);
            for (uint32_t i = 0; i < n; i++) e = std::max(e, std::abs(o[i] - ref[ch * D + i]));
        }
        verdict("matvec_rect_d_to_dff_chunked", e < 1e-3 && ys.size() == NCHUNK, e, true);
    }
    // ---- 3c. transpose direction dff -> d: chunks summed ------------------
    {
        std::vector<double> W((size_t)d * dff);
        for (uint32_t i = 0; i < d; i++)
            for (uint32_t j = 0; j < dff; j++)
                W[(size_t)i * dff + j] = std::cos(0.05 * i + 0.13 * j) * 0.15;
        std::vector<double> hid(dff);
        for (uint32_t j = 0; j < dff; j++) hid[j] = std::sin(0.21 * j) * 0.4;
        // encrypt hidden as NCHUNK period-D ciphertexts
        TrackedCt accum; bool have = false;
        for (uint32_t ch = 0; ch < NCHUNK; ch++) {
            uint32_t n = std::min(D, dff - ch * D);
            std::vector<double> part(hid.begin() + ch * D, hid.begin() + ch * D + n);
            auto cpart = enc(packCh(part));
            // column-block of W for this chunk, as a (d x D) row-major slice
            std::vector<double> Wb((size_t)d * D, 0.0);
            for (uint32_t i = 0; i < d; i++)
                for (uint32_t j = 0; j < n; j++) Wb[(size_t)i * D + j] = W[(size_t)i * dff + ch * D + j];
            auto y = matvec(cpart, Wb, d, D, 0);
            if (!have) { accum = y; have = true; }
            else { alignTo(accum, y.used); alignTo(y, accum.used);
                   accum.ct = cc->EvalAdd(accum.ct, y.ct); }
        }
        auto o = dec(accum);
        std::vector<double> ref(d, 0.0);
        for (uint32_t i = 0; i < d; i++)
            for (uint32_t j = 0; j < dff; j++) ref[i] += W[(size_t)i * dff + j] * hid[j];
        double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - ref[i]));
        verdict("matvec_dff_to_d_summed", e < 1e-3, e, true);
    }

    // ---- 3d. BATCHED matvec: encode each diagonal ONCE, apply to all tokens.
    // This is the throughput fix in gpu_real_model.cu (matvecBatch). It is a
    // pure loop reordering — the per-token math is untouched — but this test
    // PROVES it: several DISTINCT token vectors are pushed through the batched
    // path and each must match its own per-token result exactly. If any token
    // leaked into another (the hazard with slot-batching), these differ.
    {
        const int NT = 5;
        std::vector<std::vector<double>> toks(NT, std::vector<double>(d));
        for (int t = 0; t < NT; t++)
            for (uint32_t i = 0; i < d; i++)
                toks[t][i] = std::sin(1.7 * (t + 1) + 0.29 * i) * (0.2 + 0.1 * t);
        std::vector<double> W((size_t)d * d);
        for (uint32_t i = 0; i < d; i++)
            for (uint32_t j = 0; j < d; j++)
                W[(size_t)i * d + j] = std::cos(0.17 * i + 0.31 * j) * 0.25;
        // per-token reference path (one matvec call each)
        std::vector<std::vector<double>> perTok(NT);
        for (int t = 0; t < NT; t++) {
            auto c = enc(packCh(toks[t]));
            auto y = matvec(c, W, d, d, 0);
            perTok[t] = dec(y);
        }
        // batched path: diagonal encoded once, reused across all NT tokens
        std::vector<TrackedCt> us; us.reserve(NT);
        for (int t = 0; t < NT; t++) us.push_back(enc(packCh(toks[t])));
        std::vector<Ciphertext<DCRTPoly>> accB(NT); std::vector<bool> haveB(NT, false);
        uint32_t lvl = us[0].used;
        std::vector<std::vector<Ciphertext<DCRTPoly>>> baby(NT);
        for (int t = 0; t < NT; t++) {
            baby[t].resize(BS); baby[t][0] = us[t].ct;
            for (int j = 1; j < BS; j++) baby[t][j] = cc->EvalRotate(us[t].ct, j);
        }
        for (uint32_t g = 0; g < D; g += BS) {
            std::vector<Ciphertext<DCRTPoly>> inner(NT); std::vector<bool> hi(NT, false);
            for (int j = 0; j < BS && g + j < D; j++) {
                uint32_t k = g + j;
                std::vector<double> diag(SLOTS, 0.0); bool nz = false;
                for (uint32_t row = 0; row < d; row++) {
                    uint32_t col = (row + k) % D; if (col >= d) continue;
                    uint32_t slot = (row + g) % D;
                    for (uint32_t r = 0; r < REP; r++) diag[r * D + slot] = W[(size_t)row * d + col];
                    nz = true;
                }
                if (!nz) continue;
                auto dpt = ptAtS(diag, lvl, -1);               // *** encoded ONCE ***  (S3.7 X2: never cached)
                for (int t = 0; t < NT; t++) {
                    auto term = cc->EvalMult(baby[t][j], dpt);
                    if (!hi[t]) { inner[t] = term; hi[t] = true; }
                    else inner[t] = cc->EvalAdd(inner[t], term);
                }
            }
            for (int t = 0; t < NT; t++) {
                if (!hi[t]) continue;
                auto blk = (g > 0) ? cc->EvalRotate(inner[t], g) : inner[t];
                if (!haveB[t]) { accB[t] = blk; haveB[t] = true; }
                else accB[t] = cc->EvalAdd(accB[t], blk);
            }
        }
        double e = 0, crossMin = 1e9;
        for (int t = 0; t < NT; t++) {
            cc->RescaleInPlace(accB[t]);
            TrackedCt yb{accB[t], lvl + 1}; auto ob = dec(yb);
            for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(ob[i] - perTok[t][i]));
            // sanity: tokens must be genuinely DIFFERENT, else the test is vacuous
            for (int t2 = 0; t2 < NT; t2++) if (t2 != t) {
                double diff = 0;
                for (uint32_t i = 0; i < d; i++) diff = std::max(diff, std::abs(perTok[t][i] - perTok[t2][i]));
                crossMin = std::min(crossMin, diff);
            }
        }
        verdict("matvec_batched_no_token_bleed", e < 1e-3 && crossMin > 1e-2, e, true);
    }

    // ---- 4. RMSNorm: rotate-sum reduction + Newton rsqrt -----------------
    // mean(x^2) over the d real channels; padding slots are 0, so summing the
    // whole D-block and scaling by 1/d is exact.
    auto rmsnorm = [&](TrackedCt x, const std::vector<double>& gpk,
                       double a0, double b0, int iters, double eps) -> TrackedCt {
        auto t = Clock::now();                                  // S3.7 V2: the rmsnorm evalMs window ...
        const double normBoot0 = bootMs;                        // ... and the .cu's snapshot (no boot in this variant: delta 0)
        TrackedCt sq{cc->EvalMult(x.ct, x.ct), x.used};
        cc->RescaleInPlace(sq.ct); sq.used = x.used + 1;
        Ciphertext<DCRTPoly> red = sq.ct;
        for (uint32_t p = 1; p < D; p <<= 1) red = cc->EvalAdd(red, cc->EvalRotate(red, p));
        std::vector<double> invd(SLOTS, 1.0 / (double)d);
        Ciphertext<DCRTPoly> msc = red; x2Mul(msc, invd, sq.used, __LINE__);   // S3.7 X2: was ptAt + EvalMult
        TrackedCt ms{msc, sq.used};
        cc->RescaleInPlace(ms.ct); ms.used = sq.used + 1;
        // S3.7 V1 (2026-09-05, S3.6 T18 §3): ms = mean(x^2) + eps, as the model defines
        // it (invd is 1/d here, so sigma = 1 and the constant is eps itself). One
        // plaintext add at ms's own level: zero levels, `used` unchanged. This is the
        // same hunk as gpu_real_model.cu's (there: sigma*eps). --no-norm-eps skips it.
        const double epsUsed = normEpsOn ? eps : 0.0;
        normEpsUsed = epsUsed;
        if (epsUsed != 0.0) {
            std::vector<double> epsv(SLOTS, epsUsed);
            x2Add(ms.ct, epsv, ms.used, __LINE__);              // S3.7 X2: was ptAt + EvalAdd
        }
        std::vector<double> b0v(SLOTS, b0), a0v(SLOTS, a0);
        Ciphertext<DCRTPoly> yc = ms.ct; x2Mul(yc, b0v, ms.used, __LINE__);     // S3.7 X2: was ptAt + EvalMult
        TrackedCt y{yc, ms.used};
        cc->RescaleInPlace(y.ct); y.used = ms.used + 1;
        x2Add(y.ct, a0v, y.used, __LINE__);                     // S3.7 X2: was ptAt + EvalAdd
        for (int it = 0; it < iters; it++) {
            TrackedCt y2{cc->EvalMult(y.ct, y.ct), y.used};
            cc->RescaleInPlace(y2.ct); y2.used = y.used + 1;
            TrackedCt msA = ms; alignTo(msA, y2.used);
            TrackedCt t{cc->EvalMult(msA.ct, y2.ct), y2.used};
            cc->RescaleInPlace(t.ct); t.used = y2.used + 1;
            std::vector<double> mhalf(SLOTS, -0.5), oneP5(SLOTS, 1.5);
            x2Mul(t.ct, mhalf, t.used, __LINE__);               // S3.7 X2: was ptAt + EvalMult
            cc->RescaleInPlace(t.ct); t.used++;
            x2Add(t.ct, oneP5, t.used, __LINE__);               // S3.7 X2: was ptAt + EvalAdd
            alignTo(y, t.used);
            y.ct = cc->EvalMult(y.ct, t.ct); cc->RescaleInPlace(y.ct); y.used++;
        }
        alignTo(x, y.used);
        TrackedCt out{cc->EvalMult(x.ct, y.ct), x.used};
        cc->RescaleInPlace(out.ct); out.used = x.used + 1;
        mulPt(out, gpk);
        evalMs += msSince(t);                                   // S3.7 V2
        normBootMs += bootMs - normBoot0;
        return out;
    };
    {
        double ms = 0; for (uint32_t i = 0; i < d; i++) ms += v[i] * v[i]; ms /= d;
        double eps = 1e-5, inv = 1.0 / std::sqrt(ms + eps);
        // linear seed on [ms/2, 2ms] (mirrors linear_rsqrt_seed in Python)
        double a0 = 1.5 / std::sqrt(ms), b0 = -0.5 / (ms * std::sqrt(ms));
        std::vector<double> g(d, 1.0);
        auto c = enc(packCh(v));
        auto y = rmsnorm(c, packCh(g), a0, b0, 3, eps);
        auto o = dec(y);
        double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - v[i] * inv));
        verdict("rmsnorm_newton", e < 5e-3, e, false);
    }
    // ---- 4b. S3.7 V1 (2026-09-05): the norm with the seed the demo runs ---------
    // Mirrors gpu_real_model.cu selftest 4b. --norm-seed A B ITERS takes a bundle
    // site's (a, b, iters) (pbd430a L0.tm: 53.12723596752366 -13744.462664572491 5) and
    // scales v so that mean(v^2) = -a/(16 b): inside the seed's design window near its
    // low end, where eps/ms ~ 4 %. Reference = the model's 1/sqrt(ms + eps). Predicted
    // by harness/norm_eps_check.py (case cpu_twin_seeded_L0tm): the no-eps chain
    // (--no-norm-eps) misses the 1e-2 bar, the eps chain sits at the Newton floor.
    if (normSeedIters > 0) {
        const double msTest = -normSeedA / (normSeedB * 16.0);
        if (!std::isfinite(msTest) || msTest <= 0.0) {
            std::cout << "{\"cell\":\"rmsnorm_seeded\",\"skipped\":\"seed has no positive ms\"}" << std::endl;
        } else {
            double msv = 0; for (uint32_t i = 0; i < d; i++) msv += v[i] * v[i]; msv /= d;
            const double sc = std::sqrt(msTest / msv);
            std::vector<double> vs(d); for (uint32_t i = 0; i < d; i++) vs[i] = v[i] * sc;
            const double eps = 1e-5, inv = 1.0 / std::sqrt(msTest + eps);
            std::vector<double> g(d, 1.0);
            auto c = enc(packCh(vs));
            auto y = rmsnorm(c, packCh(g), normSeedA, normSeedB, normSeedIters, eps);
            auto o = dec(y);
            double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - vs[i] * inv));
            std::cout << "{\"cell\":\"rmsnorm_seeded\",\"a\":" << normSeedA << ",\"b\":" << normSeedB
                      << ",\"iters\":" << normSeedIters << ",\"msTest\":" << msTest << ",\"scale\":" << sc
                      << ",\"normEps\":" << normEpsUsed << ",\"maxAbsErr\":" << e << "}" << std::endl;
            verdict("rmsnorm_seeded", e < 1e-2, e, false);
        }
    }

    // ---- 5. learned degree-3 gate: g = z (.) (p0 + p1 z + p2 z^2) --------
    {
        std::vector<double> p0(d), p1(d), p2(d);
        for (uint32_t i = 0; i < d; i++) { p0[i] = 0.5 + 0.01 * i / d; p1[i] = 0.25; p2[i] = 0.1; }
        auto z = enc(packCh(v));
        TrackedCt z2{cc->EvalMult(z.ct, z.ct), z.used};
        cc->RescaleInPlace(z2.ct); z2.used = z.used + 1;
        TrackedCt t2 = z2; mulPt(t2, packCh(p2));
        TrackedCt t1 = z;  mulPt(t1, packCh(p1));
        alignTo(t1, t2.used);
        t1.ct = cc->EvalAdd(t1.ct, t2.ct);
        x2Add(t1.ct, packCh(p0), t1.used, __LINE__);            // S3.7 X2: was ptAt + EvalAdd
        TrackedCt zA = z; alignTo(zA, t1.used);
        TrackedCt gct{cc->EvalMult(zA.ct, t1.ct), zA.used};
        cc->RescaleInPlace(gct.ct); gct.used = zA.used + 1;
        auto o = dec(gct);
        double e = 0;
        for (uint32_t i = 0; i < d; i++) {
            double ref = v[i] * (p0[i] + p1[i] * v[i] + p2[i] * v[i] * v[i]);
            e = std::max(e, std::abs(o[i] - ref));
        }
        verdict("poly_gate_deg3", e < 1e-3, e, true);
    }

    // ---- 6. Lemma-2 diagonal scan: s_t = a (.) s_{t-1} + b (.) x_t -------
    {
        const uint32_t Tn = 8;
        std::vector<double> a(d), b(d);
        for (uint32_t i = 0; i < d; i++) { a[i] = 0.90 + 0.09 * (i % 7) / 7.0; b[i] = 1.0 - a[i]; }
        std::vector<std::vector<double>> xs(Tn, std::vector<double>(d));
        for (uint32_t t = 0; t < Tn; t++)
            for (uint32_t i = 0; i < d; i++) xs[t][i] = std::sin(0.3 * t + 0.11 * i) * 0.4;
        std::vector<double> zero(SLOTS, 0.0);
        TrackedCt s = enc(zero);
        for (uint32_t t = 0; t < Tn; t++) {
            TrackedCt x = enc(packCh(xs[t]));
            mulPt(s, packCh(a));
            TrackedCt bx = x; mulPt(bx, packCh(b));
            alignTo(s, bx.used); alignTo(bx, s.used);
            s.ct = cc->EvalAdd(s.ct, bx.ct);
        }
        auto o = dec(s);
        std::vector<double> ref(d, 0.0);
        for (uint32_t t = 0; t < Tn; t++)
            for (uint32_t i = 0; i < d; i++) ref[i] = a[i] * ref[i] + b[i] * xs[t][i];
        double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - ref[i]));
        verdict("lemma2_scan", e < 1e-3, e, true);
    }

    // ---- 7. optional: bootstrap a depleted ciphertext --------------------
    if (doBootstrap) {
        auto c = enc(packCh(v));
        std::vector<double> one(SLOTS, 1.0);
        for (int i = 0; i < 3; i++) { x2Mul(c.ct, one, c.used, __LINE__);   // S3.7 X2: was ptAt + EvalMult
                                      cc->RescaleInPlace(c.ct); c.used++; }
        c.ct = cc->EvalBootstrap(c.ct); c.used = depth - levelsAfter;
        auto o = dec(c);
        double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - v[i]));
        verdict("bootstrap", e < 5e-3, e, true);
    }


    // =====================================================================
    // S3.3 validation cells (results/theory/SCHEDULE_DESIGN_20260903.md)
    // =====================================================================

    if (cells.count("level-probe")) {
        // FLEXIBLEAUTO semantics of the primitives the replay models (Lemma 7 of
        // TICK_SCALING_BOUNDS_20260903.md): GetLevel() and GetNoiseScaleDeg() after
        // ct x pt, ct x ct, rotate, add, RescaleInPlace, and a ct x pt whose plaintext
        // was encoded at a DIFFERENT level than the ciphertext.
        auto rep = [&](const std::string& what, const Ciphertext<DCRTPoly>& c) {
            std::cout << "{\"levelProbe\":\"" << what << "\",\"level\":" << c->GetLevel() << ",\"deg\":" << c->GetNoiseScaleDeg() << "}" << std::endl;
        };
        std::vector<double> one(SLOTS, 1.0), half(SLOTS, 0.5);
        auto c0 = enc(packCh(v)).ct; rep("fresh", c0);
        auto p0 = cc->MakeCKKSPackedPlaintext(half, 1, c0->GetLevel());
        auto c1 = cc->EvalMult(c0, p0); rep("ct*pt(level-matched)", c1);
        auto r1 = cc->EvalRotate(c1, 1); rep("rotate(deg2)", r1);
        auto a1 = cc->EvalAdd(c1, r1); rep("add(deg2,deg2)", a1);
        auto s1 = a1; cc->RescaleInPlace(s1); rep("RescaleInPlace(deg2)", s1);
        auto p1 = cc->MakeCKKSPackedPlaintext(half, 1, s1->GetLevel());
        auto c2 = cc->EvalMult(s1, p1); rep("ct*pt on deg2 input (level-matched pt)", c2);
        auto cc2 = cc->EvalMult(c2, c2); rep("ct*ct on deg2 inputs", cc2);
        auto p_lo = cc->MakeCKKSPackedPlaintext(half, 1, 0);
        auto c3 = cc->EvalMult(c2, p_lo); rep("ct*pt with pt encoded at level 0 (deeper ct)", c3);
        TrackedCt t3{c3, 0}, t2{c2, 0};
        auto d3 = dec(t3); auto d2 = dec(t2);
        double e = 0; for (uint32_t i = 0; i < d; i++) e = std::max(e, std::abs(d3[i] - 0.5 * d2[i]));
        std::cout << "{\"levelProbe\":\"value check pt@0 vs pt@level\",\"maxAbsErr\":" << e << "}" << std::endl;
        auto r2 = cc->EvalRotate(c2, 1); rep("rotate(deg2, level>0)", r2);
        auto a2 = cc->EvalAdd(c2, r2); rep("add after rotate", a2);
        auto g2 = cc->EvalRotate(a2, 8); rep("giant rotate of the sum", g2);
        verdict("level_probe_ran", true, 0.0, false);
    }

    if (cells.count("shared-babies")) {
        // The d -> dff map as NCHUNK chunk calls on ONE input: (a) stock, every
        // call computes its own baby rotations; (b) shared, one baby set feeds
        // every call. Bit-identity = the output ciphertext polynomials are
        // equal element-wise (DCRTPoly::operator==) for every chunk.
        std::vector<double> W((size_t)dff * d);
        for (uint32_t i = 0; i < dff; i++)
            for (uint32_t j = 0; j < d; j++)
                W[(size_t)i * d + j] = std::sin(0.07 * i + 0.19 * j) * 0.2;
        auto c = enc(packCh(v));
        babyShared = nullptr;
        auto ysStock = matvecChunked(c, W, dff, d);
        std::vector<Ciphertext<DCRTPoly>> baby(BS);
        baby[0] = c.ct;
        for (int j = 1; j < BS; j++) baby[j] = cc->EvalRotate(c.ct, j);
        babyShared = &baby;
        auto ysShared = matvecChunked(c, W, dff, d);
        babyShared = nullptr;
        bool polysEqual = ysStock.size() == ysShared.size();
        double maxDiff = 0; int chunksEq = 0;
        for (size_t ch = 0; polysEqual && ch < ysStock.size(); ch++) {
            const auto& ea = ysStock[ch].ct->GetElements();
            const auto& eb = ysShared[ch].ct->GetElements();
            bool eq = ea.size() == eb.size();
            for (size_t k = 0; eq && k < ea.size(); k++) eq = (ea[k] == eb[k]);
            if (eq) chunksEq++; else polysEqual = false;
            auto oa = dec(ysStock[ch]); auto ob = dec(ysShared[ch]);
            for (uint32_t i = 0; i < D; i++) maxDiff = std::max(maxDiff, std::abs(oa[i] - ob[i]));
        }
        // CONTROL: decrypting the SAME ciphertext twice differs at the same
        // magnitude -- OpenFHE's CKKS decryption adds flooding noise (a decrypt-
        // side property, not a ciphertext difference). The bit-identity claim is
        // therefore the polynomial equality, and the decrypted diff is reported
        // beside its control, never used as the verdict.
        double sameCtTwice = 0;
        { auto o1 = dec(ysStock[0]); auto o2 = dec(ysStock[0]);
          for (uint32_t i = 0; i < D; i++) sameCtTwice = std::max(sameCtTwice, std::abs(o1[i] - o2[i])); }
        std::cout << "{\"cell\":\"shared-babies\",\"chunks\":" << ysStock.size() << ",\"chunksPolyEqual\":" << chunksEq
                  << ",\"polysEqual\":" << (polysEqual ? "true" : "false") << ",\"maxAbsDiffDecrypted\":" << maxDiff
                  << ",\"controlSameCtDecryptedTwice\":" << sameCtTwice
                  << ",\"babyRotationsStock\":" << (BS - 1) * (int)ysStock.size() << ",\"babyRotationsShared\":" << (BS - 1) << "}" << std::endl;
        verdict("shared_babies_bit_identical", polysEqual, maxDiff, true);
    }

    if (cells.count("newton-depth2")) {
        // Same seed (a0, b0), same iteration count, same input: stock chain vs the
        // depth-2 re-association. Reports relErr of each vs exact rsqrt, the
        // rms difference between the two, and the levels consumed by the loop.
        auto runNorm = [&](const std::vector<double>& xv, int iters, bool nd2, double& lvlLoop) -> std::vector<double> {
            double ms0 = 0; for (uint32_t i = 0; i < d; i++) ms0 += xv[i] * xv[i]; ms0 /= d;
            double a0 = 1.5 / std::sqrt(ms0), b0 = -0.5 / (ms0 * std::sqrt(ms0));
            TrackedCt x = enc(packCh(xv));
            TrackedCt sq{cc->EvalMult(x.ct, x.ct), 0}; cc->RescaleInPlace(sq.ct);
            Ciphertext<DCRTPoly> red = sq.ct;
            for (uint32_t p = 1; p < D; p <<= 1) red = cc->EvalAdd(red, cc->EvalRotate(red, p));
            std::vector<double> invd(SLOTS, 1.0 / (double)d);
            auto invdpt = cc->MakeCKKSPackedPlaintext(invd, 1, sq.ct->GetLevel());
            TrackedCt ms{cc->EvalMult(red, invdpt), 0}; cc->RescaleInPlace(ms.ct);
            std::vector<double> b0v(SLOTS, b0), a0v(SLOTS, a0);
            auto b0pt = cc->MakeCKKSPackedPlaintext(b0v, 1, ms.ct->GetLevel());
            TrackedCt y{cc->EvalMult(ms.ct, b0pt), 0}; cc->RescaleInPlace(y.ct);
            auto a0pt = cc->MakeCKKSPackedPlaintext(a0v, 1, y.ct->GetLevel());
            y.ct = cc->EvalAdd(y.ct, a0pt);
            const uint32_t lvl0 = y.ct->GetLevel();
            TrackedCt msh;
            if (nd2) { std::vector<double> mh(SLOTS, -0.5);
                       auto mhp = cc->MakeCKKSPackedPlaintext(mh, 1, ms.ct->GetLevel());
                       msh.ct = cc->EvalMult(ms.ct, mhp); cc->RescaleInPlace(msh.ct); }
            for (int it = 0; it < iters; it++) {
                if (nd2) {   // mirrors gpu_real_model.cu --newton-depth2
                    TrackedCt a{cc->EvalMult(msh.ct, y.ct), 0}; cc->RescaleInPlace(a.ct);
                    TrackedCt b{cc->EvalMult(y.ct, y.ct), 0};   cc->RescaleInPlace(b.ct);
                    TrackedCt tt{cc->EvalMult(a.ct, b.ct), 0};  cc->RescaleInPlace(tt.ct);
                    std::vector<double> o15(SLOTS, 1.5);
                    auto op15 = cc->MakeCKKSPackedPlaintext(o15, 1, y.ct->GetLevel());
                    TrackedCt y15{cc->EvalMult(y.ct, op15), 0}; cc->RescaleInPlace(y15.ct);
                    y15.ct = cc->EvalAdd(y15.ct, tt.ct);
                    y = y15;
                } else {     // mirrors the stock loop (:2973-2992)
                    TrackedCt y2{cc->EvalMult(y.ct, y.ct), 0}; cc->RescaleInPlace(y2.ct);
                    TrackedCt tms{cc->EvalMult(ms.ct, y2.ct), 0}; cc->RescaleInPlace(tms.ct);
                    std::vector<double> half(SLOTS, -0.5), oneP5(SLOTS, 1.5);
                    auto halfpt = cc->MakeCKKSPackedPlaintext(half, 1, tms.ct->GetLevel());
                    tms.ct = cc->EvalMult(tms.ct, halfpt); cc->RescaleInPlace(tms.ct);
                    auto op5 = cc->MakeCKKSPackedPlaintext(oneP5, 1, tms.ct->GetLevel());
                    tms.ct = cc->EvalAdd(tms.ct, op5);
                    y.ct = cc->EvalMult(y.ct, tms.ct); cc->RescaleInPlace(y.ct);
                }
            }
            lvlLoop = (double)y.ct->GetLevel() - (double)lvl0;
            TrackedCt out{cc->EvalMult(x.ct, y.ct), 0}; cc->RescaleInPlace(out.ct);
            return dec(out);
        };
        std::vector<std::vector<double>> inputs = {v};
        { std::vector<double> w(d); for (uint32_t i = 0; i < d; i++) w[i] = 0.7 * std::cos(1.3 * i + 0.4) + 0.05 * std::sin(7.1 * i); inputs.push_back(w); }
        int idx = 0;
        for (const auto& xv : inputs) {
            double ms0 = 0; for (uint32_t i = 0; i < d; i++) ms0 += xv[i] * xv[i]; ms0 /= d;
            const double inv = 1.0 / std::sqrt(ms0);
            double lvS = 0, lvN = 0;
            auto oS = runNorm(xv, cellIters, false, lvS);
            auto oN = runNorm(xv, cellIters, true, lvN);
            double eS = 0, eN = 0, dSN = 0, ref2 = 0;
            for (uint32_t i = 0; i < d; i++) {
                const double r = xv[i] * inv;
                eS += (oS[i] - r) * (oS[i] - r); eN += (oN[i] - r) * (oN[i] - r); dSN += (oS[i] - oN[i]) * (oS[i] - oN[i]); ref2 += r * r;
            }
            const double relS = std::sqrt(eS / ref2), relN = std::sqrt(eN / ref2), relD = std::sqrt(dSN / ref2);
            std::cout << "{\"cell\":\"newton-depth2\",\"input\":" << idx << ",\"iters\":" << cellIters
                      << ",\"relErrRmsStock\":" << relS << ",\"relErrRmsDepth2\":" << relN << ",\"relErrRmsStockVsDepth2\":" << relD
                      << ",\"levelsPerIterStock\":" << lvS / cellIters << ",\"levelsPerIterDepth2\":" << lvN / cellIters << "}" << std::endl;
            // the re-association must be within the stock chain's own error class (R3: rms), and consume 2 levels/iteration
            verdict("newton_depth2_input" + std::to_string(idx), relN <= 4.0 * std::max(relS, 1e-9) && std::abs(lvN / cellIters - 2.0) < 1e-9
                    && std::abs(lvS / cellIters - 4.0) < 1e-9, relN, true);
            idx++;
        }
    }

    // ==== S3.7 X1 boot schedule (2026-09-10): the two algebra cells (no bootstrap) ====
    if (cells.count("deferred-scan")) {
        // The Lemma-2 scan of test 6 (Tn = 16 steps) run EAGERLY (s <- decay . s + b . x_t in place, one
        // level per step) and DEFERRED with period M (S3.6 T03 §1.3; canonical_carry_trace.py
        // SimDeferredScan): between folds the carry s_base is only READ -- the readout at step j of the
        // period is S = decay^(j+1) . s_base + sum_{i<=j} (decay^(j-i) b) . x_i on COPIES -- and every
        // M-th step folds decay^M . s_base + sum_i (decay^(M-1-i) b) . x_i in place. The kept term is the
        // raw x_i; b and the decay power ride in ONE plaintext, so every term sits at level(x)+1 exactly
        // as the eager b . x does and the readout's level is max(level(s_base)+1, level(x)+1): the
        // downstream trajectory is the eager one while the base consumes one level per FOLD, not per
        // step. The same ciphertexts x_t feed both paths. Reported per M: max |S_t(deferred) -
        // S_t(eager)| over all steps and channels, both against the float64 recurrence (the noise-growth
        // record vs M), the base after the last fold vs the eager state, and the end levels.
        const uint32_t Tn = 16;
        std::vector<double> a(d), b(d);
        for (uint32_t i = 0; i < d; i++) { a[i] = 0.90 + 0.09 * (i % 7) / 7.0; b[i] = 1.0 - a[i]; }
        std::vector<std::vector<double>> xsv(Tn, std::vector<double>(d));
        for (uint32_t t = 0; t < Tn; t++) for (uint32_t i = 0; i < d; i++) xsv[t][i] = std::sin(0.3 * t + 0.11 * i) * 0.4;
        std::vector<double> zero(SLOTS, 0.0);
        auto ptL2 = [&](const std::vector<double>& packed, const Ciphertext<DCRTPoly>& ref) { return cc->MakeCKKSPackedPlaintext(packed, 1, ref->GetLevel()); };
        auto mulPtL2 = [&](TrackedCt& c, const std::vector<double>& packed) { c.ct = cc->EvalMult(c.ct, ptL2(packed, c.ct)); cc->RescaleInPlace(c.ct); };
        auto powv = [&](int p, const std::vector<double>* times) {                 // decay^p (.) times, packed
            std::vector<double> o(d, 1.0);
            for (uint32_t i = 0; i < d; i++) { for (int k = 0; k < p; k++) o[i] *= a[i]; if (times) o[i] *= (*times)[i]; }
            return packCh(o);
        };
        std::vector<std::vector<double>> ref(Tn, std::vector<double>(d, 0.0)), eag(Tn);
        { std::vector<double> r(d, 0.0); for (uint32_t t = 0; t < Tn; t++) { for (uint32_t i = 0; i < d; i++) r[i] = a[i] * r[i] + b[i] * xsv[t][i]; ref[t] = r; } }
        std::vector<TrackedCt> xct(Tn); for (uint32_t t = 0; t < Tn; t++) xct[t] = enc(packCh(xsv[t]));
        uint32_t eagerLevelEnd = 0; double eagerErr = 0;
        { TrackedCt s = enc(zero);
          for (uint32_t t = 0; t < Tn; t++) { mulPtL2(s, packCh(a)); TrackedCt bx = xct[t]; mulPtL2(bx, packCh(b)); s.ct = cc->EvalAdd(s.ct, bx.ct); eag[t] = dec(s); }
          eagerLevelEnd = s.ct->GetLevel();
          for (uint32_t t = 0; t < Tn; t++) for (uint32_t i = 0; i < d; i++) eagerErr = std::max(eagerErr, std::abs(eag[t][i] - ref[t][i])); }
        std::vector<int> Ms = {2, 4, 8, 16}; if (x1DeferredScan > 1) Ms = {x1DeferredScan};
        for (int M : Ms) {
            TrackedCt base = enc(zero); std::vector<TrackedCt> kept; int folds = 0;
            double dEager = 0, dRef = 0, dBase = 0; uint32_t readoutLvlMax = 0;
            for (uint32_t t = 0; t < Tn; t++) {
                const int j = (int)(t % (uint32_t)M);
                kept.push_back(xct[t]);
                TrackedCt Sb = base; mulPtL2(Sb, powv(j + 1, nullptr));
                for (int i = 0; i <= j; i++) { TrackedCt term = kept[i]; mulPtL2(term, powv(j - i, &b)); Sb.ct = cc->EvalAdd(Sb.ct, term.ct); }
                readoutLvlMax = std::max(readoutLvlMax, (uint32_t)Sb.ct->GetLevel());
                auto o = dec(Sb);
                for (uint32_t i = 0; i < d; i++) { dEager = std::max(dEager, std::abs(o[i] - eag[t][i])); dRef = std::max(dRef, std::abs(o[i] - ref[t][i])); }
                if (j == M - 1) {
                    mulPtL2(base, powv(M, nullptr));
                    for (int i = 0; i < M; i++) { TrackedCt term = kept[i]; mulPtL2(term, powv(M - 1 - i, &b)); base.ct = cc->EvalAdd(base.ct, term.ct); }
                    kept.clear(); folds++;
                    if (t + 1 == Tn) { auto ob = dec(base); for (uint32_t i = 0; i < d; i++) dBase = std::max(dBase, std::abs(ob[i] - eag[t][i])); }
                }
            }
            std::cout << "{\"cell\":\"deferred-scan\",\"M\":" << M << ",\"steps\":" << Tn << ",\"folds\":" << folds
                      << ",\"maxAbsDiffVsEager\":" << dEager << ",\"maxAbsErrVsRef\":" << dRef << ",\"eagerMaxAbsErrVsRef\":" << eagerErr
                      << ",\"baseAfterLastFoldVsEager\":" << dBase << ",\"baseLevelEnd\":" << base.ct->GetLevel()
                      << ",\"eagerLevelEnd\":" << eagerLevelEnd << ",\"readoutLevelMax\":" << readoutLvlMax
                      << ",\"keptAtEnd\":" << kept.size() << "}" << std::endl;
            verdict("deferred_scan_m" + std::to_string(M), dEager < 1e-3 && dRef < 1e-3 && dBase < 1e-3, dEager, true);
        }
    }
    if (cells.count("u-drop")) {
        // The exact drop of the u carry (S3.6 T03 §2.2, the `mixed` rule) as a VALUE question. A fresh
        // encryption of v is walked down by multiply-by-one to l_exit = --u-drop-exit (default 7, the
        // norm-output level of the one-layer cell), then LevelReduceInternalInPlace'd to C =
        // --u-drop-level (default 19 = depth 30 - the pf-trace post-boot lam 11) and consumed exactly as
        // the layer loop consumes uPrev: b = carry . (1-mix); a = uNew . mix; a += b. Reported: the level
        // after the drop (must be C); |dec(dropped) - dec(kept)| beside the decrypt-twice control (OpenFHE
        // floods the decryption with noise, a decrypt-side effect, so equality is judged against that
        // control); the shift-mix outputs through the dropped and the kept carry against each other and
        // against float64; and the FLEXIBLEAUTO scale-bookkeeping bound |sf(l_exit)/sf(C) - 1| . max|b-term|
        // from the context's own scaling factors (ct x pt records sf_ct^2, ckksrns-leveledshe.cpp:157-162,
        // so a dropped carry's consumer is off by exactly that ratio and by nothing else).
        const uint32_t C = x1UDropLevel > 0 ? (uint32_t)x1UDropLevel : 19u;
        const uint32_t lExit = (uint32_t)x1UDropExit;
        if (C < lExit || C >= depth) {
            std::cout << "{\"cell\":\"u-drop\",\"skipped\":\"need l_exit <= C < depth\",\"C\":" << C << ",\"lExit\":" << lExit << "}" << std::endl;
        } else {
            std::vector<double> mix(d), imix(d), w(d);
            for (uint32_t i = 0; i < d; i++) { mix[i] = 0.5 + 0.2 * std::sin(0.3 * i); imix[i] = 1.0 - mix[i]; w[i] = 0.7 * std::cos(1.3 * i + 0.4); }
            auto ptL2 = [&](const std::vector<double>& packed, const Ciphertext<DCRTPoly>& ref) { return cc->MakeCKKSPackedPlaintext(packed, 1, ref->GetLevel()); };
            auto mulPtL2 = [&](TrackedCt& c, const std::vector<double>& packed) { c.ct = cc->EvalMult(c.ct, ptL2(packed, c.ct)); cc->RescaleInPlace(c.ct); };
            auto alignToL2 = [&](TrackedCt& c, uint32_t target) { std::vector<double> one(SLOTS, 1.0);
                while (c.ct->GetLevel() < target) { c.ct = cc->EvalMult(c.ct, ptL2(one, c.ct)); cc->RescaleInPlace(c.ct); } };
            TrackedCt u = enc(packCh(v)); alignToL2(u, lExit);
            TrackedCt uKeep = u; TrackedCt uDrop; uDrop.ct = u.ct->Clone();
            // the walked-down u carries a PENDING rescale (FLEXIBLEAUTO's RescaleInPlace is a no-op): materialise it
            // first, exactly as the multi-tick cell and the .cu do, so "level C" is C for the consumer
            const uint32_t degBefore = uDrop.ct->GetNoiseScaleDeg();
            if (degBefore > 1) cc->GetScheme()->ModReduceInternalInPlace(uDrop.ct, 1);
            const uint32_t lvlSettled = uDrop.ct->GetLevel();
            if (lvlSettled < C) cc->GetScheme()->LevelReduceInternalInPlace(uDrop.ct, C - lvlSettled);   // the exact drop
            const uint32_t lvlAfter = uDrop.ct->GetLevel();
            auto dk = dec(uKeep); auto dk2 = dec(uKeep); auto ddr = dec(uDrop);
            double diffDrop = 0, ctrl = 0;
            for (uint32_t i = 0; i < d; i++) { diffDrop = std::max(diffDrop, std::abs(ddr[i] - dk[i])); ctrl = std::max(ctrl, std::abs(dk2[i] - dk[i])); }
            auto consume = [&](TrackedCt carry) {
                TrackedCt bb = carry; mulPtL2(bb, packCh(imix));
                TrackedCt uNew = enc(packCh(w)); alignToL2(uNew, lExit);
                TrackedCt aa = uNew; mulPtL2(aa, packCh(mix));
                aa.ct = cc->EvalAdd(aa.ct, bb.ct); return aa; };
            TrackedCt aKeep = consume(uKeep), aDrop = consume(uDrop);
            auto oK = dec(aKeep); auto oD = dec(aDrop);
            double dKD = 0, eK = 0, eD = 0, bmax = 0;
            for (uint32_t i = 0; i < d; i++) {
                const double r = w[i] * mix[i] + v[i] * imix[i];
                dKD = std::max(dKD, std::abs(oD[i] - oK[i])); eK = std::max(eK, std::abs(oK[i] - r)); eD = std::max(eD, std::abs(oD[i] - r));
                bmax = std::max(bmax, std::abs(v[i] * imix[i]));
            }
            const auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
            const double sfExit = cp->GetScalingFactorReal(lExit), sfC = cp->GetScalingFactorReal(C);
            const double ratioMinus1 = sfExit / sfC - 1.0;
            char sfbuf[160]; std::snprintf(sfbuf, sizeof sfbuf, "%.17g,\"sfC\":%.17g,\"scaleRatioMinus1\":%.6e,\"scaleBoundOnConsumer\":%.6e",
                                           sfExit, sfC, ratioMinus1, std::abs(ratioMinus1) * bmax);
            std::cout << "{\"cell\":\"u-drop\",\"lExit\":" << lExit << ",\"C\":" << C << ",\"levelAfterDrop\":" << lvlAfter
                      << ",\"noiseDegBefore\":" << degBefore << ",\"levelSettled\":" << lvlSettled << ",\"noiseDegAfter\":" << uDrop.ct->GetNoiseScaleDeg()
                      << ",\"limbsDropped\":" << (lvlAfter - lvlSettled) << ",\"maxAbsDiffDropVsKeep\":" << diffDrop << ",\"controlSameCtDecryptedTwice\":" << ctrl
                      << ",\"consumerMaxAbsDiffDropVsKeep\":" << dKD << ",\"consumerErrKeep\":" << eK << ",\"consumerErrDrop\":" << eD
                      << ",\"sfExit\":" << sfbuf
                      << ",\"levelKeepConsumer\":" << aKeep.ct->GetLevel() << ",\"levelDropConsumer\":" << aDrop.ct->GetLevel() << "}" << std::endl;
            verdict("u_drop_level", lvlAfter == C, (double)lvlAfter, true);
            verdict("u_drop_value_exact", diffDrop < 1e-6, diffDrop, true);
            verdict("u_drop_consumer_exact", dKD < 1e-6 && eD < 1e-3, dKD, true);
        }
    }

    // ==== S3.7 X8 drops (2026-09-10; S3.6 T07 §1.2): the algebra cell -- the exact drop of a matvec INPUT ====
    if (cells.count("x8-drop")) {
        // A fresh encryption of v is walked down by multiply-by-one until GetLevel() = l_exit (it then carries a
        // PENDING rescale: noiseScaleDeg 2, settled level l_exit + 1, X1 REPORT §0) and fed to the selftest matvec
        // three ways at the .cu's convention (diagonals encoded at the input's GetLevel()):
        //   ref    the un-dropped input;
        //   disc   the DISCIPLINED drop: ModReduceInternalInPlace(1) (level +1, deg 1), then
        //          LevelReduceInternalInPlace to level depth - R (R remaining, R + 1 limbs) -- what x8DropTo does;
        //   naive  LevelReduceInternalInPlace to depth - R WITHOUT the settle (deg 2 kept): the consumer's own
        //          ModReduce then lands the output one level deeper than disc's -- "the drop keeps a level".
        // Reported per (R, l_exit): the input's level/degree/limbs on each path, the matvec OUTPUT levels,
        // max |dec(disc) - dec(ref)| and max |dec(naive) - dec(ref)| over all slots beside the decrypt-twice
        // control and the float64 reference, and |sf(l_exit + 1)/sf(depth - R) - 1| from the context's own
        // scaling factors (the ct x pt records sf_ct^2, ckksrns-leveledshe.cpp:157-162: the ONLY effect of the
        // kept tracked scale, so it bounds the disc-vs-ref difference). A case whose settled level is already
        // >= depth - R is the "deeper" case: untouched and counted, as x8DropTo does.
        std::vector<double> Wx((size_t)d * d);
        for (uint32_t i = 0; i < d; i++) for (uint32_t j = 0; j < d; j++) Wx[(size_t)i * d + j] = std::cos(0.11 * i + 0.23 * j) * 0.1;
        std::vector<double> refF(d, 0.0);   // float64 matvec of v
        for (uint32_t i = 0; i < d; i++) for (uint32_t j = 0; j < d; j++) refF[i] += Wx[(size_t)i * d + j] * v[j];
        double refMax = 0; for (uint32_t i = 0; i < d; i++) refMax = std::max(refMax, std::abs(refF[i]));
        const auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
        auto ptX = [&](const std::vector<double>& packed, const Ciphertext<DCRTPoly>& r) { return cc->MakeCKKSPackedPlaintext(packed, 1, r->GetLevel()); };
        auto walkTo = [&](TrackedCt& c, uint32_t target) { std::vector<double> one(SLOTS, 1.0);
            while (c.ct->GetLevel() < target) { c.ct = cc->EvalMult(c.ct, ptX(one, c.ct)); cc->RescaleInPlace(c.ct); } };
        auto limbsOf = [&](const TrackedCt& c) { return (uint32_t)c.ct->GetElements()[0].GetNumOfElements(); };
        auto maxDiff = [&](const std::vector<double>& a, const std::vector<double>& b) { double m = 0; for (size_t i = 0; i < a.size() && i < b.size(); i++) m = std::max(m, std::abs(a[i] - b[i])); return m; };
        auto errF = [&](const std::vector<double>& a) { double m = 0; for (uint32_t i = 0; i < d; i++) m = std::max(m, std::abs(a[i] - refF[i])); return m; };
        const int casesR[] = {7, 4, 7, 4, 4};
        const uint32_t casesE[] = {7, 7, 15, 20, 26};
        for (int k = 0; k < 5; k++) {
            const int R = casesR[k]; const uint32_t lExit = casesE[k];
            const uint32_t target = depth - (uint32_t)R;
            TrackedCt u = enc(packCh(v)); walkTo(u, lExit);
            const uint32_t degIn = u.ct->GetNoiseScaleDeg(), lvlIn = u.ct->GetLevel();
            const uint32_t lvlSettled = lvlIn + (degIn > 1 ? 1u : 0u);
            TrackedCt uR = u; uR.used = uR.ct->GetLevel();                       // ref: the .cu's convention
            TrackedCt oR = matvec(uR, Wx, d, d, 0);
            auto dR = dec(oR); auto dR2 = dec(oR);
            const double ctrl = maxDiff(dR, dR2), eRef = errF(dR);
            const std::string tag = "x8_drop_R" + std::to_string(R) + "_e" + std::to_string(lExit);
            if (lvlSettled >= target) {
                std::cout << "{\"cell\":\"x8-drop\",\"R\":" << R << ",\"lExit\":" << lExit << ",\"target\":" << target << ",\"levelIn\":" << lvlIn << ",\"noiseDegIn\":" << degIn
                          << ",\"levelSettled\":" << lvlSettled << ",\"skippedDeeper\":true,\"limbsIn\":" << limbsOf(u) << ",\"refOutLevel\":" << oR.ct->GetLevel()
                          << ",\"refErrVsFloat64\":" << eRef << "}" << std::endl;
                verdict(tag + "_skipped_deeper", lvlSettled >= target, (double)lvlSettled, false);
                continue;
            }
            TrackedCt uD; uD.ct = u.ct->Clone();                                  // disc: settle, then reduce
            if (uD.ct->GetNoiseScaleDeg() > 1) cc->GetScheme()->ModReduceInternalInPlace(uD.ct, 1);
            const uint32_t lvlAfterSettle = uD.ct->GetLevel();
            if (lvlAfterSettle < target) cc->GetScheme()->LevelReduceInternalInPlace(uD.ct, target - lvlAfterSettle);
            const uint32_t lvlD = uD.ct->GetLevel(), degD = uD.ct->GetNoiseScaleDeg(), limbsD = limbsOf(uD);
            uD.used = lvlD;
            TrackedCt oD = matvec(uD, Wx, d, d, 0);
            auto dD = dec(oD);
            TrackedCt uN; uN.ct = u.ct->Clone();                                  // naive: reduce without the settle
            if (lvlIn < target) cc->GetScheme()->LevelReduceInternalInPlace(uN.ct, target - lvlIn);
            const uint32_t lvlN = uN.ct->GetLevel(), degN = uN.ct->GetNoiseScaleDeg(), limbsN = limbsOf(uN);
            uN.used = lvlN;
            TrackedCt oN = matvec(uN, Wx, d, d, 0);
            auto dN = dec(oN);
            const double diffD = maxDiff(dD, dR), diffN = maxDiff(dN, dR), eD = errF(dD), eN = errF(dN);
            const double sfS = cp->GetScalingFactorReal(lvlSettled), sfT = cp->GetScalingFactorReal(target);
            const double ratioMinus1 = sfS / sfT - 1.0;
            char sfbuf[200]; std::snprintf(sfbuf, sizeof sfbuf, "\"sfSettled\":%.17g,\"sfTarget\":%.17g,\"scaleRatioMinus1\":%.6e,\"scaleBoundOnOutput\":%.6e",
                                           sfS, sfT, ratioMinus1, std::abs(ratioMinus1) * refMax);
            std::cout << "{\"cell\":\"x8-drop\",\"R\":" << R << ",\"lExit\":" << lExit << ",\"target\":" << target << ",\"levelIn\":" << lvlIn << ",\"noiseDegIn\":" << degIn
                      << ",\"levelSettled\":" << lvlSettled << ",\"limbsIn\":" << limbsOf(u)
                      << ",\"disc\":{\"levelIn\":" << lvlD << ",\"noiseDeg\":" << degD << ",\"limbs\":" << limbsD << ",\"outLevel\":" << oD.ct->GetLevel()
                      << ",\"outNoiseDeg\":" << oD.ct->GetNoiseScaleDeg() << ",\"maxAbsDiffVsRef\":" << diffD << ",\"errVsFloat64\":" << eD << "}"
                      << ",\"naive\":{\"levelIn\":" << lvlN << ",\"noiseDeg\":" << degN << ",\"limbs\":" << limbsN << ",\"outLevel\":" << oN.ct->GetLevel()
                      << ",\"outNoiseDeg\":" << oN.ct->GetNoiseScaleDeg() << ",\"maxAbsDiffVsRef\":" << diffN << ",\"errVsFloat64\":" << eN << "}"
                      << ",\"ref\":{\"outLevel\":" << oR.ct->GetLevel() << ",\"outNoiseDeg\":" << oR.ct->GetNoiseScaleDeg() << ",\"errVsFloat64\":" << eRef
                      << ",\"controlDecryptedTwice\":" << ctrl << "}," << sfbuf << "}" << std::endl;
            verdict(tag + "_disc_input_level_deg1", lvlD == target && degD == 1, (double)lvlD, true);
            verdict(tag + "_disc_limbs_R_plus_1", limbsD == (uint32_t)R + 1u, (double)limbsD, true);
            verdict(tag + "_disc_value_exact", diffD < 1e-6 && eD < 1e-3, diffD, true);
            verdict(tag + "_disc_out_level_target", oD.ct->GetLevel() == target, (double)oD.ct->GetLevel(), true);
            verdict(tag + "_naive_out_one_deeper", oN.ct->GetLevel() == target + 1u, (double)oN.ct->GetLevel(), false);
            verdict(tag + "_naive_value_exact", diffN < 1e-6 && eN < 1e-3, diffN, false);
        }
    }
    if (cells.count("pf-trace")) {
        // ONE layer of the real-model circuit, T = 1, h = 1 (no lane masks), shift-mix
        // without a carry (u_{-1} = 0), the stock scan on a fresh zero state, the
        // learned gates, K = dff/D chunks -- the statement order of gpu_real_model.cu's
        // layer loop, with refresh()/boot() placed per --schedule {stock,parent-first}.
        // Every refresh/boot prints (site, GetLevel, remaining). Compare with:
        //   python3 results/theory/tick_level_trace.py --depth 30 --lam <observed> --boot-floor 3 --margin 4
        //       --dpad 64 --d 48 --dff 128 --layers 1 --iters 3 --ticks 1 --lazy --trace-levels [--schedule parent-first]
        const uint32_t F = (uint32_t)cellF, margin = (uint32_t)cellM;
        int boots = 0; uint32_t lamObserved = 0;
        // S3.7 V2: an INDEPENDENT tally of the norm-internal boots (set by rmsnormL
        // while its window is open) to check the .cu's snapshot subtraction: the same
        // dt goes into bootMs, into normBootDirectMs when inside a norm window, and
        // -- via the window's bootMs delta -- into normBootMs; the two must agree exactly.
        bool inNormWindow = false; double normBootDirectMs = 0; int bootsInNorm = 0;
        std::vector<std::string> traceOut;
        auto emit = [&](const std::string& site, const std::string& kind, uint32_t lvl) {
            std::ostringstream o; o << "{\"pfTrace\":\"" << kind << "\",\"site\":\"" << site << "\",\"level\":" << lvl
                                    << ",\"remaining\":" << (int)depth - (int)lvl << "}";
            traceOut.push_back(o.str()); std::cout << o.str() << std::endl;
        };
        auto boot = [&](TrackedCt& c, const std::string& site) {
            const uint32_t pre = c.ct->GetLevel();
            syncDev();
            auto tB = Clock::now();                             // S3.7 V2: boot()'s bootMs window
            c.ct = cc->EvalBootstrap(c.ct); boots++;
            const double dt = msSince(tB);
            bootMs += dt;
            if (inNormWindow) { normBootDirectMs += dt; bootsInNorm++; }
            const uint32_t post = c.ct->GetLevel();
            if (!lamObserved) lamObserved = depth - post;
            std::ostringstream o; o << "{\"pfTrace\":\"boot\",\"site\":\"" << site << "\",\"lvlPre\":" << pre << ",\"lvlPost\":" << post
                                    << ",\"remainingAfter\":" << (int)depth - (int)post << "}";
            traceOut.push_back(o.str()); std::cout << o.str() << std::endl;
        };
        auto refresh = [&](TrackedCt& c, uint32_t need, const std::string& site) {   // gpu_real_model.cu:2156-2182
            const uint32_t lvl = c.ct->GetLevel();
            emit(site, "refresh", lvl);
            if (lvl > depth) { boot(c, site); return; }
            if (depth - lvl < need + F) boot(c, site);
        };
        const bool pf = (cellSchedule == "parent-first");
        const uint32_t nb = 3, dN = cellNd2 ? 2u : 4u;
        auto needTail = [&](int kind) -> uint32_t { return kind == 1 ? 12u : kind == 2 ? 2u + 1u + 4u + margin : 4u; };
        auto needEntry = [&](int kind, int iters) -> uint32_t {
            const uint32_t loop = nb + dN * (uint32_t)std::max(iters - 1, 0) + 5u, branch = needTail(kind) + 2u;
            if (pfEntry == "loop") return loop; if (pfEntry == "branch") return branch;
            if (pfEntry == "both") return std::max(loop, branch); return depth + 1u; };
        auto ptL = [&](const std::vector<double>& packed, const Ciphertext<DCRTPoly>& ref) {
            ptCalls++;                                                                          // S3.7 X2: the census
            if (x2PtCache) { Plaintext hit = x2Lookup(packed, ref->GetLevel(), __LINE__); if (hit) return hit; }   // S3.7 X2
            return cc->MakeCKKSPackedPlaintext(packed, 1, ref->GetLevel()); };
        // S3.7 X2: the cell's scalar rule (the same rule as x2Mul/x2Add, on ptL's level convention)
        auto x2MulL = [&](Ciphertext<DCRTPoly>& ct, const std::vector<double>& v) {
            double cval = 0.0;
            if (x2PtScalar && x2IsConst(v, cval) && x2ScalarOp(ct, cval, false, [&]() { return ptL(v, ct); })) return;
            ct = cc->EvalMult(ct, ptL(v, ct)); };
        auto x2AddL = [&](Ciphertext<DCRTPoly>& ct, const std::vector<double>& v) {
            double cval = 0.0;
            if (x2PtScalar && x2IsConst(v, cval) && x2ScalarOp(ct, cval, true, [&]() { return ptL(v, ct); })) return;
            auto p_ = ptL(v, ct); ct = cc->EvalAdd(ct, p_); };                          // EvalAdd wants an lvalue plaintext
        auto mulPtL = [&](TrackedCt& c, const std::vector<double>& packed) {
            x2MulL(c.ct, packed); cc->RescaleInPlace(c.ct); };                                   // S3.7 X2: was EvalMult(ptL)
        auto alignToL = [&](TrackedCt& c, uint32_t target) {
            std::vector<double> one(SLOTS, 1.0);
            while (c.ct->GetLevel() < target) { x2MulL(c.ct, one); cc->RescaleInPlace(c.ct); } };   // S3.7 X2
        auto addAlignedL = [&](TrackedCt& a, TrackedCt& b, const std::string& site) {
            refresh(a, 2, site + ".a"); refresh(b, 2, site + ".b"); a.ct = cc->EvalAdd(a.ct, b.ct); };
        // rmsnorm with the harness's refresh placements (kind: 1 tm, 2 cm, 0 out)
        auto rmsnormL = [&](TrackedCt x, const std::vector<double>& gpk, double a0, double b0, int iters, int kind, const std::string& site) -> TrackedCt {
            auto tN = Clock::now();                             // S3.7 V2: the rmsnorm window with its own boots inside
            const double normBoot0 = bootMs; inNormWindow = true;
            TrackedCt sq{cc->EvalMult(x.ct, x.ct), 0}; cc->RescaleInPlace(sq.ct);
            Ciphertext<DCRTPoly> red = sq.ct;
            for (uint32_t p = 1; p < D; p <<= 1) red = cc->EvalAdd(red, cc->EvalRotate(red, p));
            std::vector<double> invd(SLOTS, 1.0 / (double)d);
            Ciphertext<DCRTPoly> msc = red; x2MulL(msc, invd);                         // S3.7 X2: was EvalMult(ptL)
            TrackedCt ms{msc, 0}; cc->RescaleInPlace(ms.ct);
            std::vector<double> b0v(SLOTS, b0), a0v(SLOTS, a0);
            Ciphertext<DCRTPoly> yc = ms.ct; x2MulL(yc, b0v);                          // S3.7 X2: was EvalMult(ptL)
            TrackedCt y{yc, 0}; cc->RescaleInPlace(y.ct);
            x2AddL(y.ct, a0v);                                                         // S3.7 X2: was EvalAdd(ptL)
            emit(site + ".y0", "born", y.ct->GetLevel());
            TrackedCt msh;
            if (cellNd2) { std::vector<double> mh(SLOTS, -0.5); msh.ct = ms.ct; x2MulL(msh.ct, mh); cc->RescaleInPlace(msh.ct); }   // S3.7 X2
            for (int it = 0; it < iters; it++) {
                if (cellNd2) {
                    refresh(y, 5, site + ".newton" + std::to_string(it) + ".y"); refresh(msh, 5, site + ".newton" + std::to_string(it) + ".ms");
                    TrackedCt a{cc->EvalMult(msh.ct, y.ct), 0}; cc->RescaleInPlace(a.ct);
                    TrackedCt b{cc->EvalMult(y.ct, y.ct), 0};   cc->RescaleInPlace(b.ct);
                    TrackedCt tt{cc->EvalMult(a.ct, b.ct), 0};  cc->RescaleInPlace(tt.ct);
                    std::vector<double> o15(SLOTS, 1.5);
                    Ciphertext<DCRTPoly> y15c = y.ct; x2MulL(y15c, o15);                   // S3.7 X2: was EvalMult(ptL)
                    TrackedCt y15{y15c, 0}; cc->RescaleInPlace(y15.ct);
                    y15.ct = cc->EvalAdd(y15.ct, tt.ct); y = y15;
                } else {
                    refresh(y, 5, site + ".newton" + std::to_string(it) + ".y"); refresh(ms, 5, site + ".newton" + std::to_string(it) + ".ms");
                    TrackedCt y2{cc->EvalMult(y.ct, y.ct), 0}; cc->RescaleInPlace(y2.ct);
                    TrackedCt tms{cc->EvalMult(ms.ct, y2.ct), 0}; cc->RescaleInPlace(tms.ct);
                    std::vector<double> half(SLOTS, -0.5), oneP5(SLOTS, 1.5);
                    x2MulL(tms.ct, half); cc->RescaleInPlace(tms.ct);                           // S3.7 X2: was EvalMult(ptL)
                    x2AddL(tms.ct, oneP5);                                                     // S3.7 X2: was EvalAdd(ptL)
                    y.ct = cc->EvalMult(y.ct, tms.ct); cc->RescaleInPlace(y.ct);
                }
            }
            refresh(y, pf ? needTail(kind) : 4u, site + ".tail.y");
            TrackedCt out{cc->EvalMult(x.ct, y.ct), 0}; cc->RescaleInPlace(out.ct);
            mulPtL(out, gpk);
            evalMs += msSince(tN);                              // S3.7 V2: the .cu's :3370 exit ...
            normBootMs += bootMs - normBoot0; inNormWindow = false;   // ... and the snapshot subtraction
            return out;
        };
        auto polyGate3L = [&](TrackedCt z, const std::vector<double>& p0, const std::vector<double>& p1, const std::vector<double>& p2) -> TrackedCt {
            TrackedCt z2{cc->EvalMult(z.ct, z.ct), 0}; cc->RescaleInPlace(z2.ct);
            TrackedCt term2 = z2; mulPtL(term2, packCh(p2));
            TrackedCt term1 = z;  mulPtL(term1, packCh(p1));
            alignToL(term1, term2.ct->GetLevel());
            term1.ct = cc->EvalAdd(term1.ct, term2.ct);
            x2AddL(term1.ct, packCh(p0));                                                  // S3.7 X2: was EvalAdd(ptL)
            TrackedCt zA = z; alignToL(zA, term1.ct->GetLevel());
            TrackedCt g{cc->EvalMult(zA.ct, term1.ct), 0}; cc->RescaleInPlace(g.ct);
            return g;
        };
        // the layer's constants (deterministic, in-range)
        const int I = cellIters;
        // S3.7 X13 (T18 §5): the per-site sidecar's L0.tm / L0.cm counts for the twin's one layer (default: I, i.e. unchanged)
        const int Itm = x13Iters.count("L0.tm") ? x13Iters.at("L0.tm") : I, Icm = x13Iters.count("L0.cm") ? x13Iters.at("L0.cm") : I;
        if (!x13SidecarPath.empty())
            std::cout << "{\"x13Sidecar\":\"" << x13SidecarPath << "\",\"itersTm\":" << Itm << ",\"itersCm\":" << Icm << ",\"itersDefault\":" << I
                      << ",\"sumItersFile\":" << x13Sum << "}" << std::endl;
        std::vector<double> g1(d, 1.0), g2(d, 1.0), mix(d), imix(d), dec_(d), bco(d), cc_(d), dd(d), p0(d), p1(d), p2(d);
        for (uint32_t i = 0; i < d; i++) { mix[i] = 0.5 + 0.2 * std::sin(0.3 * i); imix[i] = 1.0 - mix[i]; dec_[i] = 0.90 + 0.09 * (i % 7) / 7.0; bco[i] = 1.0 - dec_[i];
                                          cc_[i] = 0.5 * std::cos(0.2 * i); dd[i] = 0.5 * std::sin(0.25 * i); p0[i] = 0.5; p1[i] = 0.25; p2[i] = 0.05 * std::cos(0.1 * i); }
        std::vector<double> q0(dff, 0.0), q1(dff, 1.0), q2(dff, 0.1), q3(dff, 0.02), r0(dff, 0.5), r1(dff, 0.25), r2(dff, 0.05);
        std::vector<double> Wsq((size_t)d * d), Wk((size_t)dff * d), Wr((size_t)dff * d), Wv((size_t)d * dff);
        for (uint32_t i = 0; i < d; i++) for (uint32_t j = 0; j < d; j++) Wsq[(size_t)i * d + j] = std::cos(0.11 * i + 0.23 * j) * 0.1;
        for (uint32_t i = 0; i < dff; i++) for (uint32_t j = 0; j < d; j++) { Wk[(size_t)i * d + j] = std::sin(0.07 * i + 0.19 * j) * 0.1; Wr[(size_t)i * d + j] = std::cos(0.05 * i + 0.13 * j) * 0.1; }
        for (uint32_t i = 0; i < d; i++) for (uint32_t j = 0; j < dff; j++) Wv[(size_t)i * dff + j] = std::sin(0.03 * i + 0.17 * j) * 0.05;
        double ms0 = 0; for (uint32_t i = 0; i < d; i++) ms0 += v[i] * v[i]; ms0 /= d;
        const double a0 = 1.5 / std::sqrt(ms0), b0 = -0.5 / (ms0 * std::sqrt(ms0));
        TrackedCt Hs = enc(packCh(v));                                   // fresh client ct, level 0
        std::vector<double> zero(SLOTS, 0.0);
        TrackedCt scan = enc(zero);                                       // cold scan state
        // S3.7 V2: the layer loop's clock and the five stage windows, placed exactly
        // where gpu_real_model.cu places them (shift-mix, scan, gate, residual, hidden,
        // ffnAccum -> residual, cm residual). Snapshot the accumulators first so the
        // timers object below reports this layer only.
        const double tEv0 = evalMs, tNb0 = normBootMs, tBo0 = bootMs, tEn0 = encMs, tDe0 = decMs;
        const double tSm0 = shiftMixMs, tSc0 = scanMs, tGa0 = gateMs, tHi0 = hiddenMs, tRe0 = residualMs;
        const long long tPc0 = hostPtEncodeCount, tPs0 = hostPtEncodeStageCount;
        if (stageTimers) syncDev();
        const auto tLoop0 = Clock::now();
        const double loopEv0 = evalMs, loopNb0 = normBootMs, loopBo0 = bootMs, loopEn0 = encMs, loopDe0 = decMs;
        // ---- time mix ----
        emit("Hs@tm.entry", "probe", Hs.ct->GetLevel());
        if (pf) refresh(Hs, needEntry(1, Itm), "L0.tm.rmsnorm.Hs"); else refresh(Hs, 3, "L0.tm.rmsnorm.Hs");   // S3.7 X13: Itm (= I without a sidecar)
        TrackedCt u = rmsnormL(Hs, packCh(g1), a0, b0, Itm, 1, "L0.tm.rmsnorm");
        const auto stShift = stageOpen();                                 // S3.7 V2: shiftMixMs
        TrackedCt ua = u; mulPtL(ua, packCh(mix));                        // shift-mix, u_{-1} = 0
        u = ua;
        stageClose(stShift, shiftMixMs);
        u.used = u.ct->GetLevel();                                       // matvec encodes at .used (review: keep it = GetLevel)
        TrackedCt x = matvec(u, Wsq, d, d, 0);                            // win
        const auto stScan = stageOpen();                                  // S3.7 V2: scanMs
        refresh(scan, 3, "L0.tm.scan.state"); refresh(x, 3, "L0.tm.scan.x");
        mulPtL(scan, packCh(dec_));
        TrackedCt bx = x; mulPtL(bx, packCh(bco));
        addAlignedL(scan, bx, "L0.tm.scan");
        TrackedCt S = scan;
        stageClose(stScan, scanMs);
        const auto stGate = stageOpen();                                  // S3.7 V2: gateMs
        TrackedCt zc = S; mulPtL(zc, packCh(cc_));
        TrackedCt zd = x; mulPtL(zd, packCh(dd));
        addAlignedL(zc, zd, "L0.tm.gate.readout");
        refresh(zc, 5, "L0.tm.gate.zc");
        TrackedCt g = polyGate3L(zc, p0, p1, p2);
        stageClose(stGate, gateMs);
        g.used = g.ct->GetLevel();
        TrackedCt attn = matvec(g, Wsq, d, d, 0);                          // wout
        const auto stResTm = stageOpen();                                 // S3.7 V2: residualMs
        addAlignedL(Hs, attn, "L0.tm.residual");
        stageClose(stResTm, residualMs);
        // ---- channel mix ----
        emit("Hs@cm.entry", "probe", Hs.ct->GetLevel());
        if (pf) refresh(Hs, needEntry(2, Icm), "L0.cm.rmsnorm.Hs"); else refresh(Hs, 3, "L0.cm.rmsnorm.Hs");   // S3.7 X13: Icm
        TrackedCt u2 = rmsnormL(Hs, packCh(g2), a0, b0, Icm, 2, "L0.cm.rmsnorm");
        if (pf) refresh(u2, needTail(2) - 2u + 1u, "L0.cm.u2");
        u2.used = u2.ct->GetLevel();
        auto kks = matvecChunked(u2, Wk, dff, d);
        auto rrs = matvecChunked(u2, Wr, dff, d);
        TrackedCt ffn; bool haveFfn = false;
        for (uint32_t ch = 0; ch < NCHUNK; ch++) {
            const uint32_t n = std::min(D, dff - ch * D);
            auto chunkOf = [&](const std::vector<double>& c) { std::vector<double> s_(n); for (uint32_t i = 0; i < n; i++) s_[i] = c[ch * D + i]; return packCh(s_); };
            const std::string sch = "L0.cm.hidden.ch" + std::to_string(ch);
            const auto stHid = stageOpen();                               // S3.7 V2: hiddenMs (per chunk)
            TrackedCt kk = kks[ch], rr = rrs[ch];
            emit("kk@birth", "probe", kk.ct->GetLevel());
            refresh(kk, 5, sch + ".kk"); refresh(rr, 4, sch + ".rr");
            TrackedCt k2{cc->EvalMult(kk.ct, kk.ct), 0}; cc->RescaleInPlace(k2.ct);
            TrackedCt actT = kk; mulPtL(actT, chunkOf(q1));
            TrackedCt t2 = k2; mulPtL(t2, chunkOf(q2));
            addAlignedL(actT, t2, sch + ".act12");
            TrackedCt kkA = kk; alignToL(kkA, k2.ct->GetLevel());
            TrackedCt k3{cc->EvalMult(k2.ct, kkA.ct), 0}; cc->RescaleInPlace(k3.ct); mulPtL(k3, chunkOf(q3));
            addAlignedL(actT, k3, sch + ".act3");
            x2AddL(actT.ct, chunkOf(q0));                                                  // S3.7 X2: was EvalAdd(ptL)
            TrackedCt r2c{cc->EvalMult(rr.ct, rr.ct), 0}; cc->RescaleInPlace(r2c.ct);
            TrackedCt gt = rr; mulPtL(gt, chunkOf(r1));
            TrackedCt g2t = r2c; mulPtL(g2t, chunkOf(r2));
            addAlignedL(gt, g2t, sch + ".gate");
            x2AddL(gt.ct, chunkOf(r0));                                                    // S3.7 X2: was EvalAdd(ptL)
            refresh(actT, 2, sch + ".actT"); refresh(gt, 2, sch + ".gt");
            const uint32_t uu = std::max(actT.ct->GetLevel(), gt.ct->GetLevel());
            alignToL(actT, uu); alignToL(gt, uu);
            TrackedCt hd{cc->EvalMult(actT.ct, gt.ct), 0}; cc->RescaleInPlace(hd.ct);
            refresh(hd, 2, sch + ".hd");
            refresh(hd, margin, "L0.cm.wv.ch" + std::to_string(ch) + ".margin");
            emit("hd@wv.in", "probe", hd.ct->GetLevel());
            std::vector<double> Wb((size_t)d * D, 0.0);
            for (uint32_t i = 0; i < d; i++) for (uint32_t j = 0; j < n; j++) Wb[(size_t)i * D + j] = Wv[(size_t)i * dff + ch * D + j];
            hd.used = hd.ct->GetLevel();
            stageClose(stHid, hiddenMs);                                  // S3.7 V2 (the Wb slice is inside, as in the .cu)
            TrackedCt part = matvec(hd, Wb, d, D, 0);
            const auto stFfn = stageOpen();                               // S3.7 V2: residualMs (ffnAccum)
            if (!haveFfn) { ffn = part; haveFfn = true; } else addAlignedL(ffn, part, "L0.cm.ffnAccum.ch" + std::to_string(ch));
            stageClose(stFfn, residualMs);
        }
        const auto stResCm = stageOpen();                                 // S3.7 V2: residualMs (cm residual)
        addAlignedL(Hs, ffn, "L0.cm.residual");
        stageClose(stResCm, residualMs);
        if (stageTimers) {                                                // S3.7 V2: close the layer's clock
            syncDev();
            layerLoopMs += msSince(tLoop0);
            layerLoopTimedMs += ((evalMs - loopEv0) - (normBootMs - loopNb0)) + (bootMs - loopBo0)
                              + (encMs - loopEn0) + (decMs - loopDe0);
        }
        emit("Hs@layer.out", "probe", Hs.ct->GetLevel());
        if (x2PtCache) x2TickLine("pf-trace.layer0");                       // S3.7 X2: the census of this layer
        auto o = dec(Hs);
        bool finite = true; for (uint32_t i = 0; i < d; i++) finite = finite && std::isfinite(o[i]);
        std::cout << "{\"cell\":\"pf-trace\",\"schedule\":\"" << cellSchedule << "\",\"pfEntry\":\"" << pfEntry << "\",\"newtonDepth2\":" << (cellNd2 ? "true" : "false")
                  << ",\"depth\":" << depth << ",\"lamObserved\":" << lamObserved << ",\"F\":" << F << ",\"margin\":" << margin << ",\"iters\":" << I
                  << ",\"Dpad\":" << D << ",\"d\":" << d << ",\"dff\":" << dff << ",\"boots\":" << boots << ",\"finite\":" << (finite ? "true" : "false")
                  << ",\"HsLevelOut\":" << Hs.ct->GetLevel() << "}" << std::endl;
        verdict("pf_trace_layer_finite", finite, 0.0, true);
        // ---- S3.7 V2: the timers object for THIS layer (deltas since the snapshot) and
        // the two accounting verdicts. (1) the norm-boot subtraction: normBootMs (the
        // window's bootMs delta) must equal normBootDirectMs (the same dt tallied at the
        // boot site while a norm window is open) EXACTLY, and be <= bootMs; the ex-boot
        // eval must be >= 0. (2) disjointness: the loop's untimed part minus the five
        // stages (the residual) is >= 0 up to rounding, and so is each stage.
        {
            const double dEv = evalMs - tEv0, dNb = normBootMs - tNb0, dBo = bootMs - tBo0, dEn = encMs - tEn0, dDe = decMs - tDe0;
            const double sSm = shiftMixMs - tSm0, sSc = scanMs - tSc0, sGa = gateMs - tGa0, sHi = hiddenMs - tHi0, sRe = residualMs - tRe0;
            const double stages = sSm + sSc + sGa + sHi + sRe;
            const double loopUntimed = layerLoopMs - layerLoopTimedMs;
            const double residual = loopUntimed - stages;
            const bool subOk = (dNb == normBootDirectMs) && (dNb <= dBo + 1e-9) && (dEv - dNb >= -1e-9);
            const bool disjOk = !stageTimers || (residual >= -1e-6 && sSm >= -1e-6 && sSc >= -1e-6 && sGa >= -1e-6 && sHi >= -1e-6 && sRe >= -1e-6
                                                 && layerLoopMs >= layerLoopTimedMs - 1e-6);
            std::cout << "{\"cell\":\"pf-trace\",\"timers\":{\"stageTimersOn\":" << (stageTimers ? "true" : "false")
                      << ",\"boots\":" << boots << ",\"bootsInNorm\":" << bootsInNorm
                      << ",\"evalMs\":" << dEv << ",\"bootMs\":" << dBo << ",\"normBootMs\":" << dNb << ",\"normBootDirectMs\":" << normBootDirectMs
                      << ",\"gpuEvalMsExBoot\":" << (dEv - dNb) << ",\"msPerTokenNaive\":" << (dEv + dBo) << ",\"msPerTokenExBoot\":" << ((dEv - dNb) + dBo)
                      << ",\"encMs\":" << dEn << ",\"decMs\":" << dDe
                      << ",\"layerLoopMs\":" << layerLoopMs << ",\"layerLoopTimedMs\":" << layerLoopTimedMs << ",\"layerLoopUntimedMs\":" << loopUntimed
                      << ",\"stageMs\":{\"shiftMix\":" << sSm << ",\"scan\":" << sSc << ",\"gate\":" << sGa << ",\"hidden\":" << sHi << ",\"residual\":" << sRe << "}"
                      << ",\"layerLoopResidualMs\":" << residual
                      << ",\"hostPtEncodeCount\":" << (hostPtEncodeCount - tPc0) << ",\"hostPtEncodeStageCount\":" << (hostPtEncodeStageCount - tPs0)
                      << ",\"hostPtEncodeMs\":" << hostPtEncodeMs << ",\"hostPtEncodeStageMs\":" << hostPtEncodeStageMs << "}}" << std::endl;
            verdict("timers_norm_boot_subtraction", subOk, dNb - normBootDirectMs, true);
            verdict("timers_stage_disjoint", disjOk, residual, true);
        }
        // ==== S3.7 X1 boot schedule (2026-09-10; S3.6 T03 §1.3-§1.4, §2.2, §5): the multi-tick carry cell ====
        // --ticks N > 1: N more ticks of the SAME one-layer circuit, each on a fresh client ciphertext, with
        // the demo's two carries across ticks -- the scan state (Lemma 2) and the pre-shift u (the shift-mix
        // boundary) -- handed off at every tick end the way gpu_real_model.cu does under --canonical-carry
        // (`canon` at HEAD: align-then-boot for a carry shallower than the post-boot level, boot-only for a
        // deeper one, the post-boot level learned from the first canonical boot). The three X1 levers act
        // here exactly as in the .cu copy (see the flag block at the top). Per tick: the boot count from this
        // process's own counter (`boots`, the pf-trace cell's), the hand-off's boots and drops, the end levels
        // of both carries, and two value checks against references built from DECRYPTED operands, so that
        // upstream CKKS error cannot masquerade as a lever error: the scan readout S vs the float64
        // recurrence driven by dec(x_t), and the shift-mix output vs dec(u) . mix + dec(uPrev) . (1-mix).
        // The one-tick cell above is untouched and prints first; with --ticks 1 (the default) none of this runs.
        if (x1Ticks > 1) {
            const int M = x1DeferredScan;
            const bool deferred = (M > 1);
            TrackedCt scanB = enc(zero);                        // s_{-1} = 0 (cold)
            TrackedCt uPrev = enc(zero);                        // u_{-1} = 0, explicit as in the .cu's canonical cold start
            std::vector<TrackedCt> xs;                          // the deferred terms (STATE: they cross ticks)
            int phase = 0;
            int canonLevelL = -1; long long canonBootsL = 0, uDrops = 0, uDropBoots = 0, folds = 0, boBoots = 0, boSkipped = 0;
            std::vector<int> bootsPerTick;
            std::vector<double> refS(d, 0.0);
            auto canonL = [&](TrackedCt& c, const std::string& what) {          // gpu_real_model.cu `canon` (HEAD rule)
                if (!x1CanonicalCarry) return;
                const uint32_t lvl0 = c.ct->GetLevel();
                if (canonLevelL >= 0 && lvl0 == (uint32_t)canonLevelL) return;
                if (canonLevelL < 0 || lvl0 < (uint32_t)canonLevelL) { const uint32_t target = depth - F; if (lvl0 < target) alignToL(c, target); }
                const uint32_t pre = c.ct->GetLevel();
                boot(c, "canon." + what);
                if (c.ct->GetLevel() >= pre) { std::cerr << "{\"fatal\":\"x1 canon: bootstrap did not restore levels\",\"what\":\"" << what << "\"}\n"; std::exit(2); }
                if (canonLevelL < 0) canonLevelL = (int)c.ct->GetLevel();
                else if (c.ct->GetLevel() < (uint32_t)canonLevelL) alignToL(c, (uint32_t)canonLevelL);
                else if (c.ct->GetLevel() > (uint32_t)canonLevelL) { std::cerr << "{\"fatal\":\"x1 canon: landed deeper than canonLevel\"}\n"; std::exit(2); }
                canonBootsL++;
            };
            auto uDropL = [&](TrackedCt& c, const std::string& what) -> bool {    // T03 `mixed` u rule
                if (x1UDropLevel <= 0) return false;
                const uint32_t C = (uint32_t)x1UDropLevel;
                c.ct = c.ct->Clone();                                                   // never mutate a shared object in place
                // FOUND 2026-09-10 (ticks_ud19_t4_prefix.jsonl): under FLEXIBLEAUTO the public RescaleInPlace is a
                // NO-OP (rns-leveledshe.cpp:354-358) -- the rescale happens lazily inside the next multiply -- so the
                // u carry leaves the layer with a PENDING rescale (noiseScaleDeg 2), which a limb drop keeps: the
                // consumer then auto-rescales first and "level C" is C+1 in effect (one extra boot per tick vs the
                // replay, whose `mixed` rule sets deg 1 at C). Materialise the pending rescale (level +1, deg 1)
                // BEFORE reading the level, so that the carry enters at C with the replay's headroom.
                auto settle = [&]() { if (c.ct->GetNoiseScaleDeg() > 1) cc->GetScheme()->ModReduceInternalInPlace(c.ct, 1); };
                settle();
                if (c.ct->GetLevel() > C) { boot(c, "canon." + what); uDropBoots++; settle(); }   // deeper than C: a drop cannot raise it
                const uint32_t lvl = c.ct->GetLevel();
                if (lvl < C) cc->GetScheme()->LevelReduceInternalInPlace(c.ct, C - lvl); // exact: limbs dropped, scale kept
                emit(what + ".uDrop", "drop", c.ct->GetLevel());
                uDrops++; return true;
            };
            auto residualX1 = [&](TrackedCt& Hs_, TrackedCt& br, const std::string& site) {
                if (x1BranchOutBoot) {
                    const int post = x1BranchOutPost >= 0 ? x1BranchOutPost
                                   : (canonLevelL >= 0 ? canonLevelL : (lamObserved ? (int)depth - (int)lamObserved : -1));
                    const int lb = (int)br.ct->GetLevel(), lh = (int)Hs_.ct->GetLevel();
                    if (post < 0) boSkipped++;
                    else if (lb - lh >= 1 && lb > post) { boot(br, site + ".branchOut"); boBoots++; }
                }
                addAlignedL(Hs_, br, site);
            };
            // ==== S3.7 X8 drops (2026-09-10; T07 §1.2): the class-site drop, the pool census, the per-class line ====
            // x8DropL mirrors gpu_real_model_x.cu's x8DropTo (settle-then-reduce; deeper than the target untouched;
            // already there untouched); x8PoolNote mirrors compPoolGet's LRU of --pool-levels resident levels (a miss
            // = one pool (re)build, the term T07 fact 3 says vanishes at the code floor); x8Site runs both at a class's
            // matvec input and prints one x8Class line (level/degree/limbs in and out). Nothing here runs unless a drop
            // is on or --x8-census is given, so the flag-off ticks cell prints exactly the X1 lines.
            std::vector<uint32_t> x8PoolLru;                    // most recently used LAST
            long long x8PoolBuilds = 0;
            std::vector<long long> x8PoolBuildsPerTick, x8DropsPerTick;
            auto x8DropL = [&](TrackedCt& c, int R, const std::string& what) {
                if (R <= 0) return;
                const uint32_t target = depth - (uint32_t)R;
                const uint32_t lvlIn = c.ct->GetLevel(), degIn = c.ct->GetNoiseScaleDeg();
                const uint32_t lvlSettled = lvlIn + (degIn > 1 ? 1u : 0u);
                if (lvlSettled > target) { x8DropsSkippedDeeper++; return; }
                if (lvlSettled == target && degIn == 1) { x8DropsAtTarget++; return; }
                c.ct = c.ct->Clone();                                                   // never mutate a shared object in place
                if (c.ct->GetNoiseScaleDeg() > 1) { cc->GetScheme()->ModReduceInternalInPlace(c.ct, 1); x8DropsSettled++; }
                const uint32_t lvl = c.ct->GetLevel();
                if (lvl < target) cc->GetScheme()->LevelReduceInternalInPlace(c.ct, target - lvl);   // exact: limbs dropped, scale kept
                x8DropsApplied++;
                emit(what + ".x8Drop", "drop", c.ct->GetLevel());
            };
            auto x8PoolNote = [&](uint32_t lvl) {
                auto it = std::find(x8PoolLru.begin(), x8PoolLru.end(), lvl);
                if (it != x8PoolLru.end()) { x8PoolLru.erase(it); x8PoolLru.push_back(lvl); return; }
                while (x8PoolLru.size() >= (size_t)x8PoolLevels) x8PoolLru.erase(x8PoolLru.begin());
                x8PoolLru.push_back(lvl); x8PoolBuilds++;
            };
            auto x8Site = [&](TrackedCt& c, int R, const std::string& cls, int tick) {
                if (!x8Census) return;
                const uint32_t lvl0 = c.ct->GetLevel(), deg0 = c.ct->GetNoiseScaleDeg(), limbs0 = (uint32_t)c.ct->GetElements()[0].GetNumOfElements();
                x8DropL(c, R, "T" + std::to_string(tick) + ".L0." + cls);
                c.used = c.ct->GetLevel();                                              // the matvec encodes at .used
                x8PoolNote(c.used);
                std::cout << "{\"x8Class\":\"" << cls << "\",\"tick\":" << tick << ",\"R\":" << R << ",\"levelIn\":" << lvl0 << ",\"noiseDegIn\":" << deg0
                          << ",\"limbsIn\":" << limbs0 << ",\"levelOut\":" << c.ct->GetLevel() << ",\"noiseDegOut\":" << c.ct->GetNoiseScaleDeg()
                          << ",\"limbsOut\":" << c.ct->GetElements()[0].GetNumOfElements() << ",\"remaining\":" << (int)depth - (int)c.ct->GetLevel() << "}" << std::endl;
            };
            auto powv = [&](int p, const std::vector<double>* times) {                // decay^p (.) times, packed
                std::vector<double> o(d, 1.0);
                for (uint32_t i = 0; i < d; i++) { for (int k = 0; k < p; k++) o[i] *= dec_[i]; if (times) o[i] *= (*times)[i]; }
                return packCh(o);
            };
            const int bootsBeforeCold = boots;
            if (x1CanonicalCarry) { canonL(scanB, "scan.cold"); if (!uDropL(uPrev, "u.cold")) canonL(uPrev, "u.cold"); }
            const int bootsCold = boots - bootsBeforeCold;
            for (int tick = 0; tick < x1Ticks; tick++) {
                const int b0 = boots; const long long cb0 = canonBootsL, ud0 = uDrops, udb0 = uDropBoots, bo0 = boBoots;
                const long long x8pb0 = x8PoolBuilds, x8dr0 = x8DropsApplied;         // S3.7 X8: this tick's pool builds and drops
                const std::string P = "T" + std::to_string(tick) + ".L0.";
                std::vector<double> vt(d); for (uint32_t i = 0; i < d; i++) vt[i] = v[i] * (1.0 + 0.1 * tick) + 0.05 * std::cos(0.7 * i + tick);
                TrackedCt Hs = enc(packCh(vt));
                double ms0t = 0; for (uint32_t i = 0; i < d; i++) ms0t += vt[i] * vt[i]; ms0t /= d;
                const double a0t = 1.5 / std::sqrt(ms0t), b0t = -0.5 / (ms0t * std::sqrt(ms0t));
                // ---- time mix ----
                if (pf) refresh(Hs, needEntry(1, Itm), P + "tm.rmsnorm.Hs"); else refresh(Hs, 3, P + "tm.rmsnorm.Hs");   // S3.7 X13: Itm
                TrackedCt u = rmsnormL(Hs, packCh(g1), a0t, b0t, Itm, 1, P + "tm.rmsnorm");
                TrackedCt uPre = u;                                                    // the carry is the PRE-shift u
                TrackedCt ua = u; mulPtL(ua, packCh(mix));
                { TrackedCt b = uPrev; refresh(b, 2, P + "tm.shiftMix.uPrev"); mulPtL(b, packCh(imix)); addAlignedL(ua, b, P + "tm.shiftMix"); }
                double smErr = 0;
                { auto du = dec(u); auto dp = dec(uPrev); auto da = dec(ua);
                  for (uint32_t i = 0; i < d; i++) smErr = std::max(smErr, std::abs(da[i] - (du[i] * mix[i] + dp[i] * imix[i]))); }
                u = ua; u.used = u.ct->GetLevel();
                x8Site(u, x8DropWin, "win", tick);                                     // S3.7 X8: the win-class drop + census
                TrackedCt x = matvec(u, Wsq, d, d, 0);                                // win
                // ---- scan: eager, or deferred (T03 §1.3 / SimDeferredScan) ----
                TrackedCt S; bool folded = false;
                if (deferred) {
                    const int j = phase;
                    refresh(x, 3, P + "tm.scan.x");
                    xs.push_back(x);                                                   // this tick's term: the post-refresh x_j
                    TrackedCt Sb = scanB; refresh(Sb, 2, P + "tm.scan.readCopy");     // guard on the COPY; never fires on a canonical base
                    mulPtL(Sb, powv(j + 1, nullptr));                                  // decay^(j+1) . s_base on a copy
                    for (int i = 0; i <= j; i++) { TrackedCt term = xs[i]; mulPtL(term, powv(j - i, &bco)); addAlignedL(Sb, term, P + "tm.scan.term" + std::to_string(i)); }
                    S = Sb;
                    if (j == M - 1) {                                                  // end of the period: fold in place
                        refresh(scanB, 3, P + "tm.scan.state");
                        mulPtL(scanB, powv(M, nullptr));
                        for (int i = 0; i < M; i++) { TrackedCt term = xs[i]; mulPtL(term, powv(M - 1 - i, &bco)); addAlignedL(scanB, term, P + "tm.scan.fold" + std::to_string(i)); }
                        xs.clear(); folds++; folded = true;
                    }
                } else {
                    refresh(scanB, 3, P + "tm.scan.state"); refresh(x, 3, P + "tm.scan.x");
                    mulPtL(scanB, packCh(dec_));
                    TrackedCt bx = x; mulPtL(bx, packCh(bco));
                    addAlignedL(scanB, bx, P + "tm.scan");
                    S = scanB;
                }
                double scanErr = 0;
                { auto dx = dec(x); auto dS = dec(S);
                  for (uint32_t i = 0; i < d; i++) refS[i] = dec_[i] * refS[i] + bco[i] * dx[i];
                  for (uint32_t i = 0; i < d; i++) scanErr = std::max(scanErr, std::abs(dS[i] - refS[i])); }
                TrackedCt zc = S; mulPtL(zc, packCh(cc_));
                TrackedCt zd = x; mulPtL(zd, packCh(dd));
                addAlignedL(zc, zd, P + "tm.gate.readout");
                refresh(zc, 5, P + "tm.gate.zc");
                TrackedCt g = polyGate3L(zc, p0, p1, p2);
                g.used = g.ct->GetLevel();
                x8Site(g, x8DropWout, "wout", tick);                                   // S3.7 X8: the wout-class drop + census
                TrackedCt attn = matvec(g, Wsq, d, d, 0);                              // wout
                residualX1(Hs, attn, P + "tm.residual");
                // ---- channel mix ----
                if (pf) refresh(Hs, needEntry(2, Icm), P + "cm.rmsnorm.Hs"); else refresh(Hs, 3, P + "cm.rmsnorm.Hs");   // S3.7 X13: Icm
                TrackedCt u2 = rmsnormL(Hs, packCh(g2), a0t, b0t, Icm, 2, P + "cm.rmsnorm");
                if (pf) refresh(u2, needTail(2) - 2u + 1u, P + "cm.u2");
                u2.used = u2.ct->GetLevel();
                x8Site(u2, x8DropWkr, "wkr", tick);                                    // S3.7 X8: the shared wk/wr input, dropped once
                auto kks = matvecChunked(u2, Wk, dff, d);
                auto rrs = matvecChunked(u2, Wr, dff, d);
                TrackedCt ffn; bool haveFfn = false;
                for (uint32_t ch = 0; ch < NCHUNK; ch++) {
                    const uint32_t n = std::min(D, dff - ch * D);
                    auto chunkOf2 = [&](const std::vector<double>& c) { std::vector<double> s_(n); for (uint32_t i = 0; i < n; i++) s_[i] = c[ch * D + i]; return packCh(s_); };
                    const std::string sch = P + "cm.hidden.ch" + std::to_string(ch);
                    TrackedCt kk = kks[ch], rr = rrs[ch];
                    refresh(kk, 5, sch + ".kk"); refresh(rr, 4, sch + ".rr");
                    TrackedCt k2{cc->EvalMult(kk.ct, kk.ct), 0}; cc->RescaleInPlace(k2.ct);
                    TrackedCt actT = kk; mulPtL(actT, chunkOf2(q1));
                    TrackedCt t2 = k2; mulPtL(t2, chunkOf2(q2));
                    addAlignedL(actT, t2, sch + ".act12");
                    TrackedCt kkA = kk; alignToL(kkA, k2.ct->GetLevel());
                    TrackedCt k3{cc->EvalMult(k2.ct, kkA.ct), 0}; cc->RescaleInPlace(k3.ct); mulPtL(k3, chunkOf2(q3));
                    addAlignedL(actT, k3, sch + ".act3");
                    x2AddL(actT.ct, chunkOf2(q0));                                         // S3.7 X2: was EvalAdd(ptL)
                    TrackedCt r2c{cc->EvalMult(rr.ct, rr.ct), 0}; cc->RescaleInPlace(r2c.ct);
                    TrackedCt gt = rr; mulPtL(gt, chunkOf2(r1));
                    TrackedCt g2t = r2c; mulPtL(g2t, chunkOf2(r2));
                    addAlignedL(gt, g2t, sch + ".gate");
                    x2AddL(gt.ct, chunkOf2(r0));                                           // S3.7 X2: was EvalAdd(ptL)
                    refresh(actT, 2, sch + ".actT"); refresh(gt, 2, sch + ".gt");
                    const uint32_t uu = std::max(actT.ct->GetLevel(), gt.ct->GetLevel());
                    alignToL(actT, uu); alignToL(gt, uu);
                    TrackedCt hd{cc->EvalMult(actT.ct, gt.ct), 0}; cc->RescaleInPlace(hd.ct);
                    refresh(hd, 2, sch + ".hd");
                    refresh(hd, margin, P + "cm.wv.ch" + std::to_string(ch) + ".margin");
                    std::vector<double> Wb((size_t)d * D, 0.0);
                    for (uint32_t i = 0; i < d; i++) for (uint32_t jj = 0; jj < n; jj++) Wb[(size_t)i * D + jj] = Wv[(size_t)i * dff + ch * D + jj];
                    hd.used = hd.ct->GetLevel();
                    x8Site(hd, x8DropWv, "wv", tick);                                  // S3.7 X8: the wv-class drop (per chunk) + census
                    TrackedCt part = matvec(hd, Wb, d, D, 0);
                    if (!haveFfn) { ffn = part; haveFfn = true; } else addAlignedL(ffn, part, P + "cm.ffnAccum.ch" + std::to_string(ch));
                }
                residualX1(Hs, ffn, P + "cm.residual");
                // ---- tick end: the hand-off (gpu_real_model.cu's tick-end canon under X1) ----
                uPrev = uPre;
                if (!deferred || folded) canonL(scanB, "scan");
                if (!uDropL(uPrev, "u")) canonL(uPrev, "u");
                if (deferred) phase = (phase + 1) % M;
                auto o = dec(Hs);
                bool finite = true; for (uint32_t i = 0; i < d; i++) finite = finite && std::isfinite(o[i]);
                bootsPerTick.push_back(boots - b0);
                if (x2PtCache) x2TickLine("x1-ticks.T" + std::to_string(tick));   // S3.7 X2: the census of this tick
                std::cout << "{\"x1Tick\":" << tick << ",\"boots\":" << (boots - b0) << ",\"canonBoots\":" << (canonBootsL - cb0)
                          << ",\"uDrops\":" << (uDrops - ud0) << ",\"uDropBoots\":" << (uDropBoots - udb0) << ",\"branchOutBoots\":" << (boBoots - bo0)
                          << ",\"folded\":" << (folded ? "true" : "false") << ",\"kept\":" << xs.size()
                          << ",\"scanLevelEnd\":" << scanB.ct->GetLevel() << ",\"uLevelEnd\":" << uPrev.ct->GetLevel()
                          << ",\"scanReadoutLevel\":" << S.ct->GetLevel()
                          << ",\"scanErrVsDecX\":" << scanErr << ",\"shiftMixErrVsDec\":" << smErr
                          << ",\"HsLevelOut\":" << Hs.ct->GetLevel() << ",\"finite\":" << (finite ? "true" : "false") << "}" << std::endl;
                verdict("x1_tick" + std::to_string(tick) + "_finite", finite, 0.0, true);
                verdict("x1_tick" + std::to_string(tick) + "_scan_vs_decx", scanErr < 1e-2, scanErr, false);
                verdict("x1_tick" + std::to_string(tick) + "_shiftmix_vs_dec", smErr < 1e-2, smErr, false);
                if (x8Census) {                                                        // S3.7 X8: this tick's pool builds and drops
                    x8PoolBuildsPerTick.push_back(x8PoolBuilds - x8pb0); x8DropsPerTick.push_back(x8DropsApplied - x8dr0);
                    std::ostringstream pr; for (size_t i = 0; i < x8PoolLru.size(); i++) pr << (i ? "," : "") << x8PoolLru[i];
                    std::cout << "{\"x8Tick\":" << tick << ",\"boots\":" << (boots - b0) << ",\"poolBuilds\":" << (x8PoolBuilds - x8pb0)
                              << ",\"drops\":" << (x8DropsApplied - x8dr0) << ",\"poolResident\":[" << pr.str() << "],\"finite\":" << (finite ? "true" : "false") << "}" << std::endl;
                }
            }
            std::ostringstream bpt; for (size_t i = 0; i < bootsPerTick.size(); i++) bpt << (i ? "," : "") << bootsPerTick[i];
            std::cout << "{\"cell\":\"x1-ticks\",\"ticks\":" << x1Ticks << ",\"schedule\":\"" << cellSchedule << "\",\"newtonDepth2\":" << (cellNd2 ? "true" : "false")
                      << ",\"iters\":" << I << ",\"depth\":" << depth << ",\"F\":" << F << ",\"margin\":" << margin
                      << ",\"lamObserved\":" << lamObserved << ",\"canonLevel\":" << canonLevelL
                      << ",\"canonicalCarry\":" << (x1CanonicalCarry ? "true" : "false")
                      << ",\"deferredScan\":" << x1DeferredScan << ",\"uDropLevel\":" << x1UDropLevel
                      << ",\"branchOutBoot\":" << (x1BranchOutBoot ? "true" : "false") << ",\"branchOutPost\":" << x1BranchOutPost
                      << ",\"bootsCold\":" << bootsCold << ",\"bootsPerTick\":[" << bpt.str() << "],\"bootsTicksTotal\":" << (boots - bootsBeforeCold - bootsCold)
                      << ",\"canonBoots\":" << canonBootsL << ",\"uDrops\":" << uDrops << ",\"uDropBoots\":" << uDropBoots
                      << ",\"folds\":" << folds << ",\"branchOutBoots\":" << boBoots << ",\"branchOutSkipped\":" << boSkipped << "}" << std::endl;
            if (x8Census) {                                                            // S3.7 X8: the cell-end line, means and ranges
                // The stock schedule is chaotic in level placement (S3.8 GATE_SWEEP_20260908: 168-430 boots per tick,
                // non-monotone), so a drop arm is read as a MEAN and a RANGE against the stock arm of the SAME cell;
                // "warm" excludes tick 0 (the cold hand-off tick).
                auto stats = [](const std::vector<long long>& a, size_t from) {
                    long long lo = 0, hi = 0, sum = 0; size_t n = 0;
                    for (size_t i = from; i < a.size(); i++) { if (n == 0) { lo = hi = a[i]; } lo = std::min(lo, a[i]); hi = std::max(hi, a[i]); sum += a[i]; n++; }
                    std::ostringstream o; o << "\"mean\":" << (n ? (double)sum / (double)n : 0.0) << ",\"min\":" << lo << ",\"max\":" << hi << ",\"n\":" << n;
                    return o.str(); };
                std::vector<long long> bpt64(bootsPerTick.begin(), bootsPerTick.end());
                std::ostringstream pbt, dpt;
                for (size_t i = 0; i < x8PoolBuildsPerTick.size(); i++) pbt << (i ? "," : "") << x8PoolBuildsPerTick[i];
                for (size_t i = 0; i < x8DropsPerTick.size(); i++) dpt << (i ? "," : "") << x8DropsPerTick[i];
                std::cout << "{\"cell\":\"x8-ticks\",\"ticks\":" << x1Ticks << ",\"schedule\":\"" << cellSchedule << "\",\"iters\":" << I
                          << ",\"depth\":" << depth << ",\"F\":" << F << ",\"margin\":" << margin << ",\"lamObserved\":" << lamObserved
                          << ",\"dropLevels\":{\"wv\":" << x8DropWv << ",\"wkr\":" << x8DropWkr << ",\"wout\":" << x8DropWout << ",\"win\":" << x8DropWin << "}"
                          << ",\"dropTargets\":{\"wv\":" << (x8DropWv > 0 ? (int)depth - x8DropWv : -1) << ",\"wkr\":" << (x8DropWkr > 0 ? (int)depth - x8DropWkr : -1)
                          << ",\"wout\":" << (x8DropWout > 0 ? (int)depth - x8DropWout : -1) << ",\"win\":" << (x8DropWin > 0 ? (int)depth - x8DropWin : -1) << "}"
                          << ",\"poolLevels\":" << x8PoolLevels
                          << ",\"bootsPerTick\":[" << bpt.str() << "],\"bootsAll\":{" << stats(bpt64, 0) << "},\"bootsWarm\":{" << stats(bpt64, 1) << "}"
                          << ",\"poolBuildsPerTick\":[" << pbt.str() << "],\"poolBuildsAll\":{" << stats(x8PoolBuildsPerTick, 0) << "},\"poolBuildsWarm\":{" << stats(x8PoolBuildsPerTick, 1) << "}"
                          << ",\"dropsPerTick\":[" << dpt.str() << "],\"dropsApplied\":" << x8DropsApplied << ",\"dropsSettled\":" << x8DropsSettled
                          << ",\"dropsSkippedDeeper\":" << x8DropsSkippedDeeper << ",\"dropsAtTarget\":" << x8DropsAtTarget << "}" << std::endl;
            }
        }
    }

    // ==== S3.7 X2 pt cache (2026-09-10): the scalar route vs the plaintext route, word for word ====
    // (T05 §5.3's elem_class_probe at this twin's ring and constants.) For each constant of the circuit
    // -- 1.0 (alignTo), -0.5 and 1.5 (Newton), 1/d (invd), 1e-5 (eps), a0 and b0 (the seed), 0.0 and
    // 0.5 (the twin's q0 / r0) -- (a) on a deg-1 operand: EvalMult(ct, pt) vs EvalMult(ct, c) and the
    // EvalAdd pair, ciphertext elements compared word for word plus level/degree/scale (the rule's
    // bit-identity claim, hard); (b) on a deg-2 operand (the same ciphertext with its rescale pending,
    // which the rule declines): the same pairs, words expected to differ, the DECODED values compared
    // (relRms, R3; bar 1e-11, the T05 class), with the decode's own control (one ciphertext decrypted
    // twice) and the level/degree/scale of both results.
    if (cells.count("pt-scalar")) {
        auto& ctSame = x2CtSame;
        auto absRms = [&](const std::vector<double>& a, const std::vector<double>& b) -> double {
            double num = 0; for (uint32_t i = 0; i < d; i++) num += (a[i] - b[i]) * (a[i] - b[i]); return std::sqrt(num / d); };
        auto rmsOf = [&](const std::vector<double>& a) -> double {
            double num = 0; for (uint32_t i = 0; i < d; i++) num += a[i] * a[i]; return std::sqrt(num / d); };
        auto relRms = [&](const std::vector<double>& a, const std::vector<double>& b) -> double {
            const double r = rmsOf(a); return r > 0 ? absRms(a, b) / r : -1.0; };
        auto meta = [&](const Ciphertext<DCRTPoly>& c) { std::ostringstream o;
            o << "{\"level\":" << c->GetLevel() << ",\"deg\":" << c->GetNoiseScaleDeg() << ",\"sf\":" << std::setprecision(17) << c->GetScalingFactor() << "}"; return o.str(); };
        double ms0 = 0; for (uint32_t i = 0; i < d; i++) ms0 += v[i] * v[i]; ms0 /= d;
        const double a0c = 1.5 / std::sqrt(ms0), b0c = -0.5 / (ms0 * std::sqrt(ms0));
        const std::vector<std::pair<std::string, double>> consts = {
            {"one", 1.0}, {"mhalf", -0.5}, {"oneP5", 1.5}, {"invd", 1.0 / (double)d}, {"eps", 1e-5},
            {"a0", a0c}, {"b0", b0c}, {"q0", 0.0}, {"r0", 0.5}};
        std::vector<double> onesv(SLOTS, 1.0);
        auto base = enc(packCh(v)).ct;                                                    // deg 1, level 0
        auto settle = [&](Ciphertext<DCRTPoly>& c) { if (c->GetNoiseScaleDeg() > 1) cc->GetScheme()->ModReduceInternalInPlace(c, 1); };
        Ciphertext<DCRTPoly> d1 = cc->EvalMult(base, cc->MakeCKKSPackedPlaintext(onesv, 1, base->GetLevel())); settle(d1);   // deg 1 at level 1
        Ciphertext<DCRTPoly> d2 = cc->EvalMult(d1, cc->MakeCKKSPackedPlaintext(onesv, 1, d1->GetLevel()));                   // deg 2 (pending) at level 1
        int deg1Fail = 0; double worstDeg2 = 0; int deg2Fail = 0;
        for (const auto& kv : consts) {
            std::vector<double> cv(SLOTS, kv.second);
            // (a) deg 1
            auto p1m = cc->EvalMult(d1, cc->MakeCKKSPackedPlaintext(cv, 1, d1->GetLevel())); auto s1m = cc->EvalMult(d1, kv.second);
            auto pt1 = cc->MakeCKKSPackedPlaintext(cv, 1, d1->GetLevel());
            auto p1a = cc->EvalAdd(d1, pt1);  auto s1a = cc->EvalAdd(d1, kv.second);
            uint64_t mm1 = 0, ma1 = 0; const bool okm1 = ctSame(p1m, s1m, mm1), oka1 = ctSame(p1a, s1a, ma1);
            // (b) deg 2 (the relaxation the rule declines)
            auto p2m = cc->EvalMult(d2, cc->MakeCKKSPackedPlaintext(cv, 1, d2->GetLevel())); auto s2m = cc->EvalMult(d2, kv.second);
            auto pt2 = cc->MakeCKKSPackedPlaintext(cv, 1, d2->GetLevel());
            auto p2a = cc->EvalAdd(d2, pt2);  auto s2a = cc->EvalAdd(d2, kv.second);
            uint64_t mm2 = 0, ma2 = 0; const bool okm2 = ctSame(p2m, s2m, mm2), oka2 = ctSame(p2a, s2a, ma2);
            TrackedCt tp2m{p2m, 0}, ts2m{s2m, 0}, tp2a{p2a, 0}, ts2a{s2a, 0};
            auto dp2m = dec(tp2m), ds2m = dec(ts2m), dp2a = dec(tp2a), ds2a = dec(ts2a), dp2mAgain = dec(tp2m), dp2aAgain = dec(tp2a);
            // relRms = rms(pt route - scalar route) / rms(pt route) [R3]; the decode control is the same ciphertext
            // decrypted twice (OpenFHE adds fresh decryption noise per call), so a route difference below the
            // control is unmeasurable here -- the verdict accepts relRms < 1e-11 OR abs diff <= 4 x the control.
            const double rm = relRms(dp2m, ds2m), ra = relRms(dp2a, ds2a);
            const double am = absRms(dp2m, ds2m), aa = absRms(dp2a, ds2a), cm = absRms(dp2m, dp2mAgain), ca = absRms(dp2a, dp2aAgain);
            const double ctrl = relRms(dp2m, dp2mAgain);
            if (!okm1 || !oka1) deg1Fail++;
            const bool mulOk = (rm >= 0 && rm < 1e-11) || am <= 4.0 * cm, addOk = (ra >= 0 && ra < 1e-11) || aa <= 4.0 * ca;
            if (kv.second != 0.0) worstDeg2 = std::max(worstDeg2, std::max(rm, ra));   // the zero constant's relRms is noise/noise
            if (!mulOk || !addOk) deg2Fail++;
            std::cout << std::setprecision(17) << "{\"cell\":\"pt-scalar\",\"const\":\"" << kv.first << "\",\"value\":" << kv.second
                      << ",\"deg1\":{\"mulWordsIdentical\":" << (okm1 ? "true" : "false") << ",\"mulMismatch\":" << mm1
                      << ",\"addWordsIdentical\":" << (oka1 ? "true" : "false") << ",\"addMismatch\":" << ma1
                      << ",\"ptMul\":" << meta(p1m) << ",\"scalarMul\":" << meta(s1m) << "}"
                      << ",\"deg2\":{\"mulWordsIdentical\":" << (okm2 ? "true" : "false") << ",\"mulMismatch\":" << mm2
                      << ",\"addWordsIdentical\":" << (oka2 ? "true" : "false") << ",\"addMismatch\":" << ma2
                      << ",\"mulDecodedRelRms\":" << rm << ",\"addDecodedRelRms\":" << ra << ",\"decodeControlRelRms\":" << ctrl
                      << ",\"mulAbsRmsDiff\":" << am << ",\"mulAbsRmsControl\":" << cm << ",\"addAbsRmsDiff\":" << aa << ",\"addAbsRmsControl\":" << ca
                      << ",\"mulOk\":" << (mulOk ? "true" : "false") << ",\"addOk\":" << (addOk ? "true" : "false")
                      << ",\"ptMul\":" << meta(p2m) << ",\"scalarMul\":" << meta(s2m) << ",\"ptAdd\":" << meta(p2a) << ",\"scalarAdd\":" << meta(s2a) << "}}"
                      << std::setprecision(6) << std::endl;   // back to the stream's default precision
        }
        // deg-1 word identity is NOT a verdict any more: it is what the self-check measures per class (the cell prints
        // it per constant; the twin's finding is that it fails for some constants on this build, X2 REPORT §3).
        std::cout << "{\"cell\":\"pt-scalar\",\"deg1ConstantsNotWordIdentical\":" << deg1Fail << ",\"constants\":" << consts.size() << "}" << std::endl;
        verdict("pt_scalar_deg2_within_1e-11_or_decode_noise", deg2Fail == 0, worstDeg2, true);
    }
    if (x2PtCacheVerify) verdict("pt_cache_verify_identical", ptCacheVerifyFail == 0, (double)ptCacheVerifyFail, true);   // S3.7 X2
    if (!x13SidecarPath.empty() && x13SumFile >= 0) verdict("x13_sidecar_sum_matches_file", x13Sum == x13SumFile, (double)x13Sum, true);   // S3.7 X13

    std::cout << "{\"summary\":true,\"hardFail\":" << g_hard << ",\"softFail\":" << g_soft
              << ",\"normEps\":" << normEpsUsed                 // S3.7 V1: eps applied in rmsnorm (0 = --no-norm-eps), additive
              // S3.7 X1 (additive): the lever flags as parsed (all default OFF)
              << ",\"deferredScan\":" << x1DeferredScan << ",\"uDropLevel\":" << x1UDropLevel
              << ",\"branchOutBoot\":" << (x1BranchOutBoot ? "true" : "false") << ",\"x1Ticks\":" << x1Ticks
              // S3.7 X2 (additive): the two lever flags as parsed (both default OFF) and their counters
              << ",\"ptCache\":" << (x2PtCache ? "true" : "false") << ",\"ptScalar\":" << (x2PtScalar ? "true" : "false")
              << ",\"ptCacheVerify\":" << (x2PtCacheVerify ? "true" : "false") << ",\"ptCacheMaxGb\":" << x2PtCacheMaxGb
              << ",\"ptCalls\":" << ptCalls << ",\"ptCacheEntries\":" << ptCacheEntries << ",\"ptCacheHits\":" << ptCacheHits
              << ",\"ptCacheMisses\":" << ptCacheMisses << ",\"ptCacheMissMs\":" << ptCacheMissMs
              << ",\"ptCacheDenseFallback\":" << ptCacheDenseFallback << ",\"ptCacheSkipped\":" << ptCacheSkipped
              << ",\"ptCacheExcluded\":" << ptCacheExcluded << ",\"ptCacheLimbs\":" << ptCacheLimbs
              << ",\"ptCacheVerified\":" << ptCacheVerified << ",\"ptCacheVerifyFail\":" << ptCacheVerifyFail
              << ",\"ptScalarOps\":" << ptScalarOps << ",\"ptScalarMul\":" << ptScalarMul << ",\"ptScalarAdd\":" << ptScalarAdd
              << ",\"ptScalarDeg2Fallback\":" << ptScalarDeg2Fallback << ",\"ptScalarClasses\":" << ptScalarClasses
              << ",\"ptScalarClassesRejected\":" << ptScalarClassesRejected << ",\"ptScalarRejectedOps\":" << ptScalarRejectedOps
              // S3.7 X8 + X13 (additive): the drop levels as parsed (0 = off), the drop counters, the pool LRU capacity, the sidecar and its sum
              << ",\"dropLevels\":{\"wv\":" << x8DropWv << ",\"wkr\":" << x8DropWkr << ",\"wout\":" << x8DropWout << ",\"win\":" << x8DropWin << "}"
              << ",\"dropsApplied\":" << x8DropsApplied << ",\"dropsSettled\":" << x8DropsSettled << ",\"dropsSkippedDeeper\":" << x8DropsSkippedDeeper
              << ",\"dropsAtTarget\":" << x8DropsAtTarget << ",\"poolLevels\":" << x8PoolLevels
              << ",\"rsqrtSidecar\":\"" << x13SidecarPath << "\",\"rsqrtItersSum\":" << x13Sum
              // S3.7 V2 (additive): the process-wide accumulators, same names as the .cu summary
              << ",\"timers\":{\"gpuEvalMs\":" << evalMs << ",\"gpuBootstrapMs\":" << bootMs << ",\"normBootMs\":" << normBootMs
              << ",\"gpuEvalMsExBoot\":" << (evalMs - normBootMs) << ",\"clientEncryptMs\":" << encMs << ",\"decryptMs\":" << decMs
              << ",\"hostPtEncodeMs\":" << hostPtEncodeMs << ",\"hostPtEncodeCount\":" << hostPtEncodeCount
              << ",\"hostPtEncodeStageMs\":" << hostPtEncodeStageMs << ",\"hostPtEncodeStageCount\":" << hostPtEncodeStageCount
              << ",\"layerLoopMs\":" << layerLoopMs << ",\"stageMs\":{\"shiftMix\":" << shiftMixMs << ",\"scan\":" << scanMs
              << ",\"gate\":" << gateMs << ",\"hidden\":" << hiddenMs << ",\"residual\":" << residualMs << "}}"
              << fhe_ssm::runStamp(argc, argv) << "}" << std::endl;
    return g_hard ? 3 : 0;
}
