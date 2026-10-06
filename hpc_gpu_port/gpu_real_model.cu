// gpu_real_model.cu — REAL trained-model forward graph in ciphertext
// (REAL_MODEL_PLAN.md, 2026-07-18). Runs the in-class FHE-native SSM LM
// (lane N, ml-eval/train_fhe_native_ssm.py) end-to-end in CKKS at 128-bit
// classical parameters, and the RWKV-4 boundary lane (--arch rwkv) for the
// out-of-class characterization.
//
// WHAT IS NEW vs gpu_full_layer.cu (which this reuses wholesale for the
// context / TrackedCt level-tracking / boot policy / selftest scaffold):
//   * a BUNDLE LOADER: reads ml-eval/artifacts/bundle_<tag>.bin + .index.txt
//     (flat little-endian float64 blob + "name offset nElems shape..." index)
//     exported by the Python side — real trained weights, not synthetic.
//   * CHANNEL PACKING: d channels padded to the power-of-two period D,
//     replicated S/D times across the ring so all rotations are mod-D and
//     act identically on every replica -> FULL-slot encoding (never sparse
//     bootstrapping). CRITICAL INVARIANT: D == the padded INPUT dimension of
//     every matvec, because the diagonal identity needs rotation modulo
//     `cols`, and a ciphertext rotation is modulo the packing period. See
//     the CORRECTNESS note below.
//   * DENSE MATVEC via the diagonal (Halevi-Shoup) method with baby-step/
//     giant-step: y = sum_k diag_k (.) rot(u,k). 1 level, O(sqrt D) rotations.
//     Rectangular maps (rows > D) are CHUNKED into ceil(rows/D) matvecs.
//
//   * RMSNorm: mean-of-squares via log2(D) rotate-adds (0 levels) + Newton
//     rsqrt from the per-layer calibrated linear seed in the bundle.
//   * LEARNED POLYNOMIAL GATES (exact in FHE — they ARE the model): the
//     time-mix deg-3 gate and the channel-mix deg-3 activation + deg-2 gate.
//   * TOKEN-SHIFT mix, Lemma-2 blocked diagonal scan, chunked head matvec.
//
// CORRECTNESS (validated on CPU BEFORE any GPU time — harness/cpu_real_model.cpp
// runs this exact algebra under OpenFHE on the author's Mac and asserts every
// primitive against a plaintext reference). That validator caught two defects
// in the first draft of THIS file, both fixed here:
//   1. BSGS giant step: the plaintext diagonal must be pre-rotated by -g,
//      since the giant rotation is applied AFTER the multiply
//      (d'_j = rot(diag_{g+j}, -g)). The draft used diag_k directly.
//   2. PERIOD vs MODULUS: the draft packed with D=1024 while using cols=768,
//      so rot(u,k) read padding zeros instead of wrapping mod 768 — every
//      matvec silently wrong, and NO exception was raised (CPU validator:
//      matvec_square err 1.17 -> 1.4e-10 after the fix). Hence the invariant
//      above, and the chunking of rectangular maps.
// Post-fix CPU verdicts: encdec 1.5e-10, rotate 5.8e-10, matvec_square
// 1.4e-10, matvec_rect_chunked 1.9e-10, matvec_dff_to_d 1.8e-10,
// rmsnorm_newton 5.6e-5, poly_gate_deg3 6.1e-11, lemma2_scan 1.2e-10.
//
// The native circuit has NO division, NO exp, NO unbounded gate, NO
// selectivity -> only bootstrap noise + Newton-rsqrt truncation contribute
// error (the RWKV contrast, which explodes: REAL_MODEL_PLAN.md section 0).
//
// AUTHORED WITHOUT A GPU (as the whole kit was). Every new FIDESlib surface
// is --selftest-gated with a fallback flag, and --probe decrypts per layer
// against the bundle's float64 reference logits to localize any divergence.
// The pod runbook (REMOTE_REAL_MODEL_PROMPT.md) drives build/test/debug.
//
// Build: added as a target in CMakeLists.txt (same deps as gpu_full_layer).
// Run:   hpc_gpu_port/run_real_model.sh (selftest -> probe -> full).
//
// DEMO client/server split (blocker 6, 2026-07-30): --keys-dir/--keys-save/
// --keys-load/--allow-secret serialize/restore all key material (public
// fideslib API, available in every build); --enc-in/--ct-in/--ct-out/--dec-out
// move ciphertexts through files so encrypt/decrypt run in a process that
// holds the secret key and the forward pass runs in one that does not.
// Ciphertext modes need a -DFHE_SSM_DEMO_SER build; driver: ml-eval/
// fhe_client.py; runbook + honest caveats: hpc_gpu_port/DEMO_README.md.

#include <cctype>
#include <fideslib.hpp>

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <condition_variable>   // T0.2 (S3.1): EncodeWorker
#include <mutex>                // T0.2 (S3.1): EncodeWorker
#include <exception>            // T0.2 (S3.1): worker exception capture
#ifndef _WIN32
#include <dlfcn.h>              // T0.1 (S3.1): NVML via dlopen (--proc-vram); no link-time dep
#include <unistd.h>             // T0.1 (S3.1): getpid for the NVML process match
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

// ---- DEMO client/server split (blocker 6, 2026-07-30) ----------------------
// Key material (context, public/secret key, relin + rotation keys) needs NO
// extra includes: FIDESlib v2.1.2's PUBLIC api already exposes
// fideslib::Serial::SerializeToFile / DeserializeFromFile for cc/pk/sk (plus a
// ".dev" sidecar carrying rotation_indexes / slots_bootstrap / keyDist) and
// CryptoContextImpl<DCRTPoly>::SerializeEvalMultKey / SerializeEvalAutomorphism-
// Key / DeserializeEvalMultKey / DeserializeEvalAutomorphismKey — confirmed
// against the installed v2.1.2 headers (api/Serialize.hpp, api/CryptoContext.hpp)
// and the library's own examples/serial/src/serial.cpp.
//
// CIPHERTEXT (de)serialization, however, has no public fideslib API. Every
// fideslib ciphertext carries an lbcrypto (OpenFHE 1.5.1) host twin in the
// PUBLIC member ct->cpu (a std::any), and the library's own Decrypt brings
// GPU-resident data home via FIDESlib::CKKS::Ciphertext::store() +
// FIDESlib::CKKS::GetOpenFHECipherText() (api/CryptoContext.cpp::Decrypt).
// We reuse exactly that mechanism — which means including OpenFHE headers and
// two INSTALLED FIDESlib private headers (the install exports src/ headers
// deliberately: "so downstream targets can include implementation details").
// The unmodified CMakeLists.txt does not put the OpenFHE include dirs on this
// target (FIDESlib links OpenFHE PRIVATE, so only its libs propagate), so all
// ciphertext-file code is opt-in behind FHE_SSM_DEMO_SER, enabled by adding
// the OpenFHE include dirs via -DCMAKE_CUDA_FLAGS at configure time — see
// hpc_gpu_port/DEMO_README.md for the exact line. A build WITHOUT the macro
// is byte-identical to the pre-demo harness plus the key-file flags.
#ifdef FHE_SSM_DEMO_SER
#include "openfhe.h"                                  // lbcrypto host types
#include "ciphertext-ser.h"                           // cereal registration:
#include "cryptocontext-ser.h"                        //   ciphertext/context
#include "key/key-ser.h"                              //   keys
#include "scheme/ckksrns/ckksrns-ser.h"               //   CKKS scheme objects
#include "CKKS/Ciphertext.cuh"                        // FIDESlib::CKKS::Ciphertext::store
#include "CKKS/openfhe-interface/RawCiphertext.cuh"   // RawCipherText, GetOpenFHECipherText
#include "CKKS/Context.cuh"                           // T1b: FIDESlib::CKKS::Context (compressed store)
#include "CKKS/Plaintext.cuh"                         // T1b: device Plaintext (public c0)
#include "CKKS/Limb.cuh"                              // T1b: Limb<T>.v device buffers
#include <any>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <variant>
#include <filesystem>          // 2026-09-02: evalrot.part*.bin discovery (--keys-load)
#include "periodic_encode.hpp"   // S3.7 V5 (2026-09-10): --periodic-encode, the N/REP-point store-entry encode
#include "seeded_keys.hpp"       // S3.7 V4 (2026-09-10): --keys-load of evalrot.partNNN.seeded.bin (seed-expanded a-halves)
#include <sys/statvfs.h>        // 2026-09-03: --store-file free-space check

// ---- T1b (S3.1, 2026-08-28, the G1-chosen route): device-side limb expand.
// Identical kernel to gpu_delivery_bench.cu's, minus the aux path: measured
// there (25/25 and 300-diag cells), constant polys allocate NO Shoup aux and
// multPt never reads one — bit-identity proved it, so the store carries v
// only (16 KiB per limb, 64× at 2^17).
__global__ void fhe_ssm_expand_limbs_v(uint64_t* const* dstV,
                                       const uint64_t* __restrict__ compV,
                                       int nSlots, int nLimbs, uint32_t N, uint32_t logRep) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int slot = blockIdx.z;                    // BS diagonals per batch
    const int l = blockIdx.y;
    if (slot >= nSlots || l >= nLimbs || i >= N) return;
    const size_t cw = (size_t)(N >> logRep);
    dstV[(size_t)slot * nLimbs + l][i] =
        compV[((size_t)slot * nLimbs + l) * cw + (i >> logRep)];
}
// 2026-09-04 (pod O-2086900, 4x RTX PRO 6000, PCIe NODE links): peer_pool_probe
// returned "wrong-data" WITH cudaMemPoolSetAccess granted -- a kernel on
// devices[0] writing a peer card's pool memory does not land the bytes on this
// box. PER-DEVICE expand instead: every device holds its own copy of the
// pointer table and of the compressed staging and expands ONLY the limbs it
// owns (limbDev[l] == myDev). No kernel ever writes peer memory, so the store
// path needs no peer access at all. Single-device builds keep the kernel above.
__global__ void fhe_ssm_expand_limbs_dev(uint64_t* const* dstV,
                                         const uint64_t* __restrict__ compV,
                                         int nSlots, int nLimbs, uint32_t N, uint32_t logRep,
                                         const int* __restrict__ limbDev, int myDev) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int slot = blockIdx.z;
    const int l = blockIdx.y;
    if (slot >= nSlots || l >= nLimbs || i >= N) return;
    if (limbDev[l] != myDev) return;
    const size_t cw = (size_t)(N >> logRep);
    dstV[(size_t)slot * nLimbs + l][i] =
        compV[((size_t)slot * nLimbs + l) * cw + (i >> logRep)];
}
#endif

// Provenance stamp appended to the summary record (harness file + its sha256,
// git commit, dirty flag, build time, exact argv). See run_stamp.h.
#include <filesystem>          // BOTH builds (2026-09-03): --keys-load's evalrot.part*
                                // discovery is outside the DEMO_SER block above, and the
                                // plain LTO build died on it (pod build_fideslib.log:945)
#include "run_stamp.h"

using namespace fideslib;
using Clock = std::chrono::steady_clock;
static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
#define CUDA_CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    std::cerr << "{\"cudaError\":\"" << cudaGetErrorString(e_) << "\",\"at\":\"" \
              << __FILE__ << ":" << __LINE__ << "\"}\n"; std::exit(1);} } while(0)

static size_t g_total_vram = 0, g_min_free = SIZE_MAX;
// ---- T0.1 (S3.1, 2026-08-25): per-PROCESS VRAM peak via NVML (--proc-vram).
// peakVramGB() below is CARD-WIDE (cudaMemGetInfo free-memory drawdown): the
// clock cell measured it growing exactly +1.0000 GB per successive run in a
// session because it charges this run for every resident neighbour
// (CLOCK_SECURE_20260823.md §4). NVML's per-process usedGpuMemory closes that
// attribution gap. dlopen'd at runtime so the build needs no NVML dev files;
// struct layout is the v2/v3 nvmlProcessInfo_t (pid, usedGpuMemory,
// gpuInstanceId, computeInstanceId) — v1-only drivers report -1 (unavailable)
// rather than risk a misread field. Everything is gated on --proc-vram;
// without the flag not one NVML symbol is touched.
static bool   g_procVramOn = false;
static double g_procVramPeakGB = -1.0;   // -1 = off or unavailable
#ifndef _WIN32
namespace fhe_ssm_nvml {
struct ProcInfoV2 { unsigned pid; unsigned long long usedGpuMemory;
                    unsigned gpuInstanceId; unsigned computeInstanceId; };
typedef int (*fnInit)();
typedef int (*fnDevGet)(unsigned, void**);
typedef int (*fnProcs)(void*, unsigned*, ProcInfoV2*);
static fnProcs g_procs = nullptr; static void* g_dev = nullptr;
static bool init(int device) {
    void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) return false;
    auto ini = (fnInit)dlsym(h, "nvmlInit_v2");
    auto dev = (fnDevGet)dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
    g_procs = (fnProcs)dlsym(h, "nvmlDeviceGetComputeRunningProcesses_v3");
    if (!g_procs) g_procs = (fnProcs)dlsym(h, "nvmlDeviceGetComputeRunningProcesses_v2");
    if (!ini || !dev || !g_procs) { g_procs = nullptr; return false; }
    if (ini() != 0) { g_procs = nullptr; return false; }
    if (dev((unsigned)device, &g_dev) != 0) { g_procs = nullptr; return false; }
    return true;
}
static void sample() {
    if (!g_procs || !g_dev) return;
    ProcInfoV2 info[64]; unsigned cnt = 64;
    if (g_procs(g_dev, &cnt, info) != 0) return;
    const unsigned myPid = (unsigned)getpid();
    for (unsigned i = 0; i < cnt && i < 64; i++)
        if (info[i].pid == myPid) {
            double gb = (double)info[i].usedGpuMemory / 1073741824.0;
            if (gb > g_procVramPeakGB) g_procVramPeakGB = gb;
        }
}
}  // namespace fhe_ssm_nvml
#endif
static void vramSample() {
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) == cudaSuccess) { g_total_vram = t; if (f < g_min_free) g_min_free = f; }
#ifndef _WIN32
    if (g_procVramOn) fhe_ssm_nvml::sample();
#endif
}
static double peakVramGB() { return (g_total_vram - g_min_free) / 1073741824.0; }

// ---- M1 (S3.7, 2026-09-14): pool-attribute decomposition of the card-wide VRAM reading.
// cudaMemGetInfo counts every byte the stream-ordered pool has RESERVED from the
// driver, so FIDESlib's per-keyswitch scratch (LimbPartition.cu:285-293, the
// special-limb extension), the harness's own transient ciphertexts and the
// pool's slab rounding are all INSIDE usedGB and indistinguishable there
// (results/dense-demo-s31/MGPU_FIT_MODEL_20260903.md s2 lists the three as
// "not modelled"). The device default pool's attributes split them:
//   poolReservedGB      bytes held from the driver  (= what usedGB sees, minus non-pool allocations)
//   poolUsedGB          bytes handed out to live allocations
//   poolSlackGB         reserved - used: slab rounding + freed-but-retained memory
//   poolTransientGB     usedHigh - usedCurrent since the previous milestone: the
//                       high-water mark of allocations that came and went between
//                       two vramTrace lines (library scratch + harness temporaries)
// The two high watermarks are RESET after every line (CUDA: setting the *High
// attribute to 0 resets it to the current value), so each line describes the
// window since the previous milestone; the run-wide maxima are kept here and
// printed once in the summary (poolHighRun). Valid only while FIDESlib allocates
// from the device default pool -- the assumption the cudaMemPoolSetAccess grant
// at poolBuild already makes; a pool swapped in with cudaDeviceSetMemPool would
// read ~0 reserved against tens of GB used, which the one-time warn below flags.
// Additive fields on the existing vramTrace line; nothing else changes.
static std::map<int, std::pair<double,double>> g_poolHighRun;   // dev -> (usedHighGB, reservedHighGB), run-wide
static bool g_poolWarned = false;

// ---- T0.2 (S3.1, 2026-08-25): bounded encode worker for --pipeline. --------
// Replaces the per-giant-step `std::thread` (T16, POD_OPERATOR_MANUAL.md §2).
// The convicted failure mode needs no data race at all: any throw out of the
// main thread's apply loop (EvalMult, mkPt's terminate-guard sibling paths)
// unwound past a JOINABLE std::thread, whose destructor calls std::terminate
// — exactly the recorded "terminate called without an active exception"
// signature. This worker is (a) one persistent thread per matvecBatchImpl
// call, not one per giant step (4,224/pass -> ~hundreds), (b) exception-safe:
// its destructor always drains and joins, so unwinding can never destroy a
// joinable thread, and (c) it CAPTURES a worker-side exception and rethrows
// it on wait() in the main thread, so an encode failure surfaces as the run's
// failure instead of a std::terminate. Single job slot by design — the
// pipeline is one giant step deep, matching the double-buffer it feeds.
// Declared AFTER the buffers it writes at the use site, so on unwind its
// join runs BEFORE those buffers are destroyed.
struct EncodeWorker {
    std::thread th;
    std::mutex m;
    std::condition_variable cv;
    std::function<void()> job;
    std::exception_ptr err;
    bool stop = false, busy = false, started = false;
    void start() {
        started = true;
        th = std::thread([this] {
            std::unique_lock<std::mutex> lk(m);
            for (;;) {
                cv.wait(lk, [&] { return stop || (bool)job; });
                if (!job) return;                      // stop && no pending job
                auto j = std::move(job); job = nullptr;
                lk.unlock();
                try { j(); } catch (...) {
                    std::lock_guard<std::mutex> g(m);
                    err = std::current_exception();
                }
                lk.lock();
                busy = false;
                cv.notify_all();
                if (stop) return;
            }
        });
    }
    void submit(std::function<void()> j) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !busy; });
        job = std::move(j); busy = true;
        cv.notify_all();
    }
    void wait() {                                      // drain, then rethrow worker errors
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !busy; });
        if (err) { auto e = err; err = nullptr; lk.unlock(); std::rethrow_exception(e); }
    }
    ~EncodeWorker() {
        if (!started) return;
        { std::lock_guard<std::mutex> g(m); stop = true; }
        cv.notify_all();
        if (th.joinable()) th.join();
        // a captured-but-unretrieved error dies here silently only on the
        // unwind path, where the main thread already has its own exception.
    }
};

// ------------------------------------------------------------- bundle loader
// Flat float64 blob + text index "<name> <offset_bytes> <nElems> <dims...>".
struct Bundle {
    std::vector<double> blob;                        // whole .bin, as doubles
    std::map<std::string, std::pair<size_t, std::vector<int>>> idx;  // name -> (elemOffset, shape)

    void load(const std::string& binPath, const std::string& idxPath) {
        std::ifstream bf(binPath, std::ios::binary | std::ios::ate);
        if (!bf) { std::cerr << "{\"fatal\":\"no bundle bin " << binPath << "\"}\n"; std::exit(2); }
        size_t nbytes = bf.tellg(); bf.seekg(0);
        blob.resize(nbytes / sizeof(double));
        bf.read(reinterpret_cast<char*>(blob.data()), nbytes);
        std::ifstream xf(idxPath);
        if (!xf) { std::cerr << "{\"fatal\":\"no bundle index " << idxPath << "\"}\n"; std::exit(2); }
        std::string line;
        while (std::getline(xf, line)) {
            std::istringstream ss(line);
            std::string name; size_t offBytes, nEl;
            if (!(ss >> name >> offBytes >> nEl)) continue;
            std::vector<int> shape; int dscratch;
            while (ss >> dscratch) shape.push_back(dscratch);
            idx[name] = {offBytes / sizeof(double), shape};
        }
    }
    bool has(const std::string& n) const { return idx.count(n) > 0; }
    const double* ptr(const std::string& n) const {
        auto it = idx.find(n);
        if (it == idx.end()) { std::cerr << "{\"fatal\":\"missing tensor " << n << "\"}\n"; std::exit(2); }
        return blob.data() + it->second.first;
    }
    std::vector<int> shape(const std::string& n) const { return idx.at(n).second; }
    // row-major (out,in) matrix as a flat vector
    std::vector<double> mat(const std::string& n, int& rows, int& cols) const {
        auto s = shape(n); rows = s[0]; cols = s.size() > 1 ? s[1] : 1;
        const double* p = ptr(n);
        return std::vector<double>(p, p + (size_t)rows * cols);
    }
    std::vector<double> vec(const std::string& n) const {
        auto s = shape(n); size_t k = 1; for (int d : s) k *= (size_t)d;
        const double* p = ptr(n); return std::vector<double>(p, p + k);
    }
};

// ---- A-P0b (2026-08-22): quality sidecar reader. Looks beside the bundle for
// bundle_<tag>.provenance.json (AGN export), <tag>_export.json / export_<tag>.json,
// <tag>_eval.json; extracts native_test_ppl and weights_sha256 by plain string
// search (no JSON dependency). Emits a JSON object; nulls when absent. Quality
// is the model's WT103 test perplexity (R4) -- it is NOT measured by this run.
static std::string fhe_ssm_quality_sidecar(const std::string& dir, const std::string& tag) {
    auto slurp = [](const std::string& p) -> std::string {
        std::ifstream f(p); if (!f) return std::string();
        std::stringstream ss; ss << f.rdbuf(); return ss.str(); };
    auto findNum = [](const std::string& text, const std::string& key) -> std::string {
        size_t k = text.find("\"" + key + "\""); if (k == std::string::npos) return "null";
        size_t c = text.find(':', k); if (c == std::string::npos) return "null";
        size_t b = c + 1; while (b < text.size() && (text[b] == ' ' || text[b] == '\t')) b++;
        size_t e = b; while (e < text.size() && (std::isdigit((unsigned char)text[e]) || text[e] == '.' || text[e] == '-' || text[e] == '+' || text[e] == 'e' || text[e] == 'E')) e++;
        return (e > b) ? text.substr(b, e - b) : std::string("null"); };
    auto findStr = [](const std::string& text, const std::string& key) -> std::string {
        size_t k = text.find("\"" + key + "\""); if (k == std::string::npos) return "null";
        size_t q1 = text.find('"', text.find(':', k) + 1); if (q1 == std::string::npos) return "null";
        size_t q2 = text.find('"', q1 + 1); if (q2 == std::string::npos) return "null";
        return "\"" + text.substr(q1 + 1, q2 - q1 - 1) + "\""; };
    const std::vector<std::string> cands = {dir + "/bundle_" + tag + ".provenance.json", dir + "/provenance_" + tag + ".json",
                                            dir + "/" + tag + "_export.json", dir + "/export_" + tag + ".json", dir + "/" + tag + "_eval.json"};
    std::string ppl = "null", sha = "null", src = "null";
    for (const auto& c : cands) {
        const std::string t = slurp(c); if (t.empty()) continue;
        if (ppl == "null") { std::string v = findNum(t, "native_test_ppl"); if (v != "null") { ppl = v; src = "\"" + c + "\""; } }
        if (sha == "null") { std::string v = findStr(t, "weights_sha256"); if (v != "null") sha = v; }
    }
    return "{\"nativeTestPpl\":" + ppl + ",\"weightsSha256\":" + sha + ",\"source\":" + src + "}";
}

int main(int argc, char** argv) {
    // total-process wall clock, for the summary's self-audit fields
    // (totalWallMs / unaccountedMs / wallMsPerToken).
    auto tRun0 = Clock::now();
    // ---- args -------------------------------------------------------------
    std::string arch = "native";                 // native | rwkv
    std::string tag = "native";                  // bundle_<tag>.bin
    std::string bundleDir = "../ml-eval/artifacts";
    int device = 0;
    std::vector<int> devices;   // --devices 0,1 : empty => {device} (single-GPU,
                                // unchanged behaviour). See the --devices parse
                                // site for why sharding makes this worth having.
    bool secure = false, selftest = false, probe = false, bootLadder = false;
    bool laneTest = false;                         // --lane-test: STAGE 2 rotation-convention probe
    double diagEps = 0.0;                          // --diag-eps: skip diagonals with max|w| <= eps
    int encThreads = 0;                            // --enc-threads N: parallel diagonal encode (0 = all cores)
    bool encodeBench = false;                      // --encode-bench: encode/transfer/multiply cost model
    // ---- T0.1 (S3.1, 2026-08-25): synchronised phase timers. ---------------
    // Every timer in this file is an ASYNC-DISPATCH wall clock: there is not
    // one cudaDeviceSynchronize in the pre-S3.1 harness, so gpuBootstrapMs is
    // a lower bound (S3.0_DECISION_PACKAGE.md B.2-6) and queued GPU work lands
    // in whichever later timer happens to synchronise first. --sync-timers
    // drains the device at phase boundaries and accumulates a per-phase split
    // (baby rotations / ct x pt apply / giant rotations+adds / rescale / boot)
    // emitted in the summary's syncSplit object. It is a DIFFERENT INSTRUMENT:
    // numbers taken with it on are comparable to each other, never to the
    // historical corpus. Off by default; the unflagged path never syncs.
    bool syncTimers = false;                       // --sync-timers
    // ---- S3.7 V2 (2026-09-05): layer-loop stage timers (S3.6 N1, COMPLEXITY_MAP §1,
    // T05 §2). The layer loop (:4410-:4916 at HEAD 1d41b90) had no Clock::now() of
    // its own, so shift-mix, scan, gates, hidden, residual adds, alignTo and the
    // ~1,100-1,700 host mkPt encodes per tick all landed in unaccountedMs (cell A:
    // true untimed 48.6 s = 21.8 % of the wall). --stage-timers (DEFAULT ON) adds
    // layerLoopMs and five per-stage accumulators that bill ONLY the part of a stage
    // that no existing timer covers, and under --sync-timers drains the device at
    // each stage boundary (as syncDev does elsewhere) so those stage times are
    // honest. --no-stage-timers restores HEAD's exact behaviour (no extra clocks,
    // no extra drains; the new fields print 0 and stageTimersOn:false) for an A/B.
    // The normBootMs correction (see the accumulators) is pure accounting and is
    // not behind a flag: no existing field changes meaning.
    bool stageTimers = true;                       // --stage-timers / --no-stage-timers
    // --proc-vram: per-process VRAM peak via NVML (see the block at the top).
    // peakVramGB stays card-wide and unchanged; this adds procPeakVramGB.
    bool procVram = false;                         // --proc-vram
    // ---- T1b (S3.1, 2026-08-28): --compressed-store — the G1-chosen route.
    // Encode each live (op, layer, chunk, level, diagonal) ONCE (first use),
    // keep only the compressed eval-domain limbs (N/REP words each — exact,
    // LIMB_VERDICT/d5map/ceriumDevice all bit-identical), and deliver every
    // subsequent use as ONE small H2D + a device expand into a per-level
    // scratch-plaintext pool, consumed by the stock multPt. Keyed by VALUE
    // IDENTITY (layer/op/chunk/rowOff/level/diag) set at the call sites —
    // never by a data pointer (HARNESS_BUGS B6). Every compress VERIFIES the
    // i/REP map on the raw limbs first; a violating diagonal (e.g. any
    // --interleave run — interleaved plaintexts are coefficient-dense and do
    // NOT compress) is delivered by the normal path and counted, so wrong
    // answers are unrepresentable. Requires the DEMO_SER build (internals).
    // Timing lands in storeFetchMs; encPtMs keeps counting miss-encodes.
    bool compressedStore = false;                  // --compressed-store
    // ---- T1c (S3.1, 2026-08-28): --lanes-block — multi-lane BLOCK layout.
    // Different tokens in different replicas of the block layout. The matvec
    // masks FOLD INTO THE DIAGONALS (dual-half BSGS: lo-half × baby(u),
    // hi-half × baby(u_wrap = rot_{−Dpad}(u)); mask condition row+g+j ≥ Dpad;
    // +0 levels — T1C_DESIGN_20260828.md §1). Requires --compressed-store
    // (the halves are ordinary store entries) and the DEMO_SER build.
    bool lanesBlock = false;                       // --lanes-block
    bool laneTestBlock = false;                    // --lane-test-block: T1c acceptance
                                                   // probe — per-lane equality of the
                                                   // multi-lane block path vs broadcast
                                                   // reference runs of the same lane
                                                   // (same circuit, same Newton error →
                                                   // tight bar). Implies --lanes-block
                                                   // + --compressed-store.
    // ---- T1c FULL-PASS driver (S3.1, 2026-08-29): --lanes-full NL runs NL
    // distinct token sequences (a LANES bundle: inputs/logits_ref lane-major,
    // `lanes` tensor [NL,Tlane]) through the ENTIRE pass in block layout —
    // Newton loops, bootstraps, shift-mix/scan carry included, which the
    // zero-bootstrap --lane-test-block probe never exercised. Replicas are
    // CYCLIC-FILLED (lane r%NL) so no slot ever carries rsqrt(eps)-class
    // garbage through a bootstrap, and replica NL+r must echo replica r
    // (reported). Per-lane logits are checked against logits_ref row
    // r*Tlane+tok and dumped lane-minor for the gate. Implies --lanes-block
    // + --compressed-store. T1C_FULLPASS_DRIVER_20260829.md.
    int lanesFull = 0;                             // --lanes-full NL (0 = off)
    int bridgeTest = -1;   // --bridge-test LANE: GPU state-layout bridge probe.
                           // Converts a BROADCAST-BLOCK state (channel c at
                           // every slot r*Dpad+c) into INTERLEAVE placement
                           // (channel c at slot c*REP + LANE) and measures it
                           // against the plaintext permutation. This is the
                           // conversion that lets a packed/interleave pass
                           // START from a block-layout stateful carry, i.e. the
                           // only route by which speculative decoding can reach
                           // a block-layout demo (D14). CPU algebra proven
                           // exact in spec_decode/bridge_probe.cpp
                           // (relErrRms 3.386e-13); this is the on-device,
                           // real-ring, real-key version.
    // --bundle-row-offset K: read inputs/logits_ref starting at row K — the
    // broadcast reference runs of lane r use K = r*Tlane on the SAME lanes
    // bundle (same binary, same store path). Forbidden with --lanes-full.
    int bundleRowOffset = 0;
    // ---- multi-lane SERVE CLIENT (S3.1, 2026-09-01, decision): ONE
    // client, ONE key, N conversations. --client-lanes N changes only the
    // CLIENT crypto steps: --enc-in reads N*T*d rows LANE-MAJOR and packs
    // slot r*Dpad+c = lane (r mod N) channel c per token (cyclic fill — same
    // rationale as --lanes-full: no rsqrt(eps) garbage through bootstraps);
    // --dec-out reads "lanes" from the ct meta and emits N logit rows per
    // token (tok-major, lane-minor). The SERVER stays lane-blind: it just
    // runs the --lanes-block circuit. 64 lanes ≠ 64 clients — one secret
    // key reads every lane (claim rules §8).
    int clientLanes = 1;                           // --client-lanes N
    bool traceBoots = false;                       // --trace-boots (A-P0b / F-lvl, 2026-08-22): one JSON line per
                                                   // EvalBootstrap with the level before/after, so the no-op trap
                                                   //  and the 19-vs-21 question are readable from records
    int bandWidth = -1;                            // --band N: keep only cyclic diagonals |k| <= N
    bool ringMix = false;                          // --ring-mix: ring-native mixing COST PROXY
    int bootFloor = -1;                            // --boot-floor N: refresh reserve (default 4)
    // --pipeline overlaps the host encode with the GPU apply and is worth 1.78x
    // (measured at P=16: 9,698 -> 5,429/5,496 ms/token). It is OPT-IN, not
    // default, because it is not proven thread-safe: the worker thread calls
    // cc->MakeCKKSPackedPlaintext while the main thread issues CUDA work, and
    // FIDESlib loads plaintexts to the device, so that call may touch GPU state.
    // One run out of ~45 aborted with "terminate called without an active
    // exception" (the signature of a joinable std::thread destructor). That run
    // also carried an unrelated logic change which is a no-op for dense weights,
    // so the abort most likely EXPOSED a latent race rather than being caused by
    // it. Until that is resolved, correctness-critical runs should leave this off.
    bool pipeline = false;                         // --pipeline: overlap encode with GPU apply
    bool staticNorm = false;                       // --static-norm: rung-2 norm-free COST PROXY
    // DEFAULT = the SERIAL carry, because the two-level scan REGRESSED (2026-07-31).
    // Measured: ring 2^15 / 8 blocks / 128 tokens went from passing 5/5 (548 boots,
    // 1179-1468 ms/token) to FAILING with 808 boots; ring 2^16 / 4 blocks stayed
    // failing (448 vs 441 boots). The algebra is exact -- see
    // twolevel_sim.py (a scratch simulator, not included), err 8.9e-16 -- and single-block controls are
    // clean, so this is a COST problem, not a correctness one: level uniformity is
    // bought by making EVERY block do the worst-case work. At B=8 the two-level
    // form issues B*(1+log2(B)+1) = 40 level-consuming multiplies against the
    // serial chain's (B-1)*2 = 14, so total ops (and boots) rise ~3x and swamp the
    // per-block depth saving. Opt in with --two-level-scan if you want to work on
    // it; the cheap version to try first is equalizing only the SHIFT-MIX (one
    // dummy level for block 0), which is 2 lines instead of 40.
    bool serialCarry = true;                       // --two-level-scan flips this off
    int scaleBits = 59;                            // --scale-bits: CKKS scaling mod size
    bool finalRefresh = false;                     // --final-refresh enables (DEFAULT OFF:
                                                   // measured 2026-07-30, booting the post-
                                                   // norm output made even the passing
                                                   // structural Delta-59 config fail decode
                                                   // on EVERY block at level 19 — the boot
                                                   // corrupts the final state, suspected
                                                   // amplitude-range violation). Kept as a
                                                   // flag for the negative-result record.
                                                   // Original idea: boot the
                                                   // post-rmsnorm output ciphertext before
                                                   // decode: at reduced --scale-bits the
                                                   // DEEPEST level can trip OpenFHE's decode
                                                   // guard (measured: full circuit healthy at
                                                   // 5e-4 wvCheck, final decode rejected at
                                                   // level 34, Delta=2^35). A boot costs
                                                   // ~0.7%-of-pass noise ~1e-6 and restores
                                                   // maximal remaining modulus for decode.
    bool equalizeBlocks = false;                   // --equalize-blocks: boot ALL blocks after
                                                   // the scan so they sit at one level BY
                                                   // CONSTRUCTION and matvecBatch's alignTo
                                                   // never chain-rescales a shallow block.
                                                   // Motivated by the b4 probe: at 4 blocks,
                                                   // blocks 0-2 decode to ALL-NaN right after
                                                   // the first channel mix while the deepest
                                                   // block 3 stays healthy -- scale corruption
                                                   // from the alignTo chain, not noise.
    int numDigits = 0;                             // --num-digits: SetNumLargeDigits (0=default).
                                                   // More digits => smaller key-switching
                                                   // auxiliary modulus P => lower logQP =>
                                                   // --secure can land on a SMALLER ring
                                                   // (at more key-switch NTTs per op).
    std::string dumpLogits;                        // --dump-logits FILE: write the decrypted
                                                   // per-token logits (float64 rows) so
                                                   // ppl/top-1/KL vs the plaintext reference
                                                   // can be computed offline.
    bool cacheDiags = false;                       // --cache-diags: keep encoded diagonals GPU-resident
    int passes = 1;                                // --passes N: repeat the forward (pass 2+ = steady state)
    int maxBatch = 0;                              // --max-batch N: cap cts per batched matvec (see matvecBatch)
    uint32_t T = 64, K = 8;                       // tokens, block size
    uint32_t Dpad = 1024;                          // channel packing period (pow2)
    uint32_t logRing = 16;                         // structural ring (2^logRing).
                                                   // Lower it to fit small cards:
                                                   // rotation keys dominate VRAM
                                                   // (~120MB/key at 2^16).
    int selfD = 0, selfDff = 0;                    // --selftest without a bundle
    double bootAmp = 1.5;                          // --amp: boot-ladder message magnitude
    int packTokens = 0;                            // --pack-tokens N: STAGE 2.
                                                   // Put N tokens in ONE ciphertext at
                                                   // interleaved lanes s = c*REP + t, so VRAM
                                                   // is O(1) in sequence length up to REP and
                                                   // encoding amortizes N-fold. Only TWO ops in
                                                   // the whole forward are cross-token
                                                   // (shift-mix and the scan); everything else
                                                   // is per-lane and already correct because
                                                   // channel rotations are by k*REP.
    bool stageProbe = false;                       // --stage-probe: per-LAYER error
    bool blockProbe = false;                       // --block-probe: per-BLOCK per-layer
                                                   // decrypted magnitude + scale factor +
                                                   // level, for the multi-block failure
                                                   // isolation (magnitude-divergence vs
                                                   // level-spread hypothesis). Decrypts
                                                   // B ciphertexts at 3 sites per layer;
                                                   // probe-only cost, off by default.
    bool interleave = false;                       // --interleave: INTERLEAVED slot layout
                                                   // s = c*REP + t (channel-major, token-minor)
                                                   // instead of block replication
                                                   // s = r*Dpad + c. A channel rotation by k
                                                   // becomes a RING rotation by k*REP, and the
                                                   // channel wraparound at Dpad coincides
                                                   // exactly with the ring wraparound at
                                                   // Dpad*REP == SLOTS -- provably bleed-free,
                                                   // unlike block-major (s = t*Dpad + c), where
                                                   // token t would pull token t+1 channels.
                                                   // STAGE 1: one token broadcast to all REP
                                                   // token-slots => mathematically IDENTICAL to
                                                   // the replicated path, so any deviation in
                                                   // the selftest is a layout bug, not physics.
    bool matvecProbe = false;                      // --matvec-probe: isolate the
                                                   // selftest-vs-graph precision cliff.
    int matvecMargin = -1;                         // --matvec-margin N: limbs to reserve
                                                   // before the big wv/wk/wr matvecs.
                                                   // EVIDENCE (2026-07-29): --rsqrt-iters 1
                                                   // (FEWER bootstraps) made wv_part
                                                   // undecryptable again, while --extra-depth
                                                   // (MORE headroom) fixed it, and BSGS
                                                   // chunking changed nothing. So the error is
                                                   // driven by REMAINING LIMBS at the matvec,
                                                   // not by noise accumulated before it.
    bool bsgsRescaleInner = false;                 // --bsgs-chunk: rescale each
                                                   // giant-step partial sum instead of
                                                   // accumulating all 1024 ct x pt terms
                                                   // at scale Delta^2 before one rescale.
    // ---- S3.3 SCHEDULE LEVERS (2026-09-03). ALL DEFAULT OFF: with none of them
    // set this file's op sequence, level trace, store keys and output are the
    // b27159c stock path (the S2.8 rule). Design, replay predictions and the
    // precision argument: results/theory/SCHEDULE_DESIGN_20260903.md; the
    // replay is results/theory/tick_level_trace.py (same flag names).
    std::string schedule = "stock";   // --schedule parent-first: bootstrap the RESIDUAL Hs before
                                      // each norm when its remaining levels cannot carry the
                                      // norm's y chain (P1, replaces refresh(Hs,3) at the tm/cm
                                      // norm entries and adds one at the output norm), lift the
                                      // norm output at the tail (P2, replaces refresh(y,4) in
                                      // rmsnorm) and guard the shared FFN input u2 once (P4,
                                      // before matvecBatchChunked(wk)). Placement of the SAME
                                      // refresh rule; no arithmetic changes. Replay [trace]:
                                      // 332 -> 111 boots/tick at the demo geometry.
    std::string pfEntry = "loop";     // --pf-entry loop|branch|both|always: P1's need formula
                                      // (SCHEDULE_DESIGN section 3.1); only read under parent-first.
    bool shareBabies = false;         // --share-babies: compute the BSGS baby set (and the T1c
                                      // wrap set) of an input ONCE and reuse it across the 2K
                                      // wk/wr chunk calls of a layer (matvecBatchImpl baby loops
                                      // + the wk/wr block). Bit-identical output; 32,564 ->
                                      // 21,980 rotations/tick [trace].
    bool newtonDepth2 = false;        // --newton-depth2 (Tier A-2): evaluate the SAME Newton step
                                      // y(1.5 - 0.5 ms y^2) as 1.5y + ((-0.5 ms) y)(y y), depth 2
                                      // per iteration instead of 4 (rmsnorm loop). Re-association
                                      // only -- same polynomial, same calibrated a0/b0/iters --
                                      // but different rounding: FIDELITY-GATED (I4).
    int poolLevels = 2;               // --pool-levels N: compressed-store pool LRU capacity
                                      // (compPoolGet; stock 2). Replay: 89.5 pool (re)builds per
                                      // tick at 2, 10.3 at 4. ~1.34 GB VRAM per extra level at
                                      // 2^17/20 limbs (64 device plaintexts). Not in the store
                                      // stamp (no level-trace effect). poolBuilds/poolBuildMs
                                      // are emitted in the summary so the pod can price it.
    std::string blockType = "auto";   // --block auto|sequential|parallel (S3.3 Tier B, TIERB_SPEC_20260903.md):
                                      // the layer's block form. `auto` reads meta[6] of the bundle (absent or 0 =
                                      // the sequential block every existing bundle has; 1 = parallel, written by
                                      // train_fhe_native_ssm.py export --block parallel). An explicit value that
                                      // disagrees with the bundle is REFUSED. The parallel circuit: one norm feeds
                                      // both branches, Hs += attn + ffn at the end of the layer; no new op type.
    int rsqrtItersOverride = -1;                   // --rsqrt-iters N: override the
                                                   // bundle Newton iteration count.
    double msNormK = 0.0;                          // --ms-norm K: per-site normalization
                                                   // of the mean-square statistic. See the
                                                   // block comment in rmsnorm().
    bool normEpsOn = true;                         // --no-norm-eps clears it. S3.7 V1 (2026-09-05,
                                                   // S3.6 T18 §3): the model normalises by
                                                   // 1/sqrt(mean(x^2) + 1e-5) (train_fhe_native_ssm.py
                                                   // :834, :927-:932) and the seeds (a0, b0) were fitted
                                                   // on mean + eps, but this circuit formed ms = mean
                                                   // only -- rmsnorm()'s `eps` argument was never read.
                                                   // The pod selftest reproduces the no-eps chain to
                                                   // seven digits (0.0310171) and the no-eps plaintext
                                                   // predicts every T = 1 record's 0.03-0.05 fidelity
                                                   // class. DEFAULT ON: rmsnorm() adds the plaintext
                                                   // constant sigma*eps to ms (a plaintext add, zero
                                                   // levels, no store-stamp change). OFF restores the
                                                   // recorded circuit for A/B against the records.
    double normEpsUsed = 0.0;                      // the eps rmsnorm() last applied (0 when off);
                                                   // emitted as the ADDITIVE summary field normEps.
    int extraDepth = 0;                            // --extra-depth: raise multiplicative
                                                   // depth. Bootstrap lands on a level set
                                                   // by the BUDGET, not by depth, so extra
                                                   // depth is extra USABLE levels after every
                                                   // boot -- and more remaining modulus for
                                                   // the deep 1024-term wv matvec, which the
                                                   // source calls a marginal precision
                                                   // threshold at level ~20-23.
    int lbA = 4, lbB = 4;                          // --level-budget: CKKS bootstrap
                                                   // levelBudget {lbA,lbB}. Held at
                                                   // {4,4} in EVERY run of this project
                                                   // so far; it is the one knob that
                                                   // lowers the ABSOLUTE bootstrap error
                                                   // floor, which is what limits the
                                                   // scale-invariant path (scan_out ~0.047
                                                   // cannot be fixed by rescaling).
    // ---- DEMO client/server split flags (blocker 6; see DEMO_README.md) ----
    // Roles are PROCESS-level: the same binary runs as key-minting client
    // (--keys-save), encrypting client (--enc-in), evaluating server
    // (--ct-in/--ct-out, no secret key in the process), and decrypting client
    // (--dec-out, requires --allow-secret). All key files travel through
    // --keys-dir; ciphertexts travel as PREFIX.N files + a PREFIX.meta JSON.
    std::string keysDir;        // --keys-dir DIR: where key material lives
    bool keysSave = false;      // --keys-save: serialize cc/pk/sk/eval keys to DIR, continue
    bool keysLoad = false;      // --keys-load: deserialize instead of KeyGen
    bool allowSecret = false;   // --allow-secret: permit loading the secret key
    std::string encInPath;      // --enc-in FILE: encrypt T x d f64 rows -> FILE.ct.N, exit
    std::string ctInPath;       // --ct-in PREFIX: server input ciphertexts
    std::string ctOutPath;      // --ct-out PREFIX: serialize final hidden-state cts, skip decode
    std::string decOutPath;     // --dec-out PREFIX: decrypt result cts, emit PREFIX.logits.f64, exit
    // ---- STATEFUL DECODE (2026-08-22) -----------------------------------
    // The demo is stateless: every generated token reruns the FULL prefix
    // because the per-layer scan state is zero-initialised each invocation
    // (see the scanState init below). Generating N tokens therefore costs
    // N(N+1)/2 scan steps instead of N. These flags persist the carry so
    // step k processes ONE token.
    //
    // THE CARRY IS TWO OBJECTS PER LAYER, not one -- a complete inventory of
    // this file's cross-token dependencies (verified by grep, 2026-08-22):
    //   1. scanState[l]  the scan accumulator          (the .tm.scan loop)
    //   2. u[T-1]        the pre-shift normalised input, because shift-mix
    //                    reads u_{t-1} and hardcodes u_{-1}=0 at t==0
    // The channel mix has NO cross-token dependency. Carrying only (1) would
    // silently corrupt the shift-mix boundary on every resumed step.
    //
    // SCOPE: --pack-tokens 1 only. scanState is live ONLY in the packTokens==1
    // branch of the scan; at packTokens>1 the blocked scan carries S[b] and
    // never touches scanState, so a carry taken from it would be wrong. That
    // is refused loudly below rather than emitted silently. This is the
    // "block-layout T=1/stateful-decode lane" of DEFERRED_WORK.md:125.
    std::string stateInPath;    // --state-in PREFIX: resume the carry from PREFIX.*
    std::string stateOutPath;   // --state-out PREFIX: persist the carry after the pass
    bool canonicalCarry = false; // --canonical-carry (2026-09-04): every tick's carry enters at the post-bootstrap
                                 // level, so the store serves every tick after the first (see `canon` below)
    bool statefulServe = false; // --stateful: in --serve mode, hold the carry in MEMORY
                                // across requests (no serialization, no transfer)
    std::string serveDir;       // --serve REQDIR: PERSISTENT server ("Fix 1"). Keep the
                                // context, rotation keys and bootstrap precompute resident
                                // and serve req.<N>.* ciphertext requests from REQDIR in a
                                // loop (each answered as resp.<N>.*), instead of paying the
                                // full deserialize+LoadContext bill per generation step
                                // (~433 s/step measured at d768, mostly setup).
    int serveTimeout = 3600;    // --serve-timeout S: exit after S seconds with no request
    // ---- DEMO-POD PREP (2026-09-02) ---------------------------------------
    // The fully-secure demo needs (a) the Mac to mint EXACTLY the automorphism
    // keys this build looks up (FIDESlib loads rotation_indexes + its own
    // bootstrap set + the conjugate key, api/CryptoContext.cpp:236-243 and
    // RawCiphertext.cu:1097-1112, and aborts with map::at on any gap), (b) a
    // key set that arrives in RAM-bounded PART files (the Mac has 17 GB, the
    // 2^17 set is ~48 GB), (c) a store that does not cost ~1 h per process
    // start, and (d) a VRAM breadcrumb at every milestone, because the
    // 2026-09-01 OOM logs carry no stage line at all.
    std::string dumpRotIndicesPath;   // --dump-rot-indices PATH: write the index contract, exit
    std::string storeFilePath;        // --store-file PATH: load the compressed store from PATH
                                      // (stamp-checked, sample-verified) else save it there
    int storeVerifySamples = 2;       // --store-verify-samples N: loaded entries re-encoded+compared,
    // S3.7 V5 (2026-09-10; S3.6 T01 section 2, S3.5 section 3): --periodic-encode builds each MISSED store
    // entry from its Dpad-period with periodic_encode.hpp -- an N/REP-point encode that is bit-identical to
    // the dense path (480/480 real entries in S3.6; this branch's periodic_encode_test at the demo ring) --
    // instead of a dense MakeCKKSPackedPlaintext + the compress/verify loop (2,131,944 ms of tick 0 on
    // demo2). DEFAULT OFF: the recorded path. --periodic-verify-every K (default 0): every K-th periodic
    // entry is ALSO built densely and compared word for word; a mismatch is fatal with the key printed.
    // The store itself does not change (same words, same stamp): a store built either way loads under
    // either flag. encPtMs/encPtCount keep their meaning (time and count of the miss-path builds).
    bool periodicEncode = false; int periodicVerifyEvery = 0;
    long long periodicEncodeCount = 0, periodicVerifyCount = 0, periodicVerifyOk = 0, periodicFallbackCount = 0;
                                      // N per distinct (layer, op, level) prefix (review 2026-09-03:
                                      // "first 64 hits" was layer 0 / win / one level only)
    bool vramTrace = true;            // --no-vram-trace: per-device cudaMemGetInfo lines (additive)
    bool storeMultiGpu = false;       // --store-multi-gpu: lift the devices>1 store refusal (global-id walk)
    bool storePeerWrites = false;     // --store-peer-writes: 2026-09-04, re-run the RETIRED peer-access grants
                                      // of the old peer-write expand (record only; the expand is per-device now)
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto nx = [&] { return std::string(argv[++i]); };
        if (s == "--arch") arch = nx();
        else if (s == "--tag") tag = nx();
        else if (s == "--bundle-dir") bundleDir = nx();
        else if (s == "--device") device = std::stoi(nx());
        else if (s == "--secure") secure = true;
        else if (s == "--selftest") selftest = true;
        else if (s == "--probe") probe = true;
        else if (s == "--boot-ladder") bootLadder = true;
        else if (s == "--trace-boots") traceBoots = true;
        else if (s == "--tokens") T = std::stoi(nx());
        else if (s == "--k") K = std::stoi(nx());
        else if (s == "--dpad") Dpad = std::stoi(nx());
        else if (s == "--log-ring") logRing = std::stoi(nx());
        else if (s == "--amp") bootAmp = std::stod(nx());
        else if (s == "--level-budget") { lbA = std::stoi(nx()); lbB = std::stoi(nx()); }
        else if (s == "--extra-depth") extraDepth = std::stoi(nx());
        else if (s == "--ms-norm") msNormK = std::stod(nx());
        else if (s == "--no-norm-eps") normEpsOn = false;     // S3.7 V1: the recorded no-eps circuit
        else if (s == "--bsgs-chunk") bsgsRescaleInner = true;
        else if (s == "--schedule") schedule = nx();          // S3.3
        else if (s == "--pf-entry") pfEntry = nx();           // S3.3
        else if (s == "--share-babies") shareBabies = true;   // S3.3
        else if (s == "--newton-depth2") newtonDepth2 = true; // S3.3 (Tier A-2)
        else if (s == "--pool-levels") poolLevels = std::stoi(nx());   // S3.3
        else if (s == "--block") blockType = nx();                     // S3.3 Tier B
        else if (s == "--matvec-margin") matvecMargin = std::stoi(nx());
        else if (s == "--matvec-probe") matvecProbe = true;
        else if (s == "--interleave") interleave = true;
        else if (s == "--stage-probe") stageProbe = true;
        else if (s == "--block-probe") blockProbe = true;
        else if (s == "--scale-bits") scaleBits = std::stoi(nx());
        else if (s == "--num-digits") numDigits = std::stoi(nx());
        else if (s == "--no-final-refresh") finalRefresh = false;
        else if (s == "--final-refresh") finalRefresh = true;
        else if (s == "--equalize-blocks") equalizeBlocks = true;
        else if (s == "--dump-logits") dumpLogits = nx();
        else if (s == "--pack-tokens") packTokens = std::stoi(nx());
        else if (s == "--lane-test") laneTest = true;
        else if (s == "--diag-eps") diagEps = std::stod(nx());
        else if (s == "--enc-threads") encThreads = std::stoi(nx());
        else if (s == "--encode-bench") encodeBench = true;
        else if (s == "--sync-timers") syncTimers = true;
        else if (s == "--stage-timers") stageTimers = true;        // S3.7 V2 (default ON)
        else if (s == "--no-stage-timers") stageTimers = false;    // S3.7 V2: HEAD behaviour, for A/B
        else if (s == "--proc-vram") procVram = true;
        else if (s == "--compressed-store") compressedStore = true;
        else if (s == "--lanes-block") lanesBlock = true;
        else if (s == "--lane-test-block") { laneTestBlock = true; lanesBlock = true; compressedStore = true; }
        else if (s == "--lanes-full") { lanesFull = std::stoi(nx()); lanesBlock = true; compressedStore = true; }
        else if (s == "--bridge-test") bridgeTest = std::stoi(nx());
        else if (s == "--devices") {                 // e.g. --devices 0,1
            // MULTI-GPU (2026-09-01). FIDESlib SHARDS rotation keys and the
            // bootstrap precomputation across devices rather than replicating
            // them -- established by measured arithmetic, not source reading:
            // in every 2- and 4-device record the max PER-CARD peak is strictly
            // below the total setup footprint, which replication cannot do
            // (l1b_4dev: 90,798 MB of keys+plaintexts loaded, hottest card
            // peaked 44.41 GiB, ran to completion).
            //
            // That is the whole reason this flag exists. 4-lane secure 2^17
            // needs 46.3 GB of rotation keys + 34.3 GB of BOOTSTRAP
            // PRECOMPUTATION plaintexts (762 StC/CtS matrices -- NOT the
            // compressed store, which is ~2-4 GB; mislabelled on 2026-09-01) =
            // 80.6 GB before working memory and OOMs one 95.6 GB card. Sharded
            // over two, the projection is (46.3 x 1.25)/2 + 34.3/2 ~= 46 GB per
            // card -- the 1.25 being the KeySwitchingKey::Initialize grow(L)
            // penalty that applies only when the device list has more than one
            // entry. NOT a security change: the parameter block is untouched.
            //
            // Ported onto the CURRENT harness deliberately. There is an older
            // mgpu fork (hpc_gpu_port/mgpu_harness/gpu_real_model_mgpu.cu) that
            // carries --devices and compiled clean at arch 120, but it is a
            // byte-copy of this file from 2026-07-30 and predates
            // --compressed-store, --lanes-block, --stateful and --serve, i.e.
            // every flag the demo is built from; its binaries also live on a
            // different (single-GPU) rental. Running it would measure the
            // pre-store circuit and call it the demo.
            devices.clear();
            const std::string list = nx();
            size_t p = 0;
            while (p < list.size()) {
                size_t c = list.find(',', p);
                if (c == std::string::npos) c = list.size();
                const std::string tok = list.substr(p, c - p);
                if (!tok.empty()) devices.push_back(std::stoi(tok));
                p = c + 1;
            }
        }
        else if (s == "--bundle-row-offset") bundleRowOffset = std::stoi(nx());
        else if (s == "--client-lanes") clientLanes = std::stoi(nx());
        else if (s == "--band") bandWidth = std::stoi(nx());
        else if (s == "--ring-mix") ringMix = true;
        else if (s == "--boot-floor") bootFloor = std::stoi(nx());
        else if (s == "--no-pipeline") pipeline = false;
        else if (s == "--pipeline") pipeline = true;
        else if (s == "--static-norm") staticNorm = true;
        else if (s == "--serial-carry") serialCarry = true;
        else if (s == "--two-level-scan") serialCarry = false;
        else if (s == "--cache-diags") cacheDiags = true;
        else if (s == "--passes") passes = std::stoi(nx());
        else if (s == "--max-batch") maxBatch = std::stoi(nx());
        else if (s == "--rsqrt-iters") rsqrtItersOverride = std::stoi(nx());
        else if (s == "--self-d") selfD = std::stoi(nx());
        else if (s == "--self-dff") selfDff = std::stoi(nx());
        else if (s == "--keys-dir") keysDir = nx();
        else if (s == "--keys-save") keysSave = true;
        else if (s == "--keys-load") keysLoad = true;
        else if (s == "--allow-secret") allowSecret = true;
        else if (s == "--enc-in") encInPath = nx();
        else if (s == "--ct-in") ctInPath = nx();
        else if (s == "--ct-out") ctOutPath = nx();
        else if (s == "--dec-out") decOutPath = nx();
        else if (s == "--state-in") stateInPath = nx();
        else if (s == "--state-out") stateOutPath = nx();
        else if (s == "--stateful") statefulServe = true;
        else if (s == "--canonical-carry") canonicalCarry = true;
        else if (s == "--serve") serveDir = nx();
        else if (s == "--serve-timeout") serveTimeout = std::stoi(nx());
        else if (s == "--dump-rot-indices") dumpRotIndicesPath = nx();
        else if (s == "--store-file") storeFilePath = nx();
        else if (s == "--store-verify-samples") storeVerifySamples = std::stoi(nx());
        else if (s == "--periodic-encode") periodicEncode = true;                        // S3.7 V5
        else if (s == "--periodic-verify-every") periodicVerifyEvery = std::stoi(nx());  // S3.7 V5
        else if (s == "--no-vram-trace") vramTrace = false;
        else if (s == "--vram-trace") vramTrace = true;
        else if (s == "--store-multi-gpu") storeMultiGpu = true;
        else if (s == "--store-peer-writes") storePeerWrites = true;
        else {
            // 2026-09-02: an unknown flag used to be IGNORED -- a typo such as
            // --store-fle silently ran the un-flagged configuration (F62 class).
            std::cerr << "{\"fatal\":\"unknown flag\",\"flag\":\"" << s << "\"}\n"; return 2;
        }
    }
    // ---- DEMO mode wiring (blocker 6) --------------------------------------
    // clientCrypto steps (--enc-in / --dec-out) never touch the GPU: an empty
    // device list routes every FIDESlib op through its CPU fallback, so the
    // client half of the demo needs no VRAM and no LoadContext.
    // canDecrypt gates EVERY decrypt-touching probe: a --keys-load process
    // without --allow-secret holds an empty-pimpl secret key sentinel, and an
    // accidental Decrypt must surface as a caught bad_any_cast, not a segfault.
    const bool clientCrypto = !encInPath.empty() || !decOutPath.empty();
    const bool canDecrypt = !keysLoad || allowSecret;
    if ((keysSave || keysLoad) && keysDir.empty()) {
        std::cerr << "{\"fatal\":\"--keys-save/--keys-load require --keys-dir\"}\n"; return 2; }
    if (keysSave && keysLoad) {
        std::cerr << "{\"fatal\":\"--keys-save and --keys-load are exclusive\"}\n"; return 2; }
    if (!encInPath.empty() && !(keysLoad || keysSave)) {
        std::cerr << "{\"fatal\":\"--enc-in needs --keys-load (reuse keys) or --keys-save (mint them)\"}\n"; return 2; }
    if (!ctInPath.empty() && !keysLoad) {
        std::cerr << "{\"fatal\":\"--ct-in needs --keys-load (input was encrypted under saved keys)\"}\n"; return 2; }
    if (!decOutPath.empty() && !(keysLoad && allowSecret)) {
        std::cerr << "{\"fatal\":\"--dec-out needs --keys-load AND --allow-secret\"}\n"; return 2; }
    if (!serveDir.empty() && !keysLoad) {
        std::cerr << "{\"fatal\":\"--serve needs --keys-load (requests are encrypted under saved keys)\"}\n"; return 2; }
    if ((int)!encInPath.empty() + (int)!decOutPath.empty() + (int)!serveDir.empty()
        + (int)(!ctInPath.empty() || !ctOutPath.empty()) > 1) {
        std::cerr << "{\"fatal\":\"pick ONE mode: --enc-in | --ct-in/--ct-out | --serve | --dec-out\"}\n"; return 2; }
    // ---- stateful-decode gating (2026-08-22) -----------------------------
    // Refuse loudly rather than emit a wrong carry. scanState is live only in
    // the packTokens==1 scan branch; the blocked scan carries S[b] instead.
    const bool statefulAny = !stateInPath.empty() || !stateOutPath.empty() || statefulServe;
    if (statefulAny && packTokens != 1) {
        std::cerr << "{\"fatal\":\"stateful decode requires --pack-tokens 1 -- scanState is live "
                     "ONLY in the packTokens==1 scan branch; at packTokens>1 the blocked scan "
                     "carries S[b] and a carry taken from scanState would be silently wrong\","
                     "\"packTokens\":" << packTokens << "}\n"; return 2; }
    if (statefulServe && serveDir.empty()) {
        std::cerr << "{\"fatal\":\"--stateful is a --serve mode (in-memory carry); for the "
                     "spawn-per-step path use --state-in/--state-out\"}\n"; return 2; }
    if (statefulServe && (!stateInPath.empty() || !stateOutPath.empty())) {
        std::cerr << "{\"fatal\":\"--stateful (in-memory) and --state-in/--state-out (on-disk) "
                     "are alternative carries; pick one\"}\n"; return 2; }
    if (statefulAny && passes > 1) {
        std::cerr << "{\"fatal\":\"stateful decode is incompatible with --passes>1 -- the pass "
                     "loop would re-derive and re-persist the carry every pass, so carryTokens "
                     "and the saved state would both count the same tokens N times\","
                     "\"passes\":" << passes << "}\n"; return 2; }
    if (!canDecrypt) { probe = false; stageProbe = false; blockProbe = false; }
#ifndef FHE_SSM_DEMO_SER
    // Without the demo-serialization build the ciphertext-file modes must be
    // REJECTED here, not silently degraded: e.g. --ct-in would otherwise fall
    // through to encrypting the bundle's own rows and "succeed" on the wrong
    // input. Key-file flags stay available (public fideslib API only).
    if (!encInPath.empty() || !ctInPath.empty() || !ctOutPath.empty() || !decOutPath.empty()
        || !serveDir.empty() || statefulAny) {
        std::cerr << "{\"fatal\":\"this binary was built without -DFHE_SSM_DEMO_SER; "
                     "--enc-in/--ct-in/--ct-out/--serve/--dec-out/--state-in/--state-out/--stateful "
                     "are unavailable (see hpc_gpu_port/DEMO_README.md)\"}\n";
        return 2;
    }
    (void)serveTimeout;   // only read by the (compiled-out) serve loop
    if (compressedStore || lanesBlock) {
        std::cerr << "{\"fatal\":\"--compressed-store/--lanes-block require a build with -DFHE_SSM_DEMO_SER "
                     "(they reach FIDESlib internals; see hpc_gpu_port/DEMO_README.md)\"}\n";
        return 2;
    }
#endif
    // AUDIT FIX 2026-09-02: REFUSE multi-device with the compressed store.
    // FIDESlib assigns Q limbs ROUND-ROBIN across the device list
    // (Context.cu:234, `dev = i % GPUid.size()`, per MGPU_README.md), so the
    // store's limb walk -- which concatenates GPU[dev].limb[k] in (dev,k)
    // order -- PERMUTES limbs under sharding. The pool would then compress
    // and expand the wrong residues: silent corruption. The walk must be
    // rewritten to place each limb by its GLOBAL index from the partition's
    // LimbRecord metadata, and verified on a box, before this is lifted.
    if (devices.size() > 1 && compressedStore && !storeMultiGpu) {
        std::cerr << "{\"fatal\":\"--devices with more than one device needs --store-multi-gpu with "
                     "--compressed-store: the pool now places every limb by its GLOBAL id from the "
                     "partition metadata (self-checking: each id filled exactly once) but the "
                     "numerics are verified only by a fit test that decodes -- read its fidelity "
                     "fields before trusting a multi-device store run\"}\n";
        return 2;
    }
    if (!storeFilePath.empty() && !compressedStore) {
        std::cerr << "{\"fatal\":\"--store-file requires --compressed-store\"}\n"; return 2;
    }
    if (storeVerifySamples < 0) {
        std::cerr << "{\"fatal\":\"--store-verify-samples must be >= 0\"}\n"; return 2;
    }
    // ---- S3.3 schedule-lever guards: refuse loudly, never degrade. ----------
    if (schedule != "stock" && schedule != "parent-first") {
        std::cerr << "{\"fatal\":\"--schedule must be stock or parent-first\",\"got\":\"" << schedule << "\"}\n"; return 2;
    }
    if (pfEntry != "loop" && pfEntry != "branch" && pfEntry != "both" && pfEntry != "always") {
        std::cerr << "{\"fatal\":\"--pf-entry must be loop|branch|both|always\",\"got\":\"" << pfEntry << "\"}\n"; return 2;
    }
    if (poolLevels < 1) { std::cerr << "{\"fatal\":\"--pool-levels must be >= 1\"}\n"; return 2; }
    if (poolLevels != 2 && !compressedStore) {
        std::cerr << "{\"fatal\":\"--pool-levels only applies with --compressed-store\"}\n"; return 2;
    }
    if (newtonDepth2 && staticNorm) {
        std::cerr << "{\"fatal\":\"--newton-depth2 has no Newton loop to act on under --static-norm\"}\n"; return 2;
    }
    if (shareBabies && maxBatch > 0) {
        // REVIEW 2026-09-03: the baby cache holds ONE input set; --max-batch splits a
        // batch into sub-vectors with different ciphertext objects, so every call
        // would miss, evict and recompute -- the lever silently inert. Refuse.
        std::cerr << "{\"fatal\":\"--share-babies is inert with --max-batch (the cache holds one sub-batch); drop one of them\"}\n"; return 2;
    }
    const bool schedParentFirst = (schedule == "parent-first");
    if (schedParentFirst && ringMix) {
        // --ring-mix is a cost proxy whose logits are meaningless by construction
        // (MEASUREMENTS.md 1a); a boot-count comparison on it would be mislabelled.
        std::cerr << "{\"warn\":\"--schedule parent-first with --ring-mix: counts only, logits meaningless by construction\"}" << std::endl;
    }
    std::cout << "{\"schedLevers\":true,\"schedule\":\"" << schedule << "\",\"pfEntry\":\"" << pfEntry
              << "\",\"shareBabies\":" << (shareBabies ? "true" : "false")
              << ",\"newtonDepth2\":" << (newtonDepth2 ? "true" : "false")
              << ",\"poolLevels\":" << poolLevels
              << ",\"stockPath\":" << ((!schedParentFirst && !shareBabies && !newtonDepth2 && poolLevels == 2) ? "true" : "false")
              << "}" << std::endl;
    if (!dumpRotIndicesPath.empty() && (keysLoad || clientCrypto || !serveDir.empty()
                                        || !ctInPath.empty() || !ctOutPath.empty() || selftest)) {
        std::cerr << "{\"fatal\":\"--dump-rot-indices is a standalone mode: pass ONLY the circuit flags "
                     "the server will use (ring/secure/depth/level-budget/lanes-block/pack-tokens/"
                     "interleave/dpad) plus --keys-dir if you also want --keys-save\"}\n"; return 2;
    }
    if (lanesBlock && (!compressedStore || interleave)) {
        std::cerr << "{\"fatal\":\"--lanes-block requires --compressed-store and BLOCK layout "
                     "(no --interleave) -- the masked halves are store entries\"}\n";
        return 2;
    }
    if (lanesFull > 0) {
        // The full-pass lane gate runs ONE well-defined shape: batch mode,
        // per-token ciphertexts, no client/serve plumbing, single pass. Every
        // other combination either rotates across lanes (packTokens>1), reads
        // a layout this mode does not produce (ct-in/state-in), or inflates
        // per-token fields (passes>1, HARNESS_BUGS B1).
        if (lanesFull < 2) { std::cerr << "{\"fatal\":\"--lanes-full needs NL >= 2\"}\n"; return 2; }
        if (packTokens != 1 || !ctInPath.empty() || !ctOutPath.empty() || !stateInPath.empty()
            || !stateOutPath.empty() || statefulServe || !serveDir.empty() || passes != 1
            || laneTestBlock || bundleRowOffset != 0 || selftest) {
            std::cerr << "{\"fatal\":\"--lanes-full is batch-mode only: requires --pack-tokens 1, "
                         "--passes 1, no ct-in/ct-out/state/serve/selftest, no --bundle-row-offset, "
                         "not combined with --lane-test-block\"}\n";
            return 2;
        }
    }
    if (bundleRowOffset < 0) { std::cerr << "{\"fatal\":\"--bundle-row-offset must be >= 0\"}\n"; return 2; }
    if (clientLanes < 1) { std::cerr << "{\"fatal\":\"--client-lanes must be >= 1\"}\n"; return 2; }
    if (clientLanes > 1 && (interleave || packTokens > 1 || encInPath.empty())) {
        // dec-out learns lanes from the ct meta; only the ENC step takes the flag.
        std::cerr << "{\"fatal\":\"--client-lanes > 1 is an --enc-in mode: BLOCK layout, "
                     "--pack-tokens 1, lanes ride the ct meta to --dec-out\"}\n";
        return 2;
    }
    CUDA_CHECK(cudaSetDevice(device));
#ifndef _WIN32
    if (procVram) {
        g_procVramOn = fhe_ssm_nvml::init(device);
        if (!g_procVramOn)
            std::cerr << "{\"warn\":\"--proc-vram: NVML unavailable (dlopen/dlsym/init failed); "
                         "procPeakVramGB will be -1\"}" << std::endl;
    }
#endif
    vramSample();

    // --selftest exercises only the primitives, so it can run WITHOUT a
    // bundle (use --self-d/--self-dff). That lets a small card validate the
    // FIDESlib API layer before any weights exist.
    const bool haveBundle = !(selftest && selfD > 0) && dumpRotIndicesPath.empty();
    Bundle B;
    int d, L, V, dff; double alphaRes; bool shiftMix; bool metaParallel = false;   // metaParallel: bundle meta[6] (S3.3 Tier B)
    if (haveBundle) {
        B.load(bundleDir + "/bundle_" + tag + ".bin", bundleDir + "/bundle_" + tag + ".index.txt");
        auto meta = B.vec("meta");
        d = (int)meta[0]; L = (int)meta[1]; V = (int)meta[2];
        dff = arch == "native" ? (int)meta[3] : 4 * d;
        alphaRes = arch == "native" ? meta[4] : 1.0;
        shiftMix = arch == "native" ? (meta.size() > 5 && meta[5] > 0.5) : false;
        metaParallel = arch == "native" && meta.size() > 6 && meta[6] > 0.5;   // S3.3 Tier B: block type
    } else {
        d = selfD > 0 ? selfD : (int)Dpad;               // --dump-rot-indices: no bundle, no --self-d
        dff = selfDff > 0 ? selfDff : 2 * d;
        L = 1; V = 1024; alphaRes = 1.0; shiftMix = false;
        while ((int)Dpad < d) Dpad <<= 1;
    }
    if ((int)Dpad < d) { std::cerr << "{\"fatal\":\"dpad<d\"}\n"; return 2; }
    // ---- S3.3 Tier B: block form. `auto` follows the bundle; an explicit value must agree.
    if (blockType != "auto" && blockType != "sequential" && blockType != "parallel") {
        std::cerr << "{\"fatal\":\"--block must be auto|sequential|parallel\",\"got\":\"" << blockType << "\"}\n"; return 2;
    }
    const bool blockParallel = (blockType == "auto") ? metaParallel : (blockType == "parallel");
    if (haveBundle && blockType != "auto" && blockParallel != metaParallel) {
        std::cerr << "{\"fatal\":\"--block disagrees with the bundle's meta[6] block type\",\"flag\":\"" << blockType
                  << "\",\"bundleParallel\":" << (metaParallel ? "true" : "false") << "}\n"; return 2;
    }
    if (blockParallel && (staticNorm || ringMix)) {
        std::cerr << "{\"fatal\":\"--block parallel is not combined with the --static-norm/--ring-mix cost proxies\"}\n"; return 2;
    }
    if (blockParallel)
        std::cout << "{\"blockType\":\"parallel\",\"source\":\"" << (blockType == "auto" ? "bundle meta[6]" : "flag") << "\"}" << std::endl;

    // Thread count for the parallel diagonal encode. --enc-threads 1 gives the
    // serial reference so the speedup is measured, not assumed.
    int ompThreads = 1;
#ifdef _OPENMP
    if (encThreads > 0) omp_set_num_threads(encThreads);
    ompThreads = omp_get_max_threads();
#endif

    // ---- CKKS context (identical policy to gpu_full_layer.cu) --------------
    auto tSetup = Clock::now();
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    std::vector<uint32_t> levelBudget = {(uint32_t)lbA, (uint32_t)lbB};
    const uint32_t levelsAfter = 10;
    const uint32_t depth = levelsAfter + 19 + extraDepth;   // 29 + extra
    if (secure) parameters.SetSecurityLevel(HEStd_128_classic);
    else { parameters.SetSecurityLevel(HEStd_NotSet); parameters.SetRingDim(1u << logRing); }
    // --scale-bits N (default 59): CKKS scaling mod size. 59 is the high-
    // precision default every historical number used. 35 exists for the
    // ring-2^16 128-bit route: logQP at depth 29 drops under the 2^16
    // ceiling (1747), so --secure lands on ring 65536 instead of 131072 --
    // 4x smaller keys/plaintexts, at reduced per-rescale precision. The
    // accuracy cost is measured, not assumed.
    parameters.SetScalingModSize((uint32_t)scaleBits);
    parameters.SetFirstModSize((uint32_t)std::min(60, scaleBits + 5));
    if (numDigits > 0) parameters.SetNumLargeDigits((uint32_t)numDigits);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(depth);
    if (devices.empty()) devices.push_back(device);
    // SetDevices on the params object takes an rvalue ref; pass a COPY because
    // `devices` is reused at the cc->SetDevices site below.
    parameters.SetDevices(std::vector<int>(devices));   // multi-GPU: FIDESlib shards keys here
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE); cc->Enable(FHE);
    // ---- --vram-trace (2026-09-02): per-device free/used at milestones. The
    // 2026-09-01 OOMs (demo_smoke/lanes17.log, mgpu17.log) carry no stage line,
    // so the allocation that crossed the ceiling is unrecorded. Additive stdout
    // JSON; cudaMemGetInfo is card-wide, like peakVramGB. Off in client-crypto
    // processes (no device).
    auto vramTraceLine = [&](const std::string& stg) {
        if (!vramTrace || clientCrypto) return;
        int cur = 0; cudaGetDevice(&cur);
        for (int dv : devices) {
            size_t fr = 0, tot = 0;
            if (cudaSetDevice(dv) != cudaSuccess || cudaMemGetInfo(&fr, &tot) != cudaSuccess) {
                cudaGetLastError(); continue;
            }
            // M1 (2026-09-14): default-pool attributes beside the card-wide reading.
            cudaMemPool_t mp = nullptr;
            unsigned long long rc = 0, rh = 0, uc = 0, uh = 0;   // the attribute type is a 64-bit unsigned (cuuint64_t)
            const bool mpOk =
                cudaDeviceGetDefaultMemPool(&mp, dv) == cudaSuccess &&
                cudaMemPoolGetAttribute(mp, cudaMemPoolAttrReservedMemCurrent, &rc) == cudaSuccess &&
                cudaMemPoolGetAttribute(mp, cudaMemPoolAttrReservedMemHigh, &rh) == cudaSuccess &&
                cudaMemPoolGetAttribute(mp, cudaMemPoolAttrUsedMemCurrent, &uc) == cudaSuccess &&
                cudaMemPoolGetAttribute(mp, cudaMemPoolAttrUsedMemHigh, &uh) == cudaSuccess;
            if (!mpOk) cudaGetLastError();
            const double G = 1073741824.0;
            std::cout << "{\"vramTrace\":\"" << stg << "\",\"dev\":" << dv
                      << ",\"usedGB\":" << (double)(tot - fr) / 1073741824.0
                      << ",\"freeGB\":" << (double)fr / 1073741824.0
                      << ",\"totalGB\":" << (double)tot / 1073741824.0
                      << ",\"poolOk\":" << (mpOk ? "true" : "false")
                      << ",\"poolReservedGB\":" << (mpOk ? (double)rc / G : -1.0)
                      << ",\"poolReservedHighGB\":" << (mpOk ? (double)rh / G : -1.0)
                      << ",\"poolUsedGB\":" << (mpOk ? (double)uc / G : -1.0)
                      << ",\"poolUsedHighGB\":" << (mpOk ? (double)uh / G : -1.0)
                      << ",\"poolSlackGB\":" << (mpOk ? (double)(rc >= uc ? rc - uc : 0) / G : -1.0)
                      << ",\"poolTransientGB\":" << (mpOk ? (double)(uh >= uc ? uh - uc : 0) / G : -1.0)
                      << "}" << std::endl;
            if (mpOk) {
                auto& hr = g_poolHighRun[dv];
                hr.first = std::max(hr.first, (double)uh / G);
                hr.second = std::max(hr.second, (double)rh / G);
                unsigned long long zero = 0;   // reset both watermarks to the current value (CUDA semantics of 0)
                cudaMemPoolSetAttribute(mp, cudaMemPoolAttrReservedMemHigh, &zero);
                cudaMemPoolSetAttribute(mp, cudaMemPoolAttrUsedMemHigh, &zero);
                if (!g_poolWarned && (double)(tot - fr) / G > 8.0 && (double)rc / G < 0.5) {
                    g_poolWarned = true;
                    std::cerr << "{\"warn\":\"vramTrace: default pool reserves < 0.5 GB while the card shows "
                              << (double)(tot - fr) / G << " GB used -- the library is not allocating from the "
                                 "device default pool; the pool* fields describe only the harness's own pool use\"}"
                              << std::endl;
                }
            }
        }
        cudaSetDevice(cur);
    };
    // Multi-device drain (2026-09-02). cudaDeviceSynchronize covers ONE device;
    // under --devices 0,1 FIDESlib enqueues each partition's work on its own
    // device's streams, so the store's expand kernel (launched on one device,
    // writing peer limb buffers) must be fenced against readers/writers on the
    // OTHER device explicitly. Single-device behaviour is unchanged.
    auto syncAllDevices = [&]() {
        if (devices.size() <= 1) { cudaDeviceSynchronize(); return; }
        int cur = 0; cudaGetDevice(&cur);
        for (int dv : devices) { CUDA_CHECK(cudaSetDevice(dv)); CUDA_CHECK(cudaDeviceSynchronize()); }
        CUDA_CHECK(cudaSetDevice(cur));
    };
    // byte-size reporter for the serialized artifacts (keys-save + demo modes)
    auto fileBytes = [](const std::string& p) -> long long {
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        return f ? (long long)f.tellg() : -1;
    };
    // ---- DEMO --keys-load: deserialize context + keys instead of KeyGen ----
    // All PUBLIC fideslib v2.1.2 API (api/Serialize.hpp + the CryptoContext
    // eval-key statics) -- key material needs no OpenFHE includes. NOTE
    // DeserializeFromFile REPLACES cc wholesale (api/Serialize.cpp builds a
    // fresh impl and restores devices/rotation_indexes/slots_bootstrap/keyDist
    // from the ".dev" sidecar), so it must happen BEFORE SLOTS/REP are derived
    // and before any lambda captures cc. The lbcrypto context arrives with its
    // serialized feature set; Enable() again anyway -- it is idempotent.
    KeyPair<DCRTPoly> keys;
    if (keysLoad) {
        if (!fideslib::Serial::DeserializeFromFile(keysDir + "/cryptocontext.bin", cc,
                                                   fideslib::SerType::BINARY)) {
            std::cerr << "{\"fatal\":\"keys-load: cryptocontext.bin (+.dev) failed\",\"dir\":\""
                      << keysDir << "\"}\n"; return 2;
        }
        cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE); cc->Enable(FHE);
        if (!fideslib::Serial::DeserializeFromFile(keysDir + "/public.key", keys.publicKey,
                                                   fideslib::SerType::BINARY)) {
            std::cerr << "{\"fatal\":\"keys-load: public.key failed\"}\n"; return 2;
        }
        if (allowSecret) {
            if (!fideslib::Serial::DeserializeFromFile(keysDir + "/secret.key", keys.secretKey,
                                                       fideslib::SerType::BINARY)) {
                std::cerr << "{\"fatal\":\"keys-load: secret.key failed\"}\n"; return 2;
            }
        } else {
            // empty-pimpl sentinel: an accidental Decrypt throws bad_any_cast
            // (caught by the probe guards) instead of dereferencing a null.
            keys.secretKey = std::make_shared<PrivateKeyImpl<DCRTPoly>>();
        }
        // devices: the .dev sidecar restored the SAVE-time list; override with
        // this invocation's role (client = CPU-only, server = this GPU).
        cc->SetDevices(clientCrypto ? std::vector<int>{} : devices);
        // The circuit's level trace and the packing geometry are driven by the
        // FLAGS, so the loaded context must match them exactly.
        if (!secure && cc->GetRingDimension() != (1u << logRing)) {
            std::cerr << "{\"fatal\":\"keys-load ring mismatch: pass the SAME --log-ring/--secure used at --keys-save\","
                      << "\"loaded\":" << cc->GetRingDimension()
                      << ",\"expected\":" << (1u << logRing) << "}" << std::endl; return 2;
        }
        if (cc->multiplicative_depth != depth) {
            std::cerr << "{\"fatal\":\"keys-load depth mismatch: pass the SAME --extra-depth/--level-budget used at --keys-save\","
                      << "\"loaded\":" << cc->multiplicative_depth
                      << ",\"expected\":" << depth << "}" << std::endl; return 2;
        }
    } else if (clientCrypto) {
        cc->SetDevices(std::vector<int>{});   // keygen-in-client-process (--keys-save --enc-in)
    }
    const uint32_t SLOTS = cc->GetRingDimension() / 2;
    const uint32_t REP = SLOTS / Dpad;              // replicas of the D-block
    if (lanesFull > 0 && (uint32_t)lanesFull > REP) {
        std::cerr << "{\"fatal\":\"--lanes-full " << lanesFull << " > REP " << REP
                  << " at this ring/Dpad\"}" << std::endl; return 2;
    }
    const int BS = (int)std::ceil(std::sqrt((double)Dpad));   // baby-step count
                                                              // (hoisted out of the keygen
                                                              // block: matvec uses it too)
    if (!keysLoad) {
    keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // rotation keys: BSGS baby steps 1..bs-1 and giant steps bs,2bs,... plus
    // the reduction rotations (powers of two up to Dpad/2) and token-shift.
    std::vector<int32_t> rotsAll;                    // 2026-09-02: hoisted for --dump-rot-indices
    {
        std::vector<int32_t> rots;
        const int RM = interleave ? (int)REP : 1;   // rotation multiplier for the layout
        for (int j = 1; j < BS; j++) rots.push_back(j * RM);
        for (int g = 0; g < (int)Dpad; g += BS) if (g) rots.push_back(g * RM);
        for (int p = 1; p < (int)Dpad; p <<= 1) rots.push_back(p * RM);   // reduction
        if (lanesBlock) { rots.push_back(-(int)Dpad);                     // T1c u_wrap
                          for (int p = 1; p < (int)Dpad; p <<= 1) rots.push_back(-p); }  // T1c norm down-tree
        if (bridgeTest >= 0) {
            // BRIDGE key set. Moving content from slot u to slot u+delta is
            // EvalRotate(-delta) in OpenFHE's convention (rot(k) maps i+k -> i),
            // so the bridge needs NEGATIVE amounts that the positive matvec set
            // does not cover: BSGS babies -(1..BS-1) and giants -(hi*BS).
            // ~BS + Dpad/BS keys, and these are RAW slot amounts -- deliberately
            // NOT multiplied by RM, because the bridge works in the slot basis
            // by construction (it is the thing that changes basis).
            for (int lo = 1; lo < BS; lo++) rots.push_back(-lo);
            for (int hi = 1; hi < (int)Dpad / BS; hi++) rots.push_back(-hi * BS);
        }
        if (packTokens > 1 || laneTest) {
            // STAGE 2: token lanes are CONTIGUOUS (stride 1) within a channel
            // group, so a shift by j tokens is a ring rotation by j -- NOT by
            // j*REP. Need +-2^j for the log-doubling scan and +-1 for shift-mix.
            for (int j = 1; j < (int)REP; j <<= 1) { rots.push_back(j); rots.push_back(-j); }
            rots.push_back(1); rots.push_back(-1);
            // multi-block: the shift-mix block boundary pulls the PREVIOUS
            // block's last lane via EvalRotate(+(packTokens-1)).
            if (packTokens > 1) rots.push_back(packTokens - 1);
        }
        // dedup
        std::sort(rots.begin(), rots.end());
        rots.erase(std::unique(rots.begin(), rots.end()), rots.end());
        cc->EvalRotateKeyGen(keys.secretKey, rots);              // TODO(verify-on-target): EvalAtIndexKeyGen
        rotsAll = rots;
    }
    cc->EvalBootstrapSetup(levelBudget, {0, 0}, SLOTS);
    cc->EvalBootstrapKeyGen(keys.secretKey, SLOTS);
    // ---- --dump-rot-indices (2026-09-02): the key CONTRACT for a client keygen.
    // What this build will look up at LoadContext, by the library's own
    // formula: for every rotation step s in rotation_indexes, the automorphism
    // index 5^s mod 2N (RawCiphertext.cu:735, FIDESlib::modpow with the step
    // cast to uint64 -- a negative step lands on the inverse because the order
    // of 5 divides 2^64); every bootstrap step from GetBootstrapIndexes (what
    // AddBootstrapKeys asks for, :1097-1112); and the conjugate key 2N-1. The
    // 2^15 sovereignty failure (six missing indices, CORRECTIONS A13) was
    // exactly a client minting OpenFHE's guess of this set instead of asking.
    // The cross-check against this process's own OpenFHE keygen is reported;
    // a non-empty missingFromOwnKeygen would mean the proven keys-save path
    // itself lacks a key (never observed; recorded so it cannot hide).
#ifdef FHE_SSM_DEMO_SER
    // (2026-09-03, pod build_fideslib.log:925-929) this block reaches OpenFHE/FIDESlib
    // internals (lbcrypto::, FIDESlib::CKKS::GetBootstrapIndexes) that only the
    // DEMO_SER build includes; the plain (LTO anchor) build failed to compile here.
    if (!dumpRotIndicesPath.empty()) {
        auto& cpuCcD = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
        const uint64_t M2 = (uint64_t)cc->GetRingDimension() * 2;
        auto autoIdx = [&](int64_t step) -> uint64_t {
            uint64_t a = 5, e = (uint64_t)step, r = 1;     // == FIDESlib::modpow(5, step, 2N)
            while (e) { if (e & 1) r = (__uint128_t)a * r % M2; e >>= 1; a = (__uint128_t)a * a % M2; }
            return r;
        };
        std::set<uint64_t> need;
        for (int32_t st : rotsAll) if (st) need.insert(autoIdx(st));
        std::vector<int> bootSteps = FIDESlib::CKKS::GetBootstrapIndexes(cpuCcD, (int)SLOTS, nullptr);
        // GetBootstrapIndexes can emit step 0 (RawCiphertext.cu:1021 pushes a literal 0);
        // AddRotationKeys skips it (`if (i && ...)`, :803), and OpenFHE never mints
        // index 1, so a contract carrying it would report missingFromOwnKeygen:[1]
        // on every run and make the client mint a useless 336 MiB key.
        for (int st : bootSteps) if (st) need.insert(autoIdx(st));
        need.insert(M2 - 1);
        std::set<uint64_t> have;
        for (auto& kv : lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetAllEvalAutomorphismKeys())
            for (auto& e : *kv.second) have.insert(e.first);
        std::vector<uint64_t> missing;
        for (auto x : need) if (!have.count(x)) missing.push_back(x);
        std::ofstream jf(dumpRotIndicesPath);
        if (!jf) { std::cerr << "{\"fatal\":\"--dump-rot-indices: cannot write " << dumpRotIndicesPath << "\"}\n"; return 2; }
        auto arr = [&](auto& os, const auto& v) { os << "["; bool f = true; for (auto x : v) { if (!f) os << ","; f = false; os << x; } os << "]"; };
        jf << "{\"contract\":\"fhe-ssm-rot-indices-v1\",\"ring\":" << cc->GetRingDimension()
           << ",\"slots\":" << SLOTS << ",\"depth\":" << depth << ",\"extraDepth\":" << extraDepth
           << ",\"levelBudget\":[" << lbA << "," << lbB << "],\"scaleBits\":" << scaleBits
           << ",\"secure\":" << (secure ? "true" : "false") << ",\"dpad\":" << Dpad << ",\"bs\":" << BS
           << ",\"lanesBlock\":" << (lanesBlock ? "true" : "false") << ",\"interleave\":" << (interleave ? "true" : "false")
           << ",\"packTokens\":" << packTokens << ",\"numDigits\":" << numDigits
           << ",\"bootstrapSlots\":" << SLOTS << ",\"rotationAmounts\":"; arr(jf, rotsAll);
        jf << ",\"bootstrapSteps\":"; arr(jf, bootSteps);
        jf << ",\"autoIndices\":"; arr(jf, need);
        jf << ",\"openfheMapCount\":" << have.size() << ",\"missingFromOwnKeygen\":"; arr(jf, missing);
        jf << "}\n";
        std::cout << "{\"dumpRotIndices\":true,\"path\":\"" << dumpRotIndicesPath << "\",\"ring\":" << cc->GetRingDimension()
                  << ",\"rotationAmounts\":" << rotsAll.size() << ",\"bootstrapSteps\":" << bootSteps.size()
                  << ",\"autoIndices\":" << need.size() << ",\"openfheMapCount\":" << have.size()
                  << ",\"missingFromOwnKeygen\":" << missing.size() << "}" << std::endl;
        if (!keysSave) return missing.empty() ? 0 : 3;
    }
#else
    if (!dumpRotIndicesPath.empty()) {
        std::cerr << "{\"fatal\":\"--dump-rot-indices needs a build with -DFHE_SSM_DEMO_SER (it reads FIDESlib's bootstrap index set)\"}\n";
        return 2;
    }
#endif
    // ---- DEMO --keys-save: AFTER every key exists (EvalBootstrapKeyGen adds
    // the bootstrap automorphism keys to the same static map the rotation keys
    // live in, and pushes slots_bootstrap -- which the context's .dev sidecar
    // records), BEFORE LoadContext (FIDESlib forbids keygen after loading, so
    // this is the natural last host-side moment). SerializeEvalMultKey /
    // SerializeEvalAutomorphismKey are the static all-tags forms, called
    // instance-style exactly as FIDESlib's own examples/serial does.
    if (keysSave) {
        bool okS = true;
        okS = okS && fideslib::Serial::SerializeToFile(keysDir + "/cryptocontext.bin", cc,
                                                       fideslib::SerType::BINARY);
        okS = okS && fideslib::Serial::SerializeToFile(keysDir + "/public.key", keys.publicKey,
                                                       fideslib::SerType::BINARY);
        okS = okS && fideslib::Serial::SerializeToFile(keysDir + "/secret.key", keys.secretKey,
                                                       fideslib::SerType::BINARY);
        {
            std::ofstream em(keysDir + "/evalmult.bin", std::ios::binary);
            okS = okS && em.good() && cc->SerializeEvalMultKey(em, fideslib::SerType::BINARY);
        }
        {
            std::ofstream ea(keysDir + "/evalrot.bin", std::ios::binary);
            okS = okS && ea.good() && cc->SerializeEvalAutomorphismKey(ea, fideslib::SerType::BINARY);
        }
        if (!okS) {
            std::cerr << "{\"fatal\":\"keys-save failed (does " << keysDir
                      << " exist and is it writable?)\"}\n"; return 2;
        }
        std::cout << "{\"keysSave\":true,\"dir\":\"" << keysDir << "\""
                  << ",\"cryptocontextBytes\":" << fileBytes(keysDir + "/cryptocontext.bin")
                  << ",\"publicKeyBytes\":" << fileBytes(keysDir + "/public.key")
                  << ",\"secretKeyBytes\":" << fileBytes(keysDir + "/secret.key")
                  << ",\"evalMultKeyBytes\":" << fileBytes(keysDir + "/evalmult.bin")
                  << ",\"evalRotKeyBytes\":" << fileBytes(keysDir + "/evalrot.bin")
                  << "}" << std::endl;
    }
    } else if (!clientCrypto) {
        // ---- DEMO server half of --keys-load: the CPU-side bootstrap
        // precompute (FHECKKSRNS::m_bootPrecomMap) is NOT part of the context
        // serialization, and LoadContext's AddBootstrapPrecomputation reads it
        // from the CPU scheme -- so re-run EvalBootstrapSetup in THIS process
        // (public API, needs no secret key), then pull the relin + rotation +
        // bootstrap keys out of the files into the lbcrypto static maps that
        // LoadContext reads (keyed by the public key's tag).
        cc->EvalBootstrapSetup(levelBudget, {0, 0}, SLOTS);
        {
            std::ifstream em(keysDir + "/evalmult.bin", std::ios::binary);
            if (!em.good() || !cc->DeserializeEvalMultKey(em, fideslib::SerType::BINARY)) {
                std::cerr << "{\"fatal\":\"keys-load: evalmult.bin failed\"}\n"; return 2;
            }
        }
        {
            // 2026-09-02: evalrot.bin and/or evalrot.partNNN.bin (sorted). Each
            // file deserializes into the SAME static map (lbcrypto
            // DeserializeEvalAutomorphismKey -> InsertEvalAutomorphismKey merges
            // per tag), so a client with 17 GB of RAM can mint the ~48 GB set in
            // batches. A missing key still aborts at LoadContext (map::at).
            std::vector<std::string> rotFiles;
            if (fileBytes(keysDir + "/evalrot.bin") > 0) rotFiles.push_back(keysDir + "/evalrot.bin");
            {
                std::vector<std::string> parts;
                std::error_code ec;
                for (const auto& de : std::filesystem::directory_iterator(keysDir, ec)) {
                    const std::string fn = de.path().filename().string();
                    // 2026-09-17: the seeded parts (evalrot.partNNN.seeded.bin, V4) also end in ".bin" --
                    // they are NOT stock parts (DeserializeEvalAutomorphismKey would fail on them); the
                    // seeded loop below owns them.
                    if (fn.rfind("evalrot.part", 0) == 0 && fn.size() > 4 && fn.substr(fn.size() - 4) == ".bin"
                        && !(fn.size() > 11 && fn.substr(fn.size() - 11) == ".seeded.bin"))
                        parts.push_back(de.path().string());
                }
                std::sort(parts.begin(), parts.end());
                rotFiles.insert(rotFiles.end(), parts.begin(), parts.end());
            }
            // S3.7 V4 (2026-09-10, S3.6 T12 section 2.3): seed-expanded parts, evalrot.partNNN.seeded.bin
            // (mac_fhe_client keygen --seeded 1): the a-halves are 32-byte seeds, so the part is half the
            // bytes on disk and on the wire (336 -> 168 MiB per key at 2^17). They are re-expanded on the
            // host below and inserted into the SAME static map the stock parts fill, before LoadContext
            // reads it; VRAM is unchanged. Loaded only in the DEMO_SER build (the expander needs the
            // OpenFHE headers); a part minted for another context is refused by its header's QP chain.
            std::vector<std::string> seededParts;
            {
                std::error_code ec2;
                for (const auto& de : std::filesystem::directory_iterator(keysDir, ec2)) {
                    const std::string fn = de.path().filename().string();
                    if (fn.rfind("evalrot.part", 0) == 0 && fn.size() > 11 && fn.substr(fn.size() - 11) == ".seeded.bin")
                        seededParts.push_back(de.path().string());
                }
                std::sort(seededParts.begin(), seededParts.end());
            }
            if (rotFiles.empty() && seededParts.empty()) {
                std::cerr << "{\"fatal\":\"keys-load: no evalrot.bin, no evalrot.part*.bin and no evalrot.part*.seeded.bin in " << keysDir << "\"}\n"; return 2;
            }
            for (const auto& rf : rotFiles) {
                std::ifstream ea(rf, std::ios::binary);
                if (!ea.good() || !cc->DeserializeEvalAutomorphismKey(ea, fideslib::SerType::BINARY)) {
                    std::cerr << "{\"fatal\":\"keys-load: evalrot file failed\",\"file\":\"" << rf << "\"}\n"; return 2;
                }
            }
            if (!seededParts.empty()) {
#ifdef FHE_SSM_DEMO_SER
                auto& cpuCcK = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
                fhe_ssm::seeded::ExpandStats sst; std::string serr;
                auto tSeed = Clock::now();
                for (const auto& sp : seededParts)
                    if (!fhe_ssm::seeded::expandSeededKeys(cpuCcK, sp, sst, &serr)) {
                        std::cerr << "{\"fatal\":\"keys-load: seeded part failed\",\"file\":\"" << sp << "\",\"why\":\"" << serr << "\"}\n"; return 2;
                    }
                std::cout << "{\"keysLoadSeeded\":true,\"seededParts\":" << seededParts.size() << ",\"seededKeys\":" << sst.keys
                          << ",\"seededInserted\":" << sst.inserted << ",\"seededBytes\":" << sst.bytes
                          << ",\"seededExpandMs\":" << (int)msSince(tSeed) << "}" << std::endl;   // S3.7 V4, additive
                rotFiles.insert(rotFiles.end(), seededParts.begin(), seededParts.end());   // counted in evalRotFiles below
#else
                std::cerr << "{\"fatal\":\"keys-load: evalrot.part*.seeded.bin needs the DEMO_SER build (host expander)\"}\n"; return 2;
#endif
            }
            // autoKeysInMap: counting the lbcrypto static map needs the OpenFHE
            // headers, i.e. the DEMO_SER build; the plain (LTO anchor) build
            // reports -1 (2026-09-03 pod build_fideslib.log:953 died here).
            long autoKeys = -1;
#ifdef FHE_SSM_DEMO_SER
            autoKeys = 0;
            for (auto& kv : lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetAllEvalAutomorphismKeys())
                autoKeys += (long)kv.second->size();
#endif
            std::cout << "{\"keysLoad\":true,\"evalRotFiles\":" << rotFiles.size()
                      << ",\"autoKeysInMap\":" << autoKeys << "}" << std::endl;
        }
        vramTraceLine("keysHost");
    }
    if (!clientCrypto) cc->LoadContext(keys.publicKey);
    vramSample();
    vramTraceLine("loadContext");
    double setupMs = msSince(tSetup);

    // ---- helpers ----------------------------------------------------------
    // REDESIGNED 2026-07-19. This used to carry `uint32_t used` -- a SHADOW
    // copy of the ciphertext's level, maintained by hand at every call site.
    // Four of the seven bugs found on 2026-07-18 were the same failure mode:
    // the shadow counter silently drifting out of sync with the real
    // ciphertext (polyGate3's unaligned EvalAdd left it a level stale;
    // matvecBatch's aggregate max pushed it past `depth`; boot() ASSUMED
    // EvalBootstrap lands exactly on depth-levelsAfter). Once it desynced,
    // plaintexts were encoded at the wrong level and the result was either an
    // undecryptable ciphertext or an unsigned-underflow crash.
    // The ciphertext already knows its own level -- so ask it. There is now
    // exactly one source of truth and desync is unrepresentable.
    struct TrackedCt {
        Ciphertext<DCRTPoly> ct;
        uint32_t used = 0;              // VESTIGIAL: no longer read anywhere.
        uint32_t level() const { return ct ? (uint32_t)ct->GetLevel() : 0u; }
    };
    // DEBUG INSTRUMENTATION (2026-07-18): every 3-arg MakeCKKSPackedPlaintext
    // call in this file was routed through here (17 sites, mechanically) to
    // localize a level-overflow crash that three targeted fixes (rmsnorm's
    // unguarded Newton loop, rmsnorm's caller-mutating reference param,
    // addAligned's unguarded align target) narrowed but did not fully close.
    // Reports the EXACT source line the moment `lvl` exceeds the valid
    // [0, depth] range, instead of letting OpenFHE's internal indexing
    // underflow into an opaque vector::at() crash several calls later.
    // ---- S3.7 V2 (2026-09-05): host plaintext-encode counters (S3.6 T05 §1, §3.2).
    // Every mkPt is a host MakeCKKSPackedPlaintext at the operand level (65,536
    // doubles -> 42 - lvl limbs of 2^17 words at the demo ring); the replay census
    // counts 1,688 per warm tick, 536 inside rmsnorm windows and 1,152 in the
    // (previously untimed) layer loop. Declared BEFORE mkPt because the lambda
    // captures by reference. THREADING: mkPt runs on the MAIN THREAD ONLY -- the
    // OpenMP diagonal encode (encodeBatch's `#pragma omp parallel for`), buildComp,
    // the pool zero-encodes (compPoolGet) and the pipelined EncodeWorker all call
    // cc->MakeCKKSPackedPlaintext directly and never mkPt (checked call site by
    // call site: grep "mkPt(" lands only in main-thread code), so plain doubles
    // are safe. These counters OVERLAP the stage timers (loop call sites) and
    // evalMs (rmsnorm, the no-store matvec, --ring-mix); the *Stage* pair is the
    // share that fell inside an open stage window = the layer loop's elementwise
    // encodes. They are always on (one clock read per encode).
    double hostPtEncodeMs = 0, hostPtEncodeStageMs = 0;
    long long hostPtEncodeCount = 0, hostPtEncodeStageCount = 0;
    int stageOpenDepth = 0;                       // > 0 while a stage window is open
    auto mkPt = [&](const std::vector<double>& v, uint32_t lvl, int srcLine) -> Plaintext {
        if (lvl > depth) {
            std::cerr << "{\"fatal\":\"level_overflow\",\"lvl\":" << lvl
                      << ",\"depth\":" << depth << ",\"line\":" << srcLine << "}" << std::endl;
            std::terminate();
        }
        auto tPt = Clock::now();                   // S3.7 V2: the host encode, see hostPtEncodeMs
        Plaintext pt = cc->MakeCKKSPackedPlaintext(v, 1, lvl);
        const double ptMs = msSince(tPt);
        hostPtEncodeMs += ptMs; hostPtEncodeCount++;
        if (stageOpenDepth > 0) { hostPtEncodeStageMs += ptMs; hostPtEncodeStageCount++; }
        return pt;
    };
    double encMs = 0, evalMs = 0, bootMs = 0, decMs = 0;
    double encPtMs = 0;              // wall time inside the parallel diagonal-encode regions
    double pipeMs = 0;               // wall time of the pipelined encode+apply stages
    long long encPtCount = 0;        // diagonals actually encoded (post value test)
    // T0.1 (S3.1): per-phase accumulators, populated ONLY under --sync-timers
    // (device drained at each boundary). All stay 0.0 when the flag is off.
    double syncBabyRotMs = 0, syncCtPtMs = 0, syncGiantMs = 0, syncRescaleMs = 0, syncBootMs = 0;
    auto syncDev = [&] { if (syncTimers) cudaDeviceSynchronize(); };
    // ---- S3.7 V2 (2026-09-05): the timer corrections (the timer-correction notes N1,
    // COMPLEXITY_MAP §1, T05_T06 §2). ALL new fields are ADDITIVE; no existing field
    // changes meaning (R6 comparability with the records).
    // (a) normBootMs. rmsnorm's evalMs window contains its own refresh() bootstraps
    //     (the Newton guards and the tail refresh), whose boot() also adds to
    //     bootMs -- the only bootMs accumulation -- so msPerToken = (evalMs +
    //     bootMs)/tok double-counts every norm-internal boot: cell A
    //     (cmp_1gpu_store_t1.log:125) 117 boots x 238.38 ms = 27,890 ms of
    //     msPerToken 191,762 (wall 222,960). rmsnorm now snapshots bootMs at its
    //     window start and adds the delta HERE; evalMs and bootMs keep their
    //     historical meaning, and gpuEvalMsExBoot = evalMs - normBootMs,
    //     msPerTokenExBoot = (gpuEvalMsExBoot + bootMs)/tokPasses are the disjoint
    //     sums. Every other refresh()/boot() site was checked: none sits inside
    //     matvec's, matvecBatch's, encCh's, decCh's or the pool's window (the
    //     matvecBatch input alignTo precedes its clock), so rmsnorm is the only
    //     window that needs the subtraction (S3.7 V2 report, hunk list).
    double normBootMs = 0;
    // (b) the layer loop's clock. layerLoopMs = the loop's whole wall (it OVERLAPS
    //     evalMs/bootMs: the matvecs, norms and boots run inside it);
    //     layerLoopTimedMs = the disjoint timed sum that accrued inside the loop,
    //     so layerLoopMs - layerLoopTimedMs is the loop's untimed part. The five
    //     stage accumulators bill ONLY the untimed part of a stage: its wall minus
    //     the delta of the existing timers inside it ((evalMs - normBootMs) +
    //     bootMs + encMs + decMs), so each is disjoint from every existing field
    //     and from the other stages; the norm stages, the matvecs and the
    //     inter-stage host work (packCh vectors, the Wb slice, stage strings,
    //     probes) are NOT wrapped and remain in layerLoopResidualMs = loop untimed
    //     minus the five stages. Under --sync-timers stageOpen/stageClose drain the
    //     device (syncDev) so queued elementwise kernels are billed to their stage
    //     and never to the next timed window. --no-stage-timers: all of this is
    //     skipped (HEAD behaviour).
    double layerLoopMs = 0, layerLoopTimedMs = 0;
    double shiftMixMs = 0, scanMs = 0, gateMs = 0, hiddenMs = 0, residualMs = 0;
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
    // --cache-diags: memoize encoded diagonals so they stay GPU-RESIDENT across
    // passes. Key is (weight block, row offset, level, diagonal) -- the level
    // trace is deterministic, so pass 2 hits 100%. Only viable when the whole
    // set fits in VRAM, i.e. for a STRUCTURED model: dense is 135,168 x 9.8 MB
    // = 1.32 TB, but --band 8 is 132 x 17 = 2,244 plaintexts = ~22 GB.
    std::map<std::tuple<const double*, int, uint32_t, int>, Plaintext> diagCache;
    long long diagCacheHits = 0;
    int boots = 0;
    const std::string* stagePtr = nullptr;   // S3.3: the `stage` breadcrumb (declared later) for bootTrace lines
    // 2026-09-17 pod-fix (S3.7): the canonical-carry counters are used by canon() and the cold-start line in BOTH builds,
    // so they live outside the DEMO_SER block (the plain build failed: "canonLevel is undefined", build_fideslib.log:922).
    long long canonBoots = 0; uint32_t canonMaxGap = 0;   // --canonical-carry: boots spent and the largest align gap
    int canonLevel = -1;                                  // the post-bootstrap level, learned from the first canonical boot
#ifdef FHE_SSM_DEMO_SER
    // ---- T1b (S3.1): the compressed store + per-level scratch pool. -------
    // Key is VALUE IDENTITY — (layer, op, chunk, rowOff, level, diagonal) —
    // set by the call-site markers below, never a data pointer (B6). A store
    // entry is the diagonal's compressed eval-domain limbs (v only, N/REP
    // words per limb; empty vector = diagonal is dead). Delivery is one
    // pinned H2D + one expand-kernel launch per giant step into a resident
    // per-level pool of BS scratch plaintexts, consumed by the stock multPt.
    // Proven bit-identical end to end by gpu_delivery_bench --cerium-device.
    struct CompKey {
        int layer, op, chunk, rowOff, diag, half; uint32_t lvl;   // half: 0=lo, 1=hi (T1c), 0 in broadcast mode
        bool operator==(const CompKey& o) const {
            return layer == o.layer && op == o.op && chunk == o.chunk
                && rowOff == o.rowOff && diag == o.diag && half == o.half && lvl == o.lvl;
        }
    };
    struct CompKeyHash {
        size_t operator()(const CompKey& k) const {
            size_t h = (size_t)k.layer * 1000003u ^ (size_t)(k.op + 1) * 10007u
                     ^ (size_t)(k.chunk + 1) * 101u ^ (size_t)(k.rowOff + 1) * 33331u
                     ^ (size_t)k.lvl * 7919u ^ (size_t)(k.diag + 1) * 31u
                     ^ (size_t)(k.half + 1) * 65537u;
            return h;
        }
    };
    std::unordered_map<CompKey, std::vector<uint64_t>, CompKeyHash> compStore;
    long long storeHits = 0, storeMisses = 0, storeDead = 0, storeUncompressible = 0;
    double storeFetchMs = 0;
    size_t storeBytes = 0;
    int mvLayer = -1, mvOpId = -1, mvChunk = -1;   // call-site markers; -1 = store bypassed
    struct PoolLvl {
        std::vector<Plaintext> wraps;                                   // BS wrapper pts (gpu handle set)
        std::vector<std::shared_ptr<FIDESlib::CKKS::Plaintext>> gpts;   // the device objects
        std::vector<uint32_t> handles;
        uint64_t** dPtrTab = nullptr;   // BS*nLimbs device limb pointers (constant)
        uint64_t* dComp = nullptr;      // BS*nLimbs*CW compressed staging (device)
        uint64_t* hStage = nullptr;     // pinned host staging, same size
        std::vector<uint64_t**> dPtrTabDev;   // multi-GPU (2026-09-04): per-device pointer tables
        std::vector<uint64_t*> dCompDev;      // multi-GPU: per-device compressed staging
        std::vector<int*> dLimbDev;           // multi-GPU: per-device limb -> partition index
        int nLimbs = 0;
        long long lastUse = 0;
    };
    std::map<uint32_t, PoolLvl> compPool;
    long long compPoolClock = 0;
    long long poolBuilds = 0; double poolBuildMs = 0;     // S3.3: pool (re)builds, timed (summary fields)
    const uint32_t COMP_CW = SLOTS * 2 / REP;       // N/REP words per limb
    uint32_t compLogRep = 0; { uint32_t r = REP; while ((1u << compLogRep) < r) compLogRep++; }
    // S3.7 V5: ONE periodic encoder per context (its FFT/NTT tables at order 2N' are built once, about
    // 1 ms per tower); constructed lazily on the first miss, so a run without --periodic-encode pays nothing.
    std::unique_ptr<fhe_ssm::PeriodicEncoder> periodicEnc;
    fhe_ssm::PeriodicEncoder::Scratch periodicScratch;
    auto compPoolDrop = [&](uint32_t lvl) {
        auto it = compPool.find(lvl);
        if (it == compPool.end()) return;
        for (auto h : it->second.handles) cc->EvictDevicePlaintext(h);
        if (it->second.dPtrTab) cudaFree(it->second.dPtrTab);
        if (it->second.dComp) cudaFree(it->second.dComp);
        for (size_t d = 0; d < it->second.dPtrTabDev.size(); d++) {   // 2026-09-04 per-device tables
            cudaSetDevice(devices[d]);
            if (it->second.dPtrTabDev[d]) cudaFree(it->second.dPtrTabDev[d]);
            if (it->second.dCompDev[d]) cudaFree(it->second.dCompDev[d]);
            if (it->second.dLimbDev[d]) cudaFree(it->second.dLimbDev[d]);
        }
        if (devices.size() > 1) cudaSetDevice(devices[0]);
        if (it->second.hStage) cudaFreeHost(it->second.hStage);
        compPool.erase(it);
    };
    auto compPoolGet = [&](uint32_t lvl) -> PoolLvl& {
        auto it = compPool.find(lvl);
        if (it == compPool.end()) {
            while (compPool.size() >= (size_t)poolLevels) {   // LRU cap: --pool-levels (stock 2) levels resident
                auto lru = compPool.begin();
                for (auto p = compPool.begin(); p != compPool.end(); ++p)
                    if (p->second.lastUse < lru->second.lastUse) lru = p;
                compPoolDrop(lru->first);
            }
            auto tPool = Clock::now();               // S3.3: time the (re)build; never measured before
            PoolLvl P;
            P.nLimbs = (int)(depth + 1 - lvl);
            const int POOLN = BS * (lanesBlock ? 2 : 1);   // T1c: lo/hi half per baby slot
            std::vector<double> zeros(SLOTS, 0.0);
            auto& cpuCcT = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
            auto& gctxT = std::any_cast<FIDESlib::CKKS::Context&>(cc->gpu);
            std::vector<uint64_t*> hostPtrs;
            hostPtrs.reserve((size_t)BS * P.nLimbs);
            std::vector<int> limbDevIdx(P.nLimbs, -1);   // 2026-09-04: global limb id -> partition index
            for (int s = 0; s < POOLN; s++) {
                auto w = cc->MakeCKKSPackedPlaintext(zeros, 1, lvl);    // metadata template at lvl
                const auto& wImpl = std::any_cast<const lbcrypto::Plaintext&>(w->cpu);
                FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cpuCcT, wImpl);
                auto gpt = std::make_shared<FIDESlib::CKKS::Plaintext>(gctxT, raw);
                // MULTI-GPU (2026-09-01). This used to be `GPU.at(0)` with the
                // limb index running to P.nLimbs, which assumes every limb of
                // the plaintext lives on device 0. FIDESlib SHARDS limbs across
                // the device list, so with --devices 0,1 that throws
                // vector::_M_range_check the moment the index passes device 0's
                // share -- observed exactly, "__n (which is 12) >= size (12)"
                // at stage L0.tm.matvecBatch.win. Nobody hit it before because
                // the older mgpu fork predates --compressed-store entirely.
                // Walk every partition in device order instead.
                {
                    // 2026-09-02: place every limb by its GLOBAL id. FIDESlib
                    // assigns limb i to device i % ndev (Context.cu:248), so the
                    // (dev, k) walk order is NOT the RNS order under sharding.
                    // LimbPartition::meta[k].id is the global id of limb[k]
                    // (LimbPartition.cuh:36/43; limbs are dropped from the END,
                    // LimbPartition.cu:2306, so k indexes both vectors alike).
                    // Self-checking: each id in [0, nLimbs) must be filled exactly
                    // once, else fatal -- a permuted pool must never run.
                    const size_t base = hostPtrs.size();
                    hostPtrs.resize(base + (size_t)P.nLimbs, nullptr);
                    int filled = 0;
                    for (size_t dev = 0; dev < gpt->c0.GPU.size(); dev++) {
                        auto& part = gpt->c0.GPU.at(dev);
                        for (size_t k = 0; k < part.limb.size(); k++) {
                            auto* lb = std::get_if<FIDESlib::CKKS::Limb<uint64_t>>(&part.limb.at(k));
                            if (!lb) { std::cerr << "{\"fatal\":\"compressed-store: non-u64 limb\"}\n"; std::exit(2); }
                            const int gid = (k < part.meta.size()) ? part.meta.at(k).id : -1;
                            // Limb<T>::primeid carries the same global id (LimbPartition.cu:221-223);
                            // the two must agree or the layout premise is broken.
                            if (gid < 0 || gid >= P.nLimbs || hostPtrs[base + gid] != nullptr || lb->primeid != gid) {
                                std::cerr << "{\"fatal\":\"compressed-store: limb id out of range or duplicate\",\"dev\":"
                                          << dev << ",\"k\":" << k << ",\"gid\":" << gid << ",\"primeid\":" << lb->primeid << ",\"nLimbs\":" << P.nLimbs
                                          << ",\"metaSize\":" << part.meta.size() << "}\n";
                                std::exit(2);
                            }
                            hostPtrs[base + gid] = lb->v.data; filled++;
                            limbDevIdx[gid] = (int)dev;
                        }
                    }
                    if (filled != P.nLimbs) {
                        // Loud, not silent: a short pool would leave the store
                        // reading uninitialised device pointers, which is the
                        // silent-wrongness class this repo keeps paying for.
                        std::cerr << "{\"fatal\":\"compressed-store: limb shard short\",\"got\":"
                                  << filled << ",\"want\":" << P.nLimbs
                                  << ",\"devices\":" << gpt->c0.GPU.size() << "}\n";
                        std::exit(2);
                    }
                }
                uint32_t h = cc->RegisterDevicePlaintext(std::shared_ptr<void>(gpt));
                w->gpu = h; w->loaded = true;
                P.wraps.push_back(w); P.gpts.push_back(std::move(gpt)); P.handles.push_back(h);
            }
            // Drain every device: the zero-fill loads of the pool plaintexts ran
            // on each partition's own streams. Then pin the pool's own buffers
            // (pointer table, compressed staging) to devices[0] so the launch
            // device is deterministic rather than level-dependent.
            syncAllDevices();
            if (devices.size() > 1) CUDA_CHECK(cudaSetDevice(devices[0]));
            const size_t compWords = (size_t)POOLN * P.nLimbs * COMP_CW;
            if (devices.size() > 1) {
                // 2026-09-04: per-device expand (fhe_ssm_expand_limbs_dev). Every
                // device gets its own pointer table, compressed staging and
                // limb->partition table; the single-device fields stay unused.
                for (int l = 0; l < P.nLimbs; l++) {
                    if (limbDevIdx[l] < 0 || limbDevIdx[l] >= (int)devices.size()) {
                        std::cerr << "{\"fatal\":\"compressed-store: limb " << l << " has no partition index\"}\n";
                        std::exit(2);
                    }
                }
                P.dPtrTabDev.assign(devices.size(), nullptr);
                P.dCompDev.assign(devices.size(), nullptr);
                P.dLimbDev.assign(devices.size(), nullptr);
                for (size_t d = 0; d < devices.size(); d++) {
                    CUDA_CHECK(cudaSetDevice(devices[d]));
                    CUDA_CHECK(cudaMalloc(&P.dPtrTabDev[d], hostPtrs.size() * sizeof(uint64_t*)));
                    CUDA_CHECK(cudaMemcpy(P.dPtrTabDev[d], hostPtrs.data(), hostPtrs.size() * sizeof(uint64_t*),
                                          cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMalloc(&P.dCompDev[d], compWords * 8));
                    CUDA_CHECK(cudaMalloc(&P.dLimbDev[d], limbDevIdx.size() * sizeof(int)));
                    CUDA_CHECK(cudaMemcpy(P.dLimbDev[d], limbDevIdx.data(), limbDevIdx.size() * sizeof(int),
                                          cudaMemcpyHostToDevice));
                }
                CUDA_CHECK(cudaSetDevice(devices[0]));
                CUDA_CHECK(cudaMallocHost(&P.hStage, compWords * 8));
            } else {
                CUDA_CHECK(cudaMalloc(&P.dPtrTab, hostPtrs.size() * sizeof(uint64_t*)));
                CUDA_CHECK(cudaMemcpy(P.dPtrTab, hostPtrs.data(), hostPtrs.size() * sizeof(uint64_t*),
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMalloc(&P.dComp, compWords * 8));
                CUDA_CHECK(cudaMallocHost(&P.hStage, compWords * 8));
            }
            if (storePeerWrites && devices.size() > 1) {
                // RETIRED 2026-09-04 (kept for the record behind --store-peer-writes):
                // The expand kernel runs on the CURRENT device and writes peer
                // limb buffers through their unified addresses; that needs peer
                // access, which FIDESlib enables for its own use ("GPU P2P? 1")
                // -- make it explicit here rather than inherit it.
                int cur = 0; cudaGetDevice(&cur);
                for (int dv : devices) {
                    if (dv == cur) continue;
                    int can = 0;
                    if (cudaDeviceCanAccessPeer(&can, cur, dv) != cudaSuccess || !can) {
                        std::cerr << "{\"fatal\":\"compressed-store: no peer access " << cur << "->" << dv << "\"}\n"; std::exit(2);
                    }
                    cudaError_t pe = cudaDeviceEnablePeerAccess(dv, 0);
                    if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) {
                        std::cerr << "{\"fatal\":\"compressed-store: cudaDeviceEnablePeerAccess failed\"}\n"; std::exit(2);
                    }
                    cudaGetLastError();
                    // REVIEW 2026-09-03: FIDESlib's limb buffers come from
                    // cudaMallocAsync (CudaUtils.cu:376-423, the stream-ordered
                    // allocator). Peer accessibility of POOL memory is NOT
                    // granted by cudaDeviceEnablePeerAccess; it is set per pool
                    // with cudaMemPoolSetAccess (CUDA guide, "Device Accessibility
                    // for Multi-GPU Support"). Without this the expand kernel's
                    // first peer write is an illegal address. Verify on hardware
                    // with hpc_gpu_port/peer_pool_probe.cu before trusting a
                    // multi-device store run.
                    cudaMemPool_t mp = nullptr;
                    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mp, dv));
                    cudaMemAccessDesc ad{};
                    ad.location.type = cudaMemLocationTypeDevice;
                    ad.location.id = cur;
                    ad.flags = cudaMemAccessFlagsProtReadWrite;
                    CUDA_CHECK(cudaMemPoolSetAccess(mp, &ad, 1));
                }
            }
            it = compPool.emplace(lvl, std::move(P)).first;
            poolBuilds++; poolBuildMs += msSince(tPool);
            vramTraceLine("poolBuild.lvl" + std::to_string(lvl));
        }
        it->second.lastUse = ++compPoolClock;
        return it->second;
    };
    // ---- --store-file (2026-09-02): persist the compressed store --------
    // The store is a function of (moduli chain, ring geometry, layout flags,
    // bundle) and of the level trace -- never of any key. Building it costs a
    // full pass (3,510 s at 2^17: prov_a2 console.log line 11), and every
    // process start paid it again. The stamp below refuses a file built for
    // a different configuration LOUDLY (a silent rebuild would hide an hour);
    // loaded entries are additionally re-encoded and compared, the first
    // --store-verify-samples hits, so a corrupt file cannot serve.
    bool storeLoaded = false, storeDirty = false, storeSaved = false;
    bool storeBaseExists = false;                 // a base file was on disk at start
    long long storeVerifyOk = 0, storeVerifyBad = 0; double storeVerifyMs = 0;
    long long storeAppends = 0, storeJournalEntries = 0;
    std::map<std::tuple<int, int, uint32_t>, int> storeVerifySeen;   // (layer, op, lvl) -> samples done
    std::vector<CompKey> storeNewKeys;            // entries added since the last save/append
    std::string storeStamp;
    // Content hash of the served bundle (+ its index): the stamp used to bind
    // the store to the bundle by tag and byte size only, so a re-export of
    // identical size would have been served from stale plaintexts (review
    // 2026-09-03). FNV-1a over 8-byte words; ~2 s for 3.4 GB from page cache.
    auto hashFile = [](const std::string& p) -> uint64_t {
        std::ifstream f(p, std::ios::binary);
        uint64_t h = 1469598103934665603ULL;
        if (!f) return 0;
        std::vector<char> buf(4u << 20);
        while (f) {
            f.read(buf.data(), (std::streamsize)buf.size());
            const size_t n = (size_t)f.gcount();
            size_t i = 0;
            for (; i + 8 <= n; i += 8) { uint64_t w; std::memcpy(&w, buf.data() + i, 8); h ^= w; h *= 1099511628211ULL; }
            for (; i < n; i++) { h ^= (unsigned char)buf[i]; h *= 1099511628211ULL; }
        }
        return h;
    };
    {
        std::ostringstream st;
        auto& cpuCcS = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
        uint64_t qh = 1469598103934665603ULL;
        const auto& prs = cpuCcS->GetCryptoParameters()->GetElementParams()->GetParams();
        for (const auto& p : prs) { qh ^= (uint64_t)p->GetModulus().ConvertToInt(); qh *= 1099511628211ULL; }
        st << "fhe-ssm-store-v1|ring=" << cc->GetRingDimension() << "|slots=" << SLOTS << "|rep=" << REP
           << "|dpad=" << Dpad << "|bs=" << BS << "|depth=" << depth << "|nq=" << prs.size() << "|qhash=" << qh
           << "|lanesBlock=" << (lanesBlock ? 1 : 0) << "|interleave=" << (interleave ? 1 : 0)
           << "|scale=" << scaleBits << "|tag=" << tag
           << "|bundleBytes=" << fileBytes(bundleDir + "/bundle_" + tag + ".bin")
           << "|bundleHash=" << (haveBundle && !storeFilePath.empty() ? hashFile(bundleDir + "/bundle_" + tag + ".bin") : 0)
           << "|indexHash=" << (haveBundle && !storeFilePath.empty() ? hashFile(bundleDir + "/bundle_" + tag + ".index.txt") : 0)
           << "|diagEps=" << diagEps << "|band=" << bandWidth << "|rowOff=" << bundleRowOffset
           << "|compCw=" << COMP_CW;
        // S3.3 (I6/I8): every flag that changes the LEVEL TRACE is part of the
        // store identity -- appended ONLY when a non-stock lever is on, so a
        // stock run's stamp is byte-identical to before (existing store files
        // keep loading) while a store built under another schedule is REFUSED
        // (stamps differ), never silently missed. --boot-floor/--matvec-margin/
        // --rsqrt-iters enter the same refresh rule and ride along in that case.
        // REVIEW 2026-09-03: stamp the EFFECTIVE quantities the refresh rule consumes
        // (default -1 and an explicit default give the same trace, so they must give
        // the same stamp), and pfEntry only when it is read (parent-first).
        if (schedParentFirst || newtonDepth2)
            st << "|sched=" << schedule << (schedParentFirst ? "/" + pfEntry : std::string())
               << "|nd2=" << (newtonDepth2 ? 1 : 0)
               << "|bootFloor=" << (bootFloor >= 0 ? bootFloor : 4)
               << "|margin=" << (matvecMargin >= 0 ? matvecMargin : 6)
               << "|rsqrtIters=" << rsqrtItersOverride;
        if (blockParallel) st << "|block=parallel";    // S3.3 Tier B: a different circuit -> a different store identity
        if (canonicalCarry) st << "|canon=1";            // 2026-09-04: a different level trajectory -> a different store
        storeStamp = st.str();
    }
    // One journal record = key(7 x int32) + len(u64) + data + FNV(record).
    auto recordHash = [](const int32_t* kf, uint64_t len, const uint64_t* data) -> uint64_t {
        uint64_t h = 1469598103934665603ULL;
        auto mixw = [&](uint64_t v) { h ^= v; h *= 1099511628211ULL; };
        for (int q = 0; q < 7; q++) mixw((uint64_t)(uint32_t)kf[q]);
        mixw(len);
        for (uint64_t i = 0; i < len; i++) mixw(data[i]);
        return h;
    };
    // --store-file journal (review 2026-09-03). The base file is written ONCE
    // (a full rewrite of 60-300 GB); every later addition -- the level-drift
    // entries a stateful session keeps producing -- is APPENDED as
    // self-checksummed records to PATH.journal after every request that added
    // any. Nothing is ever rewritten inside a session, a killed process loses
    // at most the last request's entries, and no 2x disk is needed.
    auto storeAppendFn = [&]() -> bool {
        if (storeFilePath.empty() || storeNewKeys.empty()) return false;
        auto t0 = Clock::now();
        const std::string jp = storeFilePath + ".journal";
        std::ofstream f(jp, std::ios::binary | std::ios::app);
        if (!f) { std::cerr << "{\"warn\":\"store-file: cannot append to " << jp << "\"}" << std::endl; return false; }
        uint64_t bytes = 0, n = 0;
        for (const CompKey& k : storeNewKeys) {
            auto it = compStore.find(k);
            if (it == compStore.end()) continue;
            const int32_t kf[7] = {k.layer, k.op, k.chunk, k.rowOff, k.diag, k.half, (int32_t)k.lvl};
            const uint64_t len = it->second.size();
            f.write((const char*)kf, sizeof kf); f.write((const char*)&len, 8);
            if (len) f.write((const char*)it->second.data(), (std::streamsize)(len * 8));
            const uint64_t rh = recordHash(kf, len, it->second.data());
            f.write((const char*)&rh, 8);
            bytes += len * 8; n++;
        }
        f.flush(); f.close();
        if (!f) { std::cerr << "{\"warn\":\"store-file: journal append failed on " << jp << "\"}" << std::endl; return false; }
        storeNewKeys.clear(); storeDirty = false; storeAppends++; storeJournalEntries += (long long)n;
        std::cout << "{\"storeAppend\":true,\"path\":\"" << jp << "\",\"entries\":" << n
                  << ",\"dataBytes\":" << bytes << ",\"ms\":" << (int)msSince(t0) << "}" << std::endl;
        return true;
    };
    auto storeSaveFn = [&]() -> bool {
        if (storeFilePath.empty()) return false;
        auto t0 = Clock::now();
        const std::string tmp = storeFilePath + ".tmp";
        {
            uint64_t need = 64;
            for (const auto& kv : compStore) need += 44 + kv.second.size() * 8;
            struct statvfs sv;
            const std::string dir = std::filesystem::path(storeFilePath).parent_path().string();
            if (statvfs(dir.empty() ? "." : dir.c_str(), &sv) == 0) {
                const uint64_t freeB = (uint64_t)sv.f_bavail * (uint64_t)sv.f_frsize;
                std::cout << "{\"storeSave\":\"begin\",\"entries\":" << compStore.size() << ",\"bytes\":" << need
                          << ",\"freeBytes\":" << freeB << "}" << std::endl;
                if (freeB < need + (1ull << 30)) {
                    std::cerr << "{\"warn\":\"store-file: not enough free space for the base save\",\"need\":" << need
                              << ",\"free\":" << freeB << "}" << std::endl;
                    return false;
                }
            }
        }
        std::ofstream f(tmp, std::ios::binary);
        if (!f) { std::cerr << "{\"warn\":\"store-file: cannot write " << tmp << "\"}" << std::endl; return false; }
        uint64_t hsh = 1469598103934665603ULL;
        auto mixw = [&](uint64_t v) { hsh ^= v; hsh *= 1099511628211ULL; };
        char magic[16] = {'F','H','E','S','S','M','S','T','O','R','E','1',0,0,0,0};
        f.write(magic, 16);
        const uint64_t sl = storeStamp.size(); f.write((const char*)&sl, 8); f.write(storeStamp.data(), (std::streamsize)sl);
        const uint64_t n = compStore.size(); f.write((const char*)&n, 8);
        uint64_t bytes = 0;
        for (const auto& kv : compStore) {
            const int32_t kf[7] = {kv.first.layer, kv.first.op, kv.first.chunk, kv.first.rowOff,
                                   kv.first.diag, kv.first.half, (int32_t)kv.first.lvl};
            f.write((const char*)kf, sizeof kf);
            for (int q = 0; q < 7; q++) mixw((uint64_t)(uint32_t)kf[q]);
            const uint64_t len = kv.second.size(); f.write((const char*)&len, 8); mixw(len);
            if (len) f.write((const char*)kv.second.data(), (std::streamsize)(len * 8));
            for (uint64_t w : kv.second) mixw(w);
            bytes += len * 8;
        }
        f.write((const char*)&hsh, 8);
        f.close();
        if (!f) { std::cerr << "{\"warn\":\"store-file: write failed on " << tmp << "\"}" << std::endl; std::remove(tmp.c_str()); return false; }
        if (std::rename(tmp.c_str(), storeFilePath.c_str()) != 0) {
            std::cerr << "{\"warn\":\"store-file: rename failed " << tmp << " -> " << storeFilePath << "\"}" << std::endl; std::remove(tmp.c_str()); return false;
        }
        storeSaved = true; storeDirty = false; storeBaseExists = true; storeNewKeys.clear();
        std::cout << "{\"storeSave\":true,\"path\":\"" << storeFilePath << "\",\"entries\":" << n
                  << ",\"dataBytes\":" << bytes << ",\"ms\":" << (int)msSince(t0) << "}" << std::endl;
        return true;
    };
    auto storeLoadFn = [&]() -> bool {
        if (storeFilePath.empty()) return false;
        std::ifstream f(storeFilePath, std::ios::binary);
        if (!f) {
            std::cout << "{\"storeLoad\":false,\"reason\":\"absent\",\"path\":\"" << storeFilePath
                      << "\",\"willSaveAfterFirstPass\":true}" << std::endl;
            return false;
        }
        auto t0 = Clock::now();
        auto die = [&](const std::string& why) {
            std::cerr << "{\"fatal\":\"store-file: " << why << "\",\"path\":\"" << storeFilePath
                      << "\",\"hint\":\"delete or rename the file to rebuild; a rebuild costs a full pass\"}" << std::endl;
            std::exit(2);
        };
        char magic[16]; f.read(magic, 16);
        if (!f || std::string(magic, 12) != "FHESSMSTORE1") die("bad magic");
        uint64_t sl = 0; f.read((char*)&sl, 8);
        if (!f || sl == 0 || sl > 4096) die("bad stamp length");
        std::string stamp(sl, '\0'); f.read(&stamp[0], (std::streamsize)sl);
        if (!f) die("truncated stamp");
        if (stamp != storeStamp) {
            std::cerr << "{\"storeStampFile\":\"" << stamp << "\",\"storeStampThisRun\":\"" << storeStamp << "\"}" << std::endl;
            die("built for a DIFFERENT configuration (stamps above differ)");
        }
        uint64_t n = 0; f.read((char*)&n, 8);
        if (!f || n > 100000000ULL) die("bad entry count");
        uint64_t hsh = 1469598103934665603ULL;
        auto mixw = [&](uint64_t v) { hsh ^= v; hsh *= 1099511628211ULL; };
        uint64_t bytes = 0, dead = 0;
        for (uint64_t i = 0; i < n; i++) {
            int32_t kf[7]; f.read((char*)kf, sizeof kf);
            uint64_t len = 0; f.read((char*)&len, 8);
            if (!f) die("truncated entry header");
            for (int q = 0; q < 7; q++) mixw((uint64_t)(uint32_t)kf[q]);
            mixw(len);
            if (kf[6] < 0 || (uint32_t)kf[6] > depth) die("entry level out of range");
            const uint64_t wantLimbs = (uint64_t)(depth + 1 - (uint32_t)kf[6]);
            if (len != 0 && len != wantLimbs * COMP_CW) die("entry length does not match its level");
            std::vector<uint64_t> v(len);
            if (len) f.read((char*)v.data(), (std::streamsize)(len * 8));
            if (!f) die("truncated entry data");
            for (uint64_t w : v) mixw(w);
            bytes += len * 8; if (!len) dead++;
            CompKey key{kf[0], kf[1], kf[2], kf[3], kf[4], kf[5], (uint32_t)kf[6]};
            if (!compStore.emplace(key, std::move(v)).second) die("duplicate entry key");
        }
        uint64_t fileHash = 0; f.read((char*)&fileHash, 8);
        if (!f || fileHash != hsh) die("checksum mismatch");
        storeBytes += bytes; storeLoaded = true; storeBaseExists = true;
        std::cout << "{\"storeLoad\":true,\"path\":\"" << storeFilePath << "\",\"entries\":" << n
                  << ",\"dead\":" << dead << ",\"dataBytes\":" << bytes << ",\"ms\":" << (int)msSince(t0)
                  << ",\"verifySamplesPerPrefix\":" << storeVerifySamples << "}" << std::endl;
        return true;
    };
    auto storeJournalLoadFn = [&]() {
        if (storeFilePath.empty()) return;
        const std::string jp = storeFilePath + ".journal";
        std::ifstream f(jp, std::ios::binary);
        if (!f) return;
        auto t0 = Clock::now();
        uint64_t good = 0, dup = 0, bytes = 0; std::streamoff goodEnd = 0;
        for (;;) {
            int32_t kf[7]; uint64_t len = 0;
            f.read((char*)kf, sizeof kf); if (!f) break;
            f.read((char*)&len, 8); if (!f) break;
            if (kf[6] < 0 || (uint32_t)kf[6] > depth) break;
            const uint64_t wantLimbs = (uint64_t)(depth + 1 - (uint32_t)kf[6]);
            if (len != 0 && len != wantLimbs * COMP_CW) break;
            std::vector<uint64_t> v(len);
            if (len) f.read((char*)v.data(), (std::streamsize)(len * 8));
            if (!f) break;
            uint64_t rh = 0; f.read((char*)&rh, 8); if (!f) break;
            if (rh != recordHash(kf, len, v.data())) break;
            goodEnd = f.tellg();
            CompKey key{kf[0], kf[1], kf[2], kf[3], kf[4], kf[5], (uint32_t)kf[6]};
            if (compStore.emplace(key, std::move(v)).second) { good++; bytes += len * 8; } else dup++;
        }
        f.close();
        std::error_code ec;
        const auto total = std::filesystem::file_size(jp, ec);
        if (!ec && (std::streamoff)total != goodEnd) {
            // a torn tail (killed mid-append): keep the good prefix, drop the rest
            std::filesystem::resize_file(jp, (uintmax_t)goodEnd, ec);
            std::cerr << "{\"warn\":\"store-file: journal had a torn tail; truncated\",\"keptBytes\":" << goodEnd
                      << ",\"droppedBytes\":" << ((std::streamoff)total - goodEnd) << "}" << std::endl;
        }
        storeBytes += bytes; storeJournalEntries += (long long)good;
        if (good || dup) storeLoaded = true;
        std::cout << "{\"storeJournalLoad\":true,\"path\":\"" << jp << "\",\"entries\":" << good
                  << ",\"duplicates\":" << dup << ",\"dataBytes\":" << bytes << ",\"ms\":" << (int)msSince(t0) << "}" << std::endl;
    };
    // Persist whatever is new: the base file once, journal appends thereafter.
    auto storePersistFn = [&]() {
        if (storeFilePath.empty() || !storeDirty) return;
        if (!storeBaseExists) storeSaveFn(); else storeAppendFn();
    };
    if (compressedStore && !clientCrypto) { storeLoadFn(); storeJournalLoadFn(); }
#endif

    // pack a length-<=Dpad channel vector into SLOTS with REP replicas
    auto packCh = [&](const double* v, int n) {
        std::vector<double> s(SLOTS, 0.0);
        if (interleave)
            for (int i = 0; i < n; i++)
                for (uint32_t t = 0; t < REP; t++) s[(size_t)i * REP + t] = v[i];
        else
            for (uint32_t r = 0; r < REP; r++)
                for (int i = 0; i < n; i++) s[r * Dpad + i] = v[i];
        return s;
    };
    // ---- STAGE 2 helpers -------------------------------------------------
    // packTok: N token rows (each length d) -> ONE interleaved slot vector.
    //   slot = channel*REP + tokenLane.  Lanes >= N are left ZERO, which is the
    //   correct "no such token" value for every op (the scan mask keeps them out
    //   of the prefix, and the norm's channel reduction sums a lane's own
    //   channels only, so an empty lane simply produces 0).
    auto packTok = [&](const double* rows, int n, int d_) {
        std::vector<double> sv(SLOTS, 0.0);
        for (int t = 0; t < n; t++)
            for (int i = 0; i < d_; i++) sv[(size_t)i * REP + t] = rows[(size_t)t * d_ + i];
        return sv;
    };
    // decrypt ONE token lane's channel vector (index by CHANNEL, like decCh).
    auto decLane = [&](TrackedCt& c, int lane) -> std::vector<double> {
        auto t0 = Clock::now();
        Plaintext p; cc->Decrypt(keys.secretKey, c.ct, &p);
        p->SetLength(SLOTS); auto v = p->GetRealPackedValue();
        decMs += msSince(t0);
        std::vector<double> out(SLOTS, 0.0);
        for (uint32_t i = 0; i < Dpad && (size_t)i * REP + lane < SLOTS; i++)
            out[i] = v[(size_t)i * REP + lane];
        return out;
    };
    // per-channel coefficient replicated across lanes, but ZEROED on the first
    // `zeroLanes` lanes. Folding the mask into the coefficient plaintext means a
    // masked shift costs ONE ct x pt multiply, not a separate masking multiply.
    auto packLaneMasked = [&](const std::vector<double>& coef, int d_, int zeroLanes) {
        std::vector<double> sv(SLOTS, 0.0);
        for (int i = 0; i < d_; i++)
            for (uint32_t t = (uint32_t)zeroLanes; t < REP; t++)
                sv[(size_t)i * REP + t] = coef[i];
        return sv;
    };
    // per-channel coefficient on ONE lane only (all other lanes zero) -- the
    // block-boundary masks for multi-block Stage 2.
    auto packLaneOnly = [&](const std::vector<double>& coef, int d_, int lane) {
        std::vector<double> sv(SLOTS, 0.0);
        for (int i = 0; i < d_; i++) sv[(size_t)i * REP + lane] = coef[i];
        return sv;
    };
    // shift by `j` TOKEN lanes: lane t receives lane t-j. Lanes are contiguous
    // (stride 1), so this is a ring rotation by -j, NOT by -j*REP.
    auto rotTok = [&](const Ciphertext<DCRTPoly>& ct, int j) {
        return cc->EvalRotate(ct, -j);
    };
    auto ptCh = [&](const std::vector<double>& packed, uint32_t used) -> Plaintext {
        return mkPt(packed, used, __LINE__);      // at operand level
    };
    auto encCh = [&](const std::vector<double>& packed) -> TrackedCt {
        auto t = Clock::now();
        auto pt = cc->MakeCKKSPackedPlaintext(packed);
        TrackedCt r{cc->Encrypt(keys.publicKey, pt), 0};
        encMs += msSince(t); return r;
    };
    auto decCh = [&](TrackedCt& c) -> std::vector<double> {
        auto t = Clock::now();
        Plaintext p; cc->Decrypt(keys.secretKey, c.ct, &p);
        p->SetLength(SLOTS); auto v = p->GetRealPackedValue();
        decMs += msSince(t);
        if (!interleave) return v;
        // De-interleave token 0: channel i lives at slot i*REP + 0. Every caller
        // indexes the result by CHANNEL, so this keeps all existing call sites
        // (dbgDec, wvCheck's hv/pv, rmsnorm's reference, the head matvec) correct.
        std::vector<double> out(SLOTS, 0.0);
        for (uint32_t i = 0; i < Dpad && (size_t)i * REP < SLOTS; i++)
            out[i] = v[(size_t)i * REP];
        return out;
    };
    // DEBUG (probe only, zero cost otherwise): attempt an intermediate decrypt
    // and report magnitude + tracked level, CATCHING the decode failure rather
    // than aborting -- so one probe run localizes exactly where the ciphertext
    // stops being decryptable, instead of only revealing that it did.
    auto dbgDec = [&](TrackedCt& c, const std::string& label) {
        if (!probe) return;
        try {
            Plaintext p; cc->Decrypt(keys.secretKey, c.ct, &p);
            p->SetLength(SLOTS); auto v = p->GetRealPackedValue();
            double m = 0; for (int i = 0; i < d; i++) m = std::max(m, std::abs(v[i]));
            std::cout << "{\"dbg\":\"" << label << "\",\"used\":" << c.level()
                      << ",\"maxAbs\":" << m << ",\"boots\":" << boots << "}" << std::endl;
        } catch (const std::exception&) {
            std::cout << "{\"dbg\":\"" << label << "\",\"used\":" << c.level()
                      << ",\"DECODE_FAIL\":true,\"boots\":" << boots << "}" << std::endl;
        }
    };

    // --block-probe: decrypt ONE block ciphertext and report per-lane magnitude
    // spread + cryptographic scale factor + level. The magnitude-divergence
    // hypothesis predicts maxAbs fanning out with block index; the scale-drift
    // hypothesis predicts scalingFactor / noiseScaleDeg diverging across blocks
    // at equal levels. Catches decode failure per block (does not abort).
    auto blockDec = [&](TrackedCt& c, int layer, int blk, const char* site) {
        try {
            Plaintext p; cc->Decrypt(keys.secretKey, c.ct, &p);
            p->SetLength(SLOTS); auto v = p->GetRealPackedValue();
            // per-lane maxAbs over channels: lane t at slots i*REP + t
            double m0 = 0, mLast = 0, mAll = 0; long long nf = 0;
            const int P = packTokens > 0 ? packTokens : 1;
            for (uint32_t i = 0; i < Dpad && (size_t)i * REP + (P - 1) < SLOTS; i++) {
                double a0 = v[(size_t)i * REP], aL = v[(size_t)i * REP + (P - 1)];
                if (!std::isfinite(a0) || !std::isfinite(aL)) { nf++; continue; }
                m0 = std::max(m0, std::abs(a0)); mLast = std::max(mLast, std::abs(aL));
            }
            for (size_t s2 = 0; s2 < v.size(); s2++)
                if (std::isfinite(v[s2])) mAll = std::max(mAll, std::abs(v[s2])); else nf++;
            std::cout << "{\"blockProbe\":true,\"layer\":" << layer << ",\"block\":" << blk
                      << ",\"site\":\"" << site << "\",\"level\":" << c.level()
                      << ",\"noiseDeg\":" << c.ct->GetNoiseScaleDeg()
                      << ",\"maxAbs\":" << mAll << ",\"maxAbsLane0\":" << m0
                      << ",\"maxAbsLaneLast\":" << mLast << ",\"nonFinite\":" << nf
                      << ",\"boots\":" << boots << "}" << std::endl;
        } catch (const std::exception&) {
            std::cout << "{\"blockProbe\":true,\"layer\":" << layer << ",\"block\":" << blk
                      << ",\"site\":\"" << site << "\",\"level\":" << c.level()
                      << ",\"DECODE_FAIL\":true,\"boots\":" << boots << "}" << std::endl;
        }
    };

    // ================= DEMO ciphertext files (blocker 6) ====================
    // Serialized as PREFIX.0 .. PREFIX.(count-1) (one OpenFHE BINARY blob per
    // block/token ciphertext) plus a one-line PREFIX.meta JSON describing the
    // packing, so every mode can verify it was invoked with matching flags.
    // Only the HOST-side lbcrypto objects are ever serialized; a GPU-resident
    // ciphertext is brought home first through the same store() path the
    // library's own Decrypt uses (found in api/CryptoContext.cpp::Decrypt --
    // documented in DEMO_README.md).
#ifdef FHE_SSM_DEMO_SER
    // GPU -> host sync. FIDESlib keeps a host twin in ct->cpu; for a
    // GPU-resident ciphertext that twin is STALE (it is the full-tower host
    // object from encryption/load time, copied along by every eval op), and
    // GetOpenFHECipherText overwrites it with the live GPU limbs, dropping the
    // surplus towers and fixing level/scalingFactor/noiseDeg/keyTag/slots.
    // The full-tower container is exactly why the library's own cpu<gpu
    // "dummy re-encrypt with the SECRET key" fallback can never trigger here:
    // our containers descend from fresh level-0 encryptions, so the host side
    // always has at least as many towers as the GPU side. No secret key
    // involved anywhere on this path.
    auto hostCt = [&](TrackedCt& c) -> lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& {
        c.ct->EnsureLazyCPUCopy();       // detach the shared stale twin first
        auto& host = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(c.ct->cpu);
        if (c.ct->loaded) {
            auto gpuCt = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(
                cc->GetDeviceCiphertext(c.ct->gpu));
            FIDESlib::CKKS::RawCipherText raw;
            gpuCt->store(raw);
            FIDESlib::CKKS::GetOpenFHECipherText(host, raw);
        }
        return host;
    };
    auto saveCt = [&](TrackedCt& c, const std::string& path) -> long long {
        auto& host = hostCt(c);
        if (!lbcrypto::Serial::SerializeToFile(path, host, lbcrypto::SerType::BINARY))
            throw std::runtime_error("ct serialize failed: " + path);
        return fileBytes(path);
    };
    auto loadCt = [&](const std::string& path) -> TrackedCt {
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly> host;
        if (!lbcrypto::Serial::DeserializeFromFile(path, host, lbcrypto::SerType::BINARY))
            throw std::runtime_error("ct deserialize failed: " + path);
        // Wrap into a fideslib ciphertext exactly the way the api itself
        // builds them (public members; ctor takes const CryptoContext&&).
        // loaded=false => the first eval op auto-stages it to the GPU via
        // LoadCiphertext; on the CPU-only client it just stays host-side.
        CryptoContext<DCRTPoly> ccRef = cc;
        Ciphertext<DCRTPoly> fc =
            std::make_shared<CiphertextImpl<DCRTPoly>>(std::move(ccRef));
        fc->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(std::move(host));
        fc->need_lazy_copy = false;
        fc->loaded = false;
        fc->gpu = 0;
        return TrackedCt{fc, 0};
    };
    // meta sidecar: written by us, parsed by us -- a flat one-line JSON with
    // integer values only, so a string-find parser is exact, not heuristic.
    auto metaInt = [](const std::string& text, const std::string& key, long long dflt) -> long long {
        auto pos = text.find("\"" + key + "\":");
        if (pos == std::string::npos) return dflt;
        return std::atoll(text.c_str() + pos + key.size() + 3);
    };
    auto writeCtMeta = [&](const std::string& prefix, int count, uint32_t tokReal, uint32_t lvl) {
        std::ofstream mf(prefix + ".meta");
        mf << "{\"count\":" << count << ",\"packTokens\":" << packTokens
           << ",\"tokensReal\":" << tokReal << ",\"dpad\":" << Dpad
           << ",\"interleave\":" << (interleave ? 1 : 0)
           << ",\"lanes\":" << clientLanes
           << ",\"ringDim\":" << cc->GetRingDimension() << ",\"level\":" << lvl << "}\n";
    };
    auto readCtMeta = [&](const std::string& prefix) -> std::string {
        std::ifstream mf(prefix + ".meta");
        if (!mf) throw std::runtime_error("missing ct meta: " + prefix + ".meta");
        std::stringstream ss; ss << mf.rdbuf(); return ss.str();
    };
#endif

    // ---- DEMO --enc-in: client encrypt step (exits before the forward) -----
    // Reads T x d float64 rows (raw little-endian, T from --tokens), encrypts
    // them EXACTLY as the bundle-input path does (packTok multi-block layout /
    // packCh per-token layout), serializes to FILE.ct.N + FILE.ct.meta, exits.
    // Requires only the public key; runs CPU-only (devices == {}).
    if (!encInPath.empty()) {
#ifdef FHE_SSM_DEMO_SER
        try {
            auto tEnc0 = Clock::now();
            std::ifstream rf(encInPath, std::ios::binary | std::ios::ate);
            if (!rf) { std::cerr << "{\"fatal\":\"enc-in: cannot open " << encInPath << "\"}\n"; return 2; }
            const long long haveB = (long long)rf.tellg();
            const long long needB = (long long)T * d * (long long)sizeof(double);
            if (T < 1 || haveB < needB) {
                std::cerr << "{\"fatal\":\"enc-in: short file\",\"haveBytes\":" << haveB
                          << ",\"needBytes\":" << needB << ",\"tokens\":" << T
                          << ",\"d\":" << d << "}" << std::endl; return 2;
            }
            rf.seekg(0);
            const long long needLanesB = (long long)clientLanes * T * d * (long long)sizeof(double);
            if (clientLanes > 1 && haveB < needLanesB) {
                std::cerr << "{\"fatal\":\"enc-in lanes: short file\",\"haveBytes\":" << haveB
                          << ",\"needBytes\":" << needLanesB << ",\"lanes\":" << clientLanes
                          << "}" << std::endl; return 2;
            }
            std::vector<double> rowsIn((size_t)(clientLanes > 1 ? needLanesB : needB) / sizeof(double));
            rf.read(reinterpret_cast<char*>(rowsIn.data()),
                    (clientLanes > 1 ? needLanesB : needB));
            const std::string prefix = encInPath + ".ct";
            int nOut = 0; long long bytes = 0;
            if (clientLanes > 1) {
                // LANE-MAJOR rows (lane r block = rows r*T..r*T+T-1). Per
                // token: slot rep*Dpad+i = lane (rep mod NL) channel i,
                // cyclic across all REP replicas (dead-lane bootstrap
                // hazard — same rationale as --lanes-full).
                if ((uint32_t)clientLanes > REP) {
                    std::cerr << "{\"fatal\":\"client-lanes > REP\",\"lanes\":" << clientLanes
                              << ",\"REP\":" << REP << "}" << std::endl; return 2;
                }
                nOut = (int)T;
                for (int t = 0; t < nOut; t++) {
                    std::vector<double> sv(SLOTS, 0.0);
                    for (uint32_t rep = 0; rep < REP; rep++) {
                        const int lane = (int)(rep % (uint32_t)clientLanes);
                        const double* rw = rowsIn.data() + ((size_t)lane * T + t) * d;
                        for (int i = 0; i < d; i++) sv[(size_t)rep * Dpad + i] = rw[i];
                    }
                    auto c = encCh(sv);
                    bytes += saveCt(c, prefix + "." + std::to_string(t));
                }
            } else if (packTokens > 1) {
                if (packTokens > (int)REP) {
                    std::cerr << "{\"fatal\":\"packTokens>REP\",\"packTokens\":" << packTokens
                              << ",\"REP\":" << REP << "}" << std::endl; return 2;
                }
                nOut = ((int)T + packTokens - 1) / packTokens;
                for (int b = 0; b < nOut; b++) {
                    // short final block: packTok zeroes lanes >= n, which is
                    // the circuit's own "no such token" value (see packTok).
                    int n = std::min(packTokens, (int)T - b * packTokens);
                    auto c = encCh(packTok(rowsIn.data() + (size_t)b * packTokens * d, n, d));
                    bytes += saveCt(c, prefix + "." + std::to_string(b));
                }
            } else {
                nOut = (int)T;
                for (int t = 0; t < nOut; t++) {
                    std::vector<double> row(rowsIn.begin() + (size_t)t * d,
                                            rowsIn.begin() + (size_t)t * d + d);
                    auto c = encCh(packCh(row.data(), d));
                    bytes += saveCt(c, prefix + "." + std::to_string(t));
                }
            }
            writeCtMeta(prefix, nOut, T, 0);
            std::cout << "{\"clientEnc\":true,\"file\":\"" << prefix << "\",\"count\":" << nOut
                      << ",\"tokens\":" << T << ",\"packTokens\":" << packTokens
                      << ",\"lanes\":" << clientLanes
                      << ",\"totalBytes\":" << bytes
                      << ",\"encMs\":" << (int)msSince(tEnc0) << "}" << std::endl;
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "{\"fatal\":\"enc-in failed\",\"what\":\"" << e.what() << "\"}\n"; return 2;
        }
#else
        std::cerr << "{\"fatal\":\"--enc-in requires a build with -DFHE_SSM_DEMO_SER "
                     "(see hpc_gpu_port/DEMO_README.md)\"}\n"; return 2;
#endif
    }

    // ---- DEMO --dec-out: client decrypt + plaintext unembedding (exits) ----
    // Deserializes the server's post-final-norm hidden-state ciphertexts,
    // decrypts with the secret key, applies the plaintext `head` matvec
    // exactly like the normal output loop, writes PREFIX.logits.f64
    // (tokensReal x vocab float64 rows) and prints one JSON line per token.
    // The logits_ref comparison is emitted only where the bundle HAS a row
    // for the position -- and it is only MEANINGFUL when the prompt replays
    // the bundle's own eval text; generated positions have no reference.
    if (!decOutPath.empty()) {
#ifdef FHE_SSM_DEMO_SER
        try {
            auto tDec0 = Clock::now();
            const std::string meta = readCtMeta(decOutPath);
            const int nCt = (int)metaInt(meta, "count", -1);
            const int mPack = (int)metaInt(meta, "packTokens", 0);
            const long long mTok = metaInt(meta, "tokensReal", -1);
            if (nCt <= 0 || mTok <= 0) {
                std::cerr << "{\"fatal\":\"dec-out: bad meta\",\"count\":" << nCt
                          << ",\"tokensReal\":" << mTok << "}" << std::endl; return 2;
            }
            if ((long long)metaInt(meta, "ringDim", 0) != (long long)cc->GetRingDimension()
                || (int)metaInt(meta, "dpad", 0) != (int)Dpad
                || (int)metaInt(meta, "interleave", -1) != (interleave ? 1 : 0)) {
                std::cerr << "{\"fatal\":\"dec-out: layout mismatch -- pass the SAME "
                             "--log-ring/--dpad/--interleave the server ran with\"}" << std::endl;
                return 2;
            }
            std::vector<TrackedCt> res(nCt);
            for (int i2 = 0; i2 < nCt; i2++) res[i2] = loadCt(decOutPath + "." + std::to_string(i2));
            int rowsH, colsH; auto headM = B.mat("head", rowsH, colsH);     // (V, d)
            const int refRows = B.has("logits_ref") ? B.shape("logits_ref")[0] : 0;
            const double* lRef = B.has("logits_ref") ? B.ptr("logits_ref") : nullptr;
            // multi-lane: lanes rides the ct meta from the enc step; each
            // decrypted slot vector carries NL conversations. Output rows are
            // tok-major, lane-minor: row index = tok*NL + lane.
            const int mLanes = std::max(1, (int)metaInt(meta, "lanes", 1));
            if (mLanes > 1 && (mPack > 1 || interleave)) {
                std::cerr << "{\"fatal\":\"dec-out: lanes>1 requires BLOCK layout, pack-tokens 1\"}\n";
                return 2;
            }
            std::vector<double> logitsAll((size_t)mTok * mLanes * V, 0.0);
            for (int tok = 0; tok < (int)mTok; tok++) {
                std::vector<double> hfv = (mPack > 1)
                    ? decLane(res[tok / mPack], tok % mPack)
                    : decCh(res[tok]);
                double mx = -1e300; int amax = 0;
                for (int lane = 0; lane < mLanes; lane++) {
                const double* hid = hfv.data() + (size_t)lane * Dpad;   // lane 0 == old path
                double lmx = -1e300; int lamax = 0;
                const size_t outRow = (size_t)tok * mLanes + lane;
                for (int r = 0; r < V; r++) {
                    double sAcc = 0;
                    for (int j = 0; j < d; j++) sAcc += headM[(size_t)r * colsH + j] * hid[j];
                    logitsAll[outRow * V + r] = sAcc;
                    if (sAcc > lmx) { lmx = sAcc; lamax = r; }
                }
                if (mLanes > 1)
                    std::cout << "{\"clientDec\":true,\"token\":" << tok << ",\"lane\":" << lane
                              << ",\"argmax\":" << lamax << ",\"maxLogit\":" << lmx << "}" << std::endl;
                if (lane == 0) { mx = lmx; amax = lamax; }
                }
                if (mLanes > 1) continue;   // per-lane lines above; skip the single-lane block
                if (lRef && tok < refRows) {
                    double e = 0, refmax = 0; long long nfLogit = 0;
                    for (int r = 0; r < V; r++) {
                        double lr = logitsAll[(size_t)tok * V + r];
                        double rr2 = lRef[(size_t)tok * V + r];
                        if (!std::isfinite(lr) || !std::isfinite(rr2)) { nfLogit++; continue; }
                        e = std::max(e, std::abs(lr - rr2));
                        refmax = std::max(refmax, std::abs(rr2));
                    }
                    std::cout << "{\"clientDec\":true,\"token\":" << tok << ",\"argmax\":" << amax
                              << ",\"maxLogit\":" << mx << ",\"nonFiniteLogits\":" << nfLogit
                              << ",\"maxAbsLogitErr\":" << e << ",\"refMaxAbsLogit\":" << refmax
                              << ",\"relLogitErr\":" << (e / (refmax + 1e-12))
                              << ",\"refNote\":\"only meaningful if the prompt replays the bundle eval text\"}"
                              << std::endl;
                } else {
                    std::cout << "{\"clientDec\":true,\"token\":" << tok << ",\"argmax\":" << amax
                              << ",\"maxLogit\":" << mx << "}" << std::endl;
                }
            }
            const std::string outPath = decOutPath + ".logits.f64";
            std::ofstream lf(outPath, std::ios::binary);
            lf.write(reinterpret_cast<const char*>(logitsAll.data()),
                     (std::streamsize)(logitsAll.size() * sizeof(double)));
            if (!lf.good()) { std::cerr << "{\"fatal\":\"dec-out: write failed: " << outPath << "\"}\n"; return 2; }
            std::cout << "{\"clientDecDone\":true,\"logits\":\"" << outPath << "\",\"tokens\":" << mTok
                      << ",\"vocab\":" << V << ",\"decMs\":" << (int)msSince(tDec0) << "}" << std::endl;
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "{\"fatal\":\"dec-out failed\",\"what\":\"" << e.what() << "\"}\n"; return 2;
        }
#else
        std::cerr << "{\"fatal\":\"--dec-out requires a build with -DFHE_SSM_DEMO_SER "
                     "(see hpc_gpu_port/DEMO_README.md)\"}\n"; return 2;
#endif
    }

    auto boot = [&](TrackedCt& c) {
        // T0.1: under --sync-timers the device is drained BEFORE starting the
        // clock (so queued prior work is not billed to the boot) and AFTER
        // EvalBootstrap (so the boot's own kernels are). syncBootMs is the
        // honest per-boot cost; bootMs keeps its historical async-dispatch
        // meaning either way (B.2-6: it is a lower bound without the flag).
        syncDev();
        auto t = Clock::now();
        const uint32_t lvlPre = c.level();
        c.ct = cc->EvalBootstrap(c.ct);
        if (syncTimers) { cudaDeviceSynchronize(); syncBootMs += msSince(t); }
        bootMs += msSince(t);
 boots++; vramSample();
        if (traceBoots) {   // A-P0b / F-lvl: bootRestored=false is the silent-no-op trap, not a pass
            const uint32_t lvlPost = c.level();
            std::cout << "{\"bootTrace\":true,\"boot\":" << boots << ",\"lvlPre\":" << lvlPre
                      << ",\"lvlPost\":" << lvlPost << ",\"bootRestored\":" << (lvlPost < lvlPre ? "true" : "false")
                      << ",\"levelsRemainingAfterBoot\":" << (int)depth - (int)lvlPost
                      << ",\"stage\":\"" << (stagePtr ? *stagePtr : std::string("")) << "\""   // S3.3: additive
                      << ",\"ms\":" << msSince(t) << "}" << std::endl;
        }
    };
    // Never hand EvalBootstrap an input this depleted: FLEXIBLEAUTO turns a
    // bootstrap of a nearly-exhausted ciphertext into garbage (the lane-1
    // scheduler bug from the previous campaign).
    //
    // 2026-07-30: this is also a SPEED knob, and a large one. The circuit needs
    // ~21 levels per layer against 15 usable per bootstrap, i.e. ~1.4
    // bootstraps/layer, but the harness performs 99/12 = 8.25 -- so most boots
    // fire on ciphertexts that still have 6-9 usable levels, because every
    // refresh(c,need) trips at `remaining < need + BOOT_FLOOR`. Once mixing is
    // ring-native the bootstrap becomes the dominant term (see --ring-mix), and
    // the bootstrap floor is boots_per_pass * bootstrapMs / lanes -- invariant
    // to ring size, since both bootstrap cost and lane count scale together. So
    // cutting redundant boots is the only way through it. Exposed as a flag
    // because lowering it trades against the FLEXIBLEAUTO hazard above, which
    // has to be measured rather than assumed.
    const uint32_t BOOT_FLOOR = (bootFloor >= 0) ? (uint32_t)bootFloor : 4u;
    auto refresh = [&](TrackedCt& c, uint32_t need) {
        // TWO BUGS FIXED HERE 2026-07-18.
        //
        // (1) UNSIGNED UNDERFLOW: `depth - c.level()` is uint32_t. If c.level() ever
        //     exceeded depth (live culprit was matvecBatch's unguarded alignTo,
        //     fixed below) this wrapped to ~4294967295, read as "abundant
        //     headroom", and SKIPPED an overdue boot -- the ciphertext then
        //     carried an invalid tracked level into OpenFHE's RNS-tower
        //     indexing (size depth+1 = 30) and threw the vector::at() crash.
        //
        // (2) UNSATISFIABLE CONDITION (the bootstrap explosion): the old test
        //     was `depth - used < need + levelsAfter`. But boot() restores
        //     `used` to depth-levelsAfter, i.e. it delivers exactly
        //     `levelsAfter` (10) usable levels -- never more. So the old
        //     condition read `10 < need + 10`, TRUE for every need >= 1: every
        //     refresh() call booted unconditionally, no matter how fresh the
        //     ciphertext already was. That is what drove 136 bootstraps through
        //     a single layer at T=2 (a fresh boot is immediately "insufficient"
        //     and boots again). The requirement was self-defeating: it demanded
        //     levelsAfter remain AFTER consuming `need`, while bootstrapping
        //     only ever yields levelsAfter in total.
        //     Correct test: boot only when the remaining levels genuinely
        //     cannot cover `need` while leaving BOOT_FLOOR in reserve.
        if (c.level() > depth) { boot(c); return; }
        const uint32_t remaining = depth - c.level();
        if (remaining < need + BOOT_FLOOR) boot(c);
    };
    // ---- S3.3 parent-first need formulas (SCHEDULE_DESIGN_20260903.md section 3).
    // Each is the `need` handed to the stock refresh() above, so the rule stays
    // "boot iff remaining < need + BOOT_FLOOR". Levels are counted from the
    // harness's own statements (Lemma 4 / Lemma 8 of TICK_SCALING_BOUNDS):
    //   nb   = levels from the norm input to y's birth: x^2, [e0 mask], 1/d, b0
    //   dN   = levels one Newton iteration consumes on y (4 stock, 2 depth-2)
    //   tail = levels the branch below the norm output needs before its own
    //          guard fires: tm shift+win+bx+readout+gate3+wout = 8 (+2 for
    //          out=x*y and the gain, +2 for the residual addAligned guard);
    //          cm wk 1 + hidden 4 (+2) then refresh(hd, margin).
    const uint32_t pfNb = lanesBlock ? 4u : 3u;
    const uint32_t pfDN = newtonDepth2 ? 2u : 4u;
    const uint32_t pfMargin = (matvecMargin >= 0) ? (uint32_t)matvecMargin : 6u;
    auto pfNeedTail = [&](int kind) -> uint32_t {        // kind: 0 out, 1 tm, 2 cm
        if (kind == 1) return 12u;
        if (kind == 2) return 2u + 1u + 4u + pfMargin;
        return 4u;                                       // output norm: the stock tail need
    };
    auto pfNeedEntry = [&](int kind, int iters) -> uint32_t {
        const uint32_t loop = pfNb + pfDN * (uint32_t)std::max(iters - 1, 0) + 5u;
        const uint32_t branch = pfNeedTail(kind) + 2u;
        if (pfEntry == "loop") return loop;
        if (pfEntry == "branch") return branch;
        if (pfEntry == "both") return std::max(loop, branch);
        return depth + 1u;                               // "always": remaining is never >= depth+1+F
    };
    const uint32_t pfNeedU2 = pfNeedTail(2) - 2u + 1u;   // P4: rem(u2) so no wk/wr child boots
    long long pfBoots = 0;                               // boots fired by P1/P2/P4 (summary field)
    // one rescale-consuming ct x pt (weight at operand level)
    auto mulPt = [&](TrackedCt& c, const std::vector<double>& wpacked) {
        auto pt = ptCh(wpacked, c.level());
        c.ct = cc->EvalMult(c.ct, pt); cc->RescaleInPlace(c.ct);
    };
    // --- level-alignment helpers (replace the repetitive inline boilerplate;
    //     a ct can only be added to / multiplied by one at the same level) ---
    std::vector<double> ONES(SLOTS, 1.0);
    auto alignTo = [&](TrackedCt& c, uint32_t target) {
        while (c.level() < target) {
            auto op = mkPt(ONES, c.level(), __LINE__);
            c.ct = cc->EvalMult(c.ct, op); cc->RescaleInPlace(c.ct);
        }
    };
    // ---- CANONICAL-LEVEL CARRY (2026-09-04, --canonical-carry) -------------
    // A stateful session's carry drifts through levels tick to tick, so every
    // tick visits (diagonal, level) pairs the store lacks: on the demo pod the
    // first stateful tick encoded for hours, and the 2^15 record converged only
    // after ~12 ticks (per-tick encode 1,008 -> 255 -> 62 ... -> 0 s, genRb).
    // With the carry entering EVERY tick at one fixed level the trajectory is
    // identical each tick and the store serves every tick after the first.
    // Canonical level = the post-bootstrap level: align the carry DOWN to
    // depth-BOOT_FLOOR with the harness's own mult-by-one path (the gap is
    // reported; large gaps are the known noise hazard of alignTo, see
    // addAligned) so that EvalBootstrap cannot be its silent no-op
    // (MEASUREMENTS.md 1d), then bootstrap. FIDESlib's exact dropToLevel is
    // NOT used: its header marks the FLEXIBLE-mode scale adjustment a todo.
    // The cold-start zero carry gets the same treatment, so tick 0 follows the
    // canonical trajectory as well (its aligned zeros carry only noise).
    // FIX 2026-09-04 19:55Z: the first version aligned EVERY carry down to depth-BOOT_FLOOR
    // (up to 38 multiply-by-one steps per carry per tick) and then bootstrapped; identical
    // lanes diverged at tick 1 and the tick-2 reply was undecryptable ("approximation error
    // is too high") -- the alignTo noise hazard. A carry that is DEEPER than the canonical
    // level needs no chain at all: EvalBootstrap acts on it directly and lands at the
    // post-boot level. The chain is kept only for carries SHALLOWER than the canonical level
    // (the cold-start zeros, where the added noise is irrelevant). The post-boot level is
    // learned from the first canonical bootstrap (19 here) and a boot landing one level
    // shallower (the 18/19 wobble, S3.3 F-lvl) is corrected with a single one-level step.
    auto canon = [&](TrackedCt& c, const char* what) {
        if (!canonicalCarry || !c.ct) return;
        if (canonLevel >= 0 && c.level() == (uint32_t)canonLevel) return;      // already canonical
        if (canonLevel < 0 || c.level() < (uint32_t)canonLevel) {
            // shallower than canonical (cold start): make it deep enough for a real bootstrap
            const uint32_t target = depth - BOOT_FLOOR;
            const uint32_t gap = c.level() < target ? target - c.level() : 0;
            if (gap > canonMaxGap) canonMaxGap = gap;
            if (c.level() < target) alignTo(c, target);
        }
        const uint32_t pre = c.level();
        boot(c);
        if (c.level() >= pre) {
            std::cerr << "{\"fatal\":\"canonical-carry: bootstrap did not restore levels (the silent no-op)\",\"what\":\""
                      << what << "\",\"levelBefore\":" << pre << ",\"levelAfter\":" << c.level() << "}\n";
            std::exit(2);
        }
        if (canonLevel < 0) canonLevel = (int)c.level();
        else if (c.level() < (uint32_t)canonLevel) alignTo(c, (uint32_t)canonLevel);   // the 18 -> 19 wobble: one step
        else if (c.level() > (uint32_t)canonLevel) {
            std::cerr << "{\"fatal\":\"canonical-carry: bootstrap landed DEEPER than the canonical level\",\"what\":\""
                      << what << "\",\"level\":" << c.level() << ",\"canonLevel\":" << canonLevel << "}\n";
            std::exit(2);
        }
        canonBoots++;
    };
    auto addAligned = [&](TrackedCt& a, TrackedCt& b) {          // a += b
        // BUG FIXED 2026-07-18: this computed target = max(a.level(), b.level())
        // and blindly aligned BOTH operands up to it via alignTo's raw
        // EvalMult+RescaleInPlace loop -- neither alignTo nor this caller
        // ever checked the target against `depth`. addAligned is the
        // workhorse behind every residual connection and gate accumulation
        // in the forward pass, so if EITHER operand had already grown too
        // high (from an unguarded chain elsewhere), that high level would
        // propagate into the OTHER operand too, and the align loop's last
        // MakeCKKSPackedPlaintext call would be issued at or past depth.
        // Fix: refresh both operands to bounded headroom before computing
        // the alignment target, so the target itself is always safe -- this
        // is the shared choke point for the residual stream, so fixing it
        // here catches the compounding-across-layers case generically
        // rather than needing to audit every individual call site.
        refresh(a, 2); refresh(b, 2);
        // BUG FIXED 2026-07-18 (localized by intermediate-decrypt tracing to
        // L0.tm.residual_out): this used to close the level gap with
        // alignTo(), i.e. EvalMult-by-1.0 + RescaleInPlace, ONE LEVEL AT A
        // TIME. The residual stream Hs sits at a very low level (it is the
        // fresh embedding) while the branch being added back has been through
        // rmsnorm + matvec + gate, so the gap here is ~22 levels. Twenty-two
        // consecutive multiply-by-one-and-rescale steps accumulate rounding
        // error and burn an RNS limb each, and the small embedding values
        // (maxAbs ~0.19) end up under the noise floor: the ciphertext stays
        // arithmetically well-formed but is NO LONGER DECRYPTABLE
        // ("Decode(): approximation error is too high"). Measured directly:
        // gate_out at used=21 decrypts fine, residual_out at used=22 fails.
        // OpenFHE's FLEXIBLEAUTO EvalAdd already reconciles differing levels
        // and scaling factors internally (AdjustForAddOrSub), so the manual
        // pre-alignment was never necessary here -- only destructive.
        // Align to a common level before adding. Removing this alignment was
        // tried on 2026-07-18 (on the theory that the repeated multiply-by-1.0
        // was destroying precision on the deep residual gap) and did NOT fix
        // the decode failure -- the real cause there was polyGate3's level
        // desync, fixed separately. Aligning is what demonstrably repaired the
        // time-mix path, so keep the operands genuinely level-matched rather
        // than relying on implicit adjustment.
        // MEASURED 2026-07-18: pre-aligning with alignTo() here is DESTRUCTIVE.
        // alignTo closes a gap one level at a time via EvalMult-by-1.0 +
        // RescaleInPlace; that is harmless for the 1-level gaps inside
        // polyGate3, but the residual stream sits ~22 levels below the branch
        // being added back, and 22 chained rescales push the small embedding
        // values under the noise floor -> "Decode(): approximation error is
        // too high". Verified both ways on the live probe: with alignTo,
        // wout_out/residual_out DECODE_FAIL; without it, they decrypt cleanly
        // (0.031 / 0.087). OpenFHE's FLEXIBLEAUTO EvalAdd reconciles the
        // levels internally, so add directly and just track the resulting
        // level.
        a.ct = cc->EvalAdd(a.ct, b.ct);

    };
    // pack a per-channel coefficient vector for chunk `ch` (n channels from
    // offset ch*Dpad), replicated with period Dpad
    auto packChunk = [&](const std::vector<double>& v, int ch, int n) {
        std::vector<double> s(SLOTS, 0.0);
        if (interleave)
            for (int i = 0; i < n; i++)
                for (uint32_t t = 0; t < REP; t++) s[(size_t)i * REP + t] = v[(size_t)ch * Dpad + i];
        else
            for (uint32_t r = 0; r < REP; r++)
                for (int i = 0; i < n; i++) s[r * Dpad + i] = v[(size_t)ch * Dpad + i];
        return s;
    };
    auto addPtV = [&](TrackedCt& c, const std::vector<double>& packed) {
        auto pt = mkPt(packed, c.level(), __LINE__);
        c.ct = cc->EvalAdd(c.ct, pt);
    };

    // rotate within the ring (mod Dpad pattern holds because of replication)
    auto rot = [&](const Ciphertext<DCRTPoly>& ct, int k) {
        // Interleaved layout: channel rotation by k == ring rotation by k*REP.
        return cc->EvalRotate(ct, interleave ? k * (int)REP : k);
    };

    // dense matvec y = W u, W is (rows,cols) row-major, cols<=Dpad channels
    // packed. Diagonal method with BSGS: y = sum_{k} diag_k (.) rot(u,k),
    // diag_k[j] = W[j, (j+k) mod cols]. Baby-step giant-step batches the
    // rotations. Costs 1 level. Weights are plaintext (public model).
    // CONTRACT (see the CORRECTNESS note at the top; validated by
    // harness/cpu_real_model.cpp): the input is packed with period Dpad, the
    // matrix is treated as (rows x Dpad) with columns >= inCols zero-padded
    // so the rotation modulus EQUALS the diagonal modulus, and rows <= Dpad.
    // Taller maps go through matvecChunked. W is row-major (totalRows x inCols);
    // rowOff selects this chunk's row block.
    auto matvec = [&](TrackedCt& u, const std::vector<double>& W, int rows,
                      int inCols, int rowOff) -> TrackedCt {
        auto t = Clock::now();
        std::vector<Ciphertext<DCRTPoly>> baby(BS);
        baby[0] = u.ct;
        for (int j = 1; j < BS; j++) baby[j] = rot(u.ct, j);
        TrackedCt acc; bool have = false;
        for (int g = 0; g < (int)Dpad; g += BS) {
            Ciphertext<DCRTPoly> inner; bool hi = false;
            for (int j = 0; j < BS && g + j < (int)Dpad; j++) {
                int k = g + j;
                std::vector<double> diag(SLOTS, 0.0);
                bool nonzero = false;
                for (int row = 0; row < rows; row++) {
                    int col = (row + k) % (int)Dpad;              // modulus == Dpad
                    if (col >= inCols) continue;                   // zero-padded column
                    double wv = W[(size_t)(row + rowOff) * inCols + col];
                    // pre-rotate by -g so the giant rot(.,g) lands the value
                    // for output `row` back on slot `row`.
                    uint32_t slot = (uint32_t)((row + g) % (int)Dpad);
                    if (interleave)
                        for (uint32_t tk = 0; tk < REP; tk++) diag[(size_t)slot * REP + tk] = wv;
                    else
                        for (uint32_t r = 0; r < REP; r++) diag[r * Dpad + slot] = wv;
                    // VALUE test, not just the padding test: an all-zero (or
                    // all-<=eps) diagonal costs a full plaintext encode today.
                    // Measured 2026-07-30 on the trained bundle: per-diagonal
                    // max|w| floor is 1.6e-2, so with eps=0 this is bit-identical
                    // on current weights -- it exists for STRUCTURED bundles
                    // (block-diagonal/banded), where it is the entire speedup.
                    if (std::abs(wv) > diagEps) nonzero = true;
                }
                if (!nonzero) continue;                            // skip empty diagonal
                auto dpt = mkPt(diag, u.level(), __LINE__);
                auto term = cc->EvalMult(baby[j], dpt);            // ct x pt (no rescale yet)
                if (!hi) { inner = term; hi = true; }
                else inner = cc->EvalAdd(inner, term);
            }
            if (!hi) continue;
            if (g > 0) inner = rot(inner, g);                      // giant step
            if (!have) { acc.ct = inner; have = true; }
            else acc.ct = cc->EvalAdd(acc.ct, inner);
        }
        cc->RescaleInPlace(acc.ct);
        evalMs += msSince(t);
        return acc;
    };
    // rows > Dpad: ceil(rows/Dpad) chunks, one output ciphertext per chunk.
    // Elementwise ops (the poly gates) act on each chunk independently, so
    // chunking is exact.
    auto matvecChunked = [&](TrackedCt& u, const std::vector<double>& W,
                             int rows, int inCols) {
        std::vector<TrackedCt> out;
        for (int off = 0; off < rows; off += (int)Dpad)
            out.push_back(matvec(u, W, std::min((int)Dpad, rows - off), inCols, off));
        return out;
    };

    // ================= BATCHED MATVEC (the throughput fix) =================
    // WHY: a plaintext at ring 2^17 is ~30 MB (measured: "Plaintexts loaded:
    // 248 ~ 7564MB"), and a dense matvec needs one per diagonal — ~1024 of
    // them, ~11 matvecs/layer, x12 layers ~= 135k encodings. They CANNOT be
    // cached (135k x 30MB ~= 4 TB). The per-token loop re-encoded all of them
    // for EVERY token: ~22-34 min/token, i.e. ~30 h for 64 tokens.
    //
    // FIX: encode each diagonal ONCE and apply it to ALL tokens before moving
    // on. Encoding cost is then paid once per forward pass instead of once
    // per token — a T-fold amortization (T=64 => ~64x).
    //
    // WHY NOT SLOT-BATCHING (64 tokens inside one ciphertext): it would also
    // cut GPU work 64x, but a ring rotation by k crosses D-block boundaries,
    // so token b's high slots would pull token b+1's data — silent CROSS-TOKEN
    // BLEED. The correct form is INTERLEAVED packing (slot s = c*B + t, so a
    // channel rotation by k is a ring rotation by k*B and the ring wraparound
    // coincides exactly with the channel wraparound — provably bleed-free),
    // but the Lemma-2 scan is then cross-token and needs masked doubling
    // (log2 B steps, 1 level each). That is a real correctness hazard, and
    // encoding — not GPU multiply — is the dominant cost, so it buys little.
    // Deferred deliberately; design recorded here so it is not lost.
    //
    // PRECONDITION: every input ciphertext is at the SAME level, so one
    // encoded plaintext is valid for all of them. The forward pass is uniform
    // across tokens, so this holds; it is enforced (not assumed) below.
    // --max-batch N (0 = off): cap the number of ciphertexts per batched
    // matvec call. Motivated by the b4p probe (2026-07-30): with HEALTHY
    // equal-level equal-noiseDeg inputs and shared plaintexts, a 4-ct batch
    // whose operands were built through bootstraps returns all-NaN for every
    // block EXCEPT THE LAST-PROCESSED, while 1- and 2-ct batches are clean in
    // every run all session. Sub-batching trades encode sharing (E -> ceil(n/N)*E
    // for the affected calls) for dodging the suspected FIDESlib async/workspace
    // interaction. Applied inside matvecBatch so every caller inherits it.
    // ---- S3.3 --share-babies: the BSGS baby set of an input, cached across the
    // 2K wk/wr chunk calls of one layer (SCHEDULE_DESIGN section 3.4). Identity
    // = the input ciphertext objects (pointers) + their common level + n +
    // lanesBlock; any mismatch recomputes. Opened/closed by the caller around
    // the wk/wr block so nothing stays resident beyond it. Bit-identical: the
    // reused rotations are the same key switches of the same objects.
    struct BabyCache {
        bool open = false;                                   // share window (set by the wk/wr caller)
        std::vector<const void*> ids; uint32_t lvl = 0; int n = 0; bool lb = false;
        std::vector<std::vector<Ciphertext<DCRTPoly>>> baby, babyW;
        bool valid = false;
        void clear() { ids.clear(); baby.clear(); babyW.clear(); valid = false; n = 0; }
    } babyCache;
    long long babyShared = 0, babyComputed = 0;              // summary fields (calls served from the cache / computed)
    std::function<std::vector<TrackedCt>(std::vector<TrackedCt>&, const std::vector<double>&, int, int, int)> matvecBatchImpl;
    auto matvecBatch = [&](std::vector<TrackedCt>& us, const std::vector<double>& W,
                           int rows, int inCols, int rowOff) -> std::vector<TrackedCt> {
        const int nAll = (int)us.size();
        if (maxBatch > 0 && nAll > maxBatch) {
            std::vector<TrackedCt> merged;
            merged.reserve(nAll);
            for (int off = 0; off < nAll; off += maxBatch) {
                std::vector<TrackedCt> sub(us.begin() + off,
                                           us.begin() + std::min(nAll, off + maxBatch));
                auto part = matvecBatchImpl(sub, W, rows, inCols, rowOff);
                for (int q = 0; q < (int)sub.size(); q++) {
                    us[off + q] = sub[q];          // alignTo side effects propagate
                    merged.push_back(part[q]);
                }
            }
            return merged;
        }
        return matvecBatchImpl(us, W, rows, inCols, rowOff);
    };
    matvecBatchImpl = [&](std::vector<TrackedCt>& us, const std::vector<double>& W,
                           int rows, int inCols, int rowOff) {
        const int n = (int)us.size();
        std::vector<TrackedCt> out(n);
        if (n == 0) return out;
        // BUG FIXED 2026-07-18 (the actual root cause, localized via a
        // stage-breadcrumb trace to L0.tm.scan.t0): lvl = max(.level() across
        // ALL n tokens) with NO refresh guard, then every token aligned up
        // to it via raw alignTo -- not even routed through addAligned's
        // (already-fixed) protection. If any one token's incoming level was
        // already high, `lvl` (and therefore
        uint32_t lvl = us[0].level();                      // enforce a common level
        uint32_t lvlMin = us[0].level();
        for (auto& u : us) { lvl = std::max(lvl, u.level()); lvlMin = std::min(lvlMin, u.level()); }
        // DIAGNOSTIC (2026-07-31): the level SPREAD across batched ciphertexts is
        // the suspect in multi-block failures. alignTo closes the gap with
        // EvalMult-by-1.0 + Rescale ONE LEVEL AT A TIME, and addAligned's own
        // comment already records that ~22 such chained rescales push small
        // values under the noise floor ("Decode(): approximation error is too
        // high") -- yet matvecBatch is still doing exactly that, and it is the
        // last instance of the pattern. Evidence it bites: at 4 blocks the L0
        // probe shows hidden at level 19 with maxAbs 1.2595, and the wv matvec
        // output then decrypts to maxAbs EXACTLY 0 -- values destroyed, not a
        // decode failure. Print the spread so this is measured, not inferred.
        if (probe && us.size() > 1 && lvl != lvlMin)
            std::cout << "{\"lvlSpread\":true,\"n\":" << us.size()
                      << ",\"min\":" << lvlMin << ",\"max\":" << lvl
                      << ",\"gap\":" << (lvl - lvlMin) << ",\"rows\":" << rows << "}" << std::endl;
        for (auto& u : us) alignTo(u, lvl);

        // ---- RING-NATIVE MIXING COST PROXY (--ring-mix) --------------------
        // Candidate #3 of the "model impersonates the scheme" thesis: replace
        // dense channel mixing with a STRUCTURED map that costs ONE ct x pt
        // multiply and ZERO rotations, instead of Dpad diagonal plaintexts and
        // 2*sqrt(Dpad) key-switches. harness/ring_matmul_bench.cpp measured
        // 4073x on that op in isolation, but it memoized the weight plaintexts
        // offline; this harness cannot (135k x 9.8 MB = 1.3 TB per pass), so
        // the end-to-end number has to be measured here.
        //
        // WHAT THIS IS: an exact cost proxy, NOT the arithmetic. It performs
        // one ct x pt multiply consuming exactly ONE level -- the same op count
        // and the same depth as the structured map -- so the level trace,
        // bootstrap count and every other stage are bit-for-bit the same
        // circuit as the dense run. Only the mixing cost changes. Logits are
        // therefore meaningless (these weights are dense-trained); the point is
        // s/token for a model whose mixing is ring-native.
        if (ringMix) {
            auto tR = Clock::now();
            std::vector<double> pv(SLOTS, 0.0);
            for (int i = 0; i < rows && i < (int)Dpad; i++) {
                int col = i % inCols;
                double wv = W[(size_t)(i + rowOff) * inCols + col];
                if (interleave)
                    for (uint32_t tk = 0; tk < REP; tk++) pv[(size_t)i * REP + tk] = wv;
                else
                    for (uint32_t r = 0; r < REP; r++) pv[r * Dpad + i] = wv;
            }
            auto pt = mkPt(pv, lvl, __LINE__);
            encPtCount++;
            for (int i = 0; i < n; i++) {
                out[i].ct = cc->EvalMult(us[i].ct, pt);
                cc->RescaleInPlace(out[i].ct);
            }
            evalMs += msSince(tR);
            return out;
        }

        syncDev();                       // T0.1: prior queued work is not this matvec's
        auto t = Clock::now();
        // baby rotations, per token (cheap vs. encoding; still O(sqrt D))
        auto tBaby = Clock::now();
        std::vector<std::vector<Ciphertext<DCRTPoly>>> baby(n);
        // S3.3 --share-babies: serve the baby set from the cache when the window
        // is open and the identity (objects, level, n, halves) matches.
        bool babyHit = false;
        if (shareBabies && babyCache.open && babyCache.valid && babyCache.n == n
            && babyCache.lvl == lvl && babyCache.lb == lanesBlock) {
            babyHit = true;
            for (int i = 0; i < n; i++) if (babyCache.ids[(size_t)i] != (const void*)us[i].ct.get()) { babyHit = false; break; }
        }
        if (babyHit) {
            baby = babyCache.baby; babyShared++;
        } else {
            if (shareBabies && babyCache.open) babyCache.clear();   // REVIEW 2026-09-03: release the stale set BEFORE
                                                                    // computing the new one (never two sets resident)
            for (int i = 0; i < n; i++) {
                baby[i].resize(BS);
                baby[i][0] = us[i].ct;
                for (int j = 1; j < BS; j++) baby[i][j] = rot(us[i].ct, j);
            }
            babyComputed++;
            if (shareBabies && babyCache.open) {           // fill the cache for the following chunk calls
                babyCache.ids.resize((size_t)n);
                for (int i = 0; i < n; i++) babyCache.ids[(size_t)i] = (const void*)us[i].ct.get();
                babyCache.lvl = lvl; babyCache.n = n; babyCache.lb = lanesBlock;
                babyCache.baby = baby; babyCache.valid = true;   // babyW filled below under --lanes-block
            }
        }
        if (syncTimers) { cudaDeviceSynchronize(); syncBabyRotMs += msSince(tBaby); }
        std::vector<Ciphertext<DCRTPoly>> acc(n);
        std::vector<bool> have(n, false);
        // PARALLEL DIAGONAL ENCODE (2026-07-30) -- the 99.7% cost.
        // Measured: gpuBootstrapMs 890 of gpuMsPerToken 354,398. The rest is
        // ~135k host-side MakeCKKSPackedPlaintext calls (DFT + CRT into
        // depth-lvl+1 RNS limbs), issued ONE AT A TIME from this loop while 31
        // of the pod's 32 cores idle. The encodes are independent -- pure
        // functions of (diag, lvl) -- so encode a whole giant step's worth up
        // front in parallel, then apply them to the GPU serially (CUDA calls
        // stay on the calling thread). Bounded at BS=32 plaintexts live at once
        // (~130 MB), not all 1024.
        if (lvl > depth) {
            std::cerr << "{\"fatal\":\"level_overflow\",\"lvl\":" << lvl
                      << ",\"depth\":" << depth << ",\"line\":" << __LINE__ << "}" << std::endl;
            std::terminate();
        }
        // The two dominant costs are DISJOINT resources -- the encode is host
        // memory bandwidth (measured 1.32 TB per pass at ~26 GB/s = 51 s) and
        // the apply is PCIe upload + GPU (~65 s) -- but they were serialized, so
        // the pass paid their SUM. Software-pipeline them one giant step apart:
        // encode batch g+BS on a worker thread while the main thread applies
        // batch g. Every CUDA call stays on the main thread (FIDESlib state is
        // not known to be thread-safe); the worker touches only OpenFHE
        // plaintext encoding, which is a pure function of (diag, lvl).
        // --no-pipeline restores the serialized order for A/B measurement.
        std::vector<Plaintext> ptCur(BS), ptNxt(BS);
        std::vector<char> liveCur(BS, 0), liveNxt(BS, 0);
        auto encodeBatch = [&](int g, std::vector<Plaintext>& dst, std::vector<char>& dlive) {
            const int jm = std::min(BS, (int)Dpad - g);
            for (int j = 0; j < BS; j++) { dlive[j] = 0; dst[j] = nullptr; }
            if (cacheDiags) {
                // Serial cache probe first: a hit skips the encode AND the
                // host->device upload (the plaintext is already resident), which
                // is the entire 9.8 MB/diagonal cost. std::map is not
                // thread-safe, so probe outside the parallel region.
                bool allHit = true;
                for (int j = 0; j < jm; j++) {
                    auto it = diagCache.find({W.data(), rowOff, lvl, g + j});
                    if (it != diagCache.end()) { dst[j] = it->second; dlive[j] = 1; diagCacheHits++; }
                    else allHit = false;
                }
                if (allHit) return;
            }
#pragma omp parallel for schedule(static)
            for (int j = 0; j < jm; j++) {
                if (dlive[j]) continue;                 // cache hit, already filled
                int k = g + j;
                // VALUE test, not just the padding test: an all-zero (or
                // all-<=eps) diagonal costs a full plaintext encode today.
                // Measured 2026-07-30 on the trained bundle: per-diagonal
                // max|w| floor is 1.6e-2, so with eps=0 this is bit-identical
                // on current weights -- it exists for STRUCTURED bundles
                // (block-diagonal/banded), where it is the entire speedup.
                //
                // NOTE (2026-07-31): hoisting this test ABOVE the slot-vector
                // construction is a real optimization for banded weights -- band 2
                // (660 live diagonals) and band 8 (2,244) both measured exactly
                // 308 ms/token because this loop builds a 131 KB vector for all
                // 135,168 CANDIDATES before discarding 98%. It was tried and
                // REVERTED: the run aborted with "terminate called without an
                // active exception". See the report; do not re-apply without
                // resolving that first.
                std::vector<double> diag(SLOTS, 0.0);
                bool nonzero = false;
                for (int row = 0; row < rows; row++) {
                    int col = (row + k) % (int)Dpad;
                    if (col >= inCols) continue;
                    double wv = W[(size_t)(row + rowOff) * inCols + col];
                    uint32_t slot = (uint32_t)((row + g) % (int)Dpad);
                    if (interleave)
                        for (uint32_t tk = 0; tk < REP; tk++) diag[(size_t)slot * REP + tk] = wv;
                    else
                        for (uint32_t r = 0; r < REP; r++) diag[r * Dpad + slot] = wv;
                    if (std::abs(wv) > diagEps) nonzero = true;
                }
                if (!nonzero) continue;
                // *** encode ONCE, reuse across every token ***
                dst[j] = cc->MakeCKKSPackedPlaintext(diag, 1, lvl);
                dlive[j] = 1;
            }
            if (cacheDiags)
                for (int j = 0; j < jm; j++)
                    if (dlive[j]) diagCache[{W.data(), rowOff, lvl, g + j}] = dst[j];
        };
#ifdef FHE_SSM_DEMO_SER
        // ---- T1b: compressed-store delivery (the G1 route). --------------
        // Engaged only when the flag is on AND a call-site marker named the
        // op (mvOpId >= 0); everything else falls through to the unchanged
        // path below. Misses build+encode the diagonal once (counted in
        // encPtMs/encPtCount), verify the i/REP map on every raw limb, and
        // store 16 KiB/limb; hits and fresh misses alike are delivered as
        // ONE pinned H2D + ONE expand-kernel launch per giant step into the
        // per-level scratch pool, consumed by the stock multPt. An
        // uncompressible diagonal (map violation — e.g. any --interleave
        // run) is delivered by the normal full plaintext for that slot and
        // counted; wrong answers are unrepresentable by construction.
        const bool useStore = compressedStore && mvOpId >= 0 && !ringMix;
        auto storeFetch = [&](int g, std::vector<char>& dlive, PoolLvl& P,
                              std::vector<char>& fromPool, std::vector<char>& fromPoolHi) {
            const int jm = std::min(BS, (int)Dpad - g);
            auto tF = Clock::now();
            // (2026-09-02) encode+compress ONE (g+jj, half hh) diagonal at this
            // level -- the former inline miss path, shared with --store-file's
            // sample verification so both sides run the identical algebra.
            auto buildComp = [&](int jj, int hh, bool& nonzero, bool& okmap, Plaintext& ptFull,
                                 std::vector<uint64_t>& comp, double& encodeMs) {
                nonzero = false; okmap = false; encodeMs = 0; comp.clear();
                std::vector<double> diag(SLOTS, 0.0);   // same algebra as encodeBatch
                for (int row = 0; row < rows; row++) {
                    int col = (row + g + jj) % (int)Dpad;
                    if (col >= inCols) continue;
                    // T1c dual-half mask, folded into the diagonal at build
                    // time (+0 levels): hi-half rows are the ones whose
                    // source index row+g+j crossed the block boundary and
                    // therefore multiply baby(u_wrap) instead of baby(u).
                    if (lanesBlock && ((row + g + jj >= (int)Dpad) ? 1 : 0) != hh) continue;
                    double wv = W[(size_t)(row + rowOff) * inCols + col];
                    uint32_t slot = (uint32_t)((row + g) % (int)Dpad);
                    if (interleave)
                        for (uint32_t tk = 0; tk < REP; tk++) diag[(size_t)slot * REP + tk] = wv;
                    else
                        for (uint32_t r = 0; r < REP; r++) diag[r * Dpad + slot] = wv;
                    if (std::abs(wv) > diagEps) nonzero = true;
                }
                if (!nonzero) return;
                // ---- S3.7 V5 (2026-09-10): the periodic path, block layout only ----------------
                // The slot vector above is Dpad-periodic (diag[r*Dpad + slot] for every replica r), so
                // replica 0 -- diag[0 .. Dpad) -- is the whole period. periodic_encode.hpp turns it into
                // exactly the words the compress loop below would extract from the dense plaintext.
                // On any status but PE_OK the dense path runs instead (it throws on the same inputs).
                if (periodicEncode && !interleave) {
                    auto tP = Clock::now();
                    if (!periodicEnc) {
                        auto& cpuCcP = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
                        periodicEnc.reset(new fhe_ssm::PeriodicEncoder(cpuCcP, compLogRep));
                        if (periodicEnc->CW != COMP_CW || periodicEnc->period != Dpad || periodicEnc->nqTotal != depth + 1) {
                            std::cerr << "{\"fatal\":\"periodic-encode geometry\",\"encCW\":" << periodicEnc->CW
                                      << ",\"COMP_CW\":" << COMP_CW << ",\"encPeriod\":" << periodicEnc->period << ",\"Dpad\":" << Dpad
                                      << ",\"encTowers\":" << periodicEnc->nqTotal << ",\"depthPlus1\":" << (depth + 1) << "}" << std::endl;
                            std::exit(2);
                        }
                    }
                    const int prc = periodicEnc->encode(diag.data(), lvl, comp, periodicScratch);
                    if (prc == fhe_ssm::PE_OK && comp.size() == (size_t)P.nLimbs * COMP_CW) {
                        okmap = true; encodeMs = msSince(tP); periodicEncodeCount++;
                        if (periodicVerifyEvery > 0 && (periodicEncodeCount % periodicVerifyEvery) == 0) {
                            // self-audit: the dense path on the same diagonal must yield the same words
                            periodicVerifyCount++;
                            Plaintext ptV = cc->MakeCKKSPackedPlaintext(diag, 1, lvl);
                            auto& cpuCcV = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
                            const auto& ptImplV = std::any_cast<const lbcrypto::Plaintext&>(ptV->cpu);
                            FIDESlib::CKKS::RawPlainText rawV = FIDESlib::CKKS::GetRawPlainText(cpuCcV, ptImplV);
                            bool same = ((int)rawV.sub_0.size() == P.nLimbs);
                            for (int l = 0; same && l < P.nLimbs; l++)
                                for (uint32_t j2 = 0; same && j2 < COMP_CW; j2++)
                                    if (rawV.sub_0[l][(size_t)j2 << compLogRep] != comp[(size_t)l * COMP_CW + j2]) same = false;
                            if (!same) {
                                std::cerr << "{\"fatal\":\"periodic-encode mismatch vs dense\",\"layer\":" << mvLayer
                                          << ",\"op\":" << mvOpId << ",\"chunk\":" << mvChunk << ",\"rowOff\":" << rowOff
                                          << ",\"diag\":" << (g + jj) << ",\"half\":" << hh << ",\"lvl\":" << lvl << "}" << std::endl;
                                std::exit(2);
                            }
                            periodicVerifyOk++;
                        }
                        return;
                    }
                    periodicFallbackCount++;   // PE_* nonzero or a limb-count mismatch: the dense path decides
                    comp.clear();
                }
                auto tE = Clock::now();
                ptFull = cc->MakeCKKSPackedPlaintext(diag, 1, lvl);
                encodeMs = msSince(tE);
                auto& cpuCcS = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
                const auto& ptImplS = std::any_cast<const lbcrypto::Plaintext&>(ptFull->cpu);
                FIDESlib::CKKS::RawPlainText rawS = FIDESlib::CKKS::GetRawPlainText(cpuCcS, ptImplS);
                comp.assign((size_t)P.nLimbs * COMP_CW, 0);
                okmap = ((int)rawS.sub_0.size() == P.nLimbs);
                for (int l = 0; okmap && l < P.nLimbs; l++) {
                    const auto& limbv = rawS.sub_0[l];
                    if (limbv.size() != ((size_t)COMP_CW << compLogRep)) { okmap = false; break; }
                    for (size_t i2 = 0; i2 < limbv.size(); i2++)
                        if (limbv[i2] != limbv[(i2 >> compLogRep) << compLogRep]) { okmap = false; break; }
                    if (okmap)
                        for (uint32_t j2 = 0; j2 < COMP_CW; j2++)
                            comp[(size_t)l * COMP_CW + j2] = limbv[(size_t)j2 << compLogRep];
                }
                if (!okmap) comp.clear();
            };
            for (int j = 0; j < BS; j++) { dlive[j] = 0; fromPool[j] = 0; fromPoolHi[j] = 0; }
            bool anyPool = false;
            const int HALVES = lanesBlock ? 2 : 1;
            for (int j = 0; j < jm; j++) {
              for (int h = 0; h < HALVES; h++) {
                CompKey key{mvLayer, mvOpId, mvChunk, rowOff, g + j, h, lvl};
                auto it = compStore.find(key);
                if (it == compStore.end()) {
                    storeMisses++;
                    bool nonzero = false, okmap = false; Plaintext ptFull; std::vector<uint64_t> comp; double ems = 0;
                    buildComp(j, h, nonzero, okmap, ptFull, comp, ems);
                    if (!nonzero) {
                        it = compStore.emplace(key, std::vector<uint64_t>{}).first;
                        storeDead++; storeDirty = true; storeNewKeys.push_back(key);
                    } else {
                        encPtMs += ems;
                        encPtCount++;
                        if (!okmap) {
                            storeUncompressible++;
                            if (lanesBlock) {
                                // no full-plaintext fallback exists for a
                                // masked HALF -- refuse loudly (this only
                                // happens if the layout premise broke).
                                std::cerr << "{\"fatal\":\"lanes-block: uncompressible masked half\"}\n";
                                std::exit(2);
                            }
                            ptCur[j] = ptFull;      // normal delivery, never stored
                            dlive[j] = 1;
                            continue;
                        }
                        storeBytes += comp.size() * 8;
                        it = compStore.emplace(key, std::move(comp)).first;
                        storeDirty = true; storeNewKeys.push_back(key);
                    }
                } else {
                    storeHits++;
                    int* seen = storeLoaded ? &storeVerifySeen[std::make_tuple(mvLayer, mvOpId, lvl)] : nullptr;
                    if (seen && *seen < storeVerifySamples) {
                        // --store-file sample verification: a loaded entry must
                        // equal a fresh encode bit for bit (dead <-> all-zero).
                        // Sampled per (layer, op, level) prefix so every matrix
                        // and every level trace gets checked, not just the first
                        // giant step of layer 0 (review 2026-09-03).
                        (*seen)++;
                        bool nz = false, okm = false; Plaintext pf; std::vector<uint64_t> cmp; double ems = 0;
                        buildComp(j, h, nz, okm, pf, cmp, ems); storeVerifyMs += ems;
                        const bool ok = it->second.empty() ? !nz : (nz && okm && cmp == it->second);
                        if (ok) storeVerifyOk++;
                        else {
                            storeVerifyBad++;
                            std::cerr << "{\"fatal\":\"store-file: loaded entry differs from a fresh encode\","
                                         "\"layer\":" << key.layer << ",\"op\":" << key.op << ",\"chunk\":" << key.chunk
                                      << ",\"rowOff\":" << key.rowOff << ",\"diag\":" << key.diag << ",\"half\":" << key.half
                                      << ",\"lvl\":" << key.lvl << ",\"fileDead\":" << (it->second.empty() ? 1 : 0)
                                      << ",\"freshNonzero\":" << (nz ? 1 : 0) << ",\"freshOkmap\":" << (okm ? 1 : 0) << "}" << std::endl;
                            std::exit(2);
                        }
                    }
                }
                if (it->second.empty()) continue;    // dead diagonal/half
                const int slot = j * HALVES + h;
                std::memcpy(P.hStage + (size_t)slot * P.nLimbs * COMP_CW,
                            it->second.data(), it->second.size() * 8);
                if (h == 0) dlive[j] = 1; else fromPoolHi[j] = 1;
                if (h == 0) fromPool[j] = 1;
                anyPool = true;
              }
            }
            if (anyPool) {
                const int nSlots = jm * HALVES;
                const size_t words = (size_t)nSlots * P.nLimbs * COMP_CW;
                dim3 grid((SLOTS * 2 + 255) / 256, P.nLimbs, nSlots);
                if (devices.size() > 1) {
                    // WAR/WAW fence: the previous giant step's multiplies on the
                    // other device(s) may still be reading these pool buffers.
                    syncAllDevices();
                    // 2026-09-04: each device expands ONLY its own limbs from its
                    // own staging copy (peer writes returned wrong data on O-2086900).
                    for (size_t d = 0; d < devices.size(); d++) {
                        CUDA_CHECK(cudaSetDevice(devices[d]));
                        CUDA_CHECK(cudaMemcpy(P.dCompDev[d], P.hStage, words * 8, cudaMemcpyHostToDevice));
                        fhe_ssm_expand_limbs_dev<<<grid, 256>>>(P.dPtrTabDev[d], P.dCompDev[d], nSlots, P.nLimbs,
                                                                SLOTS * 2, compLogRep, P.dLimbDev[d], (int)d);
                        CUDA_CHECK(cudaGetLastError());
                    }
                    syncAllDevices();
                    CUDA_CHECK(cudaSetDevice(devices[0]));
                } else {
                    CUDA_CHECK(cudaMemcpy(P.dComp, P.hStage, words * 8, cudaMemcpyHostToDevice));
                    fhe_ssm_expand_limbs_v<<<grid, 256>>>(P.dPtrTab, P.dComp, nSlots, P.nLimbs,
                                                          SLOTS * 2, compLogRep);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
            }
            storeFetchMs += msSince(tF);
        };
        if (useStore) {
            PoolLvl& P = compPoolGet(lvl);
            std::vector<char> fromPool(BS, 0), fromPoolHi(BS, 0);
            // T1c: the wrap baby set — hi-half diagonals multiply
            // baby(u_wrap = rot_{−Dpad}(u)), so a crossed source index reads
            // the OWN replica's wrapped element (T1C_DESIGN §1).
            std::vector<std::vector<Ciphertext<DCRTPoly>>> babyW;
            if (lanesBlock) {
                if (babyHit && babyCache.babyW.size() == (size_t)n) {
                    babyW = babyCache.babyW;                 // S3.3 --share-babies: wrap set from the cache
                } else {
                    babyW.resize(n);
                    for (int i = 0; i < n; i++) {
                        auto uw = rot(us[i].ct, -(int)Dpad);
                        babyW[i].resize(BS);
                        babyW[i][0] = uw;
                        for (int j = 1; j < BS; j++) babyW[i][j] = rot(uw, j);
                    }
                    if (shareBabies && babyCache.open && babyCache.valid && !babyHit) babyCache.babyW = babyW;
                }
                if (syncTimers) { cudaDeviceSynchronize(); }
            }
            const int HH = lanesBlock ? 2 : 1;
            for (int g = 0; g < (int)Dpad; g += BS) {
                std::vector<Ciphertext<DCRTPoly>> inner(n);
                std::vector<bool> hi(n, false);
                const int jm = std::min(BS, (int)Dpad - g);
                storeFetch(g, liveCur, P, fromPool, fromPoolHi);
                syncDev();
                auto tApply = Clock::now();
                for (int j = 0; j < jm; j++) {
                    const bool loLive = liveCur[j] != 0;
                    const bool hiLive = lanesBlock && fromPoolHi[j] != 0;
                    if (!loLive && !hiLive) continue;
                    for (int i = 0; i < n; i++) {
                        Ciphertext<DCRTPoly> term; bool haveTerm = false;
                        if (loLive) {
                            Plaintext& op = fromPool[j] ? P.wraps[(size_t)j * HH] : ptCur[j];
                            term = cc->EvalMult(baby[i][j], op); haveTerm = true;
                        }
                        if (hiLive) {
                            auto t2 = cc->EvalMult(babyW[i][j], P.wraps[(size_t)j * HH + 1]);
                            if (haveTerm) term = cc->EvalAdd(term, t2);
                            else { term = t2; haveTerm = true; }
                        }
                        if (!hi[i]) { inner[i] = term; hi[i] = true; }
                        else inner[i] = cc->EvalAdd(inner[i], term);
                    }
                }
                if (syncTimers) { cudaDeviceSynchronize(); syncCtPtMs += msSince(tApply); }
                auto tGiantS = Clock::now();
                for (int i = 0; i < n; i++) {
                    if (!hi[i]) continue;
                    if (bsgsRescaleInner) cc->RescaleInPlace(inner[i]);
                    auto blk = (g > 0) ? rot(inner[i], g) : inner[i];
                    if (!have[i]) { acc[i] = blk; have[i] = true; }
                    else acc[i] = cc->EvalAdd(acc[i], blk);
                }
                if (syncTimers) { cudaDeviceSynchronize(); syncGiantMs += msSince(tGiantS); }
            }
        } else {
#endif
        // T0.2 (S3.1): the encode worker replaces the per-giant-step
        // std::thread (T16). Declared AFTER encodeBatch and the double-buffers
        // it writes, so on an exception unwind its destructor drains and joins
        // BEFORE those objects are destroyed — the joinable-thread-destructor
        // terminate path is unrepresentable. Worker-side exceptions are
        // captured and rethrown from wait() on the main thread.
        EncodeWorker encWorker;
        if (pipeline) encWorker.start();
        {
            auto tEnc0 = Clock::now();
            encodeBatch(0, ptCur, liveCur);
            encPtMs += msSince(tEnc0);
        }
        for (int g = 0; g < (int)Dpad; g += BS) {
            std::vector<Ciphertext<DCRTPoly>> inner(n);
            std::vector<bool> hi(n, false);
            const int jm = std::min(BS, (int)Dpad - g);
            const int gNext = g + BS;
            auto tPipe = Clock::now();
            if (pipeline && gNext < (int)Dpad)
                encWorker.submit([&, gNext]() {
                    auto tw = Clock::now();
                    encodeBatch(gNext, ptNxt, liveNxt);
                    encPtMs += msSince(tw);     // single worker; read only after wait()
                });
            syncDev();                          // T0.1 boundary: bill only the apply loop
            auto tApply = Clock::now();
            for (int j = 0; j < jm; j++) {
                if (!liveCur[j]) continue;
                encPtCount++;
                for (int i = 0; i < n; i++) {
                    auto term = cc->EvalMult(baby[i][j], ptCur[j]);
                    if (!hi[i]) { inner[i] = term; hi[i] = true; }
                    else inner[i] = cc->EvalAdd(inner[i], term);
                }
            }
            if (syncTimers) { cudaDeviceSynchronize(); syncCtPtMs += msSince(tApply); }
            if (pipeline) encWorker.wait();     // drain + rethrow worker errors
            else if (gNext < (int)Dpad) {       // --no-pipeline: encode now, serialized
                auto tEncS = Clock::now();
                encodeBatch(gNext, ptNxt, liveNxt);
                encPtMs += msSince(tEncS);
            }
            pipeMs += msSince(tPipe);
            std::swap(ptCur, ptNxt); std::swap(liveCur, liveNxt);
            auto tGiant = Clock::now();
            for (int i = 0; i < n; i++) {
                if (!hi[i]) continue;
                // OBJECTIVE 2 (2026-07-29): the baseline sums ALL ~1024 ct x pt
                // terms at scale Delta^2 and rescales ONCE at the very end, so
                // the intermediate magnitude grows ~1024x (noise ~sqrt(1024)=32x)
                // before any reduction. The source itself blames this depth for
                // wv_part failing on 2 of 3 statistically identical chunks
                // ("a marginal precision threshold"). Rescaling each giant-step
                // partial bounds growth to BS=32 terms (~sqrt(32)=5.7x), a ~5.6x
                // noise reduction, at IDENTICAL multiplicative depth: the 32
                // partials are parallel branches, each taking exactly one
                // rescale, so the chain still costs one level.
                if (bsgsRescaleInner) cc->RescaleInPlace(inner[i]);
                auto blk = (g > 0) ? rot(inner[i], g) : inner[i];
                if (!have[i]) { acc[i] = blk; have[i] = true; }
                else acc[i] = cc->EvalAdd(acc[i], blk);
            }
            if (syncTimers) { cudaDeviceSynchronize(); syncGiantMs += msSince(tGiant); }
        }
#ifdef FHE_SSM_DEMO_SER
        }   // close the T1b else-branch (normal delivery path)
#endif
        auto tResc = Clock::now();
        for (int i = 0; i < n; i++) {
            if (!bsgsRescaleInner) cc->RescaleInPlace(acc[i]);   // already rescaled per giant step
            out[i].ct = acc[i];
        }
        if (syncTimers) { cudaDeviceSynchronize(); syncRescaleMs += msSince(tResc); }
        evalMs += msSince(t);
        return out;
    };
    auto matvecBatchChunked = [&](std::vector<TrackedCt>& us, const std::vector<double>& W,
                                  int rows, int inCols) {
        std::vector<std::vector<TrackedCt>> chunks;
        for (int off = 0; off < rows; off += (int)Dpad)
            chunks.push_back(matvecBatch(us, W, std::min((int)Dpad, rows - off), inCols, off));
        return chunks;   // [chunk][token]
    };

    // RMSNorm: y = x * rsqrt(mean(x^2)+eps) * g. mean over the `d` real
    // channels via log2(Dpad) rotate-adds (padding channels are 0). Newton
    // rsqrt from the calibrated linear seed [a,b,iters].
    // BUG FIXED 2026-07-18: `x` was `TrackedCt&` (mutable reference). The
    // "align x up to y.level()" step below MUTATES x.level() in place, so calling
    // rmsnorm(Hs[t], ...) was silently bumping the CALLER's residual-stream
    // level to match whatever the internal Newton loop happened to need --
    // an untracked side effect that then compounds across every remaining
    // op on Hs[t] for the rest of the layer (residual add, the second
    // rmsnorm call for channel-mix, etc.), eventually overflowing `depth`
    // somewhere refresh() was never told to look. TrackedCt's `ct` member is
    // an OpenFHE smart pointer, so by-value here is a cheap refcount bump,
    // not a real ciphertext copy -- this makes rmsnorm's internal alignment
    // purely local, as it should be.
    int pfNormKind = 0;   // S3.3 P2: 1 = tm norm, 2 = cm norm, 0 = output/other. Set by the caller
                          // right before rmsnorm() (the mvOpId pattern) and reset on entry, so a
                          // caller that does not set it (output norm, --lane-test-block) gets the
                          // stock tail need.
    auto rmsnorm = [&](TrackedCt x, const std::vector<double>& g,
                       double a0, double b0, int iters, double eps) -> TrackedCt {
        auto t = Clock::now();
        // S3.7 V2 (2026-09-05): this window's own bootstraps (the Newton guards and
        // the tail refresh below) are also added to bootMs by boot(); the delta is
        // moved to normBootMs at the window's exits so evalMs - normBootMs is
        // disjoint from bootMs (S3.6 N1: 117 boots = 27,890 ms per cell-A tick).
        const double normBoot0 = bootMs;
        const int normKind = pfNormKind; pfNormKind = 0;
        // ---- STATIC SCALE INSTEAD OF RMSNORM (--static-norm) ----------------
        // Quality-ladder rung 2 ("-norm": RMSNorm -> static scale), motivated by
        // FHE candidate #1 (no data-dependent division). This is the other half
        // of the throughput story that --ring-mix starts: once mixing is
        // ring-native, the ONLY remaining rotations in the whole forward pass
        // are this function's mean-square reduction -- log2(Dpad) = 10
        // rotate-adds per norm site, 25 sites -> ~250 key-switches per pass --
        // and the Newton loop is also the depth hog that drives the bootstrap
        // count. Replacing it with one plaintext multiply costs 1 level, 0
        // rotations, 0 bootstraps.
        //
        // The scale is the seed-implied typical mean-square, recovered exactly
        // the way the --ms-norm path recovers sigma (sigma = -K*b/a ~ 1/ms_typ),
        // so no bundle-format change is needed. Like --ring-mix this is a COST
        // proxy on dense-trained weights: it has the true op count and depth of
        // the norm-free architecture, but its logits are meaningless because
        // these weights were trained WITH the norm.
        if (staticNorm) {
            double sc = 1.0;
            if (a0 != 0.0) {
                double sg = -(msNormK > 0.0 ? msNormK : 5.5) * b0 / a0;   // ~1/ms_typ
                if (std::isfinite(sg) && sg > 0.0) sc = std::sqrt(sg);
            }
            std::vector<double> gs(g.size());
            for (size_t i = 0; i < g.size(); i++) gs[i] = g[i] * sc;
            TrackedCt outS = x;
            mulPt(outS, gs);
            evalMs += msSince(t);
            normBootMs += bootMs - normBoot0;     // S3.7 V2 (0 here: no boot on this path; kept uniform)
            return outS;
        }
        // ---- PER-SITE NORMALIZATION OF THE MEAN-SQUARE (2026-07-29) ----------
        // The Newton loop below bootstraps BOTH ms and y every iteration
        // (refresh(y,5); refresh(ms,5)). But y = 1/sqrt(ms), so their magnitudes
        // are reciprocal-ish: measured across the 25 norm sites of this model the
        // in-loop ratio max(ms,y)/min(ms,y) reaches 2.97e4, while the measured
        // usable CKKS bootstrap window is only ~67x wide (amp 0.3..20, see
        // results/gpu-verify-20260728/exp7_amp.jsonl). Both operands are out of
        // range simultaneously and NO choice of residual scale c can fix it:
        // raising c pushes ms up as c^2 and y down as 1/c.
        //
        // Fix: run Newton on the NORMALIZED statistic msn = sigma*ms with
        // sigma = 1/ms_typ, so msn ~ 1 and rsqrt(msn) ~ 1 at every site.
        //     rsqrt(ms) = sqrt(sigma) * rsqrt(msn)
        //     seed:  a_n = a/sqrt(sigma),  b_n = b/sigma^1.5
        //     tail:  fold sqrt(sigma) into the plaintext gain g  (free)
        // sigma is recovered from the seed alone as -K*b/a (ms_typ ~ (-a/b)/K,
        // K ~ 5.5 across all 25 sites, spread 3.3x -- ample against a 67x window),
        // so no bundle-format change is needed. Verified: worst in-loop ratio
        // 2.97e4 -> 4.02.
        double sigma = 1.0, rtSigma = 1.0;
        if (msNormK > 0.0 && a0 != 0.0) {
            double sg = -msNormK * b0 / a0;
            if (std::isfinite(sg) && sg > 0.0) {
                sigma = sg; rtSigma = std::sqrt(sigma);
                a0 = a0 / rtSigma;
                b0 = b0 / (sigma * rtSigma);
            }
        }
        // s = x^2  (1 level)
        TrackedCt sq{cc->EvalMult(x.ct, x.ct), x.level()};
        cc->RescaleInPlace(sq.ct);
        // block sum over Dpad slots -> replicated block-sum in every slot
        Ciphertext<DCRTPoly> red = sq.ct;
        for (int p = 1; p < (int)Dpad; p <<= 1) red = cc->EvalAdd(red, rot(red, p));
        uint32_t redLvl = sq.level();
#ifdef FHE_SSM_DEMO_SER
        if (lanesBlock) {
            // T1c §2 (T1C_DESIGN_20260828.md): with DIFFERENT tokens per
            // replica the unmasked up-tree is only pure at slot c=0 of each
            // replica (its window [0,Dpad) never crosses). Isolate c=0 with
            // the ONE constant mask this norm needs (+1 level), then
            // replicate DOWN with rot(-p): every pull from below block-start
            // reads the previous replica's top slots, which are ZERO after
            // e0 — bleed-safe with no further masks while the spread <= Dpad.
            std::vector<double> e0v(SLOTS, 0.0);
            for (uint32_t r = 0; r < REP; r++) e0v[(size_t)r * Dpad] = 1.0;
            auto e0pt = mkPt(e0v, redLvl, __LINE__);
            red = cc->EvalMult(red, e0pt); cc->RescaleInPlace(red);
            redLvl += 1;
            for (int p = 1; p < (int)Dpad; p <<= 1) red = cc->EvalAdd(red, rot(red, -p));
        }
#endif
        // ms = red / d  (plaintext scale, folded into the seed evaluation)
        // Newton rsqrt: y_{n+1} = y_n (1.5 - 0.5 * ms * y_n^2)
        std::vector<double> invd(SLOTS, sigma / (double)d);   // msn = sigma*mean(x^2)
        auto invdpt = mkPt(invd, redLvl, __LINE__);
        TrackedCt ms{cc->EvalMult(red, invdpt), redLvl};
        cc->RescaleInPlace(ms.ct);         // ms = mean(x^2)
        // S3.7 V1 (2026-09-05, S3.6 T18 §3): ms = sigma * (mean(x^2) + eps). Until now
        // `eps` was accepted and never read, so the circuit normalised by
        // 1/sqrt(mean) while the model, the export's seed fit and the trainer's
        // own circuit sim all use mean + 1e-5 (train_fhe_native_ssm.py :153-:155,
        // :834, :927-:932). In the --ms-norm frame the seed is a/sqrt(sigma) +
        // (b/sigma^1.5) * msn, so with msn = sigma*(mean + eps) it reads exactly
        // a + b*(mean + eps) -- the frame the seeds and the iteration counts were
        // calibrated in. One plaintext add at ms's own level: zero levels, no
        // rotation, no change to the level trajectory (so no store-stamp change).
        // --no-norm-eps skips it and reproduces the recorded circuit.
        const double epsUsed = normEpsOn ? eps : 0.0;
        normEpsUsed = epsUsed;
        if (epsUsed != 0.0) {
            std::vector<double> epsv(SLOTS, sigma * epsUsed);
            auto epspt = mkPt(epsv, ms.level(), __LINE__);
            ms.ct = cc->EvalAdd(ms.ct, epspt);   // ms = sigma*(mean(x^2) + eps)
        }
        // seed y0 = a0 + b0*ms
        std::vector<double> a0v(SLOTS, a0), b0v(SLOTS, b0);
        auto b0pt = mkPt(b0v, ms.level(), __LINE__);
        TrackedCt y{cc->EvalMult(ms.ct, b0pt), ms.level()};
        cc->RescaleInPlace(y.ct);
        auto a0pt = mkPt(a0v, y.level(), __LINE__);
        y.ct = cc->EvalAdd(y.ct, a0pt);
        // S3.3 --newton-depth2 (Tier A-2): msh = (-0.5) ms, formed ONCE per site on a
        // copy of ms, so each iteration below is a = msh*y, b = y*y, t = a*b (depth 2
        // above y), y' = 1.5*y + t. Same polynomial as the stock chain, re-associated.
        TrackedCt msh;
        if (newtonDepth2) {
            std::vector<double> mhalf(SLOTS, -0.5);
            msh = ms; mulPt(msh, mhalf);
        }
        for (int it = 0; it < iters; it++) {
            // BUG FIXED 2026-07-18: this loop previously had no bootstrap
            // guard, so y.level() grew by ~4 EVERY iteration with no ceiling.
            // At the calibrated worst case (iters=7, seen on lane N's real
            // trained weights) that is ~28+ levels just for this loop, on
            // top of everything before/after it -- mathematically cannot fit
            // in depth=29 no matter how much headroom the CALLER's refresh()
            // reserves. It crashed with an unsigned-underflow vector::at()
            // inside OpenFHE's plaintext encoder once `used` exceeded depth
            // (the size-30 vector in the crash is the depth+1 RNS tower
            // array). The CPU validator (harness/cpu_real_model.cpp) never
            // caught this because its rmsnorm test used a fixed iters=3,
            // which happens to fit -- iters=7 was never exercised there.
            // Fix: refresh both operands to a bounded gap before each
            // iteration, so the per-iteration cost (~4-6 levels) is bounded
            // regardless of `iters`.
            if (newtonDepth2) {
                // S3.3 Tier A-2 iteration (depth 2 on y). The guards are the stock
                // ones (need 5) on y and on msh, which replaces ms in this form.
                refresh(y, 5); refresh(msh, 5);
                TrackedCt a{cc->EvalMult(msh.ct, y.ct), y.level()}; cc->RescaleInPlace(a.ct);      // (-0.5 ms) y
                TrackedCt b{cc->EvalMult(y.ct, y.ct), y.level()};   cc->RescaleInPlace(b.ct);      // y y
                TrackedCt tt{cc->EvalMult(a.ct, b.ct), a.level()};  cc->RescaleInPlace(tt.ct);     // -0.5 ms y^3
                std::vector<double> oneP5v(SLOTS, 1.5);
                auto op15 = mkPt(oneP5v, y.level(), __LINE__);
                TrackedCt y15{cc->EvalMult(y.ct, op15), y.level()}; cc->RescaleInPlace(y15.ct);    // 1.5 y
                y15.ct = cc->EvalAdd(y15.ct, tt.ct);            // FLEXIBLEAUTO reconciles the one-level gap
                y = y15;                                        // y' = 1.5 y - 0.5 ms y^3
                continue;
            }
            refresh(y, 5); refresh(ms, 5);
            // y2 = y*y ; t = ms*y2 ; y = y*(1.5 - 0.5 t)
            TrackedCt y2{cc->EvalMult(y.ct, y.ct), y.level()}; cc->RescaleInPlace(y2.ct);
            // align ms to y2.level()
            // 2026-07-19: the three manual "align one level at a time via
            // EvalMult-by-1.0 + Rescale" loops that used to live in this
            // function are GONE. They were the last instances of the pattern
            // already removed from addAligned: each step burns a level and
            // compounds rounding, so with iters=7 this function alone drove
            // ~16 bootstraps and still ran the ciphertext to used=28 of
            // depth=29 -- undecryptable. OpenFHE's FLEXIBLEAUTO reconciles
            // operand levels inside EvalMult/EvalAdd itself, so multiply
            // directly and let it do that.
            auto msAt = ms;
            TrackedCt tms{cc->EvalMult(msAt.ct, y2.ct), y2.level()}; cc->RescaleInPlace(tms.ct);
            std::vector<double> half(SLOTS, -0.5), oneP5(SLOTS, 1.5);
            auto halfpt = mkPt(half, tms.level(), __LINE__);
            tms.ct = cc->EvalMult(tms.ct, halfpt); cc->RescaleInPlace(tms.ct);
            auto op5 = mkPt(oneP5, tms.level(), __LINE__);
            tms.ct = cc->EvalAdd(tms.ct, op5);
            // y = y * tms  (align y up to tms.level())
            y.ct = cc->EvalMult(y.ct, tms.ct); cc->RescaleInPlace(y.ct);
        }
        // 2026-07-19: bound the level the Newton loop hands to the tail
        // multiplies. Without this, rmsnorm returned a ciphertext at used=28
        // of depth=29 (remaining=1): the NEXT refresh() then had no choice but
        // to bootstrap an almost-exhausted ciphertext, which FLEXIBLEAUTO
        // turns into garbage -- BOOT_FLOOR cannot save a ciphertext that is
        // already that deep, it can only stop one from getting there. Refresh
        // while there is still room, so the value handed downstream is always
        // bootstrappable.
        if (schedParentFirst) {                       // S3.3 P2 tail lift: need 12 (tm) / 7+margin (cm) / 4 (out)
            const int b0 = boots; refresh(y, pfNeedTail(normKind)); pfBoots += boots - b0;
        } else refresh(y, 4);
        // out = x * y * g
        TrackedCt out{cc->EvalMult(x.ct, y.ct), x.level()}; cc->RescaleInPlace(out.ct);
        if (rtSigma != 1.0) {                 // rsqrt(ms) = sqrt(sigma)*rsqrt(msn)
            std::vector<double> gs(g.size());
            for (size_t i = 0; i < g.size(); i++) gs[i] = g[i] * rtSigma;
            mulPt(out, gs);
        } else {
            mulPt(out, g);
        }
        evalMs += msSince(t);
        normBootMs += bootMs - normBoot0;         // S3.7 V2: the window's own boots, counted once in bootMs
        return out;
    };

#ifdef FHE_SSM_DEMO_SER
    // ---------------- T1c ACCEPTANCE PROBE (--lane-test-block) ---------------
    // Four DISTINCT lanes in block layout run ONE dual-half matvec + ONE
    // e0-masked rmsnorm; each lane is then compared against a BROADCAST
    // reference run of that lane alone (the certified path). Same circuit,
    // same seeds, same Newton error on both sides, so the equality bar is
    // tight. Distinct mvLayer markers keep the two modes' store keys apart.
    if (laneTestBlock) {
        const int NL = 4;
        std::vector<std::vector<double>> lanev(NL, std::vector<double>(Dpad, 0.0));
        for (int r = 0; r < NL; r++)
            for (int c = 0; c < d; c++)
                lanev[r][c] = 0.4 * std::sin(0.7 * c + 1.3 * r) + 0.2 * std::cos(0.11 * c - 0.5 * r);
        std::vector<double> W((size_t)d * Dpad, 0.0);
        for (int r = 0; r < d; r++)
            for (int c = 0; c < d; c++)
                W[(size_t)r * Dpad + c] = 0.004 * std::cos(0.3 * r - 0.9 * c);
        std::vector<double> gOnes(SLOTS, 1.0);
        const double pa0 = 1.2, pb0 = -0.35; const int pit = 3; const double peps = 1e-5;
        auto runOnce = [&](const std::vector<double>& sv, int marker,
                           bool multiLane) -> std::vector<double> {
            const bool savedLB = lanesBlock;
            lanesBlock = multiLane;
            auto pt0 = cc->MakeCKKSPackedPlaintext(sv);
            TrackedCt u{cc->Encrypt(keys.publicKey, pt0), 0};
            mvLayer = marker; mvOpId = 7; mvChunk = 0;
            std::vector<TrackedCt> us{u};
            auto yv = matvecBatch(us, W, d, (int)Dpad, 0);
            mvOpId = -1;
            auto yn = rmsnorm(yv[0], gOnes, pa0, pb0, pit, peps);
            lanesBlock = savedLB;
            Plaintext p; cc->Decrypt(keys.secretKey, yn.ct, &p);
            p->SetLength(SLOTS);
            return p->GetRealPackedValue();
        };
        // multi-lane pass: slot r*Dpad + c = lane r channel c
        std::vector<double> svAll(SLOTS, 0.0);
        for (int r = 0; r < NL; r++)
            for (uint32_t c = 0; c < Dpad; c++)
                svAll[(size_t)r * Dpad + c] = lanev[r][c];
        auto vAll = runOnce(svAll, 99, true);
        double worst = 0.0; bool anyNonFinite = false;
        for (int r = 0; r < NL; r++) {
            auto vRef = runOnce(packCh(lanev[r].data(), (int)Dpad), 90 + r, false);
            double num = 0, den = 0;
            for (int c = 0; c < d; c++) {
                const double a = vAll[(size_t)r * Dpad + c];
                const double b = vRef[c];   // broadcast: channel c of replica 0
                if (!std::isfinite(a) || !std::isfinite(b)) anyNonFinite = true;
                num += (a - b) * (a - b); den += b * b;
            }
            const double rel = std::sqrt(num / std::max(den, 1e-300));
            std::cout << "{\"laneTestBlock\":true,\"lane\":" << r
                      << ",\"relErrRmsVsBroadcast\":" << rel << "}" << std::endl;
            worst = std::max(worst, rel);
        }
        std::cout << "{\"laneTestBlockSummary\":true,\"lanes\":" << NL
                  << ",\"worstRelErrRms\":" << worst
                  << ",\"nonFinite\":" << (anyNonFinite ? 1 : 0)
                  << ",\"storeHits\":" << storeHits << ",\"storeMisses\":" << storeMisses
                  << ",\"pass\":" << ((worst < 5e-3 && !anyNonFinite) ? "true" : "false")
                  << "}" << std::endl;
        return (worst < 5e-3 && !anyNonFinite) ? 0 : 3;
    }
#endif

    // ---------------- MATVEC PRECISION PROBE (--matvec-probe) ----------------
    // The selftest matvec reports 4.4e-12 while the in-graph wv matvec reports
    // 1.5e-2 -- 10 orders of magnitude, same function. THREE variables differ
    // simultaneously, so nothing so far isolates the cause:
    //   (1) DENSITY. The selftest W is BIDIAGONAL (W[i][i]=0.9, W[i][i+1]=0.1),
    //       so `if (!nonzero) continue` skips 1022 of 1024 diagonals: it sums
    //       2 terms. The real wv sums 1024. This confound was never controlled.
    //   (2) LEVEL. Selftest runs at level ~0 on a fresh ciphertext; in-graph
    //       runs at level ~20.
    //   (3) BOOTSTRAP HISTORY. In-graph inputs have been through the Newton
    //       rsqrt loop and at least one bootstrap; selftest inputs have not.
    // This sweeps all three independently (2 x 3 = 6 cells).
    if (matvecProbe) {
        std::vector<double> v(Dpad, 0.0);
        for (int i = 0; i < d; i++) v[i] = 0.5 * std::sin(0.7 * i) + 0.3 * std::cos(0.13 * i);
        auto mkW = [&](bool dense) {
            std::vector<double> W((size_t)d * Dpad, 0.0);
            if (!dense) {   // the SELFTEST matrix: 2 nonzero diagonals
                for (int i = 0; i < d; i++) { W[(size_t)i * Dpad + i] = 0.9;
                    if (i + 1 < (int)Dpad) W[(size_t)i * Dpad + i + 1] = 0.1; }
            } else {        // dense, rms matched to the real scaled wv (~0.228)
                uint64_t st = 12345;
                for (size_t i = 0; i < (size_t)d * Dpad; i++) {
                    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
                    W[i] = 0.228 * (2.0 * ((double)((st >> 33) & 0xFFFFFF) / 16777216.0) - 1.0) * 1.732;
                }
            }
            return W;
        };
        auto refOf = [&](const std::vector<double>& W, const std::vector<double>& in) {
            std::vector<double> r(d, 0.0);
            for (int i = 0; i < d; i++)
                for (uint32_t j = 0; j < Dpad; j++) r[i] += W[(size_t)i * Dpad + j] * in[j];
            return r;
        };
        std::cout << "{\"probe\":\"matvec\",\"ringDim\":" << cc->GetRingDimension()
                  << ",\"depth\":" << depth << ",\"d\":" << d << ",\"Dpad\":" << Dpad << "}" << std::endl;
        for (int dense = 0; dense < 2; dense++) {
            auto W = mkW(dense != 0);
            for (int cond = 0; cond < 3; cond++) {
                const char* nm = (cond == 0 ? "fresh" : cond == 1 ? "deep_noboot" : "bootstrapped");
                try {
                    auto c = encCh(packCh(v.data(), d));
                    // the ciphertext that FEEDS the matvec; its plaintext value is
                    // whatever we decrypt from it, so the reference is exact and
                    // input error can never masquerade as matvec error.
                    if (cond == 1) {              // drive DOWN to ~20 with no bootstrap
                        for (int k = 0; k < 20; k++) {
                            auto one = mkPt(ONES, c.level(), __LINE__);
                            c.ct = cc->EvalMult(c.ct, one); cc->RescaleInPlace(c.ct);
                        }
                    } else if (cond == 2) {       // exactly one bootstrap
                        for (int k = 0; k < 20; k++) {
                            auto one = mkPt(ONES, c.level(), __LINE__);
                            c.ct = cc->EvalMult(c.ct, one); cc->RescaleInPlace(c.ct);
                        }
                        boot(c);
                    }
                    auto inv = decCh(c);                       // TRUE matvec input
                    auto y = matvec(c, W, d, (int)Dpad, 0);
                    auto o = decCh(y);
                    auto r = refOf(W, inv);
                    double num = 0, den = 0; long long nf = 0;
                    for (int i = 0; i < d; i++) {
                        if (!std::isfinite(o[i]) || !std::isfinite(r[i])) { nf++; continue; }
                        num = std::max(num, std::abs(o[i] - r[i]));
                        den = std::max(den, std::abs(r[i]));
                    }
                    std::cout << "{\"probe\":\"matvec\",\"dense\":" << (dense ? "true" : "false")
                              << ",\"cond\":\"" << nm << "\",\"inLevel\":" << c.level()
                              << ",\"outLevel\":" << y.level() << ",\"nonFinite\":" << nf
                              << ",\"maxAbsErr\":" << num << ",\"refMax\":" << den
                              << ",\"relErr\":" << num / (den + 1e-12) << "}" << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "{\"probe\":\"matvec\",\"dense\":" << (dense ? "true" : "false")
                              << ",\"cond\":\"" << nm << "\",\"throw\":\"" << e.what() << "\"}" << std::endl;
                }
            }
        }
        return 0;
    }

    // ------------------------------------------------------------- selftest
    auto maxAbs = [&](const std::vector<double>& a, const std::vector<double>& b, int n) {
        double e = 0; for (int i = 0; i < n; i++) e = std::max(e, std::abs(a[i] - b[i])); return e; };
    std::cout << "{\"harness\":\"gpu_real_model\",\"arch\":\"" << arch << "\",\"tag\":\"" << tag
              << "\",\"secure\":" << (secure ? "true" : "false") << ",\"ringDim\":" << cc->GetRingDimension()
              << ",\"slots\":" << SLOTS << ",\"d\":" << d << ",\"Dpad\":" << Dpad << ",\"REP\":" << REP
              << ",\"L\":" << L << ",\"dff\":" << dff << ",\"BS\":" << BS
              << ",\"haveBundle\":" << (haveBundle ? "true" : "false")
              << ",\"ompMaxThreads\":" << ompThreads
              << ",\"scaleBits\":" << scaleBits
              << ",\"normEpsOn\":" << (normEpsOn ? "true" : "false")   // S3.7 V1, additive
              << ",\"setupMs\":" << (int)setupMs << "}" << std::endl;

    // ---- bootstrap PRECISION ladder (--boot-ladder) ------------------------
    // Identical cycle to harness/cpu_boot_ladder.cpp so the two backends are
    // directly comparable: same ring/depth/levelBudget/scaling/key dist, and
    // the same consume-then-bootstrap pattern the real circuit performs.
    // Motivation: the composed layer passes under stock CPU OpenFHE, but here
    // the wv matvec returns ~12% relative error vs a plaintext reference --
    // real corruption. FIDESlib patches ckksrns-fhe.cpp (the bootstrapping
    // implementation), so per-bootstrap precision is the prime suspect.
    if (bootLadder) {
        // 2026-07-29: --amp sweeps the MESSAGE MAGNITUDE. The ladder was
        // hard-coded to 1.5, but the real model's layer-0 activations have
        // RMS ~0.028 (emb.weight std 0.031 => ms ~8e-4, confirmed against
        // native_seeds.json L0.tm ms_range [5.1e-4, 1.5e-3]). If CKKS
        // bootstrap error is ABSOLUTE, a 53x smaller message means a 53x
        // larger RELATIVE error -- which would explain why the synthetic
        // harness (activations ~0.29) survives and the trained model does not.
        // This flag measures that curve instead of assuming it.
        std::vector<double> ref(SLOTS);
        for (uint32_t i = 0; i < SLOTS; i++) ref[i] = std::sin(0.37 * (i % 512)) * bootAmp;
        auto pt0 = cc->MakeCKKSPackedPlaintext(ref);
        TrackedCt c{cc->Encrypt(keys.publicKey, pt0)};
        std::cout << "{\"harness\":\"gpu_boot_ladder\",\"backend\":\"fideslib\",\"ringDim\":"
                  << cc->GetRingDimension() << ",\"depth\":" << depth
                  << ",\"amp\":" << bootAmp
                  << ",\"consumePerCycle\":5}" << std::endl;
        for (uint32_t n = 1; n <= 50; n++) {
            for (int k = 0; k < 5; k++) {
                auto op = mkPt(ONES, c.level(), __LINE__);
                c.ct = cc->EvalMult(c.ct, op); cc->RescaleInPlace(c.ct);
            }
            boot(c);
            if (n == 1 || n == 5 || n == 10 || n == 20 || n == 35 || n == 50) {
                try {
                    auto v = decCh(c);
                    double e = 0; for (uint32_t i = 0; i < SLOTS; i++) e = std::max(e, std::abs(v[i] - ref[i]));
                    std::cout << "{\"boots\":" << n << ",\"level\":" << c.level()
                              << ",\"amp\":" << bootAmp
                              << ",\"maxAbsErr\":" << e
                              << ",\"relErr\":" << e / bootAmp << "}" << std::endl;
                } catch (const std::exception&) {
                    std::cout << "{\"boots\":" << n << ",\"level\":" << c.level()
                              << ",\"DECODE_FAIL\":true}" << std::endl;
                }
            }
        }
        return 0;
    }

    // ---- --bridge-test LANE: the state-layout bridge, ON DEVICE -------------
    // Block layout keeps channel c at slot r*Dpad+c for EVERY replica r; the
    // interleave layout wants channel c at slot c*REP + LANE. Bridging lets one
    // packed pass start from the session carry, which after D14 (block is the
    // demo) is the only way speculative decoding reaches the demo at all.
    //
    // Construction, same as the CPU probe: for channel c the target slot is
    // c*REP + LANE, and the source has s_c at every replica, so pick the
    // replica r(c) = floor((c*(REP-1)+LANE)/Dpad) that puts the move within one
    // Dpad stride, giving delta(c) = (c*(REP-1)+LANE) mod Dpad. Group channels
    // by delta, BSGS-decompose delta = BS*hi + lo, mask once per (lo,hi) group
    // and rotate. ONE mask level total.
    if (bridgeTest >= 0) {
        const int LANE = bridgeTest;
        if (LANE >= (int)REP) {
            std::cerr << "{\"fatal\":\"--bridge-test LANE \" " << LANE << " >= REP " << REP << "}\n";
            return 2;
        }
        auto tB0 = Clock::now();
        std::vector<double> s((size_t)Dpad);
        for (int c = 0; c < (int)Dpad; c++)
            s[c] = 0.3 * std::sin(0.11 * c) + 0.05 * std::cos(0.7 * c);
        std::vector<double> src(SLOTS, 0.0);
        for (uint32_t r = 0; r < REP; r++)
            for (int c = 0; c < (int)Dpad; c++) src[(size_t)r * Dpad + c] = s[c];
        auto ct = encCh(src);                       // encCh takes a SLOT vector

        std::map<int, std::map<int, std::vector<double>>> masks;   // lo -> hi -> mask
        for (int c = 0; c < (int)Dpad; c++) {
            const long long t = (long long)((int)REP - 1) * c + LANE;
            const int r = (int)(t / (long long)Dpad);
            const int delta = (int)(t % (long long)Dpad);
            auto& m = masks[delta % BS][delta / BS];
            if (m.empty()) m.assign(SLOTS, 0.0);
            m[(size_t)r * Dpad + c] = 1.0;          // pick THAT broadcast copy
        }
        Ciphertext<DCRTPoly> out;
        int nRot = 0, nMask = 0;
        const uint32_t srcLvl = ct.ct->GetLevel();
        for (auto& loEntry : masks) {
            const int lo = loEntry.first;
            Ciphertext<DCRTPoly> acc;
            for (auto& hiEntry : loEntry.second) {
                const int hi = hiEntry.first;
                // ct x pt exactly as mulPt does it: the plaintext must be minted
                // AT the operand's level (the 3-arg MakeCKKSPackedPlaintext),
                // then rescaled. The 1-arg form does not match any FIDESlib
                // EvalMult overload.
                auto pt = ptCh(hiEntry.second, srcLvl);
                auto piece = cc->EvalMult(ct.ct, pt);
                cc->RescaleInPlace(piece);
                nMask++;
                if (hi) { piece = cc->EvalRotate(piece, -hi * BS); nRot++; }
                acc = acc ? cc->EvalAdd(acc, piece) : piece;
            }
            if (lo) { acc = cc->EvalRotate(acc, -lo); nRot++; }
            out = out ? cc->EvalAdd(out, acc) : acc;
        }
        const double bridgeMs = msSince(tB0);

        Plaintext dec;
        cc->Decrypt(keys.secretKey, out, &dec);
        dec->SetLength(SLOTS);
        const auto v = dec->GetRealPackedValue();
        double num = 0, den = 0, leak = 0;
        for (int c = 0; c < (int)Dpad; c++) {
            const double got = v[(size_t)c * REP + LANE];
            num += (got - s[c]) * (got - s[c]);
            den += s[c] * s[c];
        }
        for (uint32_t i = 0; i < SLOTS; i++) {
            const bool isTarget = (i % REP) == (uint32_t)LANE && (i / REP) < (uint32_t)Dpad;
            if (!isTarget) leak = std::max(leak, std::abs(v[i]));
        }
        const double rel = std::sqrt(num / std::max(den, 1e-300));
        const bool pass = (rel < 1e-4 && leak < 1e-4);   // GPU bar: the CPU probe
                                                         // hit 3.4e-13, but this
                                                         // runs at real depth on
                                                         // real keys, so the bar
                                                         // is the circuit's own
                                                         // fidelity class, not
                                                         // CPU exactness.
        std::cout << "{\"bridgeTest\":true,\"device\":true"
                  << ",\"ringDim\":" << cc->GetRingDimension()
                  << ",\"REP\":" << REP << ",\"Dpad\":" << Dpad
                  << ",\"lane\":" << LANE
                  << ",\"relErrRms\":" << rel
                  << ",\"maxOffTargetLeak\":" << leak
                  << ",\"rotations\":" << nRot << ",\"maskMults\":" << nMask
                  << ",\"maskLevels\":1"
                  << ",\"levelIn\":" << ct.ct->GetLevel()
                  << ",\"levelOut\":" << out->GetLevel()
                  << ",\"bridgeMs\":" << bridgeMs
                  << ",\"pass\":" << (pass ? "true" : "false") << "}" << std::endl;
        return pass ? 0 : 3;
    }

    if (selftest) {
        int hard = 0, soft = 0;
        auto verdict = [&](const char* n, bool ok, double err, bool isHard) {
            std::cout << "{\"selftest\":\"" << n << "\",\"result\":\"" << (ok ? "ok" : "fail")
                      << "\",\"err\":" << err << "}" << std::endl;
            if (!ok) (isHard ? hard : soft)++;
        };
        // random channel vector
        std::vector<double> v(Dpad); for (int i = 0; i < (int)Dpad; i++) v[i] = (i < d) ? std::sin(0.1 * i) * 0.5 : 0.0;
        // 1. pack/enc/dec roundtrip
        try { auto c = encCh(packCh(v.data(), d)); auto o = decCh(c);
              // decCh returns CHANNEL-indexed data (it de-interleaves), so the
              // reference must be v itself, not the slot-basis packCh(v). Under
              // the replicated layout these coincided (slot i == channel i for
              // r=0); under interleaving they do not. Test-side fix only.
              std::vector<double> ref = interleave ? v : packCh(v.data(), d);
              const uint32_t NCMP = interleave ? (uint32_t)d : SLOTS;
              verdict("encdec", maxAbs(o, ref, NCMP) < 1e-6, maxAbs(o, ref, NCMP), true);
        } catch (const std::exception& e) { std::cout << "{\"selftest\":\"encdec\",\"throw\":\"" << e.what() << "\"}\n"; hard++; }
        // 2. rotation correctness within the replicated block
        try { auto c = encCh(packCh(v.data(), d)); TrackedCt rc{rot(c.ct, 1), 0}; auto o = decCh(rc);
              double e = std::abs(o[0] - v[1 % d]); verdict("rotate", e < 1e-4, e, false);
        } catch (const std::exception& e) { std::cout << "{\"selftest\":\"rotate\",\"throw\":\"" << e.what() << "\"}\n"; soft++; }
        // 3. dense matvec vs plaintext reference (small identity-ish W)
        try {
            int rows = d, cols = d;
            std::vector<double> W((size_t)rows * cols, 0.0);
            for (int i = 0; i < d; i++) { W[(size_t)i * cols + i] = 0.9; if (i + 1 < d) W[(size_t)i * cols + i + 1] = 0.1; }
            auto c = encCh(packCh(v.data(), d));
            auto y = matvec(c, W, rows, cols, 0);
            auto o = decCh(y);
            std::vector<double> ref(d, 0.0);
            for (int i = 0; i < d; i++) for (int j = 0; j < cols; j++) ref[i] += W[(size_t)i * cols + j] * (j < d ? v[j] : 0.0);
            double e = 0; for (int i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - ref[i]));
            verdict("matvec", e < 1e-3, e, true);
        } catch (const std::exception& e) { std::cout << "{\"selftest\":\"matvec\",\"throw\":\"" << e.what() << "\"}\n"; hard++; }
        // 4. RMSNorm vs plaintext reference
        try {
            std::vector<double> g(Dpad, 1.0);
            auto c = encCh(packCh(v.data(), d));
            auto y = rmsnorm(c, packCh(g.data(), d), 1.0, 0.0, 4, 1e-5);
            auto o = decCh(y);
            double ms = 0; for (int i = 0; i < d; i++) ms += v[i] * v[i]; ms /= d;
            double inv = 1.0 / std::sqrt(ms + 1e-5);
            double e = 0; for (int i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - v[i] * inv));
            verdict("rmsnorm", e < 1e-2, e, false);   // Newton from generic seed: loose bar
        } catch (const std::exception& e) { std::cout << "{\"selftest\":\"rmsnorm\",\"throw\":\"" << e.what() << "\"}\n"; soft++; }
        // 4b. S3.7 V1 (2026-09-05): the SAME norm with the seed the demo actually runs.
        // Test 4's generic seed (1, 0) is under-converged after 4 steps -- 0.0310645
        // with eps, 0.0310171 without (harness/norm_eps_check.py; S3.6 T18 §3) -- so
        // its verdict never said anything about the circuit. With a bundle loaded,
        // take L0.tm's (a, b, iters) and scale v so that mean(v^2) = -a/(16 b): inside
        // the seed's design window near its low end (the zero crossing -a/b sits at
        // 1.34-1.45x the window's top, T18 §4; pbd430a L0.tm: 2.4e-4 in [1.1e-4, 2.5e-3]),
        // where eps/ms ~ 4 %. Predicted: the no-eps chain misses the 1e-2 bar (0.028),
        // the eps chain sits at the Newton floor -- a real pass/fail. Skipped, i.e.
        // the old behaviour, when no bundle is loaded (--self-d) or the bundle has no
        // L0.tm.rsqrt (B.has(): Bundle::ptr() exits the process on a missing tensor,
        // so the try/catch below would not cover it).
        if (haveBundle && B.has("L0.tm.rsqrt")) {
            try {
                auto tr = B.vec("L0.tm.rsqrt");
                const double sa = tr[0], sb = tr[1]; const int sit = (int)tr[2];
                const double msTest = -sa / (sb * 16.0);
                if (!std::isfinite(msTest) || msTest <= 0.0) throw std::runtime_error("seed has no positive ms");
                double msv = 0; for (int i = 0; i < d; i++) msv += v[i] * v[i]; msv /= d;
                const double sc = std::sqrt(msTest / msv);
                std::vector<double> vs(Dpad, 0.0); for (int i = 0; i < d; i++) vs[i] = v[i] * sc;
                std::vector<double> g(Dpad, 1.0);
                auto c = encCh(packCh(vs.data(), d));
                auto y = rmsnorm(c, packCh(g.data(), d), sa, sb, sit, 1e-5);
                auto o = decCh(y);
                double inv = 1.0 / std::sqrt(msTest + 1e-5);
                double e = 0; for (int i = 0; i < d; i++) e = std::max(e, std::abs(o[i] - vs[i] * inv));
                std::cout << "{\"selftestSeeded\":true,\"site\":\"L0.tm\",\"a\":" << sa << ",\"b\":" << sb
                          << ",\"iters\":" << sit << ",\"msTest\":" << msTest << ",\"normEps\":" << normEpsUsed
                          << "}" << std::endl;
                verdict("rmsnormSeeded", e < 1e-2, e, false);
            } catch (const std::exception& e) { std::cout << "{\"selftest\":\"rmsnormSeeded\",\"throw\":\"" << e.what() << "\"}\n"; soft++; }
        }
        // 6. LEVEL-MISMATCH ADD -- the decisive backend test (added 2026-07-19).
        // harness/cpu_layer_compose.cpp runs this EXACT composed layer under
        // plain CPU OpenFHE, same depth/levelsAfter/iters/ring/op-order, and
        // every stage decrypts (failedStages 0, 49 boots) -- including the
        // ffn accumulation that fails here. So the circuit algebra and level
        // management are correct, and the difference must be the backend.
        // The one assumption addAligned makes is that EvalAdd reconciles
        // operands at DIFFERENT levels internally (OpenFHE FLEXIBLEAUTO does).
        // This isolates that, and separately checks whether the align-by-
        // multiply-by-1.0 path is itself lossy, by decrypting the driven-down
        // operand BEFORE the add.
        for (int gap : {1, 5, 22}) {
            try {
                auto a = encCh(packCh(v.data(), d));
                auto b = encCh(packCh(v.data(), d));
                for (int i = 0; i < gap; i++) {
                    auto op = mkPt(ONES, b.level(), __LINE__);
                    b.ct = cc->EvalMult(b.ct, op); cc->RescaleInPlace(b.ct);
                }
                auto ob = decCh(b);                       // did alignTo itself destroy it?
                double eb = 0; for (int i = 0; i < d; i++) eb = std::max(eb, std::abs(ob[i] - v[i]));
                TrackedCt sum{cc->EvalAdd(a.ct, b.ct)};   // now the mismatched add
                auto os = decCh(sum);
                double es = 0; for (int i = 0; i < d; i++) es = std::max(es, std::abs(os[i] - 2.0 * v[i]));
                std::cout << "{\"selftest\":\"levelMismatch\",\"gap\":" << gap
                          << ",\"aLvl\":" << a.level() << ",\"bLvl\":" << b.level()
                          << ",\"alignOnlyErr\":" << eb << ",\"mismatchedAddErr\":" << es
                          << ",\"result\":\"" << ((eb < 1e-2 && es < 1e-2) ? "ok" : "fail")
                          << "\"}" << std::endl;
                if (!(eb < 1e-2 && es < 1e-2)) soft++;
            } catch (const std::exception& ex) {
                std::cout << "{\"selftest\":\"levelMismatch\",\"gap\":" << gap
                          << ",\"throw\":\"" << ex.what() << "\"}" << std::endl;
                soft++;
            }
        }
        // 5. bootstrap of a used ciphertext
        try { auto c = encCh(packCh(v.data(), d));
              std::vector<double> ref = interleave ? v : packCh(v.data(), d);   // channel basis
              for (int i = 0; i < 3; i++) { std::vector<double> one(SLOTS,1.0); auto op=mkPt(one, c.level(), __LINE__); c.ct=cc->EvalMult(c.ct,op); cc->RescaleInPlace(c.ct); }
              boot(c); auto o = decCh(c); verdict("bootstrap", maxAbs(o, ref, d) < 5e-3, maxAbs(o, ref, d), true);
        } catch (const std::exception& e) { std::cout << "{\"selftest\":\"bootstrap\",\"throw\":\"" << e.what() << "\"}\n"; hard++; }
        std::cout << "{\"selftestSummary\":true,\"hardFail\":" << hard << ",\"softFail\":" << soft
                  << ",\"peakVramGB\":" << peakVramGB() << "}" << std::endl;
        return hard ? 3 : (soft ? 4 : 0);
    }

    // ------------------------------------------------ ENCODE COST MODEL
    // --encode-bench. The forward pass spends 99.7% of its time outside
    // bootstrapping (gpuBootstrapMs 890 of 354,398) in ~135k diagonal
    // plaintexts, but "encoding" lumps together three separable costs:
    //   (1) MakeCKKSPackedPlaintext   -- host DFT + CRT into (depth+1-lvl) limbs
    //   (2) host->device plaintext load  -- ~9.8 MB each ("Plaintexts loaded:
    //       378 ~ 3685MB"), charged on first use
    //   (3) the GPU ct x pt multiply itself
    // Optimizing the wrong one wastes the session, so measure all three
    // directly, and measure (1) vs level (limb count) and vs sparse slots.
    if (encodeBench) {
        const int NREP = 24;
        std::vector<double> diag(SLOTS, 0.0);
        for (uint32_t i = 0; i < SLOTS; i++) diag[i] = 0.01 * (double)((i * 37) % 97);
        // (1) encode cost vs level -> limbs = depth+1-lvl
        for (uint32_t lvl : {0u, 5u, 10u, 15u, 20u, 25u, (uint32_t)(depth > 30 ? 30 : depth)}) {
            if (lvl > depth) continue;
            auto t0 = Clock::now();
            for (int r = 0; r < NREP; r++) { auto p = cc->MakeCKKSPackedPlaintext(diag, 1, lvl); (void)p; }
            double ms = msSince(t0) / NREP;
            std::cout << "{\"encodeBench\":\"level\",\"level\":" << lvl
                      << ",\"limbs\":" << (depth + 1 - lvl) << ",\"msPerEncode\":" << ms
                      << ",\"projFwdSec\":" << ms * 135168 / 1000.0 << "}" << std::endl;
        }
        // (1b) sparse slots: the diagonal only has Dpad distinct values, so ask
        // whether a slots=Dpad encode is cheaper than a full slots=SLOTS one.
        for (uint32_t sl : {(uint32_t)Dpad, SLOTS / 4, SLOTS}) {
            if (sl > SLOTS) continue;
            std::vector<double> v(sl, 0.0);
            for (uint32_t i = 0; i < sl; i++) v[i] = 0.01 * (double)((i * 37) % 97);
            auto t0 = Clock::now();
            for (int r = 0; r < NREP; r++) { auto p = cc->MakeCKKSPackedPlaintext(v, 1, 20, nullptr, sl); (void)p; }
            std::cout << "{\"encodeBench\":\"slots\",\"slots\":" << sl
                      << ",\"msPerEncode\":" << (msSince(t0) / NREP) << "}" << std::endl;
        }
        // T0.1 (S3.1, --sync-timers only; the unflagged bench is untouched):
        // (1c) encode of a BLOCK-LAYOUT (Dpad-periodic) diagonal at the same
        // levels. D-5 measured its coefficient form exactly stride-REP sparse
        // (LIMB_VERDICT_20260814.md); this arm asks whether OpenFHE's encoder
        // sees any of that structure (expected: no — same iFFT either way),
        // which is the honest baseline for a custom periodic encoder.
        if (syncTimers) {
            std::vector<double> pdiag(SLOTS, 0.0);
            for (uint32_t i = 0; i < SLOTS; i++) pdiag[i] = 0.01 * (double)(((i % Dpad) * 37) % 97);
            for (uint32_t lvl : {0u, 15u, 25u}) {
                if (lvl > depth) continue;
                auto t0 = Clock::now();
                for (int r = 0; r < NREP; r++) { auto p = cc->MakeCKKSPackedPlaintext(pdiag, 1, lvl); (void)p; }
                std::cout << "{\"encodeBench\":\"levelPeriodic\",\"level\":" << lvl
                          << ",\"limbs\":" << (depth + 1 - lvl)
                          << ",\"msPerEncode\":" << (msSince(t0) / NREP) << "}" << std::endl;
            }
        }
        // (2)+(3) fresh plaintext per multiply (encode + load + mult) vs the
        // SAME plaintext reused (load charged once) vs multiply only.
        // T0.1: under --sync-timers each timed region is bracketed by device
        // drains, so the fresh/reuse split (and loadMsImplied) measures the
        // work itself rather than async dispatch. The overnight's 4.79/4.98/
        // 0.0122 split was taken WITHOUT sync — different instrument.
        {
            std::vector<double> ones(SLOTS, 1.0);
            auto cbase = encCh(ones);
            syncDev();
            auto t0 = Clock::now();
            for (int r = 0; r < NREP; r++) {
                auto p = cc->MakeCKKSPackedPlaintext(diag, 1, cbase.level());
                auto y = cc->EvalMult(cbase.ct, p); (void)y;
            }
            syncDev();
            double freshMs = msSince(t0) / NREP;
            auto pshared = cc->MakeCKKSPackedPlaintext(diag, 1, cbase.level());
            { auto y0 = cc->EvalMult(cbase.ct, pshared); (void)y0; }   // pay the load once
            syncDev();
            auto t1 = Clock::now();
            for (int r = 0; r < NREP; r++) { auto y = cc->EvalMult(cbase.ct, pshared); (void)y; }
            syncDev();
            double reuseMs = msSince(t1) / NREP;
            auto t2 = Clock::now();
            for (int r = 0; r < NREP; r++) { auto p = cc->MakeCKKSPackedPlaintext(diag, 1, cbase.level()); (void)p; }
            double encOnlyMs = msSince(t2) / NREP;
            std::cout << "{\"encodeBench\":\"applyMix\",\"level\":" << cbase.level()
                      << ",\"sync\":" << (syncTimers ? "true" : "false")
                      << ",\"encodeOnlyMs\":" << encOnlyMs
                      << ",\"multReusedPtMs\":" << reuseMs
                      << ",\"encodePlusLoadPlusMultMs\":" << freshMs
                      << ",\"loadMsImplied\":" << (freshMs - reuseMs - encOnlyMs)
                      << ",\"projFwdSecFresh\":" << freshMs * 135168 / 1000.0 << "}" << std::endl;
        }
        std::cout << "{\"encodeBenchDone\":true}" << std::endl;
        return 0;
    }

    // ------------------------------------------------ STAGE 2 lane test
    // Empirically settles the EvalRotate sign convention for LANE (stride-1)
    // rotations on THIS backend. Stage 1 only ever issues positive rotation
    // indices (multiples of REP and the BSGS steps), so the negative-index
    // path that rotTok relies on has never been exercised on FIDESlib.
    // Pattern: slot[c*REP + t] = 100*c + t. rotTok(ct,1) claims lane t
    // receives lane t-1, i.e. ch0 should read [garbage, 0, 1, 2, ...].
    if (laneTest) {
        std::vector<double> pat(SLOTS, 0.0);
        for (int c2 = 0; c2 < 3; c2++)
            for (uint32_t t2 = 0; t2 < REP; t2++)
                pat[(size_t)c2 * REP + t2] = 100.0 * c2 + t2;
        auto ct0 = encCh(pat);
        auto dump = [&](Ciphertext<DCRTPoly> c3, const std::string& label) {
            Plaintext p; cc->Decrypt(keys.secretKey, c3, &p);
            p->SetLength(SLOTS); auto vv = p->GetRealPackedValue();
            const int NT = (int)std::min<uint32_t>(REP, 8);
            std::cout << "{\"laneTest\":\"" << label << "\",\"ch0\":[";
            for (int t2 = 0; t2 < NT; t2++) std::cout << (t2 ? "," : "") << std::round(vv[t2] * 100) / 100;
            std::cout << "],\"ch1\":[";
            for (int t2 = 0; t2 < NT; t2++) std::cout << (t2 ? "," : "") << std::round(vv[REP + t2] * 100) / 100;
            std::cout << "]}" << std::endl;
        };
        dump(ct0.ct, "identity");
        dump(rotTok(ct0.ct, 1), "rotTok_j1");            // expect ch0 = [*, 0, 1, 2, ...]
        dump(rotTok(ct0.ct, 2), "rotTok_j2");            // expect ch0 = [*, *, 0, 1, ...]
        dump(cc->EvalRotate(ct0.ct, 1), "EvalRotate_plus1"); // doc: expect lane t <- t+1
        dump(rot(ct0.ct, 1), "chanRot_k1");              // channel rot: ch0 should become 100+t
        std::cout << "{\"laneTestDone\":true}" << std::endl;
        return 0;
    }

    if (arch != "native") {
        std::cout << "{\"note\":\"rwkv boundary lane runs via the CPU/sim path "
                     "(ml-eval/fhe_circuit_sim.py); this GPU harness implements the "
                     "in-class native forward. See REMOTE_REAL_MODEL_PROMPT.md.\"}" << std::endl;
        return 0;
    }

    // ------------------------------------------------ the native forward
    // Inputs: bundle "inputs" is (T,d) embedded rows (client-side embedding).
    auto inShape = B.shape("inputs");
    int Ttot = inShape[0];
    const double* inp = B.ptr("inputs");
    const double* logitsRef = B.has("logits_ref") ? B.ptr("logits_ref") : nullptr;
    int refRowsAvail = logitsRef ? B.shape("logits_ref")[0] : 0;
    // --bundle-row-offset: shift the read window (broadcast reference runs of
    // a LANES bundle: lane r lives at rows r*Tlane..). Applied to inputs AND
    // logits_ref identically so token t always compares against its own row.
    if (bundleRowOffset > 0) {
        if (bundleRowOffset >= Ttot) {
            std::cerr << "{\"fatal\":\"--bundle-row-offset " << bundleRowOffset
                      << " >= bundle rows " << Ttot << "\"}" << std::endl; return 2;
        }
        inp += (size_t)bundleRowOffset * d;
        Ttot -= bundleRowOffset;
        if (logitsRef) {
            if (bundleRowOffset >= refRowsAvail) { logitsRef = nullptr; refRowsAvail = 0; }
            else { logitsRef += (size_t)bundleRowOffset * V; refRowsAvail -= bundleRowOffset; }
        }
    }
    // --lanes-full: the bundle must be a LANES bundle (lane-major rows) whose
    // shape matches the flag — a plain bundle read as lanes would silently
    // compare lane r against unrelated text.
    int laneT = 0;                 // per-lane token count (rows per lane)
    if (lanesFull > 0) {
        if (!B.has("lanes")) {
            std::cerr << "{\"fatal\":\"--lanes-full needs a LANES bundle (tensor 'lanes' [NL,Tlane]; "
                         "export with --lanes)\"}" << std::endl; return 2;
        }
        auto lm = B.vec("lanes");
        const int nlB = (int)lm[0]; laneT = (int)lm[1];
        if (nlB != lanesFull) {
            std::cerr << "{\"fatal\":\"--lanes-full " << lanesFull << " != bundle lanes "
                      << nlB << "\"}" << std::endl; return 2;
        }
        if ((size_t)lanesFull * laneT != (size_t)Ttot) {
            std::cerr << "{\"fatal\":\"lanes bundle rows " << Ttot << " != NL*Tlane "
                      << (size_t)lanesFull * laneT << "\"}" << std::endl; return 2;
        }
        if ((int)T > laneT) T = laneT;   // T = tokens PER LANE in this mode
    } else
    // DEMO --ct-in: the token count is the CLIENT's (verified against the ct
    // meta below), not the bundle's -- only clamp when we actually read the
    // bundle's own pre-embedded rows.
    if (ctInPath.empty() && (int)T > Ttot) T = Ttot;

    // load per-layer weights once (host), matvecs encode diagonals on demand
    struct Layer {
        std::vector<double> decay, bcoef, g1, mix, c, dd, p0, p1, p2, g2;
        std::vector<double> q0, q1, q2, q3, r0, r1, r2;
        std::vector<double> win, wout, wk, wr, wv;
        int winR, winC, woutR, woutC, wkR, wkC, wrR, wrC, wvR, wvC;
        double tmA, tmB; int tmIt; double cmA, cmB; int cmIt;
    };
    std::vector<Layer> Ls(L);
    for (int l = 0; l < L; l++) {
        auto& ly = Ls[l]; std::string p = "L" + std::to_string(l) + ".";
        ly.decay = B.vec(p + "decay"); ly.bcoef = B.vec(p + "b");
        ly.g1 = B.vec(p + "tm.g1"); if (shiftMix) ly.mix = B.vec(p + "tm.mix");
        ly.c = B.vec(p + "tm.c"); ly.dd = B.vec(p + "tm.dd");
        ly.p0 = B.vec(p + "tm.p0"); ly.p1 = B.vec(p + "tm.p1"); ly.p2 = B.vec(p + "tm.p2");
        ly.g2 = B.vec(p + "cm.g2");
        ly.q0 = B.vec(p + "cm.q0"); ly.q1 = B.vec(p + "cm.q1"); ly.q2 = B.vec(p + "cm.q2"); ly.q3 = B.vec(p + "cm.q3");
        ly.r0 = B.vec(p + "cm.r0"); ly.r1 = B.vec(p + "cm.r1"); ly.r2 = B.vec(p + "cm.r2");
        ly.win = B.mat(p + "tm.win", ly.winR, ly.winC);
        ly.wout = B.mat(p + "tm.wout", ly.woutR, ly.woutC);
        ly.wk = B.mat(p + "cm.wk", ly.wkR, ly.wkC);
        ly.wr = B.mat(p + "cm.wr", ly.wrR, ly.wrC);
        ly.wv = B.mat(p + "cm.wv", ly.wvR, ly.wvC);
        auto tr = B.vec(p + "tm.rsqrt"); ly.tmA = tr[0]; ly.tmB = tr[1]; ly.tmIt = (int)tr[2];
        auto cr = B.vec(p + "cm.rsqrt"); ly.cmA = cr[0]; ly.cmB = cr[1]; ly.cmIt = (int)cr[2];
        // P2: the Newton rsqrt is the depth hog -- 4..7 iters x ~3 levels x 2 norms
        // per layer = 24..42 levels/layer, which is what forces ~190 boots/token.
        // iters=1 IS the calibrated affine gain a+b*ms with the guard clamp, i.e.
        // exactly the `newton1` swap already priced at +7.07% ppl on the ladder.
        // Fewer iterations also means fewer bootstraps, hence LESS accumulated
        // noise -- so this trades rsqrt accuracy against noise accumulation and
        // the net sign has to be measured, not assumed.
        if (rsqrtItersOverride >= 0) { ly.tmIt = rsqrtItersOverride; ly.cmIt = rsqrtItersOverride; }
        // ---- STRUCTURED MIXING (--band N) -----------------------------------
        // The measured cost model says the forward pass is dominated by the
        // ~135k dense diagonal plaintexts (encode + 9.8 MB H2D each), and the
        // sparsity census found ZERO droppable diagonals in the trained
        // weights, so no post-hoc pruning can help. The only structural way to
        // cut the plaintext COUNT is a weight matrix with few nonzero
        // diagonals. --band N keeps only the cyclic band |k| <= N (where the
        // BSGS diagonal index is k = (col - row) mod Dpad), leaving exactly
        // 2N+1 diagonals instead of Dpad.
        // This measures the SPEED a structured-mixing model would achieve on
        // this harness. It does NOT preserve accuracy on weights trained dense
        // -- logits are expected to be wrong, and the quality of a genuinely
        // band-trained model has to come from training, not from this flag.
        if (bandWidth >= 0) {
            auto bandify = [&](std::vector<double>& W, int rows, int cols, int colBlock) {
                // colBlock = the modulus the BSGS builder uses for this matrix
                // (Dpad for the d x d maps; the chunk width for wv's slices).
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++) {
                        int k = ((c % colBlock) - (r % colBlock) + colBlock) % colBlock;
                        int dist = std::min(k, colBlock - k);
                        if (dist > bandWidth) W[(size_t)r * cols + c] = 0.0;
                    }
            };
            bandify(ly.win,  ly.winR,  ly.winC,  (int)Dpad);
            bandify(ly.wout, ly.woutR, ly.woutC, (int)Dpad);
            bandify(ly.wk,   ly.wkR,   ly.wkC,   (int)Dpad);
            bandify(ly.wr,   ly.wrR,   ly.wrC,   (int)Dpad);
            bandify(ly.wv,   ly.wvR,   ly.wvC,   (int)Dpad);
        }
    }
    auto gOut = B.vec("g_out");
    auto outR = B.vec("ln_out.rsqrt");
    if (rsqrtItersOverride >= 0) outR[2] = (double)rsqrtItersOverride;   // final norm too

    // ---- hoisted per-pass locals (DEMO --serve) -----------------------------
    // worstLogitErr/failed/stage/tokensReal used to be declared just before
    // the forward `try`; the persistent-server loop below re-runs the whole
    // pass region once per request, and the summary AFTER the loop still
    // reads these -- so the DECLARATIONS live here and the old sites became
    // per-iteration resets. The non-serve path is one loop iteration with
    // identical initial values, i.e. unchanged behavior.
    double worstLogitErr = -1;
    // A-P0b (2026-08-22): fidelity accumulators for the summary -- rms-class logit
    // error per token (R3: report relErrRms, not max) and top-1 agreement vs the
    // bundle's OWN logits_ref (R4: this is fidelity, never quality).
    long long lcTokens = 0, lcTop1 = 0; double lcRms2Sum = 0.0, lcRmsMax = 0.0;
    // --lanes-full per-lane accumulators (index = lane) + replica-echo worst.
    std::vector<long long> lfTokens, lfTop1; std::vector<double> lfRms2Sum, lfRmsMax;
    double lfEchoWorst = 0.0;
    if (lanesFull > 0) {
        lfTokens.assign(lanesFull, 0); lfTop1.assign(lanesFull, 0);
        lfRms2Sum.assign(lanesFull, 0.0); lfRmsMax.assign(lanesFull, 0.0);
    }
    bool failed = false;
    std::string stage = "init";   // DEBUG breadcrumb: which sub-block was executing, incl. layer/token
    stagePtr = &stage;            // S3.3: bootTrace lines carry the breadcrumb (site histogram from the log)
    uint32_t tokensReal = T;      // real token count for the summary (T gets reset in packed mode)
#ifdef FHE_SSM_DEMO_SER
    // ---- DEMO --serve REQDIR: PERSISTENT server loop ("Fix 1") -------------
    // The measured per-step server cost (433 s at d768) is dominated by
    // context/eval-key deserialization + EvalBootstrapSetup + LoadContext --
    // all of which happened ABOVE this line and are therefore paid ONCE for
    // the whole generation. Protocol, all inside REQDIR:
    //   client:  writes req.<N>.0..count-1 + req.<N>.meta (the --enc-in file
    //            format), then touches req.<N>.ready
    //   server:  runs the forward exactly as --ct-in/--ct-out would (the
    //            prologue literally sets ctInPath/ctOutPath and reuses those
    //            paths), writes resp.<N>.* + resp.<N>.done (or .error),
    //            deletes the request files, waits for req.<N+1>.ready
    //   exit:    a file named `stop` in REQDIR, or --serve-timeout seconds
    //            with no request.
    // T VARIES PER REQUEST (the sequence grows every generation step): it is
    // re-derived from the request meta here, and everything T-derived
    // (nBlocks, packLane, tokensReal, H, the whole forward) is declared
    // inside this loop. The startup work above depends only on
    // packTokens/REP/ring -- the rotation-key set is T-independent -- and the
    // --ct-in meta check still enforces per-request consistency. A malformed
    // request that fails meta/loadCt checks exits the process (fail-fast; the
    // client watches server liveness). Requests must satisfy the same bounds
    // as a fresh run: ceil(T/packTokens) blocks at ~2 GB VRAM each.
    const bool serveMode = !serveDir.empty();
    int serveReqN = 0;            // next request index (client counts monotonically)
    int serveDone = 0;
    bool serveAnyFailed = false;
    double serveEval0 = 0, serveBoot0 = 0, serveEncPt0 = 0;
    // S3.7 V2: per-request snapshots of the new accumulators (the serve line's req* deltas)
    double serveNormBoot0 = 0, serveLoop0 = 0, serveLoopTimed0 = 0, serveShift0 = 0, serveScan0 = 0,
           serveGate0 = 0, serveHid0 = 0, serveRes0 = 0, servePtEnc0 = 0;
    long long servePtEncN0 = 0;
    int serveBoots0 = 0;
    auto touchFile = [](const std::string& p, const char* msg) {
        std::ofstream f(p); f << msg << "\n";
    };
    auto fileExists = [](const std::string& p) {
        std::ifstream f(p); return f.good();
    };
    // ---- STATEFUL DECODE: the in-memory carry, held ACROSS serve requests --
    // Declared outside the serve loop on purpose: with --serve --stateful the
    // state never leaves the process, so a resumed step costs no serialization
    // and no transfer at all. carryScan[l] is the scan accumulator, carryU[l]
    // the pre-shift u of the last token (the shift-mix boundary).
    std::vector<TrackedCt> carryScan, carryU;
    bool haveCarry = false;
    long long carryTokens = 0;      // how many tokens the carry has absorbed
    for (;;) {
    if (serveMode) {
        const std::string reqPrefix = serveDir + "/req." + std::to_string(serveReqN);
        std::cout << "{\"serve\":\"waiting\",\"req\":" << serveReqN
                  << ",\"dir\":\"" << serveDir << "\",\"timeoutSec\":" << serveTimeout
                  << ",\"served\":" << serveDone << "}" << std::endl;
        // serve.status (review 2026-09-03): the request counter is per PROCESS
        // while every client numbers from 0, so a second session against a
        // live server used to wait forever for resp.0. The client reads this
        // file at preflight and adopts nextReq as its base.
        {
            std::ofstream sf(serveDir + "/serve.status.tmp");
            sf << "{\"nextReq\":" << serveReqN << ",\"served\":" << serveDone << ",\"pid\":" << (long long)getpid()
               << ",\"stateful\":" << (statefulServe ? "true" : "false") << ",\"storeFile\":\"" << storeFilePath << "\"}\n";
            sf.close();
            std::rename((serveDir + "/serve.status.tmp").c_str(), (serveDir + "/serve.status").c_str());
        }
        auto tWait = Clock::now();
        bool gotReq = false, gotStop = false;
        for (;;) {
            if (fileExists(serveDir + "/stop")) { gotStop = true; break; }
            if (fileExists(reqPrefix + ".ready")) { gotReq = true; break; }
            if (msSince(tWait) > 1000.0 * (double)serveTimeout) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (!gotReq) {
            storePersistFn();
            std::cout << "{\"serve\":\"exit\",\"reason\":\"" << (gotStop ? "stop" : "idleTimeout")
                      << "\",\"served\":" << serveDone << "}" << std::endl;
            break;
        }
        long long mTok = -1;
        try { mTok = metaInt(readCtMeta(reqPrefix), "tokensReal", -1);
              // ECHO the request's lane count into everything this request
              // produces (F62, 2026-09-01): writeCtMeta stamps clientLanes
              // into the RESPONSE meta, and without this echo the server's
              // default (1) made --dec-out extract only lane 0 — four
              // "distinct lane texts" that were really lane-0 predictions
              // from four positions. The server stays blind to lane CONTENT;
              // this is plumbing, not lane logic.
              clientLanes = (int)metaInt(readCtMeta(reqPrefix), "lanes", 1); }
        catch (const std::exception&) { mTok = -1; }
        if (mTok < 1) {
            std::cout << "{\"serve\":\"badRequest\",\"req\":" << serveReqN << "}" << std::endl;
            touchFile(serveDir + "/resp." + std::to_string(serveReqN) + ".error",
                      "bad or missing request meta");
            std::remove((reqPrefix + ".ready").c_str());
            serveReqN++;
            continue;
        }
        T = (uint32_t)mTok;                                          // per-request token count
        // SESSION BOUNDARY for --stateful. Without this a persistent stateful
        // server carries one conversation's state into the next request
        // forever -- silently, since the circuit stays well-formed. The client
        // touches req.<N>.reset on its PREFILL step; we drop the carry and
        // cold-start. Same file-marker protocol as .ready/.done.
        if (fileExists(reqPrefix + ".reset")) {
            carryScan.clear(); carryU.clear(); haveCarry = false; carryTokens = 0;
            std::remove((reqPrefix + ".reset").c_str());
            std::cout << "{\"serve\":\"stateReset\",\"req\":" << serveReqN << "}" << std::endl;
        }
        ctInPath = reqPrefix;                                        // reuse the --ct-in path
        ctOutPath = serveDir + "/resp." + std::to_string(serveReqN); // reuse the --ct-out path
        serveEval0 = evalMs; serveBoot0 = bootMs; serveEncPt0 = encPtMs; serveBoots0 = boots;
        serveNormBoot0 = normBootMs; serveLoop0 = layerLoopMs; serveLoopTimed0 = layerLoopTimedMs;   // S3.7 V2
        serveShift0 = shiftMixMs; serveScan0 = scanMs; serveGate0 = gateMs; serveHid0 = hiddenMs; serveRes0 = residualMs;
        servePtEnc0 = hostPtEncodeMs; servePtEncN0 = hostPtEncodeCount;
        stageOpenDepth = 0;                                          // S3.7 V2: never carry an open stage window across a failed request
        failed = false; stage = "serve.req";                         // per-request status
        babyCache.open = false; babyCache.clear(); pfNormKind = 0;   // S3.3 (review 2026-09-03): never carry a share
                                                                     // window or a norm kind across a failed request
        vramTraceLine("serve.req." + std::to_string(serveReqN));
        std::cout << "{\"serve\":\"request\",\"req\":" << serveReqN
                  << ",\"tokens\":" << T << "}" << std::endl;
    }
#endif

    // encrypt the T token rows
    // STAGE 2: all packTokens tokens live in ONE ciphertext, so the per-token
    // loops below execute exactly once. That is what makes VRAM O(1) in sequence
    // length (up to REP) while keeping the encode amortization: one encoded
    // diagonal still serves every token, because they share the ciphertext.
    int packLane = 0;                       // last occupied lane in the LAST block
    tokensReal = T;                         // (re)set per pass/request -- see hoist note
    int nBlocks = 1;
    std::vector<TrackedCt> H;
    if (packTokens > 1) {
        if (packTokens > (int)REP) {
            std::cerr << "{\"fatal\":\"packTokens>REP\",\"packTokens\":" << packTokens
                      << ",\"REP\":" << REP << "}" << std::endl;
            return 2;
        }
        // MULTI-BLOCK (2026-07-30): ceil(T/packTokens) ciphertexts of up to
        // packTokens lanes each. Lanes amortize the encode wall at ~zero GPU
        // cost (one ct), blocks amortize it further at ~2 GB VRAM per block --
        // tokens/pass = packTokens * nBlocks. The two cross-token ops get an
        // exact cross-BLOCK carry below (the recurrence is LTI, so the carry
        // is a masked broadcast + plaintext decay powers, ~2 levels per
        // boundary). Only the last block may be short; every boundary
        // therefore reads lane packTokens-1 of a FULL predecessor.
        nBlocks = ((int)T + packTokens - 1) / packTokens;
        tokensReal = T;
        packLane = (int)T - 1 - (nBlocks - 1) * packTokens;
        H.resize(nBlocks);
#ifdef FHE_SSM_DEMO_SER
        // DEMO server input: deserialize the client's ciphertexts instead of
        // encrypting the bundle rows. The level trace and packing geometry are
        // flag-driven, so the meta must match this invocation exactly.
        if (!ctInPath.empty()) {
            try {
                const std::string meta = readCtMeta(ctInPath);
                // AUDIT FIX 2026-09-02: the F62 lane echo lived only in the
                // --serve loop; the single-shot --ct-in server stamped lanes:1
                // on a multi-lane request. Echo here as well.
                clientLanes = std::max(1, (int)metaInt(meta, "lanes", 1));
                if ((int)metaInt(meta, "count", -1) != nBlocks
                    || (int)metaInt(meta, "packTokens", -1) != packTokens
                    || metaInt(meta, "tokensReal", -1) != (long long)T
                    || metaInt(meta, "ringDim", 0) != (long long)cc->GetRingDimension()
                    || (int)metaInt(meta, "dpad", 0) != (int)Dpad        // F68
                    || (int)metaInt(meta, "interleave", -1) != (interleave ? 1 : 0)) {
                    std::cerr << "{\"fatal\":\"ct-in meta mismatch: run the server with the SAME "
                                 "--tokens/--pack-tokens/--dpad/--interleave/--log-ring the client encrypt used\","
                              << "\"expectedCount\":" << nBlocks << ",\"expectedTokens\":" << T
                              << ",\"expectedDpad\":" << Dpad
                              << ",\"expectedPack\":" << packTokens << "}" << std::endl;
                    return 2;
                }
                for (int b = 0; b < nBlocks; b++)
                    H[b] = loadCt(ctInPath + "." + std::to_string(b));
            } catch (const std::exception& e) {
                std::cerr << "{\"fatal\":\"ct-in load failed\",\"what\":\"" << e.what() << "\"}\n";
                return 2;
            }
        } else
#endif
        for (int b = 0; b < nBlocks; b++) {
            int n = std::min(packTokens, (int)T - b * packTokens);
            H[b] = encCh(packTok(inp + (size_t)b * packTokens * d, n, d));
        }
        T = (uint32_t)nBlocks;              // per-"token" loops become per-BLOCK loops
    } else {
        H.resize(T);
#ifdef FHE_SSM_DEMO_SER
        if (!ctInPath.empty()) {
            try {
                const std::string meta = readCtMeta(ctInPath);
                // AUDIT FIX 2026-09-02: the F62 lane echo lived only in the
                // --serve loop; the single-shot --ct-in server stamped lanes:1
                // on a multi-lane request. Echo here as well.
                clientLanes = std::max(1, (int)metaInt(meta, "lanes", 1));
                // F68 (2026-09-01): dpad was checked on the --dec-out path
                // (line ~1477) but NOT here, so a request packed at one slot
                // stride could be served by a context using another. The
                // rotations then move the wrong slots and the reply decrypts
                // to well-formed, wrong hidden vectors -- no error anywhere.
                if ((int)metaInt(meta, "count", -1) != (int)T
                    || (int)metaInt(meta, "packTokens", -1) > 1
                    || metaInt(meta, "ringDim", 0) != (long long)cc->GetRingDimension()
                    || (int)metaInt(meta, "dpad", 0) != (int)Dpad
                    || (int)metaInt(meta, "interleave", -1) != (interleave ? 1 : 0)) {
                    std::cerr << "{\"fatal\":\"ct-in meta mismatch (unpacked): run the server with "
                                 "the SAME --tokens/--dpad/--interleave/--log-ring the client encrypt used\","
                              << "\"expectedCount\":" << T
                              << ",\"expectedDpad\":" << Dpad
                              << ",\"metaDpad\":" << metaInt(meta, "dpad", 0) << "}" << std::endl;
                    return 2;
                }
                for (int t = 0; t < (int)T; t++)
                    H[t] = loadCt(ctInPath + "." + std::to_string(t));
            } catch (const std::exception& e) {
                std::cerr << "{\"fatal\":\"ct-in load failed\",\"what\":\"" << e.what() << "\"}\n";
                return 2;
            }
        } else
#endif
        if (lanesFull > 0) {
            // LANES bundle: slot r*Dpad+c = lane (r mod NL), channel c, for
            // ALL REP replicas (cyclic fill). Zero-filled dead replicas would
            // push rsqrt(eps)-class values (~1/sqrt(1e-5) ≈ 316) through every
            // bootstrap; cyclic fill keeps every slot inside the calibrated
            // activation range and makes replica NL+r a free echo check of
            // replica r (verified at decode).
            for (int t = 0; t < (int)T; t++) {
                std::vector<double> sv(SLOTS, 0.0);
                for (uint32_t r = 0; r < REP; r++) {
                    const int lane = (int)(r % (uint32_t)lanesFull);
                    const double* rw = inp + ((size_t)lane * laneT + t) * d;
                    for (int i = 0; i < d; i++) sv[(size_t)r * Dpad + i] = rw[i];
                }
                H[t] = encCh(sv);
            }
        } else
        for (int t = 0; t < (int)T; t++) {
            std::vector<double> row(inp + (size_t)t * d, inp + (size_t)t * d + d);
            H[t] = encCh(packCh(row.data(), d));
        }
    }

    // gate helpers: deg-3 (p0 + p1 z + p2 z^2) applied as z*(...) ; deg-3 poly
    auto polyGate3 = [&](TrackedCt z, const std::vector<double>& p0,
                         const std::vector<double>& p1, const std::vector<double>& p2) -> TrackedCt {
        // g = z*(p0 + p1 z + p2 z^2). depth 2.
        // BUG FIXED 2026-07-30: this lambda used two CONTRADICTORY argument
        // conventions. It packed p0 itself (packCh below) but handed p1/p2
        // straight to mulPt, which requires an ALREADY-PACKED SLOTS-length
        // vector. Callers pass the raw length-d coefficient vectors, so p1/p2
        // reached MakeCKKSPackedPlaintext as a length-768 slim plaintext: only
        // the first lane carried the coefficient and every other lane got ZERO.
        // Under the replicated layout that was survivable (decCh only ever reads
        // replica 0), which is why it went unnoticed. Under the INTERLEAVED
        // layout channel i lives at slot i*REP, so it is destructive -- and
        // exp28's gateCheck duly reports 86-114% relative error on tm.gate at
        // EVERY layer, against 0.02% for cm.gate (which evaluates its polynomial
        // inline and never touches this lambda).
        // Fix: one convention. Take RAW length-d vectors and pack all three here.
        std::vector<double> p1pk = packCh(p1.data(), d);
        std::vector<double> p2pk = packCh(p2.data(), d);
        TrackedCt z2{cc->EvalMult(z.ct, z.ct), z.level()}; cc->RescaleInPlace(z2.ct);
        TrackedCt term2 = z2; mulPt(term2, p2pk);            // p2 z^2 @ z.level()+1
        TrackedCt term1 = z;  mulPt(term1, p1pk);            // p1 z   @ z.level()+1
        // BUG FIXED 2026-07-18: term1 (z.level()+1) and term2 (z.level()+2) are ONE
        // LEVEL APART, and this was a raw EvalAdd with no alignment -- so
        // term1.level() kept reporting z.level()+1 while the actual ciphertext had
        // moved to z.level()+2. Every later step trusted that stale number: the
        // p0 plaintext was encoded a level too low, zA was aligned to the
        // wrong target, and the returned g carried a tracked level that did
        // not match reality. matvecBatch(wout) then encoded all 1024 of its
        // diagonals at that wrong level, producing an undecryptable result --
        // while matvecBatch(win), fed by rmsnorm's carefully aligned output,
        // worked fine. That asymmetry is what localized this.
        alignTo(term1, term2.level());
        term1.ct = cc->EvalAdd(term1.ct, term2.ct);          // p1 z + p2 z^2
        std::vector<double> p0pk = packCh(p0.data(), d);     // + p0 (fresh add at level)
        auto p0pt = mkPt(p0pk, term1.level(), __LINE__);
        term1.ct = cc->EvalAdd(term1.ct, p0pt);              // p0 + p1 z + p2 z^2
        // multiply by z (align z up)
        TrackedCt zA = z; while (zA.level() < term1.level()) { std::vector<double> one(SLOTS,1.0); auto op=mkPt(one, zA.level(), __LINE__); zA.ct=cc->EvalMult(zA.ct,op); cc->RescaleInPlace(zA.ct); }
        TrackedCt g{cc->EvalMult(zA.ct, term1.ct), zA.level()}; cc->RescaleInPlace(g.ct);
        return g;
    };

    // (worstLogitErr/failed/stage DECLARATIONS hoisted above the serve loop --
    // see the hoist note; these are now the per-iteration resets. Any earlier
    // request's failure is already latched in serveAnyFailed by the loop tail.)
    worstLogitErr = -1;
    failed = false;
    stage = "init";                // DEBUG breadcrumb: which sub-block was executing, incl. layer/token
    try {
        // ---- LAYER-MAJOR forward (all T tokens through layer l, then l+1).
        // This ordering is what makes matvecBatch possible: every token in a
        // layer sits at the same level, so each weight diagonal is encoded
        // ONCE and applied to all T tokens (see the batching note above).
        // It also makes the token-shift exact and cache-free: within a layer
        // we hold every token's u, so u[t-1] is simply the previous element.
        // Each token remains its OWN ciphertext -> no cross-token bleed is
        // possible by construction (no rotation ever crosses tokens).
        // --passes N runs the forward N times. The point is NOT repetition for
        // its own sake: with --cache-diags the encoded diagonal plaintexts stay
        // alive (and hence GPU-resident) across passes, so pass 2+ measures the
        // STEADY-STATE serving cost with the plaintext set already on the
        // device. That is the regime the measured numbers say matters --
        // multiplying by an already-loaded plaintext costs 0.0122 ms against
        // 4.79 ms to encode it and 4.98 ms to upload it -- and for a
        // structured (banded) model the whole plaintext set fits in VRAM.
        std::vector<TrackedCt> Hs;
        for (int pass = 0; pass < passes; pass++) {
            vramTraceLine("pass." + std::to_string(pass));
        const double pe0 = evalMs, pb0 = bootMs, pp0 = encPtMs;
        const double pn0 = normBootMs, pl0 = layerLoopMs;   // S3.7 V2: per-pass ex-boot fields
        const int pbt0 = boots;
        const long long pc0 = encPtCount;
        Hs = H;                                        // residual stream / token
        // per-layer carried scan state (zero-initialised, one ciphertext each)
        // STATEFUL DECODE (2026-08-22): zero-init is the COLD START only. When
        // a carry is present the state resumes from it and this invocation
        // processes just the new token(s) -- turning N(N+1)/2 replayed scan
        // steps into N. `uPrev[l]` is the second half of the carry (the
        // shift-mix boundary); `resumed` gates the u_{-1}=0 hardcode below.
        std::vector<TrackedCt> scanState(L);
        std::vector<TrackedCt> uPrev(L);
        bool resumed = false;
#ifdef FHE_SSM_DEMO_SER
        if (haveCarry) {                       // --serve --stateful: in-memory
            scanState = carryScan; uPrev = carryU; resumed = true;
        } else if (!stateInPath.empty()) {     // --state-in: on-disk carry
            const std::string meta = readCtMeta(stateInPath);
            if ((long long)metaInt(meta, "ringDim", 0) != (long long)cc->GetRingDimension()
                || (int)metaInt(meta, "dpad", 0) != (int)Dpad
                || (int)metaInt(meta, "interleave", -1) != (interleave ? 1 : 0)
                || (int)metaInt(meta, "layers", -1) != L) {
                std::cerr << "{\"fatal\":\"state-in: layout mismatch -- pass the SAME "
                             "--log-ring/--dpad/--interleave/model the state was written with\"}"
                          << std::endl; return 2; }
            // The u half exists only when the model HAS shift-mix; a model
            // without it writes no .u.* files and loading them would throw.
            const bool metaShift = metaInt(meta, "shiftMix", 0) != 0;
            if (metaShift != shiftMix) {
                std::cerr << "{\"fatal\":\"state-in: shiftMix mismatch -- the carry was written by "
                             "a model with shiftMix=\"" << (metaShift ? 1 : 0)
                          << ", this one has " << (shiftMix ? 1 : 0) << "}" << std::endl; return 2; }
            for (int l = 0; l < L; l++) {
                scanState[l] = loadCt(stateInPath + ".scan." + std::to_string(l));
                if (metaShift) uPrev[l] = loadCt(stateInPath + ".u." + std::to_string(l));
            }
            carryTokens = metaInt(meta, "carryTokens", 0);
            resumed = true;
            std::cout << "{\"stateIn\":true,\"file\":\"" << stateInPath << "\",\"layers\":" << L
                      << ",\"carryTokens\":" << carryTokens
                      << ",\"scanLevel\":" << scanState[0].level()
                      << ",\"uLevel\":" << uPrev[0].level() << "}" << std::endl;
        }
#endif
        if (!resumed) {                        // cold start: s_{-1}=0, u_{-1}=0
            std::vector<double> z(SLOTS, 0.0);
            for (int l = 0; l < L; l++) scanState[l] = encCh(z);
            if (canonicalCarry) {
                // canonical cold start: zero carries at the canonical level, and an
                // explicit zero u_{-1} so the shift-mix takes the SAME code path
                // (and level trajectory) as every resumed tick.
                for (int l = 0; l < L; l++) {
                    canon(scanState[l], "scan.cold");
                    uPrev[l] = encCh(z); canon(uPrev[l], "u.cold");
                }
                resumed = true;
                std::cout << "{\"canonicalCarry\":\"cold\",\"scanLevel\":" << scanState[0].level()
                          << ",\"uLevel\":" << uPrev[0].level() << ",\"maxGap\":" << canonMaxGap << "}" << std::endl;
            }
        }
        // S3.7 V2 (2026-09-05): the layer loop's own clock (whole wall + the timed
        // sum that accrues inside it); the five stage windows are inside. See the
        // accumulator note (normBootMs / layerLoopMs) for what each field means.
        if (stageTimers) syncDev();
        const auto tLoop0 = Clock::now();
        const double loopEv0 = evalMs, loopNb0 = normBootMs, loopBo0 = bootMs, loopEn0 = encMs, loopDe0 = decMs;
        for (int l = 0; l < L; l++) {
            auto& ly = Ls[l];

            // ---- time mix ----
            std::vector<TrackedCt> u(T);
            std::vector<TrackedCt> uNorm(blockParallel ? T : 0);   // S3.3 Tier B: the pre-shift norm output both branches read
            for (uint32_t t = 0; t < T; t++) {
                stage = "L" + std::to_string(l) + ".tm.rmsnorm.t" + std::to_string(t);
                if (schedParentFirst) {              // S3.3 P1: parent refresh, need = nb + dN(I-1) + 5
                    const int b0 = boots; refresh(Hs[t], pfNeedEntry(1, ly.tmIt)); pfBoots += boots - b0;
                } else refresh(Hs[t], 3);            // rmsnorm self-refreshes internally
                pfNormKind = 1;                      // S3.3 P2 tail need for a tm norm (read+reset by rmsnorm)
                u[t] = rmsnorm(Hs[t], packCh(ly.g1.data(), d), ly.tmA, ly.tmB, ly.tmIt, 1e-5);
                if (blockParallel) uNorm[t] = u[t];   // S3.3 Tier B: captured BEFORE the shift-mix
                if (stageProbe && t == 0) {
                    // Isolate the NORM. wvCheck only ever measured the channel-mix
                    // matvec; interleaving cut that to 0.01% while the logit error
                    // barely moved (20.9 -> 19.2), so the dominant per-layer
                    // distortion is elsewhere. Reference from the DECRYPTED input,
                    // so upstream error cannot masquerade as norm error.
                    try {
                        auto hv = decCh(Hs[0]); auto uv = decCh(u[0]);
                        double ms = 0; for (int i = 0; i < d; i++) ms += hv[i]*hv[i];
                        ms /= d; double invr = 1.0/std::sqrt(ms + 1e-5);
                        double num = 0, den = 0; long long nf = 0;
                        for (int i = 0; i < d; i++) {
                            double r = hv[i]*invr*ly.g1[i];
                            if (!std::isfinite(uv[i]) || !std::isfinite(r)) { nf++; continue; }
                            num = std::max(num, std::abs(uv[i]-r)); den = std::max(den, std::abs(r));
                        }
                        std::cout << "{\"normCheck\":true,\"layer\":" << l << ",\"site\":\"tm\""
                                  << ",\"nonFinite\":" << nf << ",\"maxAbsErr\":" << num
                                  << ",\"refMax\":" << den << ",\"relErr\":" << num/(den+1e-12)
                                  << "}" << std::endl;
                    } catch (const std::exception&) {
                        std::cout << "{\"normCheck\":true,\"layer\":" << l
                                  << ",\"site\":\"tm\",\"undecryptable\":true}" << std::endl;
                    }
                }
            }
            stage = "L" + std::to_string(l) + ".tm.shiftMix";
            const auto stShift = stageOpen();            // S3.7 V2: shiftMixMs (mulPt x2, addAligned; the carry refresh is bootMs)
            if (shiftMix) {
                // u'_t = u_t (.) mix + u_{t-1} (.) (1-mix); u_{-1} = 0.
                std::vector<double> imix(d);
                for (int i = 0; i < d; i++) imix[i] = 1.0 - ly.mix[i];
                auto mixpk = packCh(ly.mix.data(), d);
                auto imixpk = packCh(imix.data(), d);
                std::vector<TrackedCt> um(T);
                if (packTokens > 1) {
                    // STAGE 2: u_{t-1} is one TOKEN LANE back inside the same
                    // ciphertext. Rotate by -1 lane and multiply by (1-mix) with
                    // lane 0 ZEROED -- that zero is exactly the u_{-1}=0 boundary
                    // condition, and folding it into the plaintext makes the whole
                    // step one ct x pt multiply.
                    // MULTI-BLOCK: for block b>0, lane 0's predecessor is the
                    // PREVIOUS block's last lane. EvalRotate(+(packTokens-1))
                    // brings lane packTokens-1 to lane 0 (same channel); the
                    // lane-0-only mask kills every other (cross-channel) slot.
                    auto imixMasked = packLaneMasked(imix, d, 1);
                    auto imixLane0 = packLaneOnly(imix, d, 0);
                    for (uint32_t t = 0; t < T; t++) {
                        TrackedCt a = u[t]; mulPt(a, mixpk);
                        TrackedCt b = u[t]; b.ct = rotTok(b.ct, 1);
                        mulPt(b, imixMasked);
                        addAligned(a, b);
                        if (t > 0) {
                            TrackedCt c2 = u[t - 1];
                            refresh(c2, 2);
                            c2.ct = cc->EvalRotate(c2.ct, packTokens - 1);
                            mulPt(c2, imixLane0);
                            addAligned(a, c2);
                        }
                        um[t] = a;
                    }
                } else {
                    for (uint32_t t = 0; t < T; t++) {
                        TrackedCt a = u[t]; mulPt(a, mixpk);
                        // u_{-1} = 0 is the COLD-START boundary only. On a
                        // resumed step the predecessor of token 0 is the last
                        // token of the PREVIOUS invocation, carried in
                        // uPrev[l]. Same code as the t>0 branch -- which is
                        // why the carry stores the RAW pre-shift u and does
                        // no positioning of its own.
                        if (t > 0) { TrackedCt b = u[t - 1]; mulPt(b, imixpk); addAligned(a, b); }
                        else if (resumed && uPrev[l].ct) {
                            TrackedCt b = uPrev[l]; refresh(b, 2);
                            mulPt(b, imixpk); addAligned(a, b);
                        }
                        um[t] = a;
                    }
                }
                // STATEFUL DECODE: capture the shift-mix carry BEFORE `u = um`.
                // The value shift-mix consumes is the PRE-shift u, so taking it
                // after the reassignment would carry the wrong tensor.
                if (statefulAny) uPrev[l] = u[T - 1];
                u = um;
            }
            stageClose(stShift, shiftMixMs);              // S3.7 V2
            stage = "L" + std::to_string(l) + ".tm.matvecBatch.win";
#ifdef FHE_SSM_DEMO_SER
            mvLayer = l; mvOpId = 0; mvChunk = 0;      // T1b key: win
#endif
            auto x = matvecBatch(u, ly.win, ly.winR, ly.winC, 0);
#ifdef FHE_SSM_DEMO_SER
            mvOpId = -1;
#endif
            if (l == 0) { dbgDec(u[0], "L0.tm.rmsnorm_shift_out"); dbgDec(x[0], "L0.tm.win_out"); }

            // ---- Lemma-2 diagonal scan, sequential across tokens ----
            // s_t = decay (.) s_{t-1} + b (.) x_t. One ciphertext per token,
            // so this is a plain per-token recurrence (no packing subtleties).
            const auto stScan = stageOpen();             // S3.7 V2: scanMs (all scan forms; refresh/equalize boots are bootMs)
            std::vector<TrackedCt> S(T);
            auto decPk = packCh(ly.decay.data(), d);
            auto bPk   = packCh(ly.bcoef.data(), d);
            if (packTokens > 1) {
                // STAGE 2: the recurrence is LTI (decay is a per-channel constant,
                // not input dependent), so s_t = sum_{j<=t} decay^(t-j) b x_j is a
                // prefix scan and the Kogge-Stone/Blelloch log-doubling form is
                // EXACT, not an approximation:
                //     h <- h + decay^(2^k) * shift(h, 2^k),  k = 0..log2(N)-1
                // (same recurrence as ml-eval/quality_ladder.py::scan_const, which
                // is validated on the CPU side). Cost: log2(N) levels instead of N
                // sequential steps -- for N=32 that is 5 levels, not 32.
                // ---- TWO-LEVEL SCAN (2026-07-31) --------------------------------
                // The old form chained the cross-block carry: block b consumed
                // block b-1's *output*, so each block sat ~2 levels deeper than
                // its predecessor. matvecBatch needs a COMMON level (one encoded
                // plaintext serves every block -- that sharing is the entire
                // reason multi-block is cheap), so it closed the gap with
                // alignTo's EvalMult-by-1.0 + Rescale chain. Measured spread:
                // 6 levels at 4 blocks, 14 at 8, 0 at 1 -- and ~14 chained
                // rescales drive the shallowest block's values under the noise
                // floor, which is the silent `maxAbs 0` and the stochastic
                // `output.decode.tok0` failures. No margin/boot-floor setting
                // fixes it (at ring 2^16, margins 0/1/4 all fail with an
                // IDENTICAL 441 boots -- the knob changes no decision).
                //
                // Note the tempting fix -- let blocks keep jagged levels -- is
                // wrong here: a CKKS plaintext is encoded AT a level, so jagged
                // levels means one plaintext per distinct level and the encode
                // cost (~95% of a pass) multiplies by the number of levels. At
                // 4 blocks that is 4E + 4G = 9.75 s/token, exactly the
                // single-block figure: the whole multi-block gain disappears.
                //
                // So keep levels common, but make them common BY CONSTRUCTION.
                // Writing L[b] for block b's local scan and C[b] for the true
                // state entering it:
                //     s_{b*P+ln} = L[b][ln] + decay^(ln+1) * C[b]
                //     C[b+1]     = e[b] + decay^P * C[b],  e[b] := L[b][P-1]
                //  => C[b] = sum_{a<b} q^(b-1-a) e[a],     q := decay^P
                // C is therefore itself a first-order LTI prefix over BLOCKS,
                // so it log-doubles in log2(B) steps instead of B. And because
                // blocks are SEPARATE ciphertexts, combining them needs no
                // rotation at all -- just a plaintext multiply and an add,
                // indexing a host array.
                //
                // Uniformity is the point: a block with no predecessor at
                // distance k multiplies a dummy source by a ZERO coefficient
                // rather than skipping the step, so every block executes an
                // identical op sequence and they all land on the same level.
                // Cost is ~3 + log2(B) levels for EVERY block, against the old
                // chain's ~2*b for block b. Validated slot-exactly against a
                // sequential reference (twolevel_sim.py (a scratch simulator, not included), non-power-of-
                // two block count and a short final block, err 8.9e-16).
                const int nB = (int)T;                  // T is the BLOCK count here
                std::vector<double> onesd(d, 1.0), zerod(d, 0.0);
                std::vector<double> dp(SLOTS, 0.0);     // per-lane decay^(ln+1)
                for (int i = 0; i < d; i++) {
                    double pw = ly.decay[i];
                    for (int ln = 0; ln < packTokens; ln++) {
                        dp[(size_t)i * REP + ln] = pw;
                        pw *= ly.decay[i];
                    }
                }
                // -- Phase A: LOCAL scan per block, no carry. Identical per block.
                for (int b = 0; b < nB; b++) {
                    stage = "L" + std::to_string(l) + ".tm.scan.local.b" + std::to_string(b);
                    refresh(x[b], 3);
                    TrackedCt h = x[b]; mulPt(h, bPk);        // h = b (.) x
                    std::vector<double> cur = ly.decay;       // decay^(2^k), host-side
                    for (int step = 1; step < packTokens; step <<= 1) {
                        refresh(h, 2);
                        TrackedCt sh = h; sh.ct = rotTok(sh.ct, step);
                        // coefficient masked to zero on the first `step` lanes:
                        // those have no predecessor at this distance.
                        mulPt(sh, packLaneMasked(cur, d, step));
                        addAligned(h, sh);
                        for (int i = 0; i < d; i++) cur[i] = cur[i] * cur[i];
                    }
                    S[b] = h;
                }
                if (nB > 1 && !serialCarry) {
                    // -- Phase B: e[b] = block b's LAST lane, broadcast to all
                    // lanes. The broadcast shifts pull from HIGHER lanes only,
                    // and only lane packTokens-1 is live, so with steps <
                    // packTokens nothing crosses a channel boundary.
                    std::vector<TrackedCt> E(nB);
                    for (int b = 0; b < nB; b++) {
                        stage = "L" + std::to_string(l) + ".tm.scan.end.b" + std::to_string(b);
                        TrackedCt cy = S[b];
                        refresh(cy, 3);
                        mulPt(cy, packLaneOnly(onesd, d, packTokens - 1));
                        for (int step = 1; step < packTokens; step <<= 1) {
                            TrackedCt sh2 = cy; sh2.ct = cc->EvalRotate(sh2.ct, step);
                            addAligned(cy, sh2);
                        }
                        E[b] = cy;
                    }
                    // -- Phase C: cross-block log-doubling prefix -> G[b] = C[b].
                    // Init G[b] = e[b-1] with G[0] = 0, done uniformly by
                    // multiplying by a 1-or-0 coefficient rather than branching.
                    std::vector<TrackedCt> G(nB);
                    for (int b = 0; b < nB; b++) {
                        TrackedCt src = E[b > 0 ? b - 1 : 0];
                        mulPt(src, packCh((b > 0 ? onesd : zerod).data(), d));
                        G[b] = src;
                    }
                    std::vector<double> qk(d);          // q^(2^k), q = decay^P
                    for (int i = 0; i < d; i++) {
                        double v = 1.0;
                        for (int p2 = 0; p2 < packTokens; p2++) v *= ly.decay[i];
                        qk[i] = v;
                    }
                    for (int k = 1; k < nB; k <<= 1) {
                        stage = "L" + std::to_string(l) + ".tm.scan.blockprefix.k" + std::to_string(k);
                        std::vector<TrackedCt> nxt(nB);
                        for (int b = 0; b < nB; b++) {
                            TrackedCt src = G[b >= k ? b - k : 0];   // dummy when b<k
                            refresh(src, 2);
                            mulPt(src, packCh((b >= k ? qk : zerod).data(), d));
                            TrackedCt acc = G[b];
                            addAligned(acc, src);
                            nxt[b] = acc;
                        }
                        G = nxt;
                        for (int i = 0; i < d; i++) qk[i] = qk[i] * qk[i];
                    }
                    // -- Phase D: S[b] = L[b] + decay^(ln+1) * C[b]
                    for (int b = 0; b < nB; b++) {
                        stage = "L" + std::to_string(l) + ".tm.scan.apply.b" + std::to_string(b);
                        TrackedCt cy = G[b];
                        mulPt(cy, dp);
                        addAligned(S[b], cy);
                    }
                } else if (nB > 1) {
                    // --serial-carry: the ORIGINAL chained form, kept for A/B.
                    // Equivalent to the old inline code: S[b-1] here already
                    // carries its own predecessor, so the chain is the same.
                    for (int b = 1; b < nB; b++) {
                        stage = "L" + std::to_string(l) + ".tm.scan.serialcarry.b" + std::to_string(b);
                        TrackedCt cy = S[b - 1];
                        refresh(cy, 3);
                        mulPt(cy, packLaneOnly(onesd, d, packTokens - 1));
                        for (int step = 1; step < packTokens; step <<= 1) {
                            TrackedCt sh2 = cy; sh2.ct = cc->EvalRotate(sh2.ct, step);
                            addAligned(cy, sh2);
                        }
                        mulPt(cy, dp);
                        addAligned(S[b], cy);
                    }
                }
            } else {
            for (uint32_t t = 0; t < T; t++) {
                stage = "L" + std::to_string(l) + ".tm.scan.t" + std::to_string(t);
                refresh(scanState[l], 3); refresh(x[t], 3);
                mulPt(scanState[l], decPk);                   // decay (.) s
                TrackedCt bx = x[t]; mulPt(bx, bPk);          // b (.) x_t
                addAligned(scanState[l], bx);
                S[t] = scanState[l];
            }
            }
            if (l == 0) dbgDec(S[0], "L0.tm.scan_out");
            if (blockProbe && packTokens > 1)
                for (uint32_t b = 0; b < T; b++) blockDec(S[b], l, (int)b, "scan_out");
            if (equalizeBlocks && packTokens > 1 && T > 1) {
                // level-sync barrier: bootstrap RESETS every ciphertext to the
                // same fresh level, so all blocks leave the scan in lockstep
                // and downstream alignTo gaps stay in the safe 1-2 range.
                stage = "L" + std::to_string(l) + ".tm.scan.equalize";
                for (uint32_t b = 0; b < T; b++) boot(S[b]);
                if (blockProbe)
                    for (uint32_t b = 0; b < T; b++) blockDec(S[b], l, (int)b, "scan_eq");
            }
            stageClose(stScan, scanMs);                   // S3.7 V2
            // ---- readout, gate, projection ----
            const auto stGate = stageOpen();             // S3.7 V2: gateMs (readout mulPt x2, addAligned, polyGate3; refresh(zc,5) is bootMs)
            std::vector<TrackedCt> g(T);
            auto cPk = packCh(ly.c.data(), d), dPk = packCh(ly.dd.data(), d);
            for (uint32_t t = 0; t < T; t++) {
                stage = "L" + std::to_string(l) + ".tm.gate.t" + std::to_string(t);
                TrackedCt zc = S[t]; mulPt(zc, cPk);
                TrackedCt zd = x[t]; mulPt(zd, dPk);
                addAligned(zc, zd);
                refresh(zc, 5);      // polyGate3 costs ~3; 5 keeps a margin
                g[t] = polyGate3(zc, ly.p0, ly.p1, ly.p2);
                if (stageProbe && t == 0) {
                    // gateCheck (instrumentation adapted from the exp28 tree):
                    // reference built from the DECRYPTED scan state and x, so
                    // upstream error cannot masquerade as gate error. This is
                    // what exposed the polyGate3 packing-convention bug.
                    try {
                        auto dS = decCh(S[0]); auto dX = decCh(x[0]); auto pv = decCh(g[0]);
                        double num = 0, den = 0; long long nf = 0;
                        for (int i = 0; i < d; i++) {
                            double zr = dS[i]*ly.c[i] + dX[i]*ly.dd[i];
                            double pr = ly.p0[i] + zr*(ly.p1[i] + zr*ly.p2[i]);
                            double ref = zr * pr;
                            if (!std::isfinite(pv[i]) || !std::isfinite(ref)) { nf++; continue; }
                            num = std::max(num, std::abs(pv[i]-ref)); den = std::max(den, std::abs(ref));
                        }
                        std::cout << "{\"gateCheck\":true,\"layer\":" << l << ",\"site\":\"tm.gate\""
                                  << ",\"nonFinite\":" << nf << ",\"maxAbsErr\":" << num
                                  << ",\"refMax\":" << den << ",\"relErr\":" << num/(den+1e-12)
                                  << "}" << std::endl;
                    } catch (const std::exception&) {
                        std::cout << "{\"gateCheck\":true,\"layer\":" << l
                                  << ",\"site\":\"tm.gate\",\"undecryptable\":true}" << std::endl;
                    }
                }
            }
            if (l == 0) dbgDec(g[0], "L0.tm.gate_out");
            stageClose(stGate, gateMs);                   // S3.7 V2
            stage = "L" + std::to_string(l) + ".tm.matvecBatch.wout";
#ifdef FHE_SSM_DEMO_SER
            mvLayer = l; mvOpId = 1; mvChunk = 0;      // T1b key: wout
#endif
            auto attn = matvecBatch(g, ly.wout, ly.woutR, ly.woutC, 0);
#ifdef FHE_SSM_DEMO_SER
            mvOpId = -1;
#endif
            if (l == 0) { dbgDec(attn[0], "L0.tm.wout_out"); dbgDec(Hs[0], "L0.tm.Hs_pre_residual"); }
            stage = "L" + std::to_string(l) + ".tm.residual";
            const auto stResTm = stageOpen();            // S3.7 V2: residualMs (addAligned; its refresh boots are bootMs)
            for (uint32_t t = 0; t < T; t++) {
                if (alphaRes != 1.0) {
                    std::vector<double> ar(SLOTS, alphaRes);
                    auto arp = mkPt(ar, attn[t].level(), __LINE__);
                    attn[t].ct = cc->EvalMult(attn[t].ct, arp);
                    cc->RescaleInPlace(attn[t].ct);
                }
                if (!blockParallel) addAligned(Hs[t], attn[t]);   // S3.3 Tier B: the parallel block adds attn after ffn
            }
            if (l == 0) dbgDec(Hs[0], "L0.tm.residual_out");
            stageClose(stResTm, residualMs);              // S3.7 V2

            // ---- channel mix (chunked over dff) ----
            std::vector<TrackedCt> u2(T);
            for (uint32_t t = 0; t < T; t++) {
                stage = "L" + std::to_string(l) + ".cm.rmsnorm.t" + std::to_string(t);
                if (blockParallel) { u2[t] = uNorm[t]; continue; }   // S3.3 Tier B: shared norm output, no cm norm
                if (schedParentFirst) {              // S3.3 P1 at the cm norm entry
                    const int b0 = boots; refresh(Hs[t], pfNeedEntry(2, ly.cmIt)); pfBoots += boots - b0;
                } else refresh(Hs[t], 3);            // rmsnorm self-refreshes internally
                pfNormKind = 2;                      // S3.3 P2 tail need for a cm norm
                u2[t] = rmsnorm(Hs[t], packCh(ly.g2.data(), d), ly.cmA, ly.cmB, ly.cmIt, 1e-5);
            }
            if (l == 0) dbgDec(u2[0], "L0.cm.rmsnorm_out");
            stage = "L" + std::to_string(l) + ".cm.matvecBatchChunked.wk_wr";
#ifdef FHE_SSM_DEMO_SER
            mvLayer = l; mvOpId = 2; mvChunk = 0;      // T1b key: wk (chunk = rowOff)
#endif
            // S3.3 P4 (parent-first): guard the SHARED input once, so the 2K
            // children born from it need no boot of their own. Fires only when
            // P2's tail lift could not carry u2 (never at the demo geometry on
            // the replay); the per-chunk refresh(kk,5)/refresh(rr,4) guards below
            // stay as they are.
            if (schedParentFirst)
                for (auto& c : u2) { const int b0 = boots; refresh(c, pfNeedU2); pfBoots += boots - b0; }
            babyCache.open = shareBabies;              // S3.3 share window: 2K calls on the same u2
            auto kks = matvecBatchChunked(u2, ly.wk, ly.wkR, ly.wkC);   // [chunk][token]
#ifdef FHE_SSM_DEMO_SER
            mvOpId = 3;                                // T1b key: wr (chunk = rowOff)
#endif
            auto rrs = matvecBatchChunked(u2, ly.wr, ly.wrR, ly.wrC);
#ifdef FHE_SSM_DEMO_SER
            mvOpId = -1;
#endif
            babyCache.open = false; babyCache.clear();  // release the cached rotations (peak VRAM unchanged)
            std::vector<TrackedCt> ffn(T); std::vector<bool> haveFfn(T, false);
            const int nChunk = (int)kks.size();
            for (int ch = 0; ch < nChunk; ch++) {
                const int n = std::min((int)Dpad, dff - ch * (int)Dpad);
                std::vector<TrackedCt> hidden(T);
                const auto stHid = stageOpen();          // S3.7 V2: hiddenMs (per chunk: 5 mulPt, 2 addPtV, alignTo, 4 ct x ct; the kk/rr/hd/margin refresh boots are bootMs)
                for (uint32_t t = 0; t < T; t++) {
                    stage = "L" + std::to_string(l) + ".cm.hidden.ch" + std::to_string(ch) + ".t" + std::to_string(t);
                    TrackedCt kk = kks[ch][t], rr = rrs[ch][t];
                    refresh(kk, 5); refresh(rr, 4);
                    if (l == 0 && ch == 0 && t == 0) { dbgDec(kk, "L0.cm.kk"); dbgDec(rr, "L0.cm.rr"); }
                    TrackedCt k2{cc->EvalMult(kk.ct, kk.ct), kk.level()};
                    cc->RescaleInPlace(k2.ct);
                    TrackedCt actT = kk; mulPt(actT, packChunk(ly.q1, ch, n));
                    TrackedCt t2 = k2;   mulPt(t2, packChunk(ly.q2, ch, n));
                    addAligned(actT, t2);
                    TrackedCt kkA = kk; alignTo(kkA, k2.level());
                    TrackedCt k3{cc->EvalMult(k2.ct, kkA.ct), k2.level()};
                    cc->RescaleInPlace(k3.ct);
                    mulPt(k3, packChunk(ly.q3, ch, n));
                    addAligned(actT, k3);
                    addPtV(actT, packChunk(ly.q0, ch, n));
                    TrackedCt r2{cc->EvalMult(rr.ct, rr.ct), rr.level()};
                    cc->RescaleInPlace(r2.ct);
                    TrackedCt gt = rr;  mulPt(gt, packChunk(ly.r1, ch, n));
                    TrackedCt g2t = r2; mulPt(g2t, packChunk(ly.r2, ch, n));
                    addAligned(gt, g2t);
                    addPtV(gt, packChunk(ly.r0, ch, n));
                    // same unguarded-alignTo pattern addAligned had (fixed
                    // above); guard it here too rather than assume it stays
                    // bounded just because it happened to so far.
                    refresh(actT, 2); refresh(gt, 2);
                    uint32_t uu = std::max(actT.level(), gt.level());
                    alignTo(actT, uu); alignTo(gt, uu);
                    if (l == 0 && ch == 0 && t == 0) { dbgDec(actT, "L0.cm.actT"); dbgDec(gt, "L0.cm.gt"); }
                    TrackedCt hd{cc->EvalMult(actT.ct, gt.ct), uu};
                    cc->RescaleInPlace(hd.ct);
                    refresh(hd, 2);
                    hidden[t] = hd;
                }
                if (l == 0) dbgDec(hidden[0], "L0.cm.hidden_out.ch" + std::to_string(ch));
                // W_v column block for this chunk, as a (d x Dpad) slice
                stage = "L" + std::to_string(l) + ".cm.matvecBatch.wv.ch" + std::to_string(ch);
                std::vector<double> Wb((size_t)ly.wvR * Dpad, 0.0);
                for (int i = 0; i < ly.wvR; i++)
                    for (int j = 0; j < n; j++)
                        Wb[(size_t)i * Dpad + j] = ly.wv[(size_t)i * ly.wvC + ch * (int)Dpad + j];
                // 2026-07-19: run the wv matvec SHALLOW. Evidence: at D=1024 a
                // BSGS matvec sums 1024 terms before its single rescale (16x the
                // 64-term CPU case, which passes), so its output carries much
                // more accumulated noise -- and the deeper the operand, the less
                // modulus remains to represent it. Measured: kk/rr (matvec out,
                // level 21) decrypt fine, wv_part (level 23) fails on 2 of 3
                // chunks whose weight blocks are statistically IDENTICAL
                // (rms 7.13-7.15e-3), and which chunk fails varies run to run --
                // i.e. a marginal precision threshold, not a logic bug.
                // Refreshing first puts part at ~20 instead of ~23.
                for (auto& h : hidden) refresh(h, matvecMargin >= 0 ? (uint32_t)matvecMargin : 6);
                stageClose(stHid, hiddenMs);              // S3.7 V2 (the Wb slice above is host work inside this stage)
#ifdef FHE_SSM_DEMO_SER
                mvLayer = l; mvOpId = 4; mvChunk = ch; // T1b key: wv per-chunk (the B6 temporary)
#endif
                auto part = matvecBatch(hidden, Wb, ly.wvR, (int)Dpad, 0);
#ifdef FHE_SSM_DEMO_SER
                mvOpId = -1;
#endif
                // the one link never instrumented on device: isolates the wv
                // matvec (1024-term un-rescaled BSGS accumulation at D=1024)
                // from the accumulation that follows it.
                if ((l == 0 || stageProbe) && canDecrypt) {   // DEMO: no wvCheck without the secret key
                    if (l == 0) dbgDec(part[0], "L0.cm.wv_part.ch" + std::to_string(ch));
                    // Is the SURVIVING chunk actually correct, or merely above
                    // OpenFHE's 5-bit precision guard? Decrypt `hidden` (which
                    // always decodes) and multiply by Wb in the clear, then
                    // compare against the ciphertext result. If relErr is tiny
                    // the circuit is right and only the margin is thin; if it
                    // is large, chunk 2 is silently WRONG and we have been
                    // reading a pass that is not one.
                    try {
                        auto hv = decCh(hidden[0]);
                        auto pv = decCh(part[0]);
                        // BUG FIXED 2026-07-29: std::max(a,b) returns `a` when b is
                        // NaN (a<NaN is false), so a NaN in the decrypted matvec left
                        // `num` at exactly 0 and this check reported PERFECT accuracy
                        // for a totally corrupt ciphertext. That is the source of the
                        // impossible {"maxAbsErr":0} readings. Count non-finite slots
                        // explicitly and report them.
                        double num = 0, den = 0; long long nonFinite = 0;
                        for (int i = 0; i < ly.wvR; i++) {
                            double ref = 0;
                            for (int j = 0; j < (int)Dpad; j++) ref += Wb[(size_t)i * Dpad + j] * hv[j];
                            if (!std::isfinite(pv[i]) || !std::isfinite(ref)) { nonFinite++; continue; }
                            num = std::max(num, std::abs(pv[i] - ref));
                            den = std::max(den, std::abs(ref));
                        }
                        std::cout << "{\"wvCheck\":true,\"layer\":" << l << ",\"ch\":" << ch
                                  << ",\"nonFinite\":" << nonFinite
                                  << ",\"maxAbsErr\":" << num << ",\"refMax\":" << den
                                  << ",\"relErr\":" << (num / (den + 1e-12)) << "}" << std::endl;
                    } catch (const std::exception&) {
                        std::cout << "{\"wvCheck\":true,\"layer\":" << l << ",\"ch\":" << ch << ",\"undecryptable\":true}" << std::endl;
                    }
                }
                if (blockProbe && packTokens > 1) {
                    // per-(block, chunk) wv output: wvCheck only ever sees
                    // block 0. This localizes WHICH blocks' wv outputs are
                    // non-finite per chunk, and the hidden input per block.
                    for (uint32_t t = 0; t < T; t++) {
                        blockDec(hidden[t], l, (int)t, ("wv_in.ch" + std::to_string(ch)).c_str());
                        blockDec(part[t], l, (int)t, ("wv_out.ch" + std::to_string(ch)).c_str());
                    }
                }
                stage = "L" + std::to_string(l) + ".cm.ffnAccum.ch" + std::to_string(ch);
                const auto stFfn = stageOpen();          // S3.7 V2: residualMs (the chunk accumulation is an addAligned)
                for (uint32_t t = 0; t < T; t++) {
                    if (!haveFfn[t]) { ffn[t] = part[t]; haveFfn[t] = true; }
                    else addAligned(ffn[t], part[t]);
                }
                stageClose(stFfn, residualMs);            // S3.7 V2
            }
            stage = "L" + std::to_string(l) + ".cm.residual";
            const auto stResCm = stageOpen();            // S3.7 V2: residualMs
            if (l == 0) dbgDec(ffn[0], "L0.cm.ffn_accum_out");
            if (blockProbe && packTokens > 1)
                for (uint32_t t = 0; t < T; t++) blockDec(ffn[t], l, (int)t, "ffn_accum");
            for (uint32_t t = 0; t < T; t++) addAligned(Hs[t], ffn[t]);
            if (blockParallel)                       // S3.3 Tier B: h' = h + attn + ffn (attn deferred from the tm block)
                for (uint32_t t = 0; t < T; t++) addAligned(Hs[t], attn[t]);
            if (l == 0) dbgDec(Hs[0], "L0.cm.residual_out");
            stageClose(stResCm, residualMs);              // S3.7 V2
            if (blockProbe && packTokens > 1)
                for (uint32_t t = 0; t < T; t++) blockDec(Hs[t], l, (int)t, "residual_out");

            if (probe) {
                auto hv = decCh(Hs[0]);
                double m = 0; for (int i = 0; i < d; i++) m = std::max(m, std::abs(hv[i]));
                std::cout << "{\"probe\":true,\"layer\":" << (l + 1)
                          << ",\"hUsed\":" << Hs[0].level() << ",\"maxAbsH_tok0\":" << m
                          << ",\"boots\":" << boots << "}" << std::endl;
            }
        }
        if (stageTimers) {                              // S3.7 V2: close the layer loop's clock
            syncDev();
            layerLoopMs += msSince(tLoop0);
            layerLoopTimedMs += ((evalMs - loopEv0) - (normBootMs - loopNb0)) + (bootMs - loopBo0)
                              + (encMs - loopEn0) + (decMs - loopDe0);
        }
#ifdef FHE_SSM_DEMO_SER
        // ---- STATEFUL DECODE: persist the carry (2026-08-22) --------------
        // Both halves, both directions. The in-memory path (--serve --stateful)
        // costs nothing; the on-disk path (--state-out) pays one serialize per
        // layer. Written HERE, inside the --passes loop, because scanState and
        // uPrev are scoped to it.
        if (statefulServe) {
            if (canonicalCarry) {
                const long long b0 = canonBoots;
                for (int l = 0; l < L; l++) { canon(scanState[l], "scan"); if (uPrev[l].ct) canon(uPrev[l], "u"); }
                std::cout << "{\"canonicalCarry\":\"tick\",\"boots\":" << (canonBoots - b0) << ",\"scanLevel\":"
                          << scanState[0].level() << ",\"uLevel\":" << (uPrev[0].ct ? uPrev[0].level() : 0)
                          << ",\"maxGap\":" << canonMaxGap << "}" << std::endl;
            }
            carryScan = scanState; carryU = uPrev;
            haveCarry = true; carryTokens += tokensReal;
        } else if (!stateOutPath.empty()) {
            if (canonicalCarry)   // same canonical hand-off for the on-disk carry (the 2^15 test path)
                for (int l = 0; l < L; l++) { canon(scanState[l], "scan"); if (uPrev[l].ct) canon(uPrev[l], "u"); }
            long long sBytes = 0;
            for (int l = 0; l < L; l++) {
                sBytes += saveCt(scanState[l], stateOutPath + ".scan." + std::to_string(l));
                if (uPrev[l].ct)
                    sBytes += saveCt(uPrev[l], stateOutPath + ".u." + std::to_string(l));
            }
            std::ofstream mf(stateOutPath + ".meta");
            mf << "{\"layers\":" << L << ",\"packTokens\":" << packTokens
               << ",\"carryTokens\":" << (carryTokens + tokensReal)
               << ",\"dpad\":" << Dpad << ",\"interleave\":" << (interleave ? 1 : 0)
               << ",\"ringDim\":" << cc->GetRingDimension()
               << ",\"shiftMix\":" << (shiftMix ? 1 : 0)
               << ",\"scanLevel\":" << scanState[0].level()
               << ",\"uLevel\":" << (uPrev[0].ct ? (int)uPrev[0].level() : -1) << "}\n";
            mf.close();
            std::cout << "{\"stateOut\":true,\"file\":\"" << stateOutPath << "\",\"layers\":" << L
                      << ",\"bytes\":" << sBytes
                      << ",\"MB\":" << (double)sBytes / 1048576.0
                      << ",\"carryTokens\":" << (carryTokens + tokensReal)
                      << ",\"scanLevel\":" << scanState[0].level() << "}" << std::endl;
        }
#endif
        if (passes > 1) {
            const double pev = evalMs - pe0, pbo = bootMs - pb0;
            std::cout << "{\"passTiming\":true,\"pass\":" << pass
                      << ",\"evalMs\":" << (int)pev << ",\"bootMs\":" << (int)pbo
                      << ",\"encPtMs\":" << (int)(encPtMs - pp0)
                      << ",\"boots\":" << (boots - pbt0)
                      << ",\"encodedThisPass\":" << (encPtCount - pc0)
                      << ",\"cacheHits\":" << diagCacheHits
                      << ",\"cacheEntries\":" << (long long)diagCache.size()
                      << ",\"msPerToken\":" << (pev + pbo) / std::max<uint32_t>(tokensReal, 1)
                      // S3.7 V2 (additive): the same pass on the disjoint basis
                      << ",\"normBootMs\":" << (int)(normBootMs - pn0)
                      << ",\"msPerTokenExBoot\":" << ((pev - (normBootMs - pn0)) + pbo) / std::max<uint32_t>(tokensReal, 1)
                      << ",\"layerLoopMs\":" << (int)(layerLoopMs - pl0)
                      << "}" << std::endl;
        }
        }   // --passes loop
#ifdef FHE_SSM_DEMO_SER
        if (!serveMode) storePersistFn();
#endif
        H = Hs;

        // final norm + head, measured vs the reference logits.
        // BUG FIXED 2026-07-30 (the Stage-2 "90.4%"): packed mode compared lane
        // packLane's logits against logitsRef row `probeTok` == 0 -- the
        // CIPHERTEXT index, not the TOKEN index. Token 1's logits vs token 0's
        // reference row is ~O(100%) relative error by construction, worsening
        // with lane count (t2: 90.4% vs ref row 0, t4: 139%) -- while every
        // per-lane internal check (norm/gate/wv) sat at 1e-11. Fingerprint in
        // the old logs: packed runs report refMaxAbsLogit 9.8947 (= token 0's)
        // where the true token-1 row has 11.5528. Fix: index the reference by
        // the token; and while here, check EVERY packed lane, which measures
        // error-vs-token-index for free.
        // stage labels HERE, not just in the layer loop: every "L11.cm.residual"
        // failure ever recorded was actually thrown below (the label was stale).
        // Check EVERY token (all lanes in packed mode, all ciphertexts
        // unpacked) -- this is the error-vs-token-index measurement, and it
        // makes packed vs unpacked comparable per-token. The extra final-norm
        // cost in unpacked mode is ~seconds against a ~354 s encode wall.
        int rows, cols; auto head = B.mat("head", rows, cols);   // head: (V, d)
        // DEMO --ct-in: T is the CLIENT's token count and is no longer clamped
        // to the bundle, so the logits_ref compare below must bound-check
        // against the reference's own row count (it indexes tok * V).
        // refRowsAvail is already --bundle-row-offset-adjusted; under
        // --lanes-full it spans all NL*Tlane lane-major rows.
        const int refRowsMain = refRowsAvail;
        std::vector<TrackedCt> hfBlocks((packTokens > 1) ? T : 0);
        if (packTokens > 1) {
            for (uint32_t b = 0; b < T; b++) {
                stage = "output.finalNorm.b" + std::to_string(b);
                if (schedParentFirst) { const int b0 = boots; refresh(H[b], pfNeedEntry(0, (int)outR[2])); pfBoots += boots - b0; }   // S3.3 P1
                hfBlocks[b] = rmsnorm(H[b], packCh(gOut.data(), d), outR[0], outR[1], (int)outR[2], 1e-5);
                if (finalRefresh) { stage = "output.finalRefresh.b" + std::to_string(b); boot(hfBlocks[b]); }
            }
        }
#ifdef FHE_SSM_DEMO_SER
        // ---- DEMO --ct-out: server output. Serialize the FINAL post-rmsnorm
        // hidden-state ciphertexts (the exact objects the decode loop below
        // would decrypt) and SKIP the decrypt + logits entirely -- this
        // process holds no secret key. The client's --dec-out step performs
        // the decrypt + head matvec on these files. With --final-refresh the
        // packed blocks were already booted above, so the shipped ciphertexts
        // carry maximal remaining modulus for the client decode; the unpacked
        // path mirrors the decode loop's per-token norm(+optional boot).
        if (!ctOutPath.empty()) {
            auto tOut0 = Clock::now();
            long long bytes = 0; uint32_t outLvl = 0; const int nOut = (int)T;
            if (packTokens > 1) {
                for (uint32_t b = 0; b < T; b++) {
                    stage = "output.serialize.b" + std::to_string(b);
                    outLvl = hfBlocks[b].level();
                    bytes += saveCt(hfBlocks[b], ctOutPath + "." + std::to_string(b));
                }
            } else {
                for (uint32_t tok = 0; tok < T; tok++) {
                    stage = "output.finalNorm.tok" + std::to_string(tok);
                    if (schedParentFirst) { const int b0 = boots; refresh(H[tok], pfNeedEntry(0, (int)outR[2])); pfBoots += boots - b0; }   // S3.3 P1
                    TrackedCt hf = rmsnorm(H[tok], packCh(gOut.data(), d), outR[0], outR[1], (int)outR[2], 1e-5);
                    if (finalRefresh) { stage = "output.finalRefresh.tok" + std::to_string(tok); boot(hf); }
                    stage = "output.serialize.tok" + std::to_string(tok);
                    outLvl = hf.level();
                    bytes += saveCt(hf, ctOutPath + "." + std::to_string(tok));
                }
            }
            writeCtMeta(ctOutPath, nOut, tokensReal, outLvl);
            std::cout << "{\"serverOut\":true,\"file\":\"" << ctOutPath << "\",\"count\":" << nOut
                      << ",\"tokensReal\":" << tokensReal << ",\"level\":" << outLvl
                      << ",\"totalBytes\":" << bytes
                      << ",\"serializeMs\":" << (int)msSince(tOut0) << "}" << std::endl;
        } else {
#endif
        // FAILURE-TOLERANT OUTPUT DECODE (2026-07-31). Previously the FIRST
        // failed token decode threw into the outer catch and abandoned this
        // loop -- so every historical "output.decode.tok0" failure only proves
        // token 0 (block 0) died; blocks 1..N were never decoded at all. The
        // "shallowest block destroyed" reading rested partly on that abort
        // artifact. Now: a failed block is reported per-block and the loop
        // continues, so one failing run localizes WHICH blocks died. A block
        // whose ciphertext fails to decode skips its remaining lanes (they
        // share the ciphertext).
        const int nProbe = (packTokens > 1) ? (int)tokensReal : (int)T;
        std::set<int> deadBlocks;
        for (int tok = 0; tok < nProbe; tok++) {
            std::vector<double> hfv;
            if (packTokens > 1) {
                const int blk = tok / packTokens;
                if (deadBlocks.count(blk)) continue;
                stage = "output.decode.tok" + std::to_string(tok);
                try {
                    hfv = decLane(hfBlocks[blk], tok % packTokens);
                } catch (const std::exception& de) {
                    deadBlocks.insert(blk);
                    failed = true;
                    std::cout << "{\"logitCheck\":true,\"token\":" << tok
                              << ",\"block\":" << blk << ",\"decodeFail\":true"
                              << ",\"level\":" << hfBlocks[blk].level()
                              << ",\"what\":\"" << de.what() << "\"}" << std::endl;
                    continue;
                }
            } else {
                stage = "output.finalNorm.tok" + std::to_string(tok);
                if (schedParentFirst) { const int b0 = boots; refresh(H[tok], pfNeedEntry(0, (int)outR[2])); pfBoots += boots - b0; }   // S3.3 P1
                TrackedCt hf = rmsnorm(H[tok], packCh(gOut.data(), d), outR[0], outR[1], (int)outR[2], 1e-5);
                if (finalRefresh) { stage = "output.finalRefresh.tok" + std::to_string(tok); boot(hf); }
                stage = "output.decode.tok" + std::to_string(tok);
                hfv = decCh(hf);
            }
            // --lanes-full: one decrypted slot vector carries every lane. Loop
            // lanes; lane<0 = the classic single-sequence path (hidden = hfv
            // channels 0..d, ref row = tok). Lane r>=0: hidden = slots
            // r*Dpad.., ref row = r*laneT+tok. Replica echo (slot NL+r vs r,
            // cyclic fill) is checked on the RAW decrypted slots once/token.
            const int nLanes = (lanesFull > 0) ? lanesFull : 1;
            if (lanesFull > 0 && (uint32_t)(2 * lanesFull) <= REP) {
                double num = 0, den = 0;
                for (int r = 0; r < lanesFull; r++)
                    for (int j = 0; j < d; j++) {
                        const double a = hfv[(size_t)(r + lanesFull) * Dpad + j];
                        const double b = hfv[(size_t)r * Dpad + j];
                        num += (a - b) * (a - b); den += b * b;
                    }
                lfEchoWorst = std::max(lfEchoWorst, std::sqrt(num / std::max(den, 1e-300)));
            }
            for (int lane = 0; lane < nLanes; lane++) {
            const double* hid = (lanesFull > 0) ? hfv.data() + (size_t)lane * Dpad : hfv.data();
            const size_t refRow = (lanesFull > 0) ? (size_t)lane * laneT + tok : (size_t)tok;
            std::vector<double> logits(V, 0.0);
            for (int r = 0; r < V; r++) { double s = 0; for (int j = 0; j < d; j++) s += head[(size_t)r * cols + j] * hid[j]; logits[r] = s; }
            if (!dumpLogits.empty()) {
                // rows appended in token order (lane-minor under --lanes-full);
                // dead blocks leave gaps -- the offline reader must use the
                // sidecar, which records "tok" or "tok lane" per row.
                std::ofstream lf(dumpLogits, std::ios::binary | std::ios::app);
                lf.write((const char*)logits.data(), (std::streamsize)(V * sizeof(double)));
                std::ofstream tf(dumpLogits + ".tokens", std::ios::app);
                if (lanesFull > 0) tf << tok << " " << lane << "\n"; else tf << tok << "\n";
            }
            if (logitsRef && (int)refRow < refRowsMain) {
                double e = 0, refmax = 0; long long nfLogit = 0;
                double sse = 0, sref = 0; int argGot = -1, argRef = -1; double mxGot = -1e300, mxRef = -1e300;   // A-P0b
                for (int r = 0; r < V; r++) {
                    double lr = logits[r], rr2 = logitsRef[refRow * V + r];
                    if (!std::isfinite(lr) || !std::isfinite(rr2)) { nfLogit++; continue; }  // see wvCheck NaN note
                    e = std::max(e, std::abs(lr - rr2));
                    refmax = std::max(refmax, std::abs(rr2));
                    sse += (lr - rr2) * (lr - rr2); sref += rr2 * rr2;
                    if (lr > mxGot) { mxGot = lr; argGot = r; }
                    if (rr2 > mxRef) { mxRef = rr2; argRef = r; }
                }
                const double relRms = (nfLogit > 0) ? -1.0 : std::sqrt(sse / (sref + 1e-300));   // -1 sentinel on any non-finite
                const bool top1 = (nfLogit == 0) && (argGot == argRef);
                lcTokens++; if (top1) lcTop1++;
                if (nfLogit == 0) { lcRms2Sum += relRms * relRms; lcRmsMax = std::max(lcRmsMax, relRms); }
                if (lanesFull > 0) {
                    lfTokens[lane]++; if (top1) lfTop1[lane]++;
                    if (nfLogit == 0) { lfRms2Sum[lane] += relRms * relRms; lfRmsMax[lane] = std::max(lfRmsMax[lane], relRms); }
                }
                // NaN TRAP CLOSED (audit 2026-07-30). Non-finite slots used to
                // `continue` past BOTH accumulators, so an all-NaN token left
                // e=0 and refmax=0 and printed relLogitErr = 0/1e-12 = 0 --
                // PERFECT accuracy for a destroyed ciphertext -- while `failed`
                // stayed false. Exactly the std::max-with-NaN family of bug the
                // wvCheck site hit (52 such records in the b4* runs). Now any
                // non-finite logit marks the run failed and reports a -1
                // sentinel instead of a flattering zero.
                worstLogitErr = std::max(worstLogitErr, e);
                if (nfLogit > 0) failed = true;
                std::cout << "{\"logitCheck\":true,\"token\":" << tok;
                if (lanesFull > 0) std::cout << ",\"lane\":" << lane;
                std::cout << ",\"nonFiniteLogits\":" << nfLogit << ",\"maxAbsLogitErr\":" << e
                          << ",\"refMaxAbsLogit\":" << refmax << ",\"relLogitErr\":"
                          << (nfLogit > 0 ? -1.0 : e / (refmax + 1e-12))
                          << ",\"relLogitErrRms\":" << relRms << ",\"top1Agree\":" << (top1 ? "true" : "false")
                          << ",\"argmax\":" << argGot << ",\"argmaxRef\":" << argRef << "}" << std::endl;
            }
            }   // lane loop
        }
#ifdef FHE_SSM_DEMO_SER
        }   // else (!ctOutPath): the normal decode loop above ran
#endif
    } catch (const std::exception& e) {
        // Do NOT return here. A run that dies in the OUTPUT decode has already
        // executed the full 12-layer circuit, so its timing is valid and is
        // often the only thing being measured (e.g. --ring-mix / --band cost
        // proxies, where the logits are meaningless by construction). Returning
        // early threw away encPtMs/gpuEvalMs and made those runs unmeasurable.
        std::cout << "{\"error\":\"" << e.what() << "\",\"stage\":\"" << stage << "\",\"boots\":" << boots
                  << ",\"peakVramGB\":" << peakVramGB() << "}" << std::endl;
        failed = true;
        babyCache.open = false; babyCache.clear(); pfNormKind = 0;   // S3.3 (review 2026-09-03): close the share window on a throw
    }

#ifdef FHE_SSM_DEMO_SER
    // ---- DEMO --serve loop tail: answer, clean up, next request ------------
    if (serveMode) {
        const std::string reqPrefix = serveDir + "/req." + std::to_string(serveReqN);
        const std::string respPrefix = serveDir + "/resp." + std::to_string(serveReqN);
        // Answer FIRST (the resp.* ciphertexts were already written by the
        // --ct-out branch inside the try), then delete the request files; the
        // client deletes its own resp files after --dec-out.
        if (failed) {
            serveAnyFailed = true;
            touchFile(respPrefix + ".error", "forward failed; see the server log");
        } else {
            touchFile(respPrefix + ".done", "ok");
        }
        long long nReq = 0;
        try { nReq = metaInt(readCtMeta(reqPrefix), "count", 0); }
        catch (const std::exception&) { nReq = 0; }
        for (long long i2 = 0; i2 < nReq; i2++)
            std::remove((reqPrefix + "." + std::to_string(i2)).c_str());
        std::remove((reqPrefix + ".meta").c_str());
        std::remove((reqPrefix + ".ready").c_str());
        std::cout << "{\"serve\":\"served\",\"req\":" << serveReqN
                  << ",\"failed\":" << (failed ? "true" : "false")
                  << ",\"tokens\":" << tokensReal
                  << ",\"reqEvalMs\":" << (int)(evalMs - serveEval0)
                  << ",\"reqBootMs\":" << (int)(bootMs - serveBoot0)
                  << ",\"reqEncPtMs\":" << (int)(encPtMs - serveEncPt0)
                  << ",\"reqBoots\":" << (boots - serveBoots0)
                  << ",\"reqMsPerToken\":"
                  << ((evalMs - serveEval0) + (bootMs - serveBoot0)) / std::max<uint32_t>(tokensReal, 1)
                  // S3.7 V2 (additive): the disjoint basis and the layer-loop stages for THIS request
                  << ",\"reqNormBootMs\":" << (int)(normBootMs - serveNormBoot0)
                  << ",\"reqMsPerTokenExBoot\":"
                  << (((evalMs - serveEval0) - (normBootMs - serveNormBoot0)) + (bootMs - serveBoot0)) / std::max<uint32_t>(tokensReal, 1)
                  << ",\"reqLayerLoopMs\":" << (int)(layerLoopMs - serveLoop0)
                  << ",\"reqLayerLoopUntimedMs\":" << (int)((layerLoopMs - serveLoop0) - (layerLoopTimedMs - serveLoopTimed0))
                  << ",\"reqStageMs\":{\"shiftMix\":" << (int)(shiftMixMs - serveShift0)
                  << ",\"scan\":" << (int)(scanMs - serveScan0) << ",\"gate\":" << (int)(gateMs - serveGate0)
                  << ",\"hidden\":" << (int)(hiddenMs - serveHid0) << ",\"residual\":" << (int)(residualMs - serveRes0) << "}"
                  << ",\"reqHostPtEncodeMs\":" << (int)(hostPtEncodeMs - servePtEnc0)
                  << ",\"reqHostPtEncodeCount\":" << (hostPtEncodeCount - servePtEncN0)
                  << ",\"peakVramGB\":" << peakVramGB() << "}" << std::endl;
        vramTraceLine("serve.done." + std::to_string(serveReqN));
        // --store-file: persist after EVERY successful request that added
        // entries -- the base file once, journal appends thereafter (a
        // stateful session drifts through levels for its first ~10 ticks).
        if (!failed) storePersistFn();
        serveDone++; serveReqN++;
        ctInPath.clear(); ctOutPath.clear();
        continue;                      // wait for the next request
    }
    break;                             // non-serve: the pass region ran once
    }   // end DEMO --serve loop
    if (serveMode) failed = serveAnyFailed;   // process verdict = any request failed
#endif

    // tokensReal, not T: in Stage-2 packed mode T is reset to the CIPHERTEXT
    // count (1), which used to misreport tokens/bootsPerToken/gpuMsPerToken.
    //
    // PASSES BUG FIXED (audit 2026-07-30) — this was an EXACTLY-Nx error of
    // the same family as paper 1's 12x errata. evalMs/bootMs/boots accumulate
    // across the whole --passes loop, but tokensReal counts ONE pass's tokens,
    // so every per-token field was `passes` times too LARGE (i.e. the run
    // looked `passes`x SLOWER than it was: w14_stack_res.jsonl reported
    // msPerToken 500.9 and bootsPerToken 15.19 where the per-pass truth was
    // ~162 and 5.06). Direction was pessimistic, so no published speed claim
    // was inflated, and REPORT_20260731_OVERNIGHT quoted the per-pass
    // passTiming lines instead -- but any tool aggregating summary lines got
    // `passes`x garbage. Denominator is now token-passes, so the field means
    // "mean ms per token across all passes" for any --passes value.
    vramTraceLine("final");
    const double tokPasses = (double)std::max<uint32_t>(tokensReal, 1) * (double)std::max(1, passes);
    std::cout << "{\"summary\":true,\"realModel\":true,\"arch\":\"" << arch << "\",\"tag\":\"" << tag << "\""
              << ",\"tokens\":" << tokensReal << ",\"packTokens\":" << packTokens
              << ",\"layers\":" << L << ",\"secure\":" << (secure ? "true" : "false")
              << ",\"boots\":" << boots << ",\"bootsPerToken\":" << (double)boots / tokPasses
              << ",\"clientEncryptMs\":" << (int)encMs << ",\"gpuEvalMs\":" << (int)evalMs
              << ",\"gpuBootstrapMs\":" << (int)bootMs << ",\"decryptMs\":" << (int)decMs
              << ",\"gpuMsPerToken\":" << (evalMs + bootMs) / tokPasses
              << ",\"encPtMs\":" << (int)encPtMs << ",\"encPtCount\":" << encPtCount
              << ",\"encPtFracOfEval\":" << (evalMs > 0 ? encPtMs / evalMs : 0.0)
              << ",\"encThreads\":" << ompThreads << ",\"band\":" << bandWidth
              << ",\"ringMix\":" << (ringMix ? "true" : "false") << ",\"bootFloor\":" << BOOT_FLOOR
              << ",\"pipeline\":" << (pipeline ? "true" : "false") << ",\"pipeMs\":" << (int)pipeMs
              << ",\"staticNorm\":" << (staticNorm ? "true" : "false")
              << ",\"normEps\":" << normEpsUsed                 // S3.7 V1: eps applied in rmsnorm (0 = --no-norm-eps), additive
              << ",\"periodicEncodeOn\":" << (periodicEncode ? "true" : "false") << ",\"periodicVerifyEvery\":" << periodicVerifyEvery   // S3.7 V5, additive
              << ",\"serialCarry\":" << (serialCarry ? "true" : "false")
              << ",\"finalRefresh\":" << (finalRefresh ? "true" : "false")
              << ",\"equalizeBlocks\":" << (equalizeBlocks ? "true" : "false")
              << ",\"scaleBits\":" << scaleBits
              << ",\"passes\":" << passes << ",\"cacheDiags\":" << (cacheDiags ? "true" : "false")
              << ",\"cacheEntries\":" << (long long)diagCache.size()
              << ",\"cacheHits\":" << diagCacheHits
              << ",\"msPerToken\":" << (evalMs + bootMs) / tokPasses
              << ",\"tokPerSec\":" << (1000.0 * tokPasses / std::max(1.0, evalMs + bootMs))
              // WALL-CLOCK SELF-CHECK (audit 2026-07-30). The timers cover only
              // matvec/matvecBatch/rmsnorm/bootstrap; the scan, shift-mix,
              // gates, elementwise channel-mix, residuals, alignTo and the
              // plaintext head matvec are in NO timer, so (evalMs+bootMs)
              // understates real cost by a one-sided margin that GROWS as the
              // encode wall shrinks (measured 5% dense secure, 13-20% for fast
              // configs). Emitting total wall and the residual makes every
              // future run self-auditing instead of requiring an external
              // /usr/bin/time reconciliation.
              << ",\"totalWallMs\":" << (int)msSince(tRun0)
              // T0.1 (S3.1): setupMs was computed at LoadContext since 2026-07
              // but emitted only in the probe-mode header; every ex-setup
              // figure in CLOCK_SECURE_20260823.md had to estimate it from a
              // residual. ADDITIVE fields, per the A-P0b precedent below.
              << ",\"setupMs\":" << (int)setupMs
              << ",\"unaccountedMs\":" << (int)(msSince(tRun0) - (evalMs + bootMs + encMs + decMs) - setupMs)
              << ",\"wallMsPerToken\":" << (msSince(tRun0) - setupMs) / tokPasses
              << ",\"peakVramGB\":" << peakVramGB()
              << ",\"procPeakVramGB\":" << g_procVramPeakGB
              // M1 (2026-09-14): run-wide pool watermarks per device (see g_poolHighRun); additive.
              << ",\"poolHighRun\":[" << [&]{ std::string s; bool first = true;
                     for (const auto& kv : g_poolHighRun) {
                         s += (first ? "" : ",") + std::string("{\"dev\":") + std::to_string(kv.first)
                            + ",\"usedHighGB\":" + std::to_string(kv.second.first)
                            + ",\"reservedHighGB\":" + std::to_string(kv.second.second) + "}";
                         first = false;
                     } return s; }() << "]"
              << ",\"syncTimersOn\":" << (syncTimers ? "true" : "false")
              << ",\"syncSplit\":{\"babyRotMs\":" << (int)syncBabyRotMs
              << ",\"ctPtApplyMs\":" << (int)syncCtPtMs
              << ",\"giantMs\":" << (int)syncGiantMs
              << ",\"rescaleMs\":" << (int)syncRescaleMs
              << ",\"bootMs\":" << (int)syncBootMs << "}"
              // ---- S3.7 V2 (2026-09-05): timer corrections, ADDITIVE (S3.6 N1 / T05). ----
              // normBootMs = the rmsnorm windows' own bootstraps (in evalMs AND bootMs);
              // gpuEvalMsExBoot / msPerTokenExBoot = the disjoint basis. The layer loop:
              // layerLoopMs is its whole wall (overlaps evalMs/bootMs); layerLoopUntimedMs
              // = wall minus the timed sum inside it; stageMs are the five disjoint
              // elementwise stage timers; layerLoopResidualMs = untimed minus the stages
              // (norm-stage host work, the Wb slices, probes). hostPtEncode* = every
              // mkPt (overlaps stageMs and evalMs; the *Stage* pair is the loop share).
              // unaccountedMsExBoot = unaccountedMs + normBootMs is the TRUE untimed
              // bucket (unaccountedMs subtracts the norm boots twice); minus the stages
              // gives unaccountedMsAfterStages.
              << ",\"normBootMs\":" << (int)normBootMs
              << ",\"gpuEvalMsExBoot\":" << (int)(evalMs - normBootMs)
              << ",\"msPerTokenExBoot\":" << ((evalMs - normBootMs) + bootMs) / tokPasses
              << ",\"stageTimersOn\":" << (stageTimers ? "true" : "false")
              << ",\"layerLoopMs\":" << (int)layerLoopMs
              << ",\"layerLoopUntimedMs\":" << (int)(layerLoopMs - layerLoopTimedMs)
              << ",\"stageMs\":{\"shiftMix\":" << (int)shiftMixMs << ",\"scan\":" << (int)scanMs
              << ",\"gate\":" << (int)gateMs << ",\"hidden\":" << (int)hiddenMs
              << ",\"residual\":" << (int)residualMs << "}"
              << ",\"layerLoopResidualMs\":" << (int)((layerLoopMs - layerLoopTimedMs)
                                                       - (shiftMixMs + scanMs + gateMs + hiddenMs + residualMs))
              << ",\"hostPtEncodeMs\":" << (int)hostPtEncodeMs << ",\"hostPtEncodeCount\":" << hostPtEncodeCount
              << ",\"hostPtEncodeStageMs\":" << (int)hostPtEncodeStageMs
              << ",\"hostPtEncodeStageCount\":" << hostPtEncodeStageCount
              << ",\"unaccountedMsExBoot\":" << (int)(msSince(tRun0) - (evalMs - normBootMs + bootMs + encMs + decMs) - setupMs)
              << ",\"unaccountedMsAfterStages\":" << (int)(msSince(tRun0) - (evalMs - normBootMs + bootMs + encMs + decMs) - setupMs
                                                            - (shiftMixMs + scanMs + gateMs + hiddenMs + residualMs))
#ifdef FHE_SSM_DEMO_SER
              << ",\"storeOn\":" << (compressedStore ? "true" : "false")
              << ",\"storeHits\":" << storeHits << ",\"storeMisses\":" << storeMisses
              << ",\"storeDead\":" << storeDead
              << ",\"storeUncompressible\":" << storeUncompressible
              << ",\"periodicEncodeCount\":" << periodicEncodeCount << ",\"periodicVerifyCount\":" << periodicVerifyCount
              << ",\"periodicVerifyOk\":" << periodicVerifyOk << ",\"periodicFallbackCount\":" << periodicFallbackCount   // S3.7 V5, additive
              << ",\"storeFetchMs\":" << (int)storeFetchMs
              << ",\"storeBytes\":" << storeBytes
              << ",\"storeFile\":\"" << storeFilePath << "\",\"storeLoaded\":" << (storeLoaded ? "true" : "false")
              << ",\"storeSaved\":" << (storeSaved ? "true" : "false")
              << ",\"storeVerifyOk\":" << storeVerifyOk << ",\"storeVerifyBad\":" << storeVerifyBad
              << ",\"storeVerifyMs\":" << (int)storeVerifyMs
              << ",\"storeAppends\":" << storeAppends << ",\"storeJournalEntries\":" << storeJournalEntries
#endif
              << ",\"failed\":" << (failed ? "true" : "false")
              << ",\"worstLogitErr\":" << worstLogitErr
              // ---- A-P0b (2026-08-22): records-schema fields. ADDITIVE ONLY -- no existing field
              // changes. (a) the placement rule's inputs, so matvecMargin/extraDepth/ring no longer
              // have to be recovered from filenames; (b) FIDELITY vs the bundle's own logits_ref
              // (R4: says the circuit is faithful, never that the model is good); (c) QUALITY =
              // native_test_ppl, read from a sidecar beside the bundle when one exists
              // (bundle_<tag>.provenance.json / <tag>_export.json / <tag>_eval.json), else null,
              // plus the weights_sha256 that binds the bundle to its checkpoint (R2).
              << ",\"logRing\":" << logRing << ",\"ringDim\":" << cc->GetRingDimension()
              << ",\"slots\":" << SLOTS << ",\"dpad\":" << Dpad << ",\"rep\":" << REP
              << ",\"levelBudget\":[" << lbA << "," << lbB << "]" << ",\"depth\":" << depth
              << ",\"extraDepth\":" << extraDepth << ",\"matvecMargin\":" << matvecMargin
              // S3.3 schedule levers (ADDITIVE): the flags that changed the trace, plus
              // the counters the pod compares against results/theory/trace_schedule_20260903.txt.
              << ",\"schedule\":\"" << schedule << "\",\"pfEntry\":\"" << pfEntry << "\""
              << ",\"shareBabies\":" << (shareBabies ? "true" : "false")
              << ",\"newtonDepth2\":" << (newtonDepth2 ? "true" : "false")
              << ",\"poolLevels\":" << poolLevels
              << ",\"blockType\":\"" << (blockParallel ? "parallel" : "sequential") << "\""
              << ",\"pfBoots\":" << pfBoots << ",\"babyShared\":" << babyShared << ",\"babyComputed\":" << babyComputed
#ifdef FHE_SSM_DEMO_SER
              << ",\"poolBuilds\":" << poolBuilds << ",\"poolBuildMs\":" << (int)poolBuildMs
              << ",\"canonicalCarry\":" << (canonicalCarry ? "true" : "false") << ",\"canonBoots\":" << canonBoots
              << ",\"canonMaxGap\":" << canonMaxGap << ",\"canonLevel\":" << canonLevel
#endif
              << ",\"interleave\":" << (interleave ? "true" : "false")
              << ",\"fidelity\":{\"logitCheckTokens\":" << lcTokens
              << ",\"top1Agree\":" << (lcTokens > 0 ? (double)lcTop1 / (double)lcTokens : -1.0)
              << ",\"relLogitErrRmsMean\":" << (lcTokens > 0 ? std::sqrt(lcRms2Sum / (double)lcTokens) : -1.0)
              << ",\"relLogitErrRmsMax\":" << (lcTokens > 0 ? lcRmsMax : -1.0)
              << ",\"worstLogitErrIsMaxClass\":true}"
              << ",\"quality\":" << fhe_ssm_quality_sidecar(bundleDir, tag)
              // PROVENANCE STAMP. Pins WHICH CODE and WHICH COMMAND produced
              // this record, the way weights_sha256 pins which model (R2).
              // Additive: no computation and no existing field changes, so it
              // does not trigger the immutability rule. See run_stamp.h.
              << fhe_ssm::runStamp(argc, argv) << "}" << std::endl;
    if (lanesFull > 0) {
        // Per-lane fidelity + the replica-echo worst. Bars live in the GATE
        // (gate_lanes_fullpass.py), not here — this line only reports.
        std::cout << "{\"lanesFullSummary\":true,\"lanes\":" << lanesFull
                  << ",\"tokensPerLane\":" << T << ",\"laneEchoRelErr\":" << lfEchoWorst
                  << ",\"perLane\":[";
        for (int r = 0; r < lanesFull; r++) {
            if (r) std::cout << ",";
            std::cout << "{\"lane\":" << r << ",\"tokens\":" << lfTokens[r]
                      << ",\"top1Agree\":" << (lfTokens[r] > 0 ? (double)lfTop1[r] / (double)lfTokens[r] : -1.0)
                      << ",\"relLogitErrRmsMean\":" << (lfTokens[r] > 0 ? std::sqrt(lfRms2Sum[r] / (double)lfTokens[r]) : -1.0)
                      << ",\"relLogitErrRmsMax\":" << (lfTokens[r] > 0 ? lfRmsMax[r] : -1.0) << "}";
        }
        std::cout << "],\"failed\":" << (failed ? "true" : "false") << "}" << std::endl;
    }
    return failed ? 1 : 0;
}
