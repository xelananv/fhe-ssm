// peer_pool_probe.cu — does a kernel on device A get to WRITE memory that
// device B allocated from its stream-ordered pool (cudaMallocAsync)?
//
// Why (review 2026-09-03, demo-pod prep). FIDESlib's limb buffers come from
// cudaMallocAsync (src/CudaUtils.cu GPUmalloc, mempool slabs). The harness's
// compressed-store expand kernel runs on devices[0] and writes every limb
// buffer of the pool plaintexts, roughly half of which live on the other
// card under --devices 0,1. cudaDeviceEnablePeerAccess covers cudaMalloc
// memory only; POOL memory is made peer-accessible per pool with
// cudaMemPoolSetAccess (CUDA C++ Programming Guide, "Stream Ordered Memory
// Allocator" -> "Device Accessibility for Multi-GPU Support"). Without the
// grant the first peer write is an illegal address, which surfaces later
// inside FIDESlib as "Cuda failure ... illegal memory access".
//
// This probe settles it in a minute on the box, before any two-card store
// run: it allocates 1 MiB on device 1 from the default pool, launches a
// writer kernel on device 0 (a) after cudaDeviceEnablePeerAccess only, in a
// child process so the sticky fault cannot poison the parent, and (b) after
// cudaMemPoolSetAccess as well, and prints one JSON verdict line.
//
// Build:  nvcc -arch=native -O2 -o peer_pool_probe peer_pool_probe.cu
// Run:    ./peer_pool_probe            (needs >= 2 devices)
// PASS:   {"peerPoolProbe":true,...,"withPoolAccess":"ok","withoutPoolAccess":"fault"|"ok"}
//         Either outcome for (a) is fine; (b) MUST be "ok" for the store path.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

__global__ void writer(unsigned long long* p, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0xC0FFEE00ULL + i;
}

static const char* attempt(bool grantPool) {
    const int A = 0, B = 1;
    const size_t n = (1u << 20) / 8;
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, A, B) != cudaSuccess || !can) return "no-peer-capability";
    if (cudaSetDevice(A) != cudaSuccess) return "setdevice-failed";
    cudaError_t pe = cudaDeviceEnablePeerAccess(B, 0);
    if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) return "enable-peer-failed";
    cudaGetLastError();
    if (grantPool) {
        cudaMemPool_t mp = nullptr;
        if (cudaDeviceGetDefaultMemPool(&mp, B) != cudaSuccess) return "getpool-failed";
        cudaMemAccessDesc ad{};
        ad.location.type = cudaMemLocationTypeDevice;
        ad.location.id = A;
        ad.flags = cudaMemAccessFlagsProtReadWrite;
        if (cudaMemPoolSetAccess(mp, &ad, 1) != cudaSuccess) return "setaccess-failed";
    }
    // allocate on B from ITS default pool (what FIDESlib's GPUmalloc does)
    if (cudaSetDevice(B) != cudaSuccess) return "setdevice-b-failed";
    cudaStream_t sb; cudaStreamCreate(&sb);
    unsigned long long* p = nullptr;
    if (cudaMallocAsync((void**)&p, n * 8, sb) != cudaSuccess) return "mallocasync-failed";
    cudaStreamSynchronize(sb);
    // write from A
    if (cudaSetDevice(A) != cudaSuccess) return "setdevice-a2-failed";
    writer<<<(unsigned)((n + 255) / 256), 256>>>(p, n);
    cudaError_t le = cudaGetLastError();
    cudaError_t se = cudaDeviceSynchronize();
    if (le != cudaSuccess || se != cudaSuccess) return "fault";
    // verify on B
    cudaSetDevice(B);
    unsigned long long h[4] = {0, 0, 0, 0};
    if (cudaMemcpy(h, p, sizeof h, cudaMemcpyDeviceToHost) != cudaSuccess) return "readback-failed";
    if (h[0] != 0xC0FFEE00ULL || h[3] != 0xC0FFEE03ULL) return "wrong-data";
    cudaFreeAsync(p, sb); cudaStreamSynchronize(sb);
    return "ok";
}

int main(int argc, char** argv) {
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd < 2) {
        printf("{\"peerPoolProbe\":false,\"reason\":\"need >= 2 devices\",\"devices\":%d}\n", nd);
        return 2;
    }
    if (argc > 1) {   // child mode: one attempt, result on stdout
        const bool grant = std::strcmp(argv[1], "grant") == 0;
        printf("%s\n", attempt(grant));
        return 0;
    }
    char res[2][64] = {{0}, {0}};
    const char* modes[2] = {"nogrant", "grant"};
    for (int m = 0; m < 2; m++) {
        int fds[2]; if (pipe(fds) != 0) return 3;
        pid_t pid = fork();
        if (pid == 0) {
            dup2(fds[1], 1); close(fds[0]);
            execl(argv[0], argv[0], modes[m], (char*)nullptr);
            _exit(127);
        }
        close(fds[1]);
        FILE* f = fdopen(fds[0], "r");
        if (!f || !fgets(res[m], sizeof res[m], f)) std::snprintf(res[m], sizeof res[m], "child-died");
        if (f) fclose(f);
        int st = 0; waitpid(pid, &st, 0);
        char* nl = std::strchr(res[m], '\n'); if (nl) *nl = 0;
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) std::snprintf(res[m], sizeof res[m], "child-exit-%d", st);
    }
    const bool pass = std::strcmp(res[1], "ok") == 0;
    printf("{\"peerPoolProbe\":true,\"devices\":%d,\"withoutPoolAccess\":\"%s\",\"withPoolAccess\":\"%s\",\"storePathViable\":%s}\n",
           nd, res[0], res[1], pass ? "true" : "false");
    return pass ? 0 : 1;
}
