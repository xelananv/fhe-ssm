# FIDESlib memory residue: lifetime analysis, fix and audit (2026-09-18)

Code reading and patch authoring only. **Nothing here was compiled or run**: this machine has no CUDA toolchain, so
both patches meet a compiler for the first time on the pod. Library lines are upstream CAPS-UMU/FIDESlib at
`fa972864ae8d624e77d3ac6ad31a1d40ef1c4d0c`, unpatched numbering (worktree `vendor/FIDESlib-L`); GitHub reports that
commit as the head of `main` on 2026-09-18, both defects still present. Measured numbers are transcribed from
`results/dense-demo-s31/pod_2xbw_20260917/MEMORY_RESIDUE_TRACE_20260918.md` ("the trace") or from the log it reads,
`pod_pull/s37/serve_A11/logs/server.jsonl` ("A11"). No timing is quoted.

## 0. Result

- The trace's attribution is confirmed at source level. `LimbPartition::modup_ksk_moddown_mgpu`
  (`src/CKKS/LimbPartitionMGPU.cu:588-1873`) allocates a `6 * cc.dnum * sizeof(void**)` table (144 B at dnum 3) with
  `cudaMallocAsync` on every call when no cached graph exists (`:655-662`) and never releases it (`:1871` is a
  comment). It is the **only unpaired raw allocation** on the per-operation path (section 3).
- **L1** (`patches/L1_mgpu_ks_table_free.patch`, +9 lines, one translation unit): `cudaFreeAsync(digits, s.ptr())` at
  the end of the function, only when graph capture is off. Predicted residue per request per card:
  **3,563,424 B -> 0 B**.
- **L2** (`patches/L2_stream_init_single_event.patch`, -1 line): a definite second leak, `Stream::init` creates two CUDA
  events into one handle (`src/CudaUtils.cu:299-300`). Per stream initialisation, not per operation: predicted effect
  on every per-request instrument, none.
- Host RSS (+20,731 +- 81 kB per request, trace section 1c): cause still not established. L1 is the test of the
  trace's candidate H-A(i), driver bookkeeping for the never-freed blocks.

## 1. Lifetime of `digits`

| step | line | stream | what |
|---|---|---|---|
| lookup | `:652-653` | - | `exec_old = map_exec.find({*level, moddown})`. Entries are inserted only at `:1803` / `:1830`, inside the block at `:1769`, which requires `GRAPH_CAPTURE`: read once from `FIDESLIB_USE_GRAPH_CAPTURE` (`:36-48`, default false), never written again (outside this file it appears only as a read at `RNSPoly.cpp:1176` and as the `extern` at `LimbPartition.cuh:19`). Capture off: the map stays empty, `exec_old == end()` on every call. |
| allocate | `:660` | `s` | `cudaMallocAsync(&digits, ...)`; with a cached graph `digits = exec_old->second.digits` instead (`:658`). |
| fill | `:664-672` | `s` | `cudaMemcpyAsync` from a pageable host vector. |
| read 1 | `:1270-1271` | `s` | `fusedDotKSK_2_`, special limbs. |
| read 2 | `:1642-1643` | `cc.digitStream2[0][id]` (`:1628`) | `fusedDotKSK_2_`, ordinary limbs, only if `limb_size > 0`. **Not on `s`.** |
| join | `:1685` / `:1709` | `s` | moddown true: `stream.wait(cc.digitStream2.at(0).at(id))` with `stream = c1.s` for `i == 0` and `c1` is `*this` (`:615`), under the same `limb_size > 0` guard as read 2. moddown false: `s.wait(...)`, unconditional. |
| stored | `:1807`, `:1833` | - | `.digits = digits` into `cached_graph`; graph capture only. |
| end | `:1866-1872` | - | five other partition streams wait on `s`; `// digits.free(s);`; `cudaEventDestroy(ev)`. **No host synchronisation**: the function returns with GPU work in flight. |

Of the 16 kernel launch sites in the function exactly two take `digits`; the only copy touching it is `:672`;
`transferKernel` and the NCCL calls never receive it. `Stream::wait` (`src/CudaUtils.cu:199-222`) records a fresh event
on the waited stream unless one was recorded since its last `ptr()` call; the launch at `:1642` calls `ptr()`, so the
join records *after* read 2. Two `Stream` objects sharing one pooled CUDA stream (`:203`; pool of 37 per device,
`:248-293`) skip the wait and FIFO order does the same job. One `cudaSetDevice` (`:611`), no early return, three
`goto skip_capture` (`:743`, `:752`, `:778`), all graph-capture only. Every live caller passes the context's persistent
`getKeySwitchAux()` poly as `this` (`Ciphertext.cpp:657`, `:1357`, `RNSPoly.cpp:1039`): one `s` per device, every call.

## 2. L1: the fix, and why it is safe

```cpp
	// digits.free(s);
	if (!GRAPH_CAPTURE && exec_old == map_exec.end()) {
		cudaFreeAsync(digits, s.ptr());
		CudaCheckErrorModNoSync;
	}
	cudaEventDestroy(ev);
```

- **Stream order.** `cudaFreeAsync`'s contract (CUDA 12.9 `cuda_runtime_api.h`) is that the memory is not accessed once
  the given stream reaches the free. Read 1 is earlier on `s`; read 2 is on another stream, and `s` waits on an event
  recorded after it before the free is queued. Once freed, the next call may get the same address back: its `:672` copy
  is queued on `s` behind the previous call's join, so it cannot overtake a pending read. Holds for moddown true and
  false and for `limb_size == 0` (read 2 is then not launched).
- **MEMCPY_PEER on or off** differ only in how limbs travel between devices (`:897-1087`, `:1329-1462`); neither
  touches `digits`, and `transferKernel` leaves the current device alone (`src/PeerUtils.cu:774-864`).
- **Graph capture: never freed, and that is required.** With `!MEMCPY_PEER` the cache owns the table and its address is
  baked into the captured kernels. With `MEMCPY_PEER` only device 0 caches (`:1769`): on the other devices `exec_old`
  stays `end()` while, as far as the code shows, their kernels are captured in global mode (`:745`) with the first
  call's address, which a `skip` replay reads again. **The one-line form in the trace (E2,
  `if (exec_old == map_exec.end()) ...`) would free that table**; hence the extra `!GRAPH_CAPTURE`. The header also
  requires a graph allocation for `cudaFreeAsync` during capture. Graph-capture behaviour, including its own residue
  (non-zero devices allocate per call; captured graphs are never destroyed), is unchanged. Not the demo configuration.
- **Compile safety.** No new declaration, so the `goto`s cross nothing new; `exec_old`, `map_exec`, `digits`, `s` are
  function scope (`:652-656`), `GRAPH_CAPTURE` is a namespace global of the same file (`:48`); `void***` converts to
  `void*` as at `:448` and `:584`. If the free fails the process prints
  `Cuda failure ...LimbPartitionMGPU.cu:1879` and exits **with status 0** (`CudaUtils.cuh:81-96`): grep the log.
- **No environment knob.** The change repeats what `fusedHoistRotate` already does for every rotation on the same
  partition stream (`:474`/`:584`; also `:405`/`:448`), so it was judged not risky; a knob would add more uncompiled
  code than the fix itself.

**Alternative, one table per (device, size) allocated once and rewritten per call: rejected.** A static table is shared
by every partition stream on the device and `:672` is queued before the function's waits, so two key switches in flight
on different `s` would overwrite each other's pointers, a silent wrong result; today's callers all use one poly, but
nothing enforces it. A per-`LimbPartition` member avoids that but changes `LimbPartition.cuh` (full rebuild of library
and harness) and the destructor. Free-at-end adds no shared state and touches one translation unit.

**History.** Four squashed release commits. `git log -S"digits.free"` finds only the 2.0.0 release (`2614633`,
2026-01-29): the line is already commented out and `digits` already a raw `void***` kept in `cached_graph`. Leftover
comments give the order: `VectorGPU<void**>` with `.free(s)`, then `std::make_shared<VectorGPU<void**>>` (`:661`) so
the graph cache could own it, then a raw pointer. The siblings got `cudaFreeAsync` in place of `.free(s)`; here the free
could not coexist with the cache and was dropped for every path. The commit messages give no reason, and upstream's
tracker returns nothing for `leak` or `cudaFreeAsync` (GitHub search API, 2026-09-18): no use-after-free report found.

## 3. Audit of the per-operation path

| site | executed per | size | released? |
|---|---|---|---|
| `LimbPartitionMGPU.cu:660` key-switch table | ciphertext multiplication (`Ciphertext.cpp:661`; the call at `:641` is in a branch disabled at `:496`; `square` forwards, `:724`/`:740`), `keySwitch` `:1360`, `dotProduct` `:1611`; per device, **one device included** (`:652-674` is taken for every device count) | 144 B | **no -> L1** |
| `:684`, `:1890` events | key switch; `modupMGPU` | 1 event | yes, `:1872`, `:2481` |
| `:405` `dotKSKfusedMGPU`; `:474` `fusedHoistRotate` | fused dot product; rotation, conjugation, hoisted batch | 144 B; (n*6*dnum + 6*dnum + 5n) x 8 = 328 B (n=1), 5,848 B (n=31) | yes, `:448`, `:584` |
| `rescaleMGPU`, `doubleRescaleMGPU` (`:51-394`); `modupMGPU`, `moddownMGPU` | rescale; rotation | no allocation | - |
| `LimbPartition.cu:2155`, `:2178`, `:2197`, `:2348`, `:2357`; `LimbPartitionBatch.cu:70`, `:158`, `:205`, `:259`, `:410`, `:588` | scalar ops, weighted sum, batched ops | small tables | yes: `:2171`, `:2190`, `:2208`, `:2363-2364`; `:119`, `:165`, `:211`, `:266`, `:490`, `:614` |
| `VectorGPU.cu:70`/`:40` -> `GPUmalloc`/`GPUfree` (`CudaUtils.cu:364-506`) | limb, pointer table | power-of-two blocks, slabs `:392` | to the library free list; slabs never return to CUDA (bounded, trace section 2) |
| **`CudaUtils.cu:299-300` `Stream::init`** | `LimbPartition` with its own stream (`LimbPartition.cu:70-81`), context streams (`Context.cu:750-808`; top-limb and gather streams are re-initialised inside the `2 * GPUid.size()` loop), limb records, allocator stream (`CudaUtils.cu:385`, `:478`) | **1 event orphaned** | **no -> L2** |
| `Ciphertext` temporaries (`Ciphertext.cpp:73-93`; `Bootstrap.cu:33,118,177,283`; `ApproxModEval.cu`, `LinearTransform.cu`; `api/CryptoContext.cpp:283,1928`) | every operation, bootstrap, API load / copy | polys from the context pool (`Context.cu:904-917`) | returned; no `LimbPartition` is built once the pool is warm |
| `Plaintext` (`Plaintext.cu:53,59`), `KeySwitchingKey` (`KeySwitchingKey.cu:39`) | device plaintext; key load | default stream, no event (`CudaUtils.cu:308-312`) | - |
| monomial cache (`Ciphertext.cpp:1631-1683`); `RNSPoly.cpp:1118-1153` | fully packed bootstrap; - | rebuilt only on a level change; commented out | bounded; - |
| `GPUfree(ptr, id, 0, ...)` (`LimbPartition.cu:138,142,145,163,817,1038`) | destruction of context-lifetime buffers | a large buffer re-enters the 1 KiB free list | latent, not per request, not patched |

`cudaEventCreate(event, flags)` is a C++ overload forwarding to `cudaEventCreateWithFlags` (CUDA 12.9 `cuda_runtime.h`;
`pod_pull/build/BUILD_IDENTITY.txt:8` gives nvcc 12.9), so `:300` does create a second event. Rotations, rescales and
bootstraps retain nothing, so the rotation count enters no prediction; it cannot be derived from the library alone (it
depends on the bootstrap precomputation's baby/giant split; A11 ran `--level-budget 3 3`). Device slope: 144 B x 24,746
key switches = 3,563,424 B, the measured increment, and nothing else. Host: 2 x 24,746 = 49,492 never-freed blocks per
request; were they the whole cause, each would cost 20,731 x 1,024 / 49,492 = 429 B of driver memory, a fit, not a
measurement. NCCL and driver internals were not audited.

## 4. Predictions (A11 configuration: 480 bootstraps, M = 1,706)

| per served request per card | before (A11) | with L1 | with L2 |
|---|---|---|---|
| pool used (`poolReservedGB - poolSlackGB`) | +3,563,424 B = 144 x (48 x 480 + 1,706) | **0 B** (trace bound on anything else: < 14 B) | no change |
| `poolSlackGB`, non-step requests | -0.0033187 (`A11:45` 0.0389521 -> `A11:56` 0.0356334) | constant to the last digit (one table = 1.3 units of it) | no change |
| `poolReservedGB`, card `usedGB` | +0.03125 on 16 of 43 requests | constant from `serve.done.1` | no change |
| `poolTransientGB` | 5.31226e-06 (5,704 B = 5,848 - 144) | 5.44637e-06 (5,848 B); secondary, assumes used bytes are counted at call time | no change |
| host VmRSS | +20,731 +- 81 kB | **unknown**: near 0 if H-A(i) holds, unchanged if not | none expected (0 `Stream::init` per warm request, from code, not measured) |

Any other arm: before = 144 x (48 x reqBoots + M), after = 0. By the trace's count (section 3b) M = 432 + 98 + 3 x 49 x n
under `--newton-robust a n`: n = 12, the long-run setting of `POD_RUN_PLAN.md`, gives M = 2,294 (derived, not measured).

## 5. Verification on the pod (about 14 minutes of serving)

1. Build. Fresh pod: `campaign/scripts/rider_build_fideslib.sh` with
   `FIDESLIB_PIN_COMMIT=fa972864ae8d624e77d3ac6ad31a1d40ef1c4d0c` and
   `FIDESLIB_EXTRA_PATCH_DIR=<repo>/results/dense-demo-s31/pod_longrun_prep/patches` prints
   `EXTRA-PATCH applied: L1_mgpu_ks_table_free.patch sha256 d64b87c54b5c6caf` and
   `... L2_stream_init_single_event.patch sha256 5e2f996b6346e458`. The hook applies **every** `*.patch` in that
   directory, L2 included: move one out to skip it. L1 and L2 each pass `git apply --check` alone on the pristine
   checkout, and together with `K1_boot_uniform_ext_env.patch` and `K2_boot_preraise_hook.patch` (different files:
   `RawCiphertext.cu`, `Bootstrap.cu`, `Bootstrap.cuh`). Existing tree: `git -C $WORK/FIDESlib apply --check <patch>`, `apply`, then
   `cmake --build $WORK/FIDESlib/build --target install -j48` and
   `cmake --build $WORK/build-demo -j48 --target gpu_real_model_x`. The harness links the static `fideslib.a`: confirm
   that `binary_x_sha256` in `BUILD_IDENTITY.txt` changed.
2. Start the server (`POD_RUN_PLAN.md` section 4: the first ticks of the long run are the check; the A11 command line
   works equally) with `--vram-trace`, `FIDESLIB_USE_MEMCPY_PEER=0 NCCL_P2P_DISABLE=1`, `FIDESLIB_USE_GRAPH_CAPTURE`
   unset, and `mem_sampler.sh`. Serve 4 requests (A11: 9,332,755 ms / 45 = 207 s each).
3. `grep -a '"vramTrace":"serve.done' server.jsonl`. **Pass:** `poolReservedGB`, `poolUsedGB`, `poolSlackGB` identical
   on `serve.done.1`, `.2`, `.3`, both devices (A11 flags: verdict at `.2`, about 10 min, `.3` confirms). With the
   plaintext cache on, compare from the first request without cache misses (`serve.done.2` in the plan; trace section
   3c: 1,140 -> 1,144 entries, then 0 misses). **Fail:** slack falling by 144 x (48 x reqBoots + M) bytes per request
   (0.0033187 GiB under A11 flags) = the patched library is not in the binary (relink); any other constant step is a
   different residue, to be recorded as such.
4. Correctness (crosses binaries): `"failed":false`, `"reqBoots":480` on every `"serve":"served"` line; `argmax` per
   lane in `t1..t3.dec.jsonl` equal to A11's for the same prompts (`maxLogit` only to CKKS noise);
   `grep -a "Cuda failure"` empty.
5. Host: `mem_sampler.log` VmRSS against 20,731 kB per request. Four requests give a first reading; the firm slope
   comes from the long run at no extra cost. Nothing CPU-heavy beside it.

## 6. Rollback

`git -C $WORK/FIDESlib apply -R <patch>`, then the two `cmake --build` commands. L1 and L2 touch different files and
revert independently. `FIDESLIB_USE_GRAPH_CAPTURE=1` is **not** a rollback: it switches the whole key-switch path.
Unfixed, the reservation grows 11.9 MiB per request per card (trace section 0), about 2.9 GiB over 250 requests against
29.69 GiB free: a long run completes without the fix, it just does not read flat.

## 7. Not determined

- Whether either patch compiles; whether the pod tree equals upstream at these lines (`git apply --check` answers).
- The cause of the host slope; the driver's host cost per live block; the host cost of one orphaned event.
- `Stream::init` calls per warm request (predicted 0) and rotations per request.
- Whether CUDA counts used bytes at call time or at stream completion (the 5,848 B fingerprint depends on it).
- The one-device path and both graph-capture paths (unchanged by design, untested). Any timing effect: unmeasured, and
  not comparable across binaries.
