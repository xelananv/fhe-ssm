# Runbook — building and running the recorded configuration

This is the order in which the recorded sessions were produced. The scripts under [results/dense-demo-s31/pod_tools/demo/](../results/dense-demo-s31/pod_tools/demo/) and [pod_longrun_prep/pod_scripts/](../results/dense-demo-s31/pod_longrun_prep/pod_scripts/) are the ones that ran;
they expect the tree at `/root/src` on the server box and take their paths from the environment (defaults in
[results/dense-demo-s31/pod_tools/demo/_lib.sh](../results/dense-demo-s31/pod_tools/demo/_lib.sh)). Paths that were personal on the recording machines appear as `$REPO`, `$MAC_ART`,
`$HOME/.ssh/<pod-key>`, `<pod-host>`.

## Hardware and software

| | server | client |
|---|---|---|
| machine used | 2× NVIDIA RTX PRO 6000 Blackwell 96 GB, 256 GB RAM, Ubuntu 24.04, CUDA 12.9 (a rented box) | Apple laptop, 16 GB RAM |
| what it needs | ≥ 180 GB host RAM for the server process, ≥ 150 GB of disk (store 68 GB + keys 50 GB + artifacts), two cards with working peer access (`nvidia-smi topo -p2p w` OK) | OpenFHE 1.5.1 CPU build, Python 3.9+ with numpy/torch/transformers, ≈ 60 GB of disk for the key set |
| libraries | FIDESlib at commit `fa972864ae8d624e77d3ac6ad31a1d40ef1c4d0c` + the patches in [results/dense-demo-s31/pod_longrun_prep/patches/](../results/dense-demo-s31/pod_longrun_prep/patches/); OpenFHE as vendored by FIDESlib + [results/dense-demo-s31/pod_longrun_prep/patches/openfhe/](../results/dense-demo-s31/pod_longrun_prep/patches/openfhe/); NCCL | OpenFHE 1.5.1 ([hpc_gpu_port/mac_build_client.sh](../hpc_gpu_port/mac_build_client.sh)) |

## 1. Train or obtain the model

The trainer is [ml-eval/train_fhe_native_ssm.py](../ml-eval/train_fhe_native_ssm.py) (config of the served model:
[ml-eval/artifacts/pbd430a_config.json](../ml-eval/artifacts/pbd430a_config.json)); the trainer writes `<tag>_weights.npz` + `<tag>_config.json` into `ml-eval/artifacts/`, and its `export` command
(`python ml-eval/train_fhe_native_ssm.py export --tag pbd430a`) reads them back and writes the weight bundle the server loads
(`bundle_<tag>.bin` + `.index.txt`, plus the Newton-seed sidecar); the plaintext reference reads the `.npz`. The trained weights
(`pbd430a`, 1.8 GB npz + 3.4 GB bundle) are distributed separately (see the top-level README); their sha256 values are in
[results/dense-demo-s31/pod_2xbw_20260917/pod_pull/](../results/dense-demo-s31/pod_2xbw_20260917/pod_pull/) (`*.VERIFIED`).

## 2. Build the server

```bash
# on the box, tree at /root/src
bash /root/src/results/dense-demo-s31/pod_tools/00_toolchain.sh                      # CUDA toolkit if nvcc is missing
FIDESLIB_PIN_COMMIT=fa972864ae8d624e77d3ac6ad31a1d40ef1c4d0c \
FIDESLIB_EXTRA_PATCH_DIR=/root/src/results/dense-demo-s31/pod_longrun_prep/patches \
OPENFHE_EXTRA_PATCH_DIR=/root/src/results/dense-demo-s31/pod_longrun_prep/patches/openfhe \
BUILD_X=1 bash /root/src/results/dense-demo-s31/pod_tools/demo/10_bringup.sh build       # -> /root/fhe-main-demo/build-demo/gpu_real_model{,_x}
```

`10_bringup.sh` calls [results/dense-demo-s31/pod_tools/s31_bootstrap.sh](../results/dense-demo-s31/pod_tools/s31_bootstrap.sh), which calls
[campaign/scripts/rider_build_fideslib.sh](../campaign/scripts/rider_build_fideslib.sh) (clone, pin, apply the patches, build FIDESlib with its
vendored OpenFHE) and [campaign/scripts/rider_build_demo_nolto_nccl_v2.sh](../campaign/scripts/rider_build_demo_nolto_nccl_v2.sh) (the no-LTO
FIDESlib archive and the harness; `-DFHE_SSM_EXPERIMENTAL=ON` builds `gpu_real_model_x`, the served binary). The build
writes `BUILD_IDENTITY.txt` (binary hashes, library commit, LTO section count, CUDA) — keep it beside every record.
Recorded build time on the box: 8 minutes.

The flight recorder (test keys only) is a harness patch applied on top: [results/dense-demo-s31/pod_longrun_prep/patches/harness/K3_harness_boot_record.patch](../results/dense-demo-s31/pod_longrun_prep/patches/harness/K3_harness_boot_record.patch);
the recorded long run's `gpu_real_model_x` (`df8ada73…`) is `c503feeb…` + K3.

## 3. Stage the artifacts, build the keys, build the store

```bash
bash /root/src/results/dense-demo-s31/pod_tools/demo/00_box_preflight.sh                # RAM, disk, peer access, bandwidth
bash /root/src/results/dense-demo-s31/pod_tools/demo/20_stage_artifacts.sh              # prints the upload commands; verifies sha256 on the pod
bash /root/src/results/dense-demo-s31/pod_tools/demo/30_dump_indices.sh                 # the rotation-index contract the keys must cover (contract_r17.json)
# on the CLIENT:
bash hpc_gpu_port/mac_build_client.sh                            # builds mac_fhe_client (OpenFHE 1.5.1)
mac_fhe_client keygen --indices contract_r17.json --batch 16 --out $HOME/mac_keys_r17   # secret + public + eval keys (51 GB)
bash results/dense-demo-s31/pod_tools/demo/mac_upload_keys.sh                           # uploads everything EXCEPT secret.key (chunked, resumable)
# on the box:
DEVICES=0,1 bash /root/src/results/dense-demo-s31/pod_tools/demo/40_fit_and_store.sh    # the 68 GB weight store along the canonical trajectory (~11 min)
DEVICES=0,1 bash /root/src/results/dense-demo-s31/pod_tools/demo/60_fidelity_gate.sh    # teacher-forced probe: top-1 and relErrRms against plaintext
```

Two-card boxes need `DEVICES=0,1`; `_lib.sh` then adds `--devices 0,1 --store-multi-gpu` and exports
`FIDESLIB_USE_MEMCPY_PEER=0 NCCL_P2P_DISABLE=1` (direct peer DMA returned garbage on the PCIe box; NCCL is the route).
Run the two-device selftest before anything else on a new box:
`$BIN --tag pbd430a --bundle-dir $ART --device 0 --devices 0,1 $DEMO_FLAGS --selftest --self-d 1024 --self-dff 4096`.

## 4. Serve a client-keyed session (the privacy demonstration)

```bash
# box: the stateful server, keys = the client's PUBLIC key set (it refuses a directory that holds secret.key)
DEVICES=0,1 KEYS=/root/demo/keys_mac_r17 bash /root/src/results/dense-demo-s31/pod_tools/demo/70_demo_server.sh start
# client: prepare 64 lanes, then generate tick by tick (encrypt -> upload -> wait -> download -> decrypt -> head -> next token)
bash results/dense-demo-s31/pod_tools/demo/71_mac_demo_session.sh all      # subcommands: env | preflight | fid | prepare | generate | audit | archive
```

`71_mac_demo_session.sh archive` writes the verification kit (keys, every request and reply ciphertext with its plaintext,
manifests) — [VERIFICATION_KIT.md](VERIFICATION_KIT.md). Stop the server with a `stop` file, never by killing it mid-tick.

## 5. The long free-running session (pod-side, instrument keys)

```bash
cp /root/src/results/dense-demo-s31/pod_longrun_prep/pod_scripts/longrun.env /root/demo/longrun.env      # flags, store path, prompts, decode rule
TICKS=400 BOOT_RECORD=1 CAP="2026-09-19 23:20:00" bash /root/src/results/dense-demo-s31/pod_longrun_prep/pod_scripts/run_all.sh
```

[results/dense-demo-s31/pod_longrun_prep/pod_scripts/run_all.sh](../results/dense-demo-s31/pod_longrun_prep/pod_scripts/run_all.sh): pod test keygen → store build → memory sampler → the run
([results/dense-demo-s31/pod_longrun_prep/pod_scripts/longrun_pod.sh](../results/dense-demo-s31/pod_longrun_prep/pod_scripts/longrun_pod.sh) → [spec_decode/fidelity_tick.py](../spec_decode/fidelity_tick.py)
`--free-run --decode-policy norepeat …`), stopped cleanly by a `stop` file at `CAP`. The bootstrap-arm selftest that chose
the extended range is [results/dense-demo-s31/pod_longrun_prep/pod_scripts/boot_arms_selftest.sh](../results/dense-demo-s31/pod_longrun_prep/pod_scripts/boot_arms_selftest.sh). Watch it with
[tools/token_panel.py](../tools/token_panel.py) (a local web panel that polls the box over ssh). Reports are produced
from the pulled records by the scripts in [results/dense-demo-s31/pod_longrun_20260919/](../results/dense-demo-s31/pod_longrun_20260919/) (`long_report.py`,
`recorder_events.py`, `fidelity_trend.py`, `error_headroom.py`, `compare_arms.py`, `make_plot_data.py`).

## 6. Without a GPU

- The CPU twin ([harness/](../harness/), OpenFHE) runs the same circuit at small rings for tests; ring 2^15 fits in 16 GB
  (2^16 needs ≈ 15 GB of footprint for a single layer; 2^17 is out of reach on a laptop).
- [spec_decode/dryrun_freerun_mock.sh](../spec_decode/dryrun_freerun_mock.sh) exercises the free-running driver end to end against
  [ml-eval/mock_fhe_server.py](../ml-eval/mock_fhe_server.py) (fake arithmetic, real protocol).
- [spec_decode/plain_recurrent.py](../spec_decode/plain_recurrent.py) is the exact plaintext reference; [spec_decode/plain_greedy.py](../spec_decode/plain_greedy.py)
  and [spec_decode/prompt_select_longrun.py](../spec_decode/prompt_select_longrun.py) run it on a laptop.
- The level-trace replayer ([results/theory/run_all.sh](../results/theory/run_all.sh)) counts the bootstraps of a configuration
  from its level chain, without any cryptography.

## 7. The November path: one card, the Perseus runtime

Both ports live on the fork [github.com/xelananv/perseus](https://github.com/xelananv/perseus): branch `fhe-ssm-speedrun` for the
405M model, branch `fhe-ssm-rwkv7-any` for RWKV-7. Each branch's example README (`python/perseus/examples/fhe_ssm/README.md`,
`python/perseus/examples/rwkv7/README.md`) is the reference for every flag; this section gives the order in which the recorded
runs were produced. Perseus itself is under the Business Source License 1.1: any commercial use of the contributed files as part of
Perseus requires prior written permission from Alessandro Zirilli or Davide Marincione under that license; on their own the files
carry the MIT license.

| | recorded box |
|---|---|
| 405M runs | 1× RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB), driver 595.84, CUDA 13.2, 20 host cores |
| RWKV-7 runs | 1× RTX PRO 6000 Blackwell Workstation (600 W, 97,887 MiB), CUDA 13.2, 32 host cores, 186,252 MiB of host RAM |

Every run goes through the same four steps:

1. **Capture.** The model runs one step on the CPU twin, which records the operation graph of every stage. No GPU is needed.
2. **Plan.** The planner places the refreshes on the captured graph and writes a plan directory.
3. **Keys.** `keyspec` turns the plan's needs into a key request; `keygen` plays the client role, samples a secret and writes the
   bundle (`keys.pckb`) with `client_secret.bin` beside it. Skip this step only for an in-process timing run (`KEYS=inproc`: the
   session samples and holds the secret, and the record says `insecure-inproc`).
4. **Run.** The session opens from the bundle (`KEYS=bundle`), compiles the stream against the plan, captures CUDA graphs on an
   untimed warm-up step and runs the timed steps, scoring each against the plaintext reference.

### 7.1 The 405M model (`tools/fhe_ssm/run_gpu.sh`, branch `fhe-ssm-speedrun`)

```bash
# on the box, the fork checked out at /root/perseus; DATA holds pbd430a_weights.npz alongside pbd430a_seeds.json + pbd430a_config.json (also in tools/fhe_ssm/configs/)
export DATA=/root/data PROFILE_EVERY=2
bash tools/fhe_ssm/run_gpu.sh box                 # record the box: GPU, driver, nvcc, RAM, disk, load, commit
bash tools/fhe_ssm/run_gpu.sh build               # uv venv + editable install + maturin develop --release
bash tools/fhe_ssm/run_gpu.sh verify              # _pc loads, the CPU compile check, the CPU test suites
bash tools/fhe_ssm/run_gpu.sh arms                # the configurations ("arms") and what each sets
bash tools/fhe_ssm/run_gpu.sh capture cfg-best-em && bash tools/fhe_ssm/run_gpu.sh plan cfg-best-em
# the publication run: client bundle, the three-document scenario as a lane configuration
LANE_CONFIG=tools/fhe_ssm/configs/own.passes.json bash tools/fhe_ssm/run_gpu.sh keyspec cfg-best-em
bash tools/fhe_ssm/run_gpu.sh keygen cfg-best-em
KEYS=bundle FHE_SSM_KEYS=$OUT/keys_cfg-best-em LANE_CONFIG=tools/fhe_ssm/configs/own.passes.json \
  RUN_EXTRA="--stc-cache 160 --pcmm-cf 8 --pcmm-cf-table tools/fhe_ssm/configs/t_mildall.json" bash tools/fhe_ssm/run_gpu.sh run cfg-best-em
# the 148-step free run: in-process keys, the screened prompts of the first run
bash tools/fhe_ssm/run_gpu.sh capture best && bash tools/fhe_ssm/run_gpu.sh plan best
KEYS=inproc TICKS=148 RUN_EXTRA="--stc-cache 160 --pcmm-cf 8 --pcmm-cf-table tools/fhe_ssm/configs/t_mildall.json --tokens serve_LONG_tokens.json --free-run" \
  bash tools/fhe_ssm/run_gpu.sh run best
```

- The lane configuration of the three-document scenario is [own.passes.json](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_best/gpu/cfgs_best/own.passes.json) (and bundled in `tools/fhe_ssm/configs/own.passes.json` on both Perseus branches).
- [t_mildall.json](../results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_cf/t_mildall.json) (also `tools/fhe_ssm/configs/t_mildall.json`) is the per-site correction-factor table: 7 for the `wk`, `wout` and `wv` products of every block, 8 for every
  other PCMM product.
- `--stc-cache 160` sizes the bootstrap stage cache for the whole model; below it a whole-model run fails in its warm-up.
- Each run writes `record.json` (build identity, plan, key mode, every pass with its seconds, fenced flag, errors and top-1)
  beside `run.log` and an `nvidia-smi` sample. Quote unfenced passes.

### 7.2 RWKV-7 (`tools/rwkv7/run_box.sh`, branch `fhe-ssm-rwkv7-any`)

The branch bundles the PyTorch polynomial harness and World tokenizer (`tools/rwkv7/harness/`, `tools/rwkv7/raw/`), the fitted coefficient/range JSONs for 0.1B, 0.4B, and 1.5B (`tools/rwkv7/coeffs/`), prompt/arrival files (`tools/rwkv7/{batch_prompts,longrun_prompts64,serve_arrivals}.txt`), and a SHA-256-verified checkpoint/dataset downloader (`tools/rwkv7/fetch_data.py`). When `/root/d_models` is not present, `run_box.sh` defaults cleanly to `.cache/rwkv7/`, `tools/rwkv7/coeffs/`, `tools/rwkv7/harness/`, and `.cache/rwkv7_out/`.

```bash
# the recorded environment: the best levers, client bundles, the stage cache at 160, 32 generated tokens
export LEVERS=best KEYS=bundle STC_CACHE=160 B_GENERATE=32 PF_GENERATE=32
bash tools/rwkv7/run_box.sh build
bash tools/rwkv7/run_box.sh fetch check           # downloads + SHA-256 verifies 0.1B (.pth) and runs the CPU checks
# decode: capture, plan, keys (KEYMODE = decode | pf | b), then the run
bash tools/rwkv7/run_box.sh capture && bash tools/rwkv7/run_box.sh plan
KEYMODE=decode bash tools/rwkv7/run_box.sh keyspec && KEYMODE=decode bash tools/rwkv7/run_box.sh keygen
TEXT=$'\n\nThe scientists discovered a new species of' TOKENS=64 RUN_EXTRA="--generate 32" RUN_TAG=P1 bash tools/rwkv7/run_box.sh run
# prefill, T = 32 per pass
bash tools/rwkv7/run_box.sh capture_pf && bash tools/rwkv7/run_box.sh plan_pf && bash tools/rwkv7/run_box.sh pfcheck
KEYMODE=pf bash tools/rwkv7/run_box.sh keyspec && KEYMODE=pf bash tools/rwkv7/run_box.sh keygen
PF_TEXT=... PF_TOKENS=... POOL_PF=256 bash tools/rwkv7/run_box.sh run_pf
# 32 conversations, filled layout, no refresh inside a conversation's state work (PLANOPT=b2)
export LAYOUT=filled PLANOPT=b2 B_K=32
bash tools/rwkv7/run_box.sh capture_b && bash tools/rwkv7/run_box.sh plan_b && bash tools/rwkv7/run_box.sh bcheck
KEYMODE=b bash tools/rwkv7/run_box.sh keyspec && KEYMODE=b bash tools/rwkv7/run_box.sh keygen
RWKV7_SHED_TWIN=1 RWKV7_LEAN_PREFIX=1 POOL_B=160 STC_CACHE_B=1 bash tools/rwkv7/run_box.sh batch
# continuous batching (in-process keys), 8 conversations from tools/rwkv7/serve_arrivals.txt
B_K=4 bash tools/rwkv7/run_box.sh scheck && RWKV7_SHED_TWIN=1 B_K=4 KEYS=inproc S_GENERATE=8 POOL_S=256 bash tools/rwkv7/run_box.sh s
```

- The 0.4B runs set the checkpoint and its coefficients (`PTH`, `COEFFS`), and both memory flags.
- `RWKV7_SHED_TWIN=1` cuts the CPU twin's host memory (149,520 → 26,425 MiB at 0.1B, 32 conversations);
  `RWKV7_LEAN_PREFIX=1` shares one encryption of the public prefix state (device 90,905 → 63,689 MiB). Neither changes a
  compiled program. Without the host flag a 32-conversation or prefill session needs roughly 150–180 GB of host RAM.
- `tools/rwkv7/size_pipeline.sh <checkpoint.pth> <tag>` runs any `RWKV-x070-World-*` size from calibration through the fit, the
  scans and the gate; it stops before any capture when the gate fails (the 1.5B did).
- Each run writes a `.jsonl`: a header (keys, `has_secret`, `input_encryption`, `params_hash`), one line per token or pass, and a
  summary. The headers carry no build stamp: record the checkout beside every run.

### 7.3 Without a GPU

Both ports run on Perseus's numpy fake with the runtime's level rules, and their CPU test suites run on a laptop
(`python -m pytest python/perseus/examples/fhe_ssm/tests`, `python -m pytest python/perseus/examples/rwkv7/tests`). The capture
and plan steps need no GPU either; the plans of the recorded runs were cut on the CPU.
