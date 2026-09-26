# fhe-ssm

**A language model that generates text under fully homomorphic encryption — for hours, for 64 conversations at once, at a cost
and a memory footprint that do not grow with the conversation.**

**Project page, with the run's figures: [xelananv.github.io/fhe-ssm](https://xelananv.github.io/fhe-ssm/). Demo video, 3½ minutes: [youtu.be/RypXQfcqSno](https://youtu.be/RypXQfcqSno).**

A 405M-parameter recurrent language model, trained from scratch so that its own arithmetic is the encrypted circuit, served
under CKKS at 128-bit classical parameters on two GPUs. The server holds the conversation state encrypted, refreshes it
every step and never sees a token, a logit or an activation; the secret key never leaves the client. Every number on this page
is transcribed from a record in this repository, and the record is linked beside it.

## In numbers

| | | record |
|---|---|---|
| **2.7 s per token per conversation** | 64 conversations advance one token every **175.6 s** (layer loop; 188 s request-to-reply), the same at step 1 and at step 148 | [LONG_REPORT.md §1](results/dense-demo-s31/pod_longrun_20260919/LONG_REPORT.md) |
| **8 h 04 min, 148 consecutive steps, 9,472 tokens (7,481 generated), 0 failures** | once past its prompt, every step fed each conversation the token the encrypted session itself had just produced — the longest encrypted generation chain reported for any language model we know of | [LONG_REPORT.md §3](results/dense-demo-s31/pod_longrun_20260919/LONG_REPORT.md) |
| **Constant memory in context length** | the carried state of all 64 conversations is 48 ciphertexts (two per block, each holding the 64 lanes), no KV cache; device pool identical after every one of 147 requests on both cards, host memory unchanged over the last 127 | [LONG_REPORT.md §2](results/dense-demo-s31/pod_longrun_20260919/LONG_REPORT.md) |
| **99.7 % top-1 fidelity to the model's own plaintext** | 9,445 / 9,472 comparisons; every one of the 27 differences is a near-tie (the plaintext model's top two within 0.03 logits). Measured on 64 prompts pre-screened in plaintext against loops, norm stress and near-ties, so the number is conditional on that selection ([how the prompts were chosen](docs/RESULTS.md)) | [FIDELITY_TREND.md](results/dense-demo-s31/pod_longrun_20260919/FIDELITY_TREND.md) |
| **529 bootstraps per step, identical every step** | the canonical carry makes every step the same circuit: precomputed weights (a 68 GB store built once), zero weight encodes after warm-up | [server.jsonl](results/dense-demo-s31/pod_longrun_20260919/pod_pull/s37/serve_LONG/logs/server.jsonl) |
| **A recorded client-keyed session, replayable** | 64 prompts, up to 8 tokens each (492 in all), keys generated on a laptop, every ciphertext that crossed the wire kept; re-decrypting the kit reproduces all 492 tokens | [docs/VERIFICATION_KIT.md](docs/VERIFICATION_KIT.md) |
| **Bootstrap overflow, measured** | 78,340 bootstraps recorded on the GPU, 71,236 of them with a non-trivial input: 17 exceeded the stock range (17.97 expected), max 552, every such step decoded; the extended range costs 2 % and cuts the logit error 1.57× | [RECORDER_EVENTS.md](results/dense-demo-s31/pod_longrun_20260919/RECORDER_EVENTS.md), [CTRL512_VS_LONG.md](results/dense-demo-s31/pod_longrun_20260919/CTRL512_VS_LONG.md) |
| **34.9 Wh per 64-token step** | 656.7 W for both cards, ≈ 1,960 J per token (board power) | [gpu_power.csv](results/dense-demo-s31/pod_longrun_20260919/pod_pull/gpu_power.csv) |

The long session ran on keys generated on the server (a measurement run: its bootstrap flight recorder reads the secret key; the keys were deleted at the end); the client-keyed session of the kit is the privacy demonstration. Hardware: 2× NVIDIA RTX PRO 6000 Blackwell 96 GB. Parameters: ring 2^17, OpenFHE `HEStd_128_classic`, uniform ternary secret,
the modulus chain 204 bits under the 128-bit ceiling ([docs/SECURITY.md](docs/SECURITY.md)). Model: 404,715,520 parameters,
WikiText-103 token perplexity 28.55, LAMBADA 9 % ([docs/RESULTS.md](docs/RESULTS.md) §4) — fidelity says the circuit computes what
the model computes; it does not make the model good.

Per generated token per GPU-second this is 5.5–5.9 GPU-s, against 24–144 for Sylph's single decode step (Llama-3-8B, one to
eight cards) and 96.6 for Cachemir's (Llama-3-8B, one A100) — and that at ring 2^17, with two cards, for a model 20× smaller;
per parameter the 8B systems' single step is cheaper. None of them reports a chain of generated tokens or a fidelity measured
under encryption. The full comparison, both normalisations side by side: [docs/COMPARISON.md](docs/COMPARISON.md).

## How it works

1. **The model is the circuit.** Every nonlinearity is a low-degree polynomial with trained coefficients (a cubic gate, a cubic
   activation, a quadratic gate) and the norm is a Newton inverse-square-root, so the encrypted evaluation computes the model's
   own float64 arithmetic. No post-hoc approximation of softmax or GELU; the only error is CKKS noise, and it is measured every
   step against the plaintext model on the same tokens.
2. **A recurrence, not attention.** Each of the 24 blocks carries one state vector per channel with a fixed, trained decay
   (S ← a·S + (1−a)·x) and one token of shift-mix history: 2 vectors of 1,024 per block, whatever the context length.
3. **Stateful serving with a canonical carry.** The state stays encrypted on the server and is returned to the same level and
   scale at the end of every step. Every step is therefore the identical circuit — the same bootstraps, the same precomputed
   weight plaintexts — so nothing in a step's cost or memory depends on how many steps have been taken (148 measured).
4. **64 conversations in one ciphertext.** 64 lanes × 1,024 channels fill the 65,536 slots of a ring-2^17 ciphertext; one
   bootstrap refreshes all 64.
5. **The client decides the tokens.** It decrypts the last hidden state, applies the output head in plaintext, picks the next
   token (greedy, or a deterministic anti-repetition rule), re-encrypts it. One 88 MB ciphertext up, one 10.5 MB ciphertext down,
   per step.
6. **Two library fixes on the way.** A device-memory leak in FIDESlib's multi-GPU key switching (+3.56 MB per request per card,
   fixed: [L1](results/dense-demo-s31/pod_longrun_prep/patches/L1_mgpu_ks_table_free.patch)), and the extended bootstrap range whose
   scalar the library rounds to an integer — folding the factor into CoeffsToSlots removes that rounding error (1.78× the stock table's error instead of 2.43× on the CPU cell;
   [K768 note](results/dense-demo-s31/pod_longrun_prep/reports/K768_SPLIT_VARIANT_NOTE.md)). An OpenFHE sparse-bootstrapping
   bug was reported upstream with a reproducer ([#1264](https://github.com/openfheorg/openfhe-development/issues/1264)).

The full picture — the block equations, the packing, the level schedule, the protocol — is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## What it is not

- Not fast for one user: a step is three minutes whatever the number of conversations, so this is throughput, not latency.
- Not a large or clever model: 405M parameters, about a hundred dollars of GPU rental for its training; fluent, often wrong.
- Not a privacy demo for the weights: the server holds the model in plaintext. What is protected is the client's text.
- Measured at one model size, one ring, one kind of machine. Limits and open questions: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## The model, the keys, the ciphertexts

- Weights, bundle, config, training log: **[huggingface.co/xelananv/fhe-ssm-pbd430a](https://huggingface.co/xelananv/fhe-ssm-pbd430a)**
  (the model card carries the training records and the quality numbers).
- The verification kit — the secret and public key of the recorded session, every request and reply ciphertext, the rows and
  the tokens — and the 51 GB of evaluation keys of that key set: **[huggingface.co/datasets/xelananv/fhe-ssm-verification-kit](https://huggingface.co/datasets/xelananv/fhe-ssm-verification-kit)**.
  What it lets anyone check, and what it gave when checked on a laptop: [docs/VERIFICATION_KIT.md](docs/VERIFICATION_KIT.md).

## Layout

The directory names are the working tree's own, so that every path inside the scripts is valid as published.

| directory | contents |
|---|---|
| [ml-eval/](ml-eval/) | the trainer (`train_fhe_native_ssm.py`), evaluation, the client session driver (`fhe_client.py`), an Ollama-compatible chat server for the plaintext model (`serve_ollama.py`), the model's config, seeds and training log |
| [hpc_gpu_port/](hpc_gpu_port/) | the GPU server (`gpu_real_model.cu`; the served binary is [experimental/gpu_real_model_x.cu](hpc_gpu_port/experimental/gpu_real_model_x.cu)), the client crypto tool (`mac_fhe_client.cpp`: keygen, encrypt, decrypt), CMake |
| [harness/](harness/) | the OpenFHE CPU twin of the circuit and the probes (bootstrap overflow, level chains, security parameters) |
| [spec_decode/](spec_decode/) | the pod-side free-running driver, the decoding rule, the exact plaintext reference (`plain_recurrent.py`), prompt selection |
| [campaign/scripts/](campaign/scripts/), [provisioning/](provisioning/), [results/dense-demo-s31/pod_tools/](results/dense-demo-s31/pod_tools/) | build scripts (FIDESlib at the pinned commit + patches), box provisioning, the scripts that ran the recorded sessions |
| [results/dense-demo-s31/pod_longrun_prep/](results/dense-demo-s31/pod_longrun_prep/) | the patches to FIDESlib and OpenFHE, the long-run scripts, the preparation reports |
| [results/dense-demo-s31/pod_longrun_20260919/](results/dense-demo-s31/pod_longrun_20260919/) | the long session: reports, plotting tables, prompt selection, raw records |
| [results/dense-demo-s31/pod_2xbw_20260917/](results/dense-demo-s31/pod_2xbw_20260917/), [sessions/](results/dense-demo-s31/sessions/) | the recorded client-keyed session, the earlier arms, the CPU overflow probes, the kit replay |
| [results/dense-demo-s31/pod/state/runs/](results/dense-demo-s31/pod/state/runs/) | the training and anneal runs, the quality anchors |
| [results/theory/](results/theory/), [estimator/](estimator/) | the level-trace replayer; the lattice-estimator runs behind the parameters |
| [docs/](docs/) | [ARCHITECTURE](docs/ARCHITECTURE.md) · [RESULTS](docs/RESULTS.md) · [COMPARISON](docs/COMPARISON.md) · [SECURITY](docs/SECURITY.md) · [RUNBOOK](docs/RUNBOOK.md) · [LIMITATIONS](docs/LIMITATIONS.md) · [VERIFICATION_KIT](docs/VERIFICATION_KIT.md) |

Comments, docstrings and help strings in the code cite working notes, earlier campaigns and internal tooling that are not part of this
release; they are listed in [docs/NOT_INCLUDED.md](docs/NOT_INCLUDED.md). Nothing needed to train, export, build, serve, replay or check the
released artifacts is among them: every released script's local imports resolve inside this tree (checked when the release is assembled), and
[docs/RUNBOOK.md](docs/RUNBOOK.md) names the script of every step.

## Running it

Two machines: a box with two 96 GB cards (server) and a laptop with the secret key (client). FIDESlib at the pinned commit
with the patches above, ≈ 180 GB of host RAM, a 68 GB weight store; on the client OpenFHE 1.5.1 and a one-time 51 GB
evaluation-key upload. [docs/RUNBOOK.md](docs/RUNBOOK.md) walks through it in the order it was done; every script it names is
in this tree. Without a GPU: the plaintext model runs on a laptop (`ml-eval/serve_ollama.py`), the CPU twin runs the circuit
at small rings, and the free-running driver has a mock-server dry run.

## License and contact

MIT ([LICENSE](LICENSE)). Alexander Ananyev, independent researcher — xelananv@protonmail.com ·
[xelananv.github.io/fhe-ssm](https://xelananv.github.io/fhe-ssm/) · [x.com/xelananv](https://x.com/xelananv) ·
[github.com/xelananv](https://github.com/xelananv).
Cite with [CITATION.cff](CITATION.cff).
