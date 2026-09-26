# Architecture: the model, the circuit, and the serving protocol

Everything here is read from the code in this tree: the trainer [ml-eval/train_fhe_native_ssm.py](../ml-eval/train_fhe_native_ssm.py),
the exact plaintext recurrent reference [spec_decode/plain_recurrent.py](../spec_decode/plain_recurrent.py) (a statement-by-statement mirror
of the trainer's forward pass, used as the reference in every fidelity comparison), the GPU server
[hpc_gpu_port/experimental/gpu_real_model_x.cu](../hpc_gpu_port/experimental/gpu_real_model_x.cu) and its CPU twin
[hpc_gpu_port/experimental/cpu_real_model_x.cpp](../hpc_gpu_port/experimental/cpu_real_model_x.cpp), and the client
[hpc_gpu_port/mac_fhe_client.cpp](../hpc_gpu_port/mac_fhe_client.cpp) / [ml-eval/fhe_client.py](../ml-eval/fhe_client.py).

## 1. The model (tag `pbd430a`)

| | |
|---|---|
| parameters | 404,715,520 (embedding and output head are one tied 50,277 × 1,024 matrix = 51.5M; the 24 blocks hold 353.2M) |
| width / depth | d = 1,024 channels, 24 blocks, channel-mix hidden width 4,096 |
| vocabulary | 50,277 (the GPT-NeoX tokenizer) |
| training context | 1,024 tokens, with the recurrent state carried across shard boundaries (`window_carry`) |
| config / seeds / log | [ml-eval/artifacts/pbd430a_config.json](../ml-eval/artifacts/pbd430a_config.json), [pbd430a_seeds.json](../ml-eval/artifacts/pbd430a_seeds.json), [pbd430a_train_log.jsonl](../ml-eval/artifacts/pbd430a_train_log.jsonl) |

Each block is a **time-mix** followed by a **channel-mix**, both residual (`h ← h + branch(rmsnorm(h))`):

```
time-mix    u  = rmsnorm(h)
            um = mix ⊙ u + (1 − mix) ⊙ u_prev          # token shift: one token of history per channel
            x  = W_in · um
            S  = a ⊙ S + (1 − a) ⊙ x                  # the recurrence; a = a_min + (a_max − a_min)·sigmoid(a_raw), per channel, TRAINED and FIXED at inference
            z  = c ⊙ S + d ⊙ x
            g  = z ⊙ (p0 + p1 z + p2 z²)               # learned cubic gate
            h  = h + W_out · g
channel-mix u2 = rmsnorm(h)
            k  = W_k · u2 ;  r = W_r · u2
            act  = q0 + q1 k + q2 k² + q3 k³            # learned cubic activation
            gate = r0 + r1 r + r2 r²                    # learned quadratic gate
            h  = h + W_v · (act ⊙ gate)
```

Every nonlinearity is a low-degree polynomial with learned coefficients, chosen so that the encrypted circuit computes
exactly what the model computes in float64. The one non-polynomial step is `rmsnorm`, whose inverse square root is
evaluated by a Newton iteration with a robust constant seed (`--newton-robust 0.1 12` in the recorded runs: seed 0.1 in the
mean-square frame, 12 steps). The decays are bounded by construction (a ∈ [0.9, 0.9995]); the trained values of this model
range from 0.9088 to 0.9918, i.e. time constants of 11 to 121 tokens ([results/dense-demo-s31/pod_longrun_20260919/ERROR_HEADROOM.md](../results/dense-demo-s31/pod_longrun_20260919/ERROR_HEADROOM.md)).

The model is closer to RWKV / Griffin-style recurrences than to Mamba: the decay is fixed per channel (not input-selective)
and the state width per channel is 1.

## 2. The encrypted circuit

**Parameters.** CKKS in FIDESlib (GPU) / OpenFHE (CPU twin), ring dimension N = 2^17, 65,536 slots, OpenFHE's
`HEStd_128_classic` parameter set with a uniform ternary secret; multiplicative depth 41 (`--extra-depth 12` above the block chain); the modulus chain measured at 3,319 bits of Q·P against the 2^17 ceiling of 3,523 ([docs/SECURITY.md](SECURITY.md)).

**Packing.** One ciphertext holds 64 independent sequences ("lanes") × 1,024 channels. The 64 lanes are 64 unrelated
prompts of one client; a matrix–vector product against W (1,024 × 4,096 etc.) is evaluated as diagonal products against
plaintext-encoded weight diagonals, replicated across lanes by the packing period (`Dpad` = 1,024).

**Weights.** The server holds the weights in plaintext. Every weight diagonal is encoded once, at the level and scale at
which it is consumed, into a **store** (688,128 plaintexts, 68.4 GB) built by the first pass; warm ticks perform zero host
encodes (`reqEncPtMs` 0 on every warm tick of the recorded runs). Encoding replicated lanes uses a periodic encoder
([hpc_gpu_port/periodic_encode.hpp](../hpc_gpu_port/periodic_encode.hpp)), 18× faster than the dense encode at this width on the CPU probe.

**Levels.** Operand levels are dropped exactly before the large multiplications (`--drop-wv 7 --drop-wkr 4 --drop-wout 7
--drop-win 4`: R = the levels that remain), and a plaintext cache keeps the elementwise constants resident on the device
(`--pt-cache`). The bootstrap schedule that results is 529 bootstraps per tick: 481 inside the 24 blocks and 48 for the canonical carry
(the level-trace replayer in [results/theory/](../results/theory/) counts them from the level chain alone).

**Bootstrapping.** FIDESlib's CKKS bootstrap, ≈ 150–160 ms each on the recorded box when measured synchronously; about
39 % of the recorded bootstraps refresh the two norm inputs per block (the Newton iteration), the rest the branch outputs and the
carried state. Bootstrapping the mod-raised input requires |I| ≤ K coefficient-wise; the stock table uses K = 512
(polynomial degree 88). Under this circuit the bound is exceeded about once per 8 ticks, so the recorded runs use the
library's extended table K = 768 (degree 118) with the scaling folded into CoeffsToSlots
([results/dense-demo-s31/pod_longrun_prep/reports/K768_SPLIT_VARIANT_NOTE.md](../results/dense-demo-s31/pod_longrun_prep/reports/K768_SPLIT_VARIANT_NOTE.md)).
A flight recorder ([results/dense-demo-s31/pod_longrun_prep/patches/harness/K3_harness_boot_record.patch](../results/dense-demo-s31/pod_longrun_prep/patches/harness/K3_harness_boot_record.patch),
test keys only) measures the overflow of every bootstrap input ([results/dense-demo-s31/pod_longrun_20260919/RECORDER_EVENTS.md](../results/dense-demo-s31/pod_longrun_20260919/RECORDER_EVENTS.md)).

**Two GPUs.** The store and the ciphertexts are split across two cards (`--devices 0,1`); peer copies go through NCCL
(`FIDESLIB_USE_MEMCPY_PEER=0 NCCL_P2P_DISABLE=1`) because direct peer DMA returned garbage on the PCIe box used.

## 3. The serving protocol (one tick)

```
client                                              server (holds: public key, eval keys, weights, the encrypted state)
------                                              ------
embed the 64 next tokens (plaintext, tied matrix)
encrypt the 64 × 1,024 rows → request (88.1 MB)  ──►  24 blocks under encryption, state carried from the last tick
                                                     canonical carry: every carried ciphertext returned to its
                                                     canonical level and scale (48 bootstraps) → next tick is the
                                                     identical circuit
decrypt the reply (10.5 MB): last hidden rows   ◄──  reply = the final rmsnorm output, 64 × 1,024
apply the output head in plaintext (64 × 50,277 logits)
choose the next token per lane (greedy, or the anti-repetition rule)
```

The server never sees a token, a logit or a plaintext activation; it sees one ciphertext in and one out per tick. The
secret key exists only on the client. Key generation, encryption and decryption are the client tool's
(`mac_fhe_client keygen | enc | dec`); the evaluation keys (multiplication key + 36 rotation-key parts, 51 GB at 2^17)
are uploaded once per key set.

**Stateful, constant cost.** Because the recurrence carries a fixed number of ciphertexts (2 per block: the scan state
and the shift-mix input = 48), the server's memory and the tick's cost are the same at token 1 and at token 148
([results/dense-demo-s31/pod_longrun_20260919/LONG_REPORT.md](../results/dense-demo-s31/pod_longrun_20260919/LONG_REPORT.md) sections 1–2). There is no KV cache.

**Decoding on the client.** [spec_decode/decode_policy.py](../spec_decode/decode_policy.py): greedy, or `norepeat` — a repetition
penalty of 1.3 over the last 64 fed tokens and a no-repeat-4-gram ban, with the end-of-text token banned. The rule is
deterministic and is applied to the decrypted logits and to the plaintext model's logits on the same history, so each
tick still yields 64 exact FHE-vs-plaintext comparisons ([spec_decode/fidelity_tick.py](../spec_decode/fidelity_tick.py)).

## 4. Why these choices

- **Polynomial gates, trained in** — so that fidelity is a property of CKKS noise alone and can be measured exactly; a
  post-hoc approximation of a pretrained transformer's softmax/GELU is where most of the error and the depth of prior
  systems goes.
- **A recurrence** — because attention's per-token cost and memory grow with context under encryption (a KV cache of
  ciphertexts); a first-order recurrence's do not.
- **Canonical carry** — so that a session is a loop of identical ticks: precomputed weights stay valid, the bootstrap count
  is the same every tick, and a session can go on indefinitely.
- **Lanes** — because one bootstrap at ring 2^17 costs the same for 1 lane or 64; 64 lanes are the throughput.
- **Client-side head and decoding** — the head is 51M parameters of plaintext arithmetic the client can afford, and it
  keeps the token choice (and any sampling rule) out of the circuit.
