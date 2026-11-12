# Architecture: the model, the circuit, and the serving protocol

Everything here is read from the code in this tree: the trainer [ml-eval/train_fhe_native_ssm.py](../ml-eval/train_fhe_native_ssm.py),
the exact plaintext recurrent reference [spec_decode/plain_recurrent.py](../spec_decode/plain_recurrent.py) (a statement-by-statement mirror
of the trainer's forward pass, used as the reference in every fidelity comparison), the GPU server
[hpc_gpu_port/experimental/gpu_real_model_x.cu](../hpc_gpu_port/experimental/gpu_real_model_x.cu) and its CPU twin
[hpc_gpu_port/experimental/cpu_real_model_x.cpp](../hpc_gpu_port/experimental/cpu_real_model_x.cpp), and the client
[hpc_gpu_port/mac_fhe_client.cpp](../hpc_gpu_port/mac_fhe_client.cpp) / [ml-eval/fhe_client.py](../ml-eval/fhe_client.py). Sections 5 and 6 are read from the
two ports on the fork [github.com/xelananv/perseus](https://github.com/xelananv/perseus) (branches `fhe-ssm-speedrun` and `fhe-ssm-rwkv7-any`).

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

## 2. The encrypted circuit of the first run (FIDESlib, ring 2^17)

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

## 3. The serving protocol of the first run (one tick)

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

## 5. The 405M model on the Perseus runtime (November 2026)

The port is `python/perseus/examples/fhe_ssm` on the branch `fhe-ssm-speedrun` of [github.com/xelananv/perseus](https://github.com/xelananv/perseus),
a fork of Perseus ([gladia-research-group/perseus](https://github.com/gladia-research-group/perseus), [arXiv 2611.04488](https://arxiv.org/abs/2611.04488)).
The model is written on Perseus's leaf-primitive operations; the box procedure is `tools/fhe_ssm/run_gpu.sh`, and the changes
to upstream files are listed in `tools/fhe_ssm/UPSTREAM_CHANGES.md` on that branch. The weights, seeds and config are the same
files as in section 1.

**Parameters.** Ring 2^16 (32,768 complex slots), the `n32` chain: 54 + 9 primes with composite scaling of
degree 2, Q·P 1,705.34 bits ([SECURITY.md](SECURITY.md)).

**Packing: the filled layout.** A complex slot carries two real values, one on each axis. 32,768 complex slots therefore hold
65,536 reals = 64 conversations × 1,024 channels, and one bootstrap refreshes all 64 conversations. The first run needed ring
2^17 to reach the same 64 lanes with one real per slot.

**Capture, plan, compile, replay.** A step is not scheduled by hand:

1. **Capture.** The model runs one step on a CPU twin (a numpy stand-in with the runtime's level rules) that records the
   operation graph of every stage (24 blocks and a tail).
2. **Plan.** A planner places the refreshes (bootstraps) on that graph: it chooses where each ciphertext is refreshed so that
   every product has the levels it needs, minimising their number (`make_plan.sh`, one pass over every stage).
3. **Compile.** `Program.compile` compiles the recorded stream against the plan; the stream is checked to be static across
   probe steps (no dynamic step, the same plaintexts).
4. **Replay.** The session uploads the weights, stages every plaintext once, captures the compiled programs as CUDA graphs on
   an untimed warm-up step, and replays them every step. The carried state is two registers per block (`cache.s`,
   `cache.n_prev`), 48 in all, returned to their canonical level each step.

**Linears.** Two methods, a flag each:
- *Diagonals*: the input is multiplied by every rotated diagonal of the weight matrix (baby-step giant-step, 32 × 32, 1,024
  products per 1,024-wide matrix).
- *PCMM* (plaintext–ciphertext matrix multiplication): the weights are packed into plaintext matrices at 24-bit precision
  (`--pcmm-wbits 24`) and the product finishes with a correction-factor rescale (`--pcmm-cf 8`). On one build it runs a pass
  2.27× faster than the diagonals, at a relErrRms of 0.0031–0.0053 against 1–2e-4 ([RESULTS.md](RESULTS.md) §5).

**The norm.** The RMSNorm's Newton iteration runs at depth 2 per site (`--newton-form depth2 --newton-sites per-site`), and the norm statistics are packed into
the two halves of one ciphertext (`--pack-stats halves`).

**A pass in any lane configuration.** With `--pass-kernels on`, every lane-slot of a pass is either a conversation's next step
or a position of a document being read, chosen per pass by plaintext masks that are re-staged on the host. One recorded stream
serves every configuration: reading 64 positions of one document, 64 conversations generating, or a mix.

**Where a pass goes.** In the fastest configuration a fenced pass is 25 stages: 24 blocks at 0.358 s mean and a tail of
0.067 s, 8.667 s in all.

**Keys.** `--keyspec` derives the key request from the plan's needs (rotation steps, encapsulation, PCMM keys); `--keygen`
plays the client role: it samples a secret and writes the public and evaluation keys as a bundle, with `client_secret.bin`
beside it. A session opened from the bundle refuses a secret (`has_secret: false`) and decrypts nothing itself; `--keys inproc`
lets the session sample and hold the secret, for timing runs only.

## 6. RWKV-7 on the Perseus runtime

The port is `python/perseus/examples/rwkv7` on the branch `fhe-ssm-rwkv7-any` of the same fork; the box procedure is
`tools/rwkv7/run_box.sh`, and `tools/rwkv7/size_pipeline.sh` takes any `RWKV-x070-World-*` checkpoint through calibration,
fitting, the scans and the gate to the encrypted runs. The checkpoints are the public `RWKV-x070-World-0.1B-v2.8` (12 blocks,
width 768) and `RWKV-x070-World-0.4B-v2.9` (24 blocks, width 1,024).

**The stage interface** is one ciphertext `z = h + i·v_first`: the residual stream on the real axis and layer 0's value (which
every later block's value residual reads) on the imaginary axis. A block reads `2h = z + conj z` and `2·v_first = i(conj z − z)`;
a conjugation and a multiplication by `i` cost no level.

**The layout.** Channel `c` of lane `l` sits at slot `c·t + l`, with 1,024 channels (768 padded to 1,024 at 0.1B) and t = 32
lanes. A 64 × 64 head state lives in two ciphertexts. A vector of width ≤ 1,024 is one ciphertext at both sizes, so the cost
grows with the block count: 2.0× from 12 to 24 blocks at the same lever settings (two points, not a law).

**The polynomial sites.** Every non-polynomial operation of RWKV-7 is replaced:

| site | what it computes | how |
|---|---|---|
| `att.w_tanh`, `att.decay` | the input-dependent decay `exp(−e^−0.5 σ(z))` after a tanh | Chebyshev series, the decay as one polynomial |
| `att.a_sigmoid`, `att.g_sigmoid`, `att.v_sigmoid` | the in-context learning rate, the output gate, the value residual's gate | Chebyshev series |
| `ffn.relu2` | the channel mix's squared ReLU | Chebyshev series |
| `ln1.rsqrt`, `ln2.rsqrt`, `att.gn_rsqrt`, `att.kk_rsqrt` | the inverse square roots of the LayerNorms, the GroupNorm and the key normalisation | Newton iteration from a constant seed (two levels per step) |

A Chebyshev series of degree d costs about 2√d ciphertext products and ⌈log₂ d⌉ + 1 levels (baby-step giant-step).

**The intervals.** A ciphertext cannot be clamped to an interval, and a high-degree Chebyshev series explodes just outside its
interval (as `cosh(d·acosh|t|)`, about 10^13 at |t| = 1.03 and d = 127). So each site's interval is its observed range on
calibration text widened by 50 % (ReLU²: 20 %), at degree ≤ 255, and the inverse square roots use a constant seed that
converges for every input below three times the widened top. Position 0 of a text leaves several intervals; every prompt
therefore starts with the public separator `"\n\n"`, whose state is computed in plaintext and encrypted, and the encrypted
decode starts at position 1.

**The gate.** Before any encrypted run, the polynomial model is evaluated in plaintext, unclamped, over the whole
WikiText-103 test and LAMBADA: perplexity within 0.5 % of the exact model, LAMBADA within 0.5 points, top-1 agreement with
exact ≥ 98 %, and no non-finite value anywhere. 0.1B and 0.4B pass; 1.5B fails ([RESULTS.md](RESULTS.md) §7.5).

**The block**, per token: LayerNorm (rotate-and-sum over the channels, Newton); PCMM on the pair `[x, x_prev]` with the token
shift folded into the matrices; the decay, gates and learning rate through their sites; the state update
`S ← S⊙w + (−S⊙κ̂)⊗(κ̂⊙a) + v⊗k̃`; the readout `y = S r`; GroupNorm; the output projection; the channel mix (LayerNorm, PCMM,
ReLU², PCMM). The carried ciphertexts (the two state halves and the two token-shift inputs, four per block) are refreshed at
block end every token, so every token runs the same op stream.

**Batched decode.** Conversation c's current token sits on lane c, so every token-side operation serves K ≤ 32 conversations
for the price of one. Each conversation keeps its own state ciphertexts per block; the token-shift carries are lane-form, one
ciphertext for all K. Because the token-side ciphertexts are shared and a refresh mixes all slots, one conversation's
out-of-interval value reaches every lane ([RESULTS.md](RESULTS.md) §7.5).

**The filled state.** A head state's two value groups share one ciphertext, `S = S_0 + i·S_1`. Every operand on the update side
is indexed by the key dimension, so it is real and the same for both groups: one product serves both. The readout splits the
two with one conjugation. Half the state ciphertexts, carry refreshes and state products; three carried ciphertexts per block
instead of four.

**Prefill.** A pass puts token l of a 32-token chunk on lane l instead of copying one token over the lanes. The linears,
polynomial sites and norms run unchanged (their reductions stay inside a lane); the token shift becomes a rotation by one lane,
and the state update runs as a sequential scan inside the pass on the decode step's own state layout. A pass therefore ends
with exactly the state a decode step carries, and decoding continues on the same registers.

**The serve loop.** One session holds four program sets: decode, prefill, the K-lane batched pass and a join. An arrival is
read alone while the batch waits, then joins a free lane: its state registers are handed over (no server arithmetic), and the
join program writes its token-shift carries into lane c with plaintext masks, one refresh per carry. A finished conversation
leaves: its lane returns to an encryption of the public prefix state.

**The client's part.** The server's last step is the last block's output, masked to the live channels and refreshed once. The
client decrypts it, applies `ln_out` (a LayerNorm) and the head in plaintext and picks the token.
