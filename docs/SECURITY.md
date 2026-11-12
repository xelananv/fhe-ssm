# Security parameters and threat model

## The November runs (Perseus runtime, ring 2^16)

The 405M publication run of 2026-11-02 and every bundle-keyed RWKV-7 run carry the same parameter hash,
`params_hash d364d0968ad178ca`: one parameter set, one chain (`n32`, 54 + 9 primes). Priced with the lattice estimator `3e48ef4`
under its default cost model, classical, IND-CPA ([battery.txt](../results/estimator-speedrun-port-20261103/battery.txt),
[battery.jsonl](../results/estimator-speedrun-port-20261103/battery.jsonl)):

| key | secret | modulus an attacker can use (log2) | weakest attack | bits |
|---|---|---|---|---|
| main key: rotations, relinearisation, conjugation, sparse → dense, public key | uniform ternary, ring 2^16 | Q·P = 1,705.34 (the HE-standard bound at this ring is 1,747) | uSVP / BDD, β 333 | **131.28** |
| dense → sparse encapsulation key | Hamming weight 32 | 307.64: the key exists only at the bottom of the chain | primal MITM hybrid | **142.97** |
| PCMM ring-switch key | uniform ternary, ring 2^12 | 86.98 | BDD, β 450 | **159.71** |

**Why these instances.** A switching key is `b = −a·s_new + e + P·g·s_old (mod Q·P)`; reduced modulo the primes where the
gadget term vanishes it is a plain RLWE sample under `s_new`, and recovering `s_new` gives `s_old`. So the key set is as strong
as its weakest new secret over the largest modulus an attacker can use: here the main key, at 2^131.28. A sparse secret would
be the weak point at a large modulus; the weight-32 key is only ever used at the bottom shape (307.64 bits), where it costs
2^142.97.

**Not covered:** quantum cost models; attacks that use decrypted values (the server never sees a decryption); the
circular-security assumption every relinearisation, rotation and PCMM key makes. A run whose session held the secret (below)
is a timing and fidelity measurement, not a deployment, whatever its parameters.

### Where the secret was, run by run

Read from each record's header: `keys` / `has_secret` / `input_encryption` in the RWKV-7 `.jsonl` files, `keys.mode` and
`setup.has_secret` in the 405M `record.json` files.

| label | what it means | runs |
|---|---|---|
| secret on another machine | the client's secret never on the GPU machine; the client encrypts and decrypts over the network | the 2026-09-18 session (18 ticks) only |
| no server secret, client role co-located, client encrypts and decrypts | the server session is verified secret-free (`has_secret: false`); a client object in the same process on the GPU machine encrypts every input and decrypts every reply (`input_encryption: client`) | RWKV-7 in `f2_01b/`: decode P1–P3, prefill L1/L5/L9/L25, every 32-conversation batch, the free run; in `f2_04b/`: prefill L1/L5, the 32-conversation batch |
| no server secret, client role co-located, server encrypts inputs | the server session is verified secret-free; it encrypts block 0's inputs with the public key from plaintext rows held in the same process, and the client secret in that process decrypts the replies | the 405M publication run (58 passes); the phase-B run `run_cfg-pcmm24_bundle` (6 passes) |
| in-process | the session holds the secret (`insecure-inproc`, or server-generated test keys) | 2026-09-19 (the flight recorder reads the secret); the 405M 148-step run and the other lever runs of 2026-11-02; the earlier RWKV-7 runs; both RWKV-7 serve loops |

A session with only the public and evaluation keys does the same work per step as one that also holds the secret. The
co-located client means the November runs demonstrate the server's side of the protocol, not the separation of machines.

### What a serve loop reveals

Beyond the batch protocol's metadata (the number of steps, the lane count and the timing), a continuous-batching server
learns each conversation's **arrival time**, its **prompt length** in tokens (the number of prefill passes and decoded
remainders), its **lane assignment**, and its **output length** (when it leaves). On the Perseus runtime a pass's mask staging
runs inside the timed path, so the host time of a pass depends on the lane configuration. None of this is hidden here.

## The first run (FIDESlib runtime, ring 2^17)

| | value | record |
|---|---|---|
| scheme | CKKS (OpenFHE 1.5.1 parameter generation; FIDESlib GPU evaluation) | [hpc_gpu_port/](../hpc_gpu_port/) |
| ring dimension | N = 2^17 (131,072), 65,536 slots | every server log line `"ringDim": 131072` |
| security level | OpenFHE `HEStd_128_classic` (`--secure`): the library refuses a modulus chain above its 128-bit table for N = 2^17 | `--secure` in every recorded command line |
| secret distribution | uniform ternary (each coefficient −1/0/1); measured Hamming weight of the test key 87,435 ≈ 2N/3 | recorder line `"hamming": 87435` in [server.jsonl](../results/dense-demo-s31/pod_longrun_20260919/pod_pull/s37/serve_LONG/logs/server.jsonl) |
| modulus chain | 3,319 bits of Q·P (42 Q limbs + 14 P limbs, dnum 3) against OpenFHE's 2^17 ceiling of 3,523 — 204 bits of margin | [harness/logqp_probe.cpp](../harness/logqp_probe.cpp) |
| multiplicative depth | 41 (`--extra-depth 12`) | server setup line |
| lattice estimator | the 2^17 ceiling itself (logQP 3,523, uniform ternary, σ = 3.19) estimates at **2^128.09** (weakest attack: primal uSVP, β = 318); the served chain is 204 bits shorter, so it is at least as hard | [results/estimator-fest/fest_battery.txt](../results/estimator-fest/fest_battery.txt) `R17-CEILING`; scripts in [estimator/](../estimator/) |

No sparse-secret or key-encapsulation shortcut is used; the evaluation keys (multiplication key + rotation keys, 51 GB at
this ring) are generated by the client tool with the library's key generation.

## Threat model

- **Parties.** A client with the secret key and a server with the model weights in plaintext. Honest-but-curious server:
  it follows the protocol and may try to learn the client's text from what it sees.
- **What the server sees.** One ciphertext in and one out per step per batch (the encrypted inputs; the final hidden state),
  the public key and the evaluation keys. It never sees a token, a logit or a plaintext activation. There is one client round
  trip per generated token and none inside a token. The client decrypts, applies the output head (for RWKV-7 also the final
  LayerNorm `ln_out`) and picks the token; no run evaluates the head or the argmax under encryption.
- **What is not protected.** The model (the server owns it). The number of steps, the lane count and the timing (metadata),
  and for a serve loop the arrival times, prompt lengths, lane assignment and output lengths (above). The lanes of a batch
  belong to one client and decrypt under one secret; this is not a multi-tenant system.
- **The protocol and the November runs.** The statements above describe the protocol. In the November runs the process that
  drove the server session also held the plaintext tokens, for scoring and for the client role; the 2026-09-18 session is the
  only run that instantiated the protocol across machines.
- **Circuit privacy** is not claimed: the reply's noise could in principle reveal something about the circuit, which is public anyway.
- **The long session of 2026-09-19 is a measurement run**, not the privacy demonstration: it used keys generated on the server
  and a flight recorder that reads the secret key on the server to measure bootstrap-input overflow. The keys were deleted
  at the end; the client-keyed session of 2026-09-18 ([VERIFICATION_KIT.md](VERIFICATION_KIT.md)) is the privacy demonstration.

## What changed in the libraries for the first run, and why it does not lower security

The patches in [results/dense-demo-s31/pod_longrun_prep/patches/](../results/dense-demo-s31/pod_longrun_prep/patches/) touch the public evaluation only: a memory leak in multi-GPU key
switching (L1), the choice of the library's own extended bootstrap table and where its scaling constant is applied (K1,
O1), a read hook for the recorder (K2, K3; a test-key instrument). Key generation, the parameter tables and the secret
distribution are the libraries' own. The extended bootstrap range (K = 768 instead of 512) changes the polynomial that
approximates the modular reduction, i.e. the accuracy of a public computation, not the hardness assumption.
