# The verification kit — the recorded client-keyed session, replayable

The session of 2026-09-18 (`demo_20260918T074952Z`: 64 lanes, up to 8 generated tokens per lane (492 in all; a lane stops at its first newline), 18 ticks, keys generated on the
client, the secret key never on the server) was recorded in full: the key files, every request ciphertext that left the
client and every reply ciphertext that came back, the plaintext rows that were encrypted, the rows the client decrypted, and
the tokens. The kit is 1.9 GB (129 files, 128 of them listed in its manifest) and is distributed separately from this repository (link in the top-level README);
its manifests are in this tree so that a downloaded kit can be checked byte for byte:

- [MANIFEST.sha256](../results/dense-demo-s31/sessions/demo_20260918T074952Z/verification/MANIFEST.sha256) — every file of the kit
- [session_cts_MANIFEST.jsonl](../results/dense-demo-s31/sessions/demo_20260918T074952Z/verification/session_cts_MANIFEST.jsonl) — one line per ciphertext: tick, request, lanes, d, bytes, sha256, UTC
- [MAC_KEYS_MANIFEST.sha256](../results/dense-demo-s31/sessions/demo_20260918T074952Z/verification/MAC_KEYS_MANIFEST.sha256) — the complete client key directory (multiplication key and the 36 rotation-key parts included, 51 GB) as it was when the evaluation keys were uploaded
- [HOWTO.md](../results/dense-demo-s31/sessions/demo_20260918T074952Z/verification/HOWTO.md) — the kit's own instructions

**Publishing the secret key burns this key set.** It was generated for this session, protected only the public prompts in
[demo_prompts_64.txt](../results/dense-demo-s31/sessions/demo_20260918T074952Z/client/demo_prompts_64.txt), and will never be used again.

## What the kit lets anyone check, and what it gave when checked (2026-09-22, a 16 GB laptop, OpenFHE 1.5.1 CPU; scripts and raw results in [kit_level1_check_20260922/](../results/dense-demo-s31/pod_2xbw_20260917/kit_level1_check_20260922/))

The client tool is [hpc_gpu_port/mac_fhe_client.cpp](../hpc_gpu_port/mac_fhe_client.cpp) (`mac_fhe_client dec --ctx keys/cryptocontext.bin
--sec keys/secret.key --in session_cts/tick_TTT_reqN/resp.N --count 1 --lanes 64 --d 1024 --dpad 1024 --require-meta …`); one
decryption of a ring-2^17 ciphertext takes about a second.

| check | result |
|---|---|
| integrity: `shasum -a 256 -c MANIFEST.sha256` | 128 of 128 files OK |
| re-decrypt every **request** (18) and compare with the recorded `input_rows.f64` | maximum relative difference 1.2 × 10⁻¹¹ (fresh-encryption error) |
| are the request rows the embeddings of the recorded tokens? (`head[ids]` with the tied matrix, from the weights whose sha256 is in the kit) | identical: difference 0 on all 18 × 64 rows |
| re-decrypt every **reply** (18) and compare with the recorded `hidden.f64` | relative rms (‖a − b‖ / ‖b‖ over the tick's 64 × 1,024 rows) 2.2 × 10⁻³ to 8.6 × 10⁻³ per tick, rising with the tick (2.3 × 10⁻³ at tick 0, 6.3 × 10⁻³ at tick 17, the largest at tick 3); largest single entry 1.9 × 10⁻³ of the tick's maximum; no NaN ([level1_reply_error.json](../results/dense-demo-s31/pod_2xbw_20260917/kit_level1_check_20260922/level1_reply_error.json)) |
| hidden rows → tokens: `argmax(hidden · headᵀ)` of the re-decrypted replies vs the recorded picks (`transcript.json`) | **492 of 492 generated tokens reproduced** — all 64 lanes' prompts and replayed outputs, verbatim: [REPLAYED_LANES.md](../results/dense-demo-s31/pod_2xbw_20260917/kit_level1_check_20260922/REPLAYED_LANES.md) |

Two decryptions of the same reply are therefore not byte-identical and not within 10⁻¹³ either — OpenFHE adds noise at
decryption, and at this reply's level that noise is of the order 10⁻³ relative. The kit's HOWTO originally gave a 10⁻¹³
tolerance — an expectation, not a measurement; it was corrected before publication (with two layout entries for files that were
never archived), and its line in the manifest above with it; no data file changed. The token decisions are unaffected: every
recorded pick is reproduced. The requests, encrypted fresh at level 0, re-decrypt to 10⁻¹¹.

What was **not** done: re-running the encrypted inference on the recorded requests. It needs the evaluation keys of the kit's
key set (51 GB, sha256 in `MAC_KEYS_MANIFEST.sha256`), the weight bundle and a server built from this tree on a two-GPU box
([RUNBOOK.md](RUNBOOK.md)); feeding the recorded requests in order from a fresh server start would produce replies that decrypt
to the recorded hidden rows within the CKKS error and the decryption noise above. Whether the replies would be byte-identical
to the recorded ones is untested (the host-side plaintext encoding uses floating-point FFTs).

The long free-running session of 2026-09-19 is **not** in a kit: it used server-generated test keys (deleted at the end)
and kept no ciphertexts — only its records ([results/dense-demo-s31/pod_longrun_20260919/](../results/dense-demo-s31/pod_longrun_20260919/)).
