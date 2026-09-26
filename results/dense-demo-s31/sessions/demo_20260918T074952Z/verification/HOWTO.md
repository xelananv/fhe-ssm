# Independent verification kit — fully-secure FHE-SSM demo session

Everything in this directory was produced on the client Mac. The pod never
held `secret.key`; it received only `cryptocontext.bin(+.dev)`, `public.key`,
`evalmult.bin` and the `evalrot*.bin` rotation-key files listed (with sha256)
in `keys/MAC_KEYS_MANIFEST.sha256`. `MANIFEST.sha256` covers this whole kit.

## Layout
- `keys/` — `cryptocontext.bin`, `cryptocontext.bin.dev`, `public.key`,
  `secret.key` (copies of the client's key files) and the sha256 manifest of
  the complete key directory as it was when the eval material was uploaded.
- `session_cts/tick_TTT_reqN/` — for every tick of the recorded session:
  `req.N.0` + `req.N.meta` (the ciphertext + layout sidecar that LEFT the Mac),
  `resp.N.0` + `resp.N.meta` (what CAME BACK from the pod), `input_rows.f64`
  (the plaintext embedded rows that were encrypted, lanes×d float64, lane-major),
  `hidden.f64` (what the Mac decrypted from the reply, lanes×d float64).
  `session_cts/MANIFEST.jsonl` has one line per file: tick, req, lanes, d,
  dpad, bytes, sha256, UTC.
- `records/` — `tokens.json`, `transcript.json`, `lanes_text.json`, the run
  logs, the 64 prompts, the model config/seeds and the `.VERIFIED` sha256 of
  the served bundle.

## 1. Check integrity
    shasum -a 256 -c MANIFEST.sha256
    python3 -c "import json,hashlib,sys;[print(r['file'], hashlib.sha256(open('session_cts/'+r['file'],'rb').read()).hexdigest()==r['sha256']) for r in map(json.loads, open('session_cts/MANIFEST.jsonl'))]"

## 2. Re-decrypt a reply and compare with what the Mac decrypted
`mac_fhe_client` is built from `hpc_gpu_port/mac_fhe_client.cpp` of the repo
(`hpc_gpu_port/mac_build_client.sh`, OpenFHE 1.5.1 CPU). LANES/D/DPAD are in
the tick's manifest line and in the `.meta` sidecar (lanes, dpad; d = 1024).
    T=session_cts/tick_000_req0
    mac_fhe_client dec --ctx keys/cryptocontext.bin --sec keys/secret.key \
      --in $T/resp.0 --count 1 --lanes 64 --d 1024 --dpad 1024 \
      --require-meta $T/resp.0.meta --out /tmp/hidden_check.f64
    python3 - <<'PY'
    import numpy as np
    a=np.fromfile('/tmp/hidden_check.f64','<f8'); b=np.fromfile('session_cts/tick_000_req0/hidden.f64','<f8')
    print('max |a-b| / max|b| =', np.max(np.abs(a-b))/np.max(np.abs(b)))
    PY
Two decryptions of one CKKS ciphertext are not byte-identical (OpenFHE adds
noise at decryption). Measured on this kit: the re-decrypted replies differ
from the recorded `hidden.f64` rows by a relative rms of 2.2e-3 to 8.6e-3 per
tick, the re-decrypted requests from `input_rows.f64` by at most 1.2e-11; every
recorded token is reproduced. Compare with a tolerance, not `cmp`.

## 3. Re-decrypt a request and compare with the plaintext that was encrypted
    mac_fhe_client dec --ctx keys/cryptocontext.bin --sec keys/secret.key \
      --in $T/req.0 --count 1 --lanes 64 --d 1024 --dpad 1024 \
      --require-meta $T/req.0.meta --out /tmp/input_check.f64
    # compare /tmp/input_check.f64 with $T/input_rows.f64 as in step 2
    # (fresh-encryption error at scale 2^59 is far below 1e-9 relative).

## 4. From hidden rows to the recorded tokens
The reply is the model's last hidden state per lane; the token is
`argmax(hidden @ head.T)` with the head matrix of the served model
(`ml-eval/fhe_client.py: head_matrix(bundle_dir, tag)` reads it from the
bundle whose sha256 is in `records/bundle_*.VERIFIED`; the same weights are
in `<tag>_weights.npz`, sha256 in `records/`). Compare the argmax per lane
with `records/tokens.json` / `transcript.json` for that tick (the client kept
a lane's prediction iff its next input was that token — D11 in the repo).

## 5. Encrypt something yourself
    mac_fhe_client enc --ctx keys/cryptocontext.bin --pub keys/public.key \
      --rows myrows.f64 --lanes 64 --tokens 1 --d 1024 --dpad 1024 --out /tmp/mine
CKKS encryption is randomised: a fresh encryption of `input_rows.f64` will not
be byte-identical to `req.N.0`; it decrypts to the same rows (step 3).
