#!/usr/bin/env bash
# 71_mac_demo_session.sh -- RUN ON THE MAC: the recorded encrypted-chat session.
#
# PURPOSE   prepare 64 lanes -> (optional) mac-mode fidelity probe + server
#           restart -> generate tick-wise against the remote stateful server
#           (encrypt/decrypt on this Mac with mac_fhe_client; the secret key
#           never leaves $MAC_KEYS) -> audit_ragged A/B/C -> archive everything
#           into the repo under results/dense-demo-s31/sessions/demo_<UTC>/.
#           Recording hook: RECORD=1 (default) re-executes itself under
#           script(1) so the whole terminal is a typescript beside the JSON
#           records. All 64 lanes are shown unedited; retakes are whole-session
#           with the count disclosed (DEMO_DECISIONS D10).
# INPUTS    SSH_KEY PORT HOST (required); REPO, PY=$REPO/.venv/bin/python
#           (torch 2.8.0 / transformers 4.57.6 there), MAC_TOOL, MAC_KEYS
#           (=~/mac_keys_r17, holds secret.key), MAC_BACKUP, MAC_ART (built here
#           from symlinks: pbd430a_weights.npz -> the SHA-verified npz, config,
#           seeds, bundle), PROMPTS (the D10 64-prompt file), LANES=64, WORDS=8
#           (--words-per-lane), STEPS=24 (hard cap on ticks = wall-time cap),
#           FID_MAC=1 (run the mac-mode probe first), EXPECT_TOP1 (from 60's
#           top1AgreeFrac; audit check C judges against it, F73), DEMO/KEYS/
#           POD_SRC = the pod paths (defaults match _lib.sh).
#           Subcommands: all (default) | env | preflight | fid | prepare |
#           generate | audit | archive | restart-server
# OUTPUTS   $SESSION/{tokens.json,transcript.json,lanes_text.json,logs/,
#           terminal.typescript,fid_mac/,audit.log}, and the repo copy at
#           $REPO/results/dense-demo-s31/sessions/demo_<UTC>/{client,pod}/.
# PASS      generate prints {"lanesDone":true,...}; audit_ragged exits 0 (A, B,
#           C all ran and passed; noise flips listed); archive complete.
# RECORD    the session dir (client) + server.log & friends (pod) + typescript.
# WIRE      one ciphertext per direction per tick, 88,088,569 B at 2^17 for a
#           1-lane T=1 request (demo_smoke/fid17_run.log:3 totalBytes); a 64-lane
#           request is the same single ciphertext. At 12.5 MB/s that is ~7 s
#           each way per tick; the tick itself is ~172 s + drift encodes.
set -uo pipefail
SCRIPT=$(basename "$0")
: "${REPO:=$REPO}"
: "${PY:=$REPO/.venv/bin/python}"
: "${MAC_TOOL:=$REPO/hpc_gpu_port/mac_fhe_client}"
: "${MAC_KEYS:=$HOME/mac_keys_r17}"
: "${MAC_BACKUP:=$HOME/Documents/fhe-ssm-backup}"
: "${MAC_ART:=$MAC_BACKUP/mac_art}"
: "${PROMPTS:=$REPO/results/dense-demo-s31/sessions/demo_prompts_64.txt}"
: "${TAG:=pbd430a}"; : "${LANES:=64}"; : "${WORDS:=8}"; : "${STEPS:=24}"; : "${FID_MAC:=1}"; : "${FID_TICKS:=8}"
: "${DEMO:=/root/demo}"; : "${KEYS:=/root/mac_keys_r17}"; : "${POD_SRC:=/root/src}"; : "${DEMO_TOOLS:=$POD_SRC/results/dense-demo-s31/pod_tools/demo}"
: "${WORK:=/root/fhe-main-demo}"; : "${BIN:=$WORK/build-demo/gpu_real_model}"; : "${ART:=$POD_SRC/ml-eval/artifacts}"
: "${SESSION_ROOT:=$HOME/demo_sessions}"
: "${SESSION:=$SESSION_ROOT/demo_$(date -u +%Y%m%dT%H%M%SZ)}"
: "${SSH_KEY:?set SSH_KEY}"; : "${PORT:?set PORT}"; : "${HOST:?set HOST}"
: "${POD_ENV:=}"   # e.g. POD_ENV="DEVICES=0,1" -- prefixed to every remote 70_demo_server.sh call
: "${TICK_TIMEOUT:=7200}"   # 2026-09-04: a fresh stateful two-card tick with level-drift encodes exceeded the client default of 1800 s
: "${STOP_NL:=1}"   # 2026-09-17 (POD_PLAN_S37 A11): STOP_NL=0 omits --stop-on-newline; with WORDS=0 no lane retires and STEPS is the tick count
# CLIENT_PY = the client module (default ml-eval/fhe_client.py); CLIENT_EXTRA = extra generate flags (word-split).
: "${CLIENT_PY:=$REPO/ml-eval/fhe_client.py}"; : "${CLIENT_EXTRA:=}"
# S3.7 V3 (2026-09-05; S3.6 N2, T12 §2.9): the recorded sessions spent 24–53 s per
# tick in the client's control traffic (8–9 ssh/scp round trips at ~4 s on this
# proxy path, POD_LOG.md:21, + poll overshoot + liveness). `remote` = ONE ssh per
# tick (markers + server-side wait + reply meta), one get, rm folded into the next
# tick; the poll wait is `WAIT_MODE=poll` (the pre-V3 path, still in the client).
# --poll-interval 5 below is used only under poll.
: "${WAIT_MODE:=remote}"
SSH=(ssh -i "$SSH_KEY" -p "$PORT" "root@$HOST")
# review 2026-09-03: one multiplexed connection instead of ~1,000 fresh ones per
# session (each poll/scp reuses the master; ServerAlive keeps it up); the client
# also retries transport failures with backoff.
mkdir -p "$HOME/.ssh"
RSSH="ssh -i $SSH_KEY -p $PORT -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-fhe-%r@%h:%p -o ControlPersist=600 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 root@$HOST"
# 2026-09-17: the circuit flags the client is told about follow LOG_RING/EXTRA_DEPTH/SECURE/LANES_BLOCK (defaults = the
# 2^17 demo config) so a 2^15 one-card session names the right ring; the client only needs --pack-tokens 1 here.
: "${LOG_RING:=17}"; : "${EXTRA_DEPTH:=12}"; : "${SECURE:=1}"; : "${LANES_BLOCK:=1}"
_SEC=""; [ "$SECURE" = 1 ] && _SEC=" --secure"
_LB=""; [ "$LANES_BLOCK" = 1 ] && _LB=" --lanes-block"
SERVER_FLAGS="$BIN --tag $TAG --bundle-dir $ART --device 0 --log-ring $LOG_RING$_SEC --level-budget 3 3 --ms-norm 5.5 --enc-threads 24 --boot-floor 3 --matvec-margin 4 --extra-depth $EXTRA_DEPTH --pack-tokens 1$_LB --compressed-store"
MODE=${1:-all}
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(utc)] $*"; }
mkdir -p "$SESSION"
# ---- recording hook: everything below runs inside script(1) -----------------
if [ "${RECORD:-1}" = 1 ] && [ -z "${UNDER_SCRIPT:-}" ] && [ "$MODE" != env ]; then
  export UNDER_SCRIPT=1 SESSION
  log "recording terminal to $SESSION/terminal.typescript (script(1)); RECORD=0 to disable"
  exec script -q "$SESSION/terminal.typescript" "$0" "$@"
fi
note() { echo "$(utc) | $*" >> "$SESSION/SESSION_LOG.md"; }

do_env() {
  mkdir -p "$MAC_ART"
  ln -sf "$MAC_BACKUP/pbd430a_weights_DEMO_MODEL_20260901.npz" "$MAC_ART/pbd430a_weights.npz"
  ln -sf "$REPO/ml-eval/artifacts/pbd430a_config.json" "$MAC_ART/pbd430a_config.json"
  ln -sf "$REPO/ml-eval/artifacts/pbd430a_seeds.json" "$MAC_ART/pbd430a_seeds.json"
  ln -sf "$MAC_BACKUP/pod_artifacts/bundle_pbd430a.bin" "$MAC_ART/bundle_pbd430a.bin"
  ln -sf "$MAC_BACKUP/pod_artifacts/bundle_pbd430a.index.txt" "$MAC_ART/bundle_pbd430a.index.txt"
  cp "$PROMPTS" "$SESSION/demo_prompts_64.txt"
  S=$(shasum -a 256 "$MAC_ART/pbd430a_weights.npz" | cut -c1-64)
  [ "$S" = 3ec3f1d8b9b71e7eee38055a7621213991c6417a89576f1c02590963293e784b ] && log "npz sha256 OK (RECOVERY_STATE:14)" || { log "npz sha256 MISMATCH: $S"; exit 2; }
  "$PY" -c 'import torch, transformers, numpy; print("mac python ok", torch.__version__, transformers.__version__)' || exit 2
  [ -x "$MAC_TOOL" ] || { log "mac_fhe_client missing: build with hpc_gpu_port/mac_build_client.sh"; exit 2; }
  [ -s "$MAC_KEYS/secret.key" ] && [ -s "$MAC_KEYS/public.key" ] || { log "Mac keys missing in $MAC_KEYS (30_dump_indices.sh step 2)"; exit 2; }
  log "env OK: SESSION=$SESSION MAC_ART=$MAC_ART"
}

do_preflight() {
  log "pod checks over ssh"
  "${SSH[@]}" "test ! -e $KEYS/secret.key && echo 'secret ABSENT on pod: OK' || { echo 'FATAL: secret.key on the pod'; exit 9; }" || exit 9
  "${SSH[@]}" "kill -0 \$(cat $DEMO/server.pid) 2>/dev/null && echo 'server pid alive' || { echo 'FATAL: no live server (70_demo_server.sh start)'; exit 9; }" || exit 9
  "${SSH[@]}" "tail -3 $DEMO/server.log | cut -c1-200; if ls $DEMO/serve/req.* $DEMO/serve/resp.* >/dev/null 2>&1; then echo 'FATAL: stale req/resp files in the serve dir (F64)'; ls $DEMO/serve | head -5; exit 9; fi; echo 'serve dir clean'" || exit 9
  "${SSH[@]}" "grep -a 'PASS sovereignty' $DEMO/CAMPAIGN_LOG.md | tail -1; grep -a 'FIDGATE:verdict' $DEMO/CAMPAIGN_LOG.md | tail -1" || true
  note "preflight OK"
}

do_restart_server() {
  log "restarting the server (request counter must start at 0 for the next client, fhe_client.py:446-450)"
  # POD_ENV (2026-09-04): env prefix for the remote scripts, e.g. "DEVICES=0,1" for the two-card route
  "${SSH[@]}" "${POD_ENV:-} bash $DEMO_TOOLS/70_demo_server.sh stop; ${POD_ENV:-} bash $DEMO_TOOLS/70_demo_server.sh start" | tail -15
  note "server restarted"
}

do_fid() {
  local O="$SESSION/fid_mac"
  # 4 prompts of >= FID_TICKS tokens: lines 1-2, 3-4, 5-6, 7-8 of the D10 pool, joined (same rule as 60_fidelity_gate.sh)
  python3 - "$PROMPTS" "$SESSION/fid_prompts4.txt" <<'PY'
import sys
l = [x.strip() for x in open(sys.argv[1]) if x.strip()]
open(sys.argv[2], "w").write("\n".join(l[2*i] + " " + l[2*i+1] for i in range(4)) + "\n")
PY
  log "mac-mode fidelity probe (fidelity_tick.py:33-43): $FID_TICKS ticks x 4 lanes through the LIVE server, enc/dec here"
  ( cd "$REPO" && PYTHONUNBUFFERED=1 "$PY" -u spec_decode/fidelity_tick.py --out "$O" --tag "$TAG" --prompts-file "$SESSION/fid_prompts4.txt" \
      --ticks "$FID_TICKS" --bundle-dir "$MAC_ART" --art-dir "$MAC_ART" --device cpu \
      --mac-tool "$MAC_TOOL" --mac-keys "$MAC_KEYS" --remote-ssh "$RSSH" --remote-reqdir "$DEMO/serve" --poll-interval 5 \
      --wait-mode "$WAIT_MODE" \
      --keep-cts "$O/cts" --tick-timeout "${TICK_TIMEOUT:-7200}" ) 2>&1 | tee "$SESSION/fid_mac_run.log"
  # same rule as 60_fidelity_gate.sh verdict (copied verbatim, pod path unavailable here)
  python3 - "$O/fidelity.json" "${FID_TOP1_MIN:--1}" "${FID_FLIP_MARGIN:-0.5}" "${FID_RELERR_MAX:-1e-2}" "${FID_RELERR_RECORD:-4e-2}" <<'PY' | tee -a "$SESSION/fid_mac_run.log"
import json, sys
d = json.load(open(sys.argv[1])); rows = d["rows"]
nmin, flip, rmax, rrec = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
n = len(rows); nmin = n - 2 if nmin < 0 else nmin
agree = sum(1 for r in rows if r["top1"])
hard = [(r["tick"], r["lane"], r["refTop1Margin"]) for r in rows if not r["top1"] and r["refTop1Margin"] >= flip]
worst = max(r["relErrRms"] for r in rows)
print("FIDGATE:rows:" + json.dumps({"comparisons": n, "top1Agree": agree, "top1Min": nmin, "hardMisses": hard, "relErrRmsWorst": worst,
      "serveSecPerTick": {r["tick"]: r["serveSec"] for r in rows}}))
ok1 = agree >= nmin and not hard
if ok1 and worst <= rmax: print("FIDGATE:verdict:GO"); sys.exit(0)
if ok1 and worst <= rrec: print("FIDGATE:verdict:CONDITIONAL -- relErrRmsWorst %.4g above %.0e but inside the on-record 2^17 single-token class (0.0122788 / 0.0374048); MANUAL CALL" % (worst, rmax)); sys.exit(2)
print("FIDGATE:verdict:NO-GO -- top1 %d/%d hard %s worst %.4g" % (agree, n, hard, worst)); sys.exit(1)
PY
  RC=${PIPESTATUS[0]}
  note "fid_mac verdict rc=$RC $(grep -ao 'FIDGATE:verdict:[A-Z-]*' "$SESSION/fid_mac_run.log" | tail -1)"
  [ "$RC" = 1 ] && { log "NO-GO from the mac-mode probe -- do not record"; exit 1; }
  do_restart_server
}

do_prepare() {
  log "prepare $LANES lanes from $PROMPTS (D10: 7 NeoX tokens each, first-64 of the pool, no selection on output)"
  # LANES=1 (2026-09-04): the client's single-lane path takes --prompt, the ragged path takes --prompts-file
  if [ "$LANES" = 1 ]; then PARGS=(--prompt "$(head -1 "$PROMPTS")"); else PARGS=(--prompts-file "$PROMPTS"); fi
  ( cd "$REPO" && "$PY" "$CLIENT_PY" prepare --lanes "$LANES" "${PARGS[@]}" \
      --bundle-dir "$MAC_ART" --tag "$TAG" --out "$SESSION" --dpad 1024 ) 2>&1 | tee "$SESSION/prepare.log"
  grep -q '"prepare": true' "$SESSION/prepare.log" || { log "prepare failed"; exit 1; }
  note "prepare OK: $(grep -o '{"prepare"[^}]*}' "$SESSION/prepare.log")"
}

do_generate() {
  local SNL=(--stop-on-newline); [ "$STOP_NL" = 1 ] || SNL=()
  local CX=(); [ -n "$CLIENT_EXTRA" ] && read -r -a CX <<< "$CLIENT_EXTRA"
  log "generate: STEPS=$STEPS ticks max, --words-per-lane $WORDS ${SNL[*]-}(STOP_NL=$STOP_NL), greedy; ~172 s/tick + drift encodes + ~14 s wire"
  log "hexdump hook (another terminal): while :; do for f in $SESSION/req.*.0; do [ -f \$f ] && { ls -l \$f; xxd \$f | head -4; }; done; sleep 2; done"
  ( cd "$REPO" && PYTHONUNBUFFERED=1 "$PY" -u "$CLIENT_PY" generate --out "$SESSION" --steps "$STEPS" --greedy ${CX[@]+"${CX[@]}"} \
      --keys-dir "$MAC_KEYS" --mac-tool "$MAC_TOOL" --mac-keys "$MAC_KEYS" \
      --remote-ssh "$RSSH" --remote-reqdir "$DEMO/serve" \
      --words-per-lane "$WORDS" ${SNL[@]+"${SNL[@]}"} --poll-interval 5 --tick-timeout "${TICK_TIMEOUT:-7200}" \
      --wait-mode "$WAIT_MODE" \
      --keep-cts "$SESSION/cts" \
      --harness-cmd "$SERVER_FLAGS" ) 2>&1 | tee "$SESSION/generate.log"
  grep -q '"lanesDone": true' "$SESSION/generate.log" || { log "generate did not finish cleanly (see generate.log)"; note "generate INCOMPLETE"; exit 1; }
  note "generate OK: $(grep -o '{"lanesDone"[^}]*}' "$SESSION/generate.log")"
  echo "--- per-tick seconds (enc / serve / dec) ---"
  python3 -c "import json;[print(t['tick'], t.get('encSec'), t.get('serveSec'), t.get('decSec'), t['phases'].count('prompt'), t['phases'].count('gen'), t['phases'].count('retired')) for t in json.load(open('$SESSION/transcript.json'))]"
}

do_audit() {
  local X=()
  [ -n "${EXPECT_TOP1:-}" ] && X=(--expect-top1 "$EXPECT_TOP1")
  log "audit_ragged A/B/C (spec_decode/audit_ragged.py; rc 0 = every check ran and passed; 3 violation; 4 check could not run; 5 refused)"
  ( cd "$REPO" && "$PY" spec_decode/audit_ragged.py --out "$SESSION" --tag "$TAG" --art-dir "$MAC_ART" --device cpu --flip-margin 0.15 --checks ABC ${X[@]+"${X[@]}"} ) 2>&1 | tee "$SESSION/audit.log"
  RC=${PIPESTATUS[0]}; note "audit rc=$RC"
  [ "$RC" = 0 ] || { log "AUDIT rc=$RC -- the recording does not ship until A-C are green (spec_decode/README.md:31-40; D on >=4 lanes is a session recipe)"; return 1; }
}

do_archive() {
  local D="$REPO/results/dense-demo-s31/sessions/$(basename "$SESSION")"
  mkdir -p "$D/client" "$D/pod"
  # cts/ (every ciphertext that left the Mac, ~176 MB per tick) goes to the
  # verification kit (do_kit), not into the repo; its manifests do.
  rsync -avh --exclude 'req.*' --exclude '*.hidden.f64' --exclude 'cts/' "$SESSION/" "$D/client/"
  log "pulling the pod-side records (everything in $DEMO except the store, pod keys and the serve dir)"
  rsync -avh --exclude 'store_*.bin' --exclude '*.journal' --exclude 'keys_pod_r17' --exclude 'serve' --exclude '*.logits.f64' -e "ssh -i $SSH_KEY -p $PORT" "root@$HOST:$DEMO/" "$D/pod/"
  "${SSH[@]}" "ls -la $KEYS; sha256sum $KEYS/cryptocontext.bin $KEYS/public.key" > "$D/pod/KEYS_ON_POD.txt" 2>&1 || true
  ( cd "$D" && find . -type f | sort | xargs shasum -a 256 > MANIFEST.sha256 )
  log "archived to $D -- commit it: git add results/dense-demo-s31/sessions/$(basename "$SESSION") && git commit -m 'demo session $(basename "$SESSION"): 64-lane stateful 2^17 recording, records + typescript'"
  note "archived to $D"
}

do_passcost() {
  # 2026-09-14 (POD_PLAN_S37 A11): the per-tick cost vs context position from the ARCHIVED
  # records (client transcript.json + pod/server.log), rendered by
  # spec_decode/pass_cost_vs_context.py into $D/client/passcost/{csv,md,svg}. Non-fatal:
  # a missing record prints a note and the recording still ships. Re-manifests $D.
  local D="$REPO/results/dense-demo-s31/sessions/$(basename "$SESSION")"
  [ -f "$D/client/transcript.json" ] || { note "passcost: no archived transcript in $D/client -- run archive first"; return 0; }
  local ID; ID="$(head -1 "$D/pod/build/BUILD_IDENTITY.txt" 2>/dev/null || true)"
  ( cd "$REPO" && python3 spec_decode/pass_cost_vs_context.py --session "$D/client" --server-log "$D/pod/server.log" \
      --out "$D/client/passcost" --label "$(basename "$SESSION") $ID" ) 2>&1 | tee -a "$SESSION/passcost.log" || true
  ( cd "$D" && find . -type f | sort | xargs shasum -a 256 > MANIFEST.sha256 )
  note "passcost: $(grep -o '"verdict": "[^"]*"' "$SESSION/passcost.log" | tail -1)"
}

write_kit_howto() {   # write_kit_howto KITDIR
  cat > "$1/HOWTO.md" <<'HOWTO'
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
- `fid_mac_cts/` — the same for the teacher-forced fidelity probe that ran
  through the live server before the session (4 lanes).
- `records/` — `tokens.json`, `transcript.json`, `lanes_text.json`,
  `state.json`, the run logs, the 64 prompts, the model config/seeds and the
  `.VERIFIED` sha256 of the served bundle.

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
Two decryptions of one CKKS ciphertext differ at the ~1e-13 level (OpenFHE
adds flooding noise at decrypt), so compare with a tolerance, not `cmp`.

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
HOWTO
}

do_kit() {
  # DECISION 2026-09-04: "save everything: exact keys and ciphertexts that
  # left our Mac, in a manner that would allow an independent reviewer to
  # encrypt/decrypt to verify for themselves." The kit lives OUTSIDE the repo
  # (~176 MB of ciphertext per tick); the repo archive gets its manifests.
  local K="${KIT_ROOT:-$MAC_BACKUP}/demo_verification_$(basename "$SESSION")"
  local D="$REPO/results/dense-demo-s31/sessions/$(basename "$SESSION")"
  mkdir -p "$K/keys" "$K/records" "$D/verification"
  log "verification kit -> $K"
  [ -d "$SESSION/cts" ] && { mv "$SESSION/cts" "$K/session_cts"; log "moved session ciphertexts ($(ls "$K/session_cts" | grep -c tick_) ticks)"; }
  [ -d "$SESSION/fid_mac/cts" ] && { mv "$SESSION/fid_mac/cts" "$K/fid_mac_cts"; log "moved fidelity-probe ciphertexts"; }
  # decrypting material (small) is COPIED; the ~48 GB eval material stays in $MAC_KEYS, pinned by hash
  for f in cryptocontext.bin cryptocontext.bin.dev public.key secret.key; do [ -f "$MAC_KEYS/$f" ] && cp -p "$MAC_KEYS/$f" "$K/keys/"; done
  cp -p "$MAC_KEYS"/*.json "$K/keys/" 2>/dev/null || true
  ( cd "$MAC_KEYS" && find . -type f | sort | xargs shasum -a 256 ) > "$K/keys/MAC_KEYS_MANIFEST.sha256"
  for f in tokens.json transcript.json lanes_text.json state.json prepare.log generate.log audit.log fid_mac_run.log fid_mac/fidelity.json fid_prompts4.txt SESSION_LOG.md; do
    [ -f "$SESSION/$f" ] && { mkdir -p "$K/records/$(dirname "$f")"; cp -p "$SESSION/$f" "$K/records/$f"; }
  done
  cp -p "$PROMPTS" "$K/records/" 2>/dev/null || true
  for f in "$MAC_BACKUP/pod_artifacts/bundle_${TAG}.bin.VERIFIED" "$MAC_ART/${TAG}_config.json" "$MAC_ART/${TAG}_seeds.json"; do [ -f "$f" ] && cp -p "$f" "$K/records/"; done
  [ -f "$MAC_ART/${TAG}_weights.npz" ] && shasum -a 256 "$MAC_ART/${TAG}_weights.npz" > "$K/records/${TAG}_weights.npz.sha256"
  write_kit_howto "$K"
  ( cd "$K" && find . -type f ! -name MANIFEST.sha256 | sort | xargs shasum -a 256 > MANIFEST.sha256 )
  find "$K" -type f -exec chmod 444 {} +
  cp -p "$K/MANIFEST.sha256" "$K/HOWTO.md" "$K/keys/MAC_KEYS_MANIFEST.sha256" "$D/verification/" 2>/dev/null || true
  for m in "$K"/session_cts/MANIFEST.jsonl "$K"/fid_mac_cts/MANIFEST.jsonl; do [ -f "$m" ] && cp -p "$m" "$D/verification/$(basename "$(dirname "$m")")_MANIFEST.jsonl"; done
  log "kit complete: $(du -sh "$K" | cut -f1) at $K (files chmod 444); manifests + HOWTO copied into $D/verification -- commit them"
  note "verification kit at $K"
}

case "$MODE" in
  env) do_env;;
  preflight) do_env; do_preflight;;
  fid) do_env; do_preflight; do_fid;;
  prepare) do_env; do_prepare;;
  generate) do_env; do_preflight; do_generate;;
  audit) do_audit;;
  archive) do_archive;;
  passcost) do_passcost;;
  kit) do_kit;;
  restart-server) do_restart_server;;
  all) do_env; do_preflight; [ "$FID_MAC" = 1 ] && do_fid; do_prepare; do_generate; do_audit || true; do_archive; do_passcost; do_kit;;
  *) echo "usage: $SCRIPT [all|env|preflight|fid|prepare|generate|audit|archive|kit|restart-server]"; exit 2;;
esac
