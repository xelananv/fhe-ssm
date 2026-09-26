#!/usr/bin/env bash
# 70_demo_server.sh -- the demo server on the MAC keys: persistent, stateful,
#                      block layout, compressed store, no secret key on the box.
#
# PURPOSE   start | stop | status of the one process that serves the recording.
#           Protocol (gpu_real_model.cu:3186-3197, DEMO_README.md:79-95): the
#           client drops req.N.0 + req.N.meta into $DEMO/serve and touches
#           req.N.ready (req.0.reset on the first tick); the server answers
#           resp.N.* + resp.N.done (or .error), prints {"serve":"served",...,
#           "reqMsPerToken":...}, deletes the request, waits for req.N+1.ready.
#           A file named `stop` in the serve dir ends it (or SERVE_TIMEOUT idle
#           seconds). N is monotonic per server process, the client always
#           numbers from 0 (fhe_client.py:446-450) -> RESTART the server between
#           any two client sessions (probe, session, retake).
# INPUTS    env per _lib.sh; SERVE_TIMEOUT=7200; SETUP_WAIT=1800 s; STOP_WAIT=900 s
#           DEVICES=0,1 for the two-card route (2026-09-03): _lib.sh then adds
#           --devices 0,1 --store-multi-gpu and STORE = the _mgpu file 41 built.
#           (a tick in flight is ~172 s + drift encodes; the server finishes it
#           before it sees `stop`); CLEAN_SERVE=1 wipes stale req/resp files.
# OUTPUTS   $DEMO/server.log, $DEMO/server.pid, $DEMO/.busy (held by the server
#           until stop), $DEMO/serve/ (the request dir the Mac client uses),
#           $DEMO/server_vram.jsonl (on stop/status).
# PASS      start: secret.key ABSENT in $KEYS (asserted), keys + bundle present,
#           the log reaches {"serve":"waiting","req":0,...} with no CUDA/fatal
#           text and the pid alive. stop: {"serve":"exit"} line present, pid
#           gone, no served line with failed:true.
# RECORD    server.log = the demo's server-side record (served lines carry
#           reqEvalMs/reqBootMs/reqEncPtMs/reqBoots/reqMsPerToken/peakVramGB per
#           tick; vramTrace lines per stage/request; the summary at exit).
# DURATION  setup: keys-load setupMs 45,100 (demo_smoke/fid17_run.log:6) +
#           store load (~61 GB); first tick: all-hits 171,941 ms (clean:16) +
#           whatever level-drift encodes the store still lacks.
set -uo pipefail
SCRIPT=$(basename "$0")
. "$(dirname "$(readlink -f "$0")")/_lib.sh"
: "${SERVE_TIMEOUT:=7200}"; : "${SETUP_WAIT:=1800}"; : "${STOP_WAIT:=900}"
SD="$DEMO/serve"; L="$DEMO/server.log"; PF="$DEMO/server.pid"
MODE=${1:-start}

alive() { [ -s "$PF" ] && kill -0 "$(cat "$PF")" 2>/dev/null; }

do_start() {
  require_bin; require_file "$ART/bundle_${TAG}.bin" "20_stage_artifacts.sh"
  if [ -e "$KEYS/secret.key" ]; then log "REFUSED: $KEYS/secret.key exists on the pod -- the demo server must not even be able to load it. Delete it."; campaign_log "REFUSED server start | secret.key present"; exit 2; fi
  for f in cryptocontext.bin cryptocontext.bin.dev public.key evalmult.bin; do require_file "$KEYS/$f" "Mac upload"; done
  ls "$KEYS"/evalrot.part*.bin >/dev/null 2>&1 || [ -s "$KEYS/evalrot.bin" ] || { log "no evalrot in $KEYS"; exit 2; }
  grep -q 'PASS sovereignty' "$DEMO/CAMPAIGN_LOG.md" 2>/dev/null || log "WARN: no 'PASS sovereignty' in CAMPAIGN_LOG.md -- run 50_sovereignty.sh first"
  # 2026-09-18 (pod_2xbw): NO_STORE=1 serves WITHOUT a store file -- the drift recording form: every plaintext is
  # periodic-encoded per tick and dropped, so host RAM stays flat (a drift session with a store journals +25-31 GB per tick).
  STORE_FLAGS=(--store-file "$STORE"); if [ "${NO_STORE:-0}" = 1 ]; then STORE_FLAGS=(); log "NO_STORE=1: serving without a store file (periodic encode every tick)"; else
  [ -f "$STORE" ] || log "WARN: $STORE missing -- request 0 will build the store (~1 h) and save it"; fi
  if alive; then log "REFUSED: server already running (pid $(cat "$PF")). $SCRIPT status | stop"; exit 3; fi
  if [ -f "$DEMO/.busy" ]; then log "REFUSED: $DEMO/.busy held ($(cat "$DEMO/.busy")) -- a GPU cell is running; wait for it or stop it"; exit 3; fi
  mkdir -p "$SD"
  # 2026-09-04: `ls a b c` fails when ANY glob has no match, so a dir holding only
  # req.* was never detected and CLEAN_SERVE=1 never wiped it (the restarted server
  # then picked up a dead session's req.0.ready). Detect any of the three.
  if ls "$SD"/req.* "$SD"/resp.* "$SD"/stop 2>/dev/null | grep -q .; then
    if [ "${CLEAN_SERVE:-0}" = 1 ]; then rm -f "$SD"/req.* "$SD"/resp.* "$SD"/stop; log "serve dir wiped"
    else log "REFUSED: stale files in $SD ($(ls "$SD" | head -5 | tr '\n' ' ')). A leftover resp.N.done replays a dead session (F64). CLEAN_SERVE=1 to wipe."; exit 3; fi
  fi
  [ -s "$L" ] && mv "$L" "$L.$(date -u +%Y%m%dT%H%M%SZ).prev"
  echo "70_demo_server.sh SERVER pid=pending since=$(utc)" > "$DEMO/.busy"
  { echo "=== SERVER START $(utc) ==="; record_load_state
    echo "CMD: $BIN ${MODEL_FLAGS[*]} ${DEMO_FLAGS[*]} ${BLOCK_FLAGS[*]} ${DEVICES_FLAGS:-} --keys-dir $KEYS --keys-load --serve $SD --serve-timeout $SERVE_TIMEOUT --stateful ${STORE_FLAGS[*]-} --vram-trace"; } > "$L"
  # argv[0] must be the binary itself (no bash -c / env wrapper): the client's
  # preflight matches '^[^ ]*gpu_real_model' + '--serve' + the reqdir on ONE
  # process line (fhe_client.py:526-556).
  nohup "$BIN" "${MODEL_FLAGS[@]}" "${DEMO_FLAGS[@]}" "${BLOCK_FLAGS[@]}" ${DEVICES_FLAGS:-} \
    --keys-dir "$KEYS" --keys-load --serve "$SD" --serve-timeout "$SERVE_TIMEOUT" --stateful \
    "${STORE_FLAGS[@]}" --vram-trace >> "$L" 2>&1 < /dev/null &
  echo $! > "$PF"; echo "70_demo_server.sh SERVER pid=$(cat "$PF") since=$(utc)" > "$DEMO/.busy"
  log "server pid $(cat "$PF"); waiting up to ${SETUP_WAIT}s for {\"serve\":\"waiting\",\"req\":0}"
  T0=$(date +%s)
  while :; do
    if grep -aq '"serve":"waiting","req":0' "$L"; then break; fi
    if ! cuda_ok "$L"; then log "SETUP FAILED: $(cuda_problems "$L" | tr '\n' ';')"; campaign_log "FAIL server start | $(cuda_problems "$L" | head -1) | $L"; rm -f "$DEMO/.busy"; exit 1; fi
    if ! alive; then log "SETUP FAILED: process exited (tail):"; tail -5 "$L"; campaign_log "FAIL server start | process exited | $L"; rm -f "$DEMO/.busy"; exit 1; fi
    [ $(( $(date +%s) - T0 )) -ge "$SETUP_WAIT" ] && { log "SETUP TIMEOUT after ${SETUP_WAIT}s (still alive; check $L)"; campaign_log "TIMEOUT server start | $L"; exit 1; }
    sleep 10
  done
  SETUP=$(grep -ao '"setupMs":[0-9]*' "$L" | head -1); SL=$(grep -ao '{"storeLoad"[^}]*}' "$L" | head -1)
  log "SERVER READY in $(( $(date +%s) - T0 ))s  $SETUP  $SL"
  grep -a '"vramTrace"' "$L" | tail -3
  campaign_log "PASS server start | pid $(cat "$PF") | $SETUP | ${SL:-no storeLoad line} | keys $KEYS | $L"
  cat <<EOF

# ===== MAC SIDE (fill KEY/PORT/HOST; the full session script is 71_mac_demo_session.sh) =====
REPO=$REPO; PY=\$REPO/.venv/bin/python; MAC_ART=\$HOME/Documents/fhe-ssm-backup/mac_art
SESSION=\$HOME/demo_sessions/demo_\$(date -u +%Y%m%dT%H%M%SZ)
cd \$REPO && \$PY ml-eval/fhe_client.py prepare --lanes 64 --prompts-file results/dense-demo-s31/sessions/demo_prompts_64.txt \\
    --bundle-dir \$MAC_ART --tag $TAG --out \$SESSION --dpad 1024
\$PY ml-eval/fhe_client.py generate --out \$SESSION --steps 24 --greedy --keys-dir \$HOME/mac_keys_r17 \\
    --mac-tool hpc_gpu_port/mac_fhe_client --mac-keys \$HOME/mac_keys_r17 \\
    --remote-ssh "ssh -i KEY -p PORT root@HOST" --remote-reqdir $SD \\
    --words-per-lane 8 --stop-on-newline --poll-interval 5 \\
    --harness-cmd "$BIN ${MODEL_FLAGS[*]} ${DEMO_FLAGS[*]} ${BLOCK_FLAGS[*]}"
# STOP (never pkill):  touch $SD/stop      or   bash $DEMO_TOOLS/$SCRIPT stop
EOF
}

do_stop() {
  if ! alive; then log "no live server (pid file: $(cat "$PF" 2>/dev/null || echo none))"; rm -f "$PF"; grep -q SERVER "$DEMO/.busy" 2>/dev/null && rm -f "$DEMO/.busy"; return 0; fi
  PID=$(cat "$PF"); mkdir -p "$SD"; echo stop > "$SD/stop"
  log "stop file written; waiting up to ${STOP_WAIT}s for the serve loop to exit (a tick in flight finishes first)"
  T0=$(date +%s)
  while alive; do
    [ $(( $(date +%s) - T0 )) -ge "$STOP_WAIT" ] && break
    sleep 5
  done
  if alive; then log "still alive after ${STOP_WAIT}s -> kill $PID (by pid, T7-safe)"; kill "$PID"; sleep 30; alive && kill -9 "$PID"; fi
  rm -f "$SD/stop" "$PF"; grep -q SERVER "$DEMO/.busy" 2>/dev/null && rm -f "$DEMO/.busy"
  grep -ao '{"serve":"exit"[^}]*}' "$L" | tail -1
  NS=$(grep -ac '"serve":"served"' "$L"); NF=$(grep -a '"serve":"served"' "$L" | grep -ac '"failed":true')
  vram_table "$L" "$DEMO/server_vram.jsonl" > /dev/null
  print_summary_fields "$L"
  campaign_log "server stop | served $NS failed $NF | $(grep -ao '{"serve":"exit"[^}]*}' "$L" | tail -1) | $L"
  [ "$NF" = 0 ] || { log "WARNING: $NF served request(s) reported failed:true"; return 1; }
}

do_status() {
  if alive; then echo "RUNNING pid $(cat "$PF") since $(sed 's/.*since=//' "$DEMO/.busy" 2>/dev/null)"; else echo "NOT RUNNING"; fi
  echo "served: $(grep -ac '"serve":"served"' "$L" 2>/dev/null)   last: $(grep -a '"serve":' "$L" 2>/dev/null | tail -1 | cut -c1-200)"
  echo "reqMsPerToken: $(grep -ao '"reqMsPerToken":[0-9.e+]*' "$L" 2>/dev/null | cut -d: -f2 | tr '\n' ' ')"
  echo "reqEncPtMs   : $(grep -ao '"reqEncPtMs":[0-9]*' "$L" 2>/dev/null | cut -d: -f2 | tr '\n' ' ')"
  cuda_ok "$L" 2>/dev/null || echo "CUDA/fatal text in log: $(cuda_problems "$L" | tr '\n' ';')"
  ls "$SD" 2>/dev/null | head -8
  vram_table "$L" "$DEMO/server_vram.jsonl" | tail -4
}

case "$MODE" in start) do_start;; stop) do_stop;; status) do_status;; *) echo "usage: $SCRIPT start|stop|status"; exit 2;; esac
