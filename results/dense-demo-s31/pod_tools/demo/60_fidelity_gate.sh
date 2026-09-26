#!/usr/bin/env bash
# 60_fidelity_gate.sh -- THE GO/NO-GO FOR RECORDING: teacher-forced fidelity of
#                        the stateful, multi-lane, block-layout server at 2^17,
#                        through the compressed store.
#
# PURPOSE   spec_decode/fidelity_tick.py, 4 lanes x 8 ticks, every lane fed its
#           own prompt token each tick, encrypted logits vs the plaintext model
#           on the same prefix. This is the first record of the COMPOSITION
#           stateful carry x --lanes-block x store at the demo ring (at 2^15/xd9
#           the circuit itself missed 25% of argmaxes on every layout,
#           CORRECTIONS_20260902.md A14, so that run is not a baseline). It is
#           also the true SERVER-FOOTPRINT fit test (keys-load + --stateful
#           carry + lanes keys + store, which 40's batch run does not exercise)
#           and the host-RAM growth measurement (store entries are keyed by
#           level, gpu_real_model.cu:1299-1300).
# INPUTS    env per _lib.sh; FID_TICKS=8; FID_LANES=4; FID_DEVICE=cpu (the
#           plaintext reference runs on CPU so GPU $DEVICE keeps every byte for
#           the server); FID_TOP1_MIN (default comparisons-2 = 30 of 32);
#           FID_FLIP_MARGIN=0.5; FID_RELERR_MAX=1e-2; FID_RELERR_RECORD=4e-2;
#           FID_PROMPTS (default: built from demo_prompts_64.txt lines 1-8,
#           joined pairwise -- the D10 prompts are exactly 7 tokens each and
#           fidelity_tick caps ticks at the shortest prompt, :114-115).
#           Modes: run (default) | keygen-only | verdict <fidelity.json> | mac-cmd
# OUTPUTS   $KEYS_POD (pod keys, ~46.5 GB: keygen_pod_r17.log), $DEMO/fid_r17/
#           {fidelity.json,logs/server.jsonl,...}, $DEMO/fid_r17_run.log,
#           $DEMO/fid_vram.jsonl
# PASS      GO   : top1Agree >= FID_TOP1_MIN over all comparisons AND every miss
#                  has refTop1Margin < FID_FLIP_MARGIN (noise flips only) AND
#                  relErrRmsWorst <= FID_RELERR_MAX (1e-2) AND the server log has
#                  no CUDA/fatal text and every served request has failed:false.
#           CONDITIONAL (exit 2): top-1 passes but 1e-2 < relErrRmsWorst <=
#                  FID_RELERR_RECORD. R5 conflict recorded: the on-record
#                  single-token STATELESS relLogitErrRms at 2^17/xd12 is
#                  0.0122788 (s31_t1b_store_secure_r17_prov_a2/console.log:17,
#                  pbd430) and 0.0374048 (s31_t1b_store_secure_r17/console.log:17,
#                  pbd430a) -- both ABOVE 1e-2; the 8.5e-4 class in the queue's
#                  pass_criteria was pilot430 (x_clockv2_dense_t1 relLogitErrRmsMean
#                  0.000858542). The stateful branch measured 2.9e-5..9.1e-5 at
#                  2^15 (queue s31_t31_stateful_secure_r17 pass_criteria). Author
#                  decides whether the record class is acceptable for recording.
#           NO-GO otherwise. R3: rms, never max.
# RECORD    fid_r17/fidelity.json (rows + verdict), server.jsonl (served lines
#           with reqMsPerToken, storeBytes growth, vramTrace), keygen_pod_r17.log.
# DURATION  keygen: s31_t31_keys_r17 launched 17:45:34Z, passed 17:47:34Z
#           (pod/state/CAMPAIGN_LOG.md:238,243) ~2 min + 46.5 GB write. Server
#           setup 45,100 ms (fid17_run.log:6) + store load (~61 GB read). Ticks:
#           171,941 ms all-hits (clean:16) PLUS level-drift encodes: at 2^15 the
#           per-request encode went 1,008,032 -> 254,664 -> 61,953 -> ~50,000 ->
#           658,363 (req 6) -> ... -> 0 by req 9 (genRb/logs/server.jsonl); a
#           full 2^17 encode is 2,579,088 ms (clean:11), so budget up to ~1.3 h
#           for 8 ticks if the same drift pattern holds. Measure it here.
set -uo pipefail
SCRIPT=$(basename "$0")
. "$(dirname "$(readlink -f "$0")")/_lib.sh"
: "${FID_TICKS:=8}"; : "${FID_LANES:=4}"; : "${FID_DEVICE:=cpu}"
: "${FID_FLIP_MARGIN:=0.5}"; : "${FID_RELERR_MAX:=1e-2}"; : "${FID_RELERR_RECORD:=4e-2}"
: "${FID_OUT:=$DEMO/fid_r${LOG_RING}}"
# 2026-09-17: the pod key set is ~46.5 GB at 2^17 (evalrot.bin) and ~7.5 GB at 2^15 (demo_smoke/keygen.log
# evalRotKeyBytes 7,525,552,448; lanes set 8.3 GB) -- the "present" / "too small" checks follow the ring.
if [ -z "${KEYS_MIN_BYTES:-}" ]; then if [ "$LOG_RING" -ge 17 ]; then KEYS_MIN_BYTES=40000000000; else KEYS_MIN_BYTES=5000000000; fi; fi
MODE=${1:-run}

verdict() {  # verdict fidelity.json [server.jsonl]
  local F=$1 SL=${2:-}
  [ -s "$F" ] || { log "no $F"; return 1; }
  local NMIN=${FID_TOP1_MIN:-}
  python3 - "$F" "${NMIN:--1}" "$FID_FLIP_MARGIN" "$FID_RELERR_MAX" "$FID_RELERR_RECORD" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); rows = d["rows"]; v = d.get("verdict", {})
nmin, flip, rmax, rrec = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
n = len(rows)
if nmin < 0: nmin = n - 2
agree = sum(1 for r in rows if r["top1"])
hard = [(r["tick"], r["lane"], r["refTop1Margin"]) for r in rows if not r["top1"] and r["refTop1Margin"] >= flip]
noise = [(r["tick"], r["lane"], r["refTop1Margin"]) for r in rows if not r["top1"] and r["refTop1Margin"] < flip]
worst = max(r["relErrRms"] for r in rows); med = sorted(r["relErrRms"] for r in rows)[n // 2]
per_tick = {}
for r in rows: per_tick.setdefault(r["tick"], []).append(r["relErrRms"])
print("FIDGATE:rows:" + json.dumps({"comparisons": n, "top1Agree": agree, "top1Min": nmin, "hardMisses": hard,
      "noiseFlips": noise, "relErrRmsWorst": worst, "relErrRmsMedian": med, "mode": v.get("mode"),
      "relErrRmsWorstPerTick": {t: max(x) for t, x in sorted(per_tick.items())},
      "serveSecPerTick": {r["tick"]: r["serveSec"] for r in rows}}))
ok1 = agree >= nmin and not hard
if ok1 and worst <= rmax:
    print("FIDGATE:verdict:GO"); sys.exit(0)
if ok1 and worst <= rrec:
    print("FIDGATE:verdict:CONDITIONAL -- top-1 passes; relErrRmsWorst %.4g is above the spec's %.0e but inside the on-record 2^17 single-token stateless class (0.0122788 prov_a2:17 / 0.0374048 clean:17). MANUAL CALL before recording." % (worst, rmax)); sys.exit(2)
print("FIDGATE:verdict:NO-GO -- top1 %d/%d (min %d), hard misses %s, relErrRmsWorst %.4g" % (agree, n, nmin, hard, worst)); sys.exit(1)
PY
  local RC=$?
  if [ -n "$SL" ] && [ -s "$SL" ]; then
    cuda_ok "$SL" || { log "SERVER LOG has CUDA/fatal text: $(cuda_problems "$SL" | tr '\n' ';')"; RC=1; }
    grep -a '"serve":"served"' "$SL" | grep -q '"failed":true' && { log "SERVER LOG: a served request failed"; RC=1; }
    echo "served lines: $(grep -ac '"serve":"served"' "$SL")  reqMsPerToken: $(grep -ao '"reqMsPerToken":[0-9.e+]*' "$SL" | cut -d: -f2 | tr '\n' ' ')"
    echo "reqEncPtMs per tick (level-drift encodes): $(grep -ao '"reqEncPtMs":[0-9]*' "$SL" | cut -d: -f2 | tr '\n' ' ')"
    grep -ao '{"storeLoad"[^}]*}' "$SL" | head -1
    local SS; SS=$(summary_line "$SL")
    if [ -n "$SS" ]; then
      python3 - "$(jget "$SS" storeBytes)" "$(grep -ac '"serve":"served"' "$SL")" <<'PY'
import sys
sb = float(sys.argv[1] or 0); n = int(sys.argv[2] or 1); base = 60850962432.0
g = (sb - base) / max(n - 1, 1)
print("STORE GROWTH (host RAM, never evicted): storeBytes %.2f GB after %d ticks; +%.2f GB/tick vs the 60.85 GB single-token record -> 64-tick upper bound %.0f GB (growth stops when every (diag,level) is seen; 2^15 reached 0 new encodes by tick 9)" % (sb/1e9, n, g/1e9, (base + g*63)/1e9))
PY
      echo "server peakVramGB $(jget "$SS" peakVramGB) procPeakVramGB $(jget "$SS" procPeakVramGB) harness $(jget "$SS" harnessSha256)"
    fi
    echo "=== SERVER VRAM TRACE ==="; vram_table "$SL" "$DEMO/fid_vram.jsonl"
  fi
  return $RC
}

keygen_pod() {
  if [ -s "$KEYS_POD/secret.key" ] && [ -s "$KEYS_POD/public.key" ] && [ -s "$KEYS_POD/cryptocontext.bin.dev" ] \
     && [ -s "$KEYS_POD/evalmult.bin" ] && [ "$(file_bytes "$KEYS_POD/evalrot.bin")" -gt "$KEYS_MIN_BYTES" ]; then
    log "pod keys present in $KEYS_POD (evalrot $(file_bytes "$KEYS_POD/evalrot.bin") B) -- keygen skipped"; return 0; fi
  require_bin; require_file "$ART/bundle_${TAG}.bin" "20_stage_artifacts.sh"
  mkdir -p "$KEYS_POD"; local L="$DEMO/keygen_pod_r17.log"
  log "minting the POD key set at the demo flags (+lanes keys) -- precedent demo_smoke/keygen_lanes.log; ~2 min + 46.5 GB disk"
  { echo "=== KEYGEN POD $(utc) ==="; echo "CMD: $BIN ${MODEL_FLAGS[*]} ${DEMO_FLAGS[*]} ${BLOCK_FLAGS[*]} --keys-dir $KEYS_POD --keys-save --selftest"; } > "$L"
  # shellcheck disable=SC2086
  "$BIN" "${MODEL_FLAGS[@]}" "${DEMO_FLAGS[@]}" "${BLOCK_FLAGS[@]}" ${DEVICES_FLAGS:-} --keys-dir "$KEYS_POD" --keys-save --selftest >> "$L" 2>&1
  echo "RC=$? (4 = T6 soft-fail; ignored)" >> "$L"; ls -la "$KEYS_POD" >> "$L"
  local P=()
  grep -aq '"keysSave":true' "$L" || P+=("no keysSave line")
  grep -aq '"selftestSummary":true,"hardFail":0' "$L" || P+=("selftest hardFail != 0 (with the secret loaded the probes are real): $(grep -a selftestSummary "$L" | tail -1)")
  cuda_ok "$L" || P+=("CUDA/fatal: $(cuda_problems "$L" | tr '\n' ';')")
  [ "$(file_bytes "$KEYS_POD/evalrot.bin")" -gt "$KEYS_MIN_BYTES" ] || P+=("evalrot.bin too small: $(file_bytes "$KEYS_POD/evalrot.bin") B")
  if [ ${#P[@]} -gt 0 ]; then for p in "${P[@]}"; do log "KEYGEN PROBLEM: $p"; done; campaign_log "FAIL keygen_pod | ${P[*]} | $L"; return 1; fi
  log "KEYGEN OK: $(grep -ao '{"keysSave"[^}]*}' "$L")"; campaign_log "PASS keygen_pod | $(grep -ao '{"keysSave"[^}]*}' "$L") | $L"
}

build_prompts() {
  if [ -n "${FID_PROMPTS:-}" ]; then [ -s "$FID_PROMPTS" ] || { log "FID_PROMPTS $FID_PROMPTS missing"; exit 2; }; return; fi
  local SRC="$DEMO/demo_prompts_64.txt"; [ -s "$SRC" ] || SRC="$POD_SRC/results/dense-demo-s31/sessions/demo_prompts_64.txt"
  require_file "$SRC" "20_stage_artifacts.sh"
  FID_PROMPTS="$DEMO/fid_prompts${FID_LANES}.txt"
  python3 - "$SRC" "$FID_LANES" "$FID_PROMPTS" <<'PY'
import sys
lines = [l.strip() for l in open(sys.argv[1]) if l.strip()]
NL = int(sys.argv[2])
out = [lines[2*i] + " " + lines[2*i+1] for i in range(NL)]   # deterministic: lines 1-2, 3-4, ... of the D10 pool
open(sys.argv[3], "w").write("\n".join(out) + "\n")
print("\n".join(out))
PY
  # token-length check with the venv tokenizer (fidelity_tick caps ticks at the shortest prompt)
  "$VENV/bin/python" - "$FID_PROMPTS" "$FID_TICKS" <<'PY' || exit 2
import sys, os
sys.path.insert(0, os.environ.get("POD_SRC", "/root/src") + "/ml-eval")
import fhe_client as FC
tok = FC.get_tokenizer()
ls = [len(tok.encode(l.rstrip("\n"))) for l in open(sys.argv[1]) if l.strip()]
print("prompt token lengths:", ls)
if min(ls) < int(sys.argv[2]):
    print("fatal: shortest prompt has %d tokens < FID_TICKS %s" % (min(ls), sys.argv[2])); sys.exit(2)
PY
}

case "$MODE" in
keygen-only) keygen_pod; exit $? ;;
verdict) verdict "${2:?fidelity.json}" "${3:-}"; exit $? ;;
mac-cmd)
  cat <<EOF
# MAC-KEYS MODE (fidelity_tick.py:33-43): the same probe THROUGH the running demo server on the Mac keys,
# enc/dec on the Mac. Run AFTER 70_demo_server.sh start, from the Mac (fill KEY/PORT/HOST):
REPO=$REPO; MAC_ART=\$HOME/Documents/fhe-ssm-backup/mac_art   # 71 builds MAC_ART
cd \$REPO && .venv/bin/python spec_decode/fidelity_tick.py --out \$HOME/demo_sessions/fid_mac_\$(date -u +%Y%m%dT%H%M%SZ) \\
  --tag $TAG --prompts-file results/dense-demo-s31/pod_tools/demo/fid_prompts4.example.txt --ticks $FID_TICKS \\
  --bundle-dir \$MAC_ART --art-dir \$MAC_ART --device cpu \\
  --mac-tool hpc_gpu_port/mac_fhe_client --mac-keys \$HOME/mac_keys_r17 \\
  --remote-ssh "ssh -i KEY -p PORT root@HOST" --remote-reqdir $DEMO/serve --poll-interval 5
# then judge it with the same rule:  bash $DEMO_TOOLS/$SCRIPT verdict <out>/fidelity.json
# and RESTART the server before the session (its request counter continued past the probe's requests;
# the client numbers from 0 -- fhe_client.py:446-450):  bash $DEMO_TOOLS/70_demo_server.sh stop && ... start
EOF
  exit 0 ;;
run) ;;
*) echo "usage: $SCRIPT [run|keygen-only|verdict F [server.jsonl]|mac-cmd]"; exit 2 ;;
esac

require_bin; require_file "$ART/bundle_${TAG}.bin" "20_stage_artifacts.sh"; require_file "$ART/${TAG}_weights.npz" "20_stage_artifacts.sh (the plaintext reference needs the npz)"
[ -f "$STORE" ] || log "WARN: $STORE missing -- the server will BUILD the store on request 0 (~1 h, clean:11 3,556 s) instead of loading it"
busy_acquire
keygen_pod || { busy_release; exit 1; }
build_prompts
RL="$DEMO/fid_r17_run.log"
[ -e "$FID_OUT" ] && mv "$FID_OUT" "$FID_OUT.$(date -u +%Y%m%dT%H%M%SZ).prev"   # never destroy a record
mkdir -p "$FID_OUT"
# DEVICES_FLAGS (2026-09-03): the two-card route passes "--devices 0,1 --store-multi-gpu"
# through 41_mgpu_ladder.sh rung 4; empty on a single card.
HC="$BIN ${MODEL_FLAGS[*]} ${DEMO_FLAGS[*]} ${BLOCK_FLAGS[*]} ${DEVICES_FLAGS:-} --store-file $STORE --vram-trace"
{ echo "=== FIDELITY GATE $(utc) ==="; record_load_state; echo "HARNESS_CMD: $HC"; echo "prompts: $FID_PROMPTS"; } > "$RL"
gpu_idle_or_warn || echo "LOADSTATE CONTENDED" >> "$RL"
T0=$(date +%s)
( cd "$POD_SRC" && "$VENV/bin/python" spec_decode/fidelity_tick.py --out "$FID_OUT" --tag "$TAG" \
    --prompts-file "$FID_PROMPTS" --ticks "$FID_TICKS" --device "$FID_DEVICE" --keys-dir "$KEYS_POD" \
    --bundle-dir "$ART" --art-dir "$ART" --serve-timeout 7200 --poll-interval 2 ${FID_EXTRA_ARGS:-} \
    --harness-cmd "$HC" ) >> "$RL" 2>&1   # FID_EXTRA_ARGS (2026-09-18): e.g. "--free-run" (autoregressive long run), word-split on purpose
echo "RC=$? WALL_S=$(( $(date +%s) - T0 ))" >> "$RL"
echo "=== VERDICT ===" | tee -a "$RL"
verdict "$FID_OUT/fidelity.json" "$FID_OUT/logs/server.jsonl" | tee -a "$RL"
RC=${PIPESTATUS[0]}
V=$(grep -ao 'FIDGATE:verdict:[A-Z-]*' "$RL" | tail -1)
ROWS=$(grep -ao 'FIDGATE:rows:.*' "$RL" | tail -1 | cut -c14-400)
case "$RC" in
  0) echo "=== GO FOR RECORDING (pod keys). Next: 70_demo_server.sh start on the MAC keys, then the mac-mode probe ($SCRIPT mac-cmd). ===";;
  2) echo "=== CONDITIONAL -- author decides (see header). ===";;
  *) echo "=== NO-GO -- do not record. Attribute: compare against a stateless control (fidelity_tick.py --stateless) before touching the circuit. ===";;
esac
campaign_log "$V | $ROWS | wall $(( $(date +%s) - T0 ))s | $FID_OUT/fidelity.json"
busy_release; exit "$RC"
