#!/usr/bin/env bash
# 92_s37_levers.sh -- the lever program of SESSION_SINGLE_RUNSHEET_20260917.md section 2, as ONE sequential chain on
#                     the pod (pod test keys, pod-side driver 91_s37_serve.sh; the Mac is not in the loop).
# usage:  . /root/demo/s37.env; ARMS="L0 L1 L2 L3 L4 L5 L6 L6s L7 L8 L8s L9" bash 92_s37_levers.sh
#         (ARMS env picks/reorders; default = all, payoff first). Each arm records under $DEMO/s37/; an arm's fresh
#         store is deleted after its TICKS.md exists (L1's store is kept: L6 reuses it). One line per arm in
#         $DEMO/s37/LEVERS.md with the verdict lines from TICKS.md / FIELDS.txt. Stops only on a missing prerequisite
#         (binary, plain store, pod keys, artifacts); a failed arm is recorded and the chain continues (cross-out rule).
set -uo pipefail
SCRIPT=$(basename "$0")
. "$(dirname "$(readlink -f "$0")")/_lib.sh"
D="$(dirname "$(readlink -f "$0")")"
: "${ARMS:=L0 L1 L2 L3 L4 L5 L6 L6s L7 L8 L8s L9}"; : "${S37:=$DEMO/s37}"; : "${UD_C:=28}"
: "${BINX:=$WORK/build-demo/gpu_real_model_x}"
PR4="$POD_SRC/results/dense-demo-s31/sessions/fid_prompts4_identical.txt"
PR4L="$POD_SRC/results/dense-demo-s31/sessions/fid_prompts4_identical_long.txt"
require_bin; [ -x "$BINX" ] || { log "no experimental binary $BINX"; exit 2; }
[ -f "$STORE" ] || { log "no plain store $STORE (run the store_plain cell first)"; exit 2; }
[ -s "$KEYS_POD/secret.key" ] || { log "no pod test keys in $KEYS_POD (60_fidelity_gate.sh keygen-only)"; exit 2; }
mkdir -p "$S37"; LV="$S37/LEVERS.md"
[ -s "$LV" ] || printf '# lever program -- %s -- binary real %s / copy %s\n\n| arm | flags | enc0From | bootsConst | lanesIdentical | fidelity | boots/ticks | store | note |\n|---|---|---|---|---|---|---|---|---|\n' "$(utc)" "$(sha256sum "$BIN" | cut -c1-16)" "$(sha256sum "$BINX" | cut -c1-16)" > "$LV"
PF="--schedule parent-first --newton-depth2 --trace-boots"
serve_arm() {   # serve_arm NAME XBIN TICKS PROMPTS STORE_OR_EMPTY FLAGS...
  local NAME=$1 XB=$2 TK=${TICKS_OVERRIDE:-$3} PRM=$4 ST=$5; shift 5; local FL="$*"
  local EFX="${EFO-$EXTRA_FLAGS}"       # EFO="" serve_arm ... drops the s37.env server flags (the drift control)
  log "=== $NAME (xbin=$XB ticks=$TK) flags: $FL ==="
  if [ -n "$ST" ]; then EXTRA_FLAGS="$EFX" ARM="$NAME" XBIN="$XB" TICKS="$TK" PROMPTS="$PRM" STORE="$ST" ARM_FLAGS="$FL" bash "$D/91_s37_serve.sh"
  else
    # FIX 2026-09-18 00:10Z: STORE is exported by s37.env, so 91's default never applied and every schedule arm was
    # pointed at the base store and refused by the stamp. A fresh-store arm gets its OWN path and the V5 periodic
    # encoder (verified 344064/344064 on the base store) so its cold tick 0 costs ~4 min of encode, not ~50.
    EXTRA_FLAGS="$EFX" ARM="$NAME" XBIN="$XB" TICKS="$TK" PROMPTS="$PRM" STORE="$S37/serve_$NAME/store.bin" ARM_FLAGS="$FL --periodic-encode --periodic-verify-every 64" bash "$D/91_s37_serve.sh"; fi
  local T="$S37/serve_$NAME/TICKS.md"
  local e0 bc li fi bt sb
  e0=$(grep -o 'VERDICT enc0From: [^ ]*' "$T" 2>/dev/null | cut -d' ' -f3); bc=$(grep -o 'VERDICT bootsConst: [^ ]*' "$T" 2>/dev/null | cut -d' ' -f3)
  li=$(grep -o 'VERDICT lanesIdentical: [^ ]*' "$T" 2>/dev/null | cut -d' ' -f3); fi=$(grep -o 'VERDICT fidelity: top1 [^;]*' "$T" 2>/dev/null | cut -c19-)
  bt=$(grep -o '"reqBoots": [0-9]*' "$S37/serve_$NAME/logs/server.jsonl" 2>/dev/null | cut -d' ' -f2 | tr '\n' '/' ); sb=$(grep -o '"storeBytes": [0-9]*' "$T" 2>/dev/null | cut -d' ' -f2)
  printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' "$NAME" "$FL" "${e0:-?}" "${bc:-?}" "${li:-?}" "${fi:-?}" "${bt:-?}" "${sb:-?}" "$(grep -c '"fatal"' "$S37/serve_$NAME/logs/server.jsonl" 2>/dev/null) fatal" >> "$LV"
  if [ -z "$ST" ] && [ -s "$T" ] && [ "$NAME" != L1 ]; then rm -f "$S37/serve_$NAME/store.bin"; log "store of $NAME deleted (record kept)"; fi
}
for a in $ARMS; do
  case "$a" in
    L0)  CELL=V2_timers STOREF="$STORE" PASSES=2 bash "$D/90_s37_cell.sh"; printf '| L0 | V2 cell (batch, --passes 2) | | | | %s | | | FIELDS.txt |\n' "$(grep -o 'fidelity .*' "$S37/V2_timers/FIELDS.txt" 2>/dev/null | cut -c1-80)" >> "$LV";;
    L1)  serve_arm L1 1 8 "$PR4" "$STORE" "";;
    L1r) serve_arm L1r 1 3 "$PR4" "$STORE" "";;                                  # 2026-09-18: L1 repeated (copy, base flags, base store): reproducible or stochastic?
    D2)  EFO="" serve_arm D2_real_drift 0 3 "$PR4" "" "";;                       # 2026-09-18: DRIFT control (no carry boots), real binary, fresh store
    D3)  serve_arm D3_x_udrop 1 4 "$PR4" "" $PF --u-drop-level "$UD_C";;
    D4)  serve_arm D4_x_udrop_stock 1 4 "$PR4" "" --u-drop-level "$UD_C" --trace-boots;;   # 2026-09-18: STOCK schedule + canonical carry, only the u carry by the exact drop (24 carry boots, not 48)        # 2026-09-18: the X1 u exact drop as a CARRY candidate (L4 at 4 ticks)
    L2)  serve_arm L2 1 8 "$PR4" "" $PF;;
    L3)  serve_arm L3 1 20 "$PR4L" "" $PF --deferred-scan 16;;
    L4)  serve_arm L4 1 8 "$PR4" "" $PF --u-drop-level "$UD_C";;
    L5)  serve_arm L5 1 20 "$PR4L" "" $PF --deferred-scan 16 --u-drop-level "$UD_C" --branch-out-boot --branch-out-post 19;;
    L6)  serve_arm L6 1 4 "$PR4" "$STORE" --pt-cache --pt-cache-verify --pt-cache-max-gb 8;;
    L6s) serve_arm L6s 1 4 "$PR4" "$STORE" --pt-cache --pt-scalar --pt-cache-max-gb 8;;
    L7)  serve_arm L7 1 10 "$PR4" "" --schedule stock --drop-wv 7 --drop-wkr 4 --drop-wout 7 --drop-win 4 --trace-boots;;
    L8)  serve_arm L8 1 10 "$PR4" "" --schedule stock --trace-boots;;
    L8s) serve_arm L8s 1 10 "$PR4" "" --schedule stock --drop-wv 7 --drop-wkr 4 --drop-wout 7 --drop-win 4 --rsqrt-iters-sidecar "$POD_SRC/results/s37-impl-20260905/X8_drops_sidecar/rsqrt_iters_Im2.json";;
    L9)  serve_arm L9 0 4 "$PR4" "$STORE" --no-norm-eps; serve_arm L9e 0 4 "$PR4" "$STORE" "";;
    *) log "unknown arm $a";;
  esac
done
log "=== LEVERS DONE: $LV ==="; cat "$LV"
campaign_log "s37 levers done | $ARMS | $LV"
