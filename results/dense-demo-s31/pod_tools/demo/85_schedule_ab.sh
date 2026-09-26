#!/usr/bin/env bash
# 85_schedule_ab.sh -- S3.3 (2026-09-03): the schedule-lever A/B on ONE binary.
#                      NEW, OPTIONAL script; the gated demo scripts 00..80 are untouched (I10).
#
# PURPOSE   Flip the S3.3 flags of hpc_gpu_port/gpu_real_model.cu one at a time,
#           on the SAME binary and in the SAME load state, and judge each flip
#           against the replay's prediction and the stock arm:
#             --schedule parent-first      (P1/P2/P4 refresh placements)
#             --share-babies               (baby rotations shared across the 2K wk/wr calls)
#             --newton-depth2              (Tier A-2 re-association; FIDELITY-gated)
#             --pool-levels 4              (compressed-store pool LRU 2 -> 4; VRAM-gated)
#             all four together
#           Predictions, rules and the precision argument:
#           results/theory/SCHEDULE_DESIGN_20260903.md; the replay that produces
#           the per-configuration prediction here is results/theory/tick_level_trace.py.
#
# MODES     quick  : ring 2^15 STRUCTURAL (--extra-depth 9, the S3.1 2^15 class), one
#                    token, --passes 2, every arm; ~10-15 min per arm. Boot counts,
#                    fidelity and wvCheck are correctness fields and cross rings
#                    (the R6 rule (never compare a timing across binaries or machine load states)); the replay predicts THIS configuration exactly.
#           secure : the demo configuration (2^17, --secure, xd12), one token,
#                    --passes 2: stock (loads $STORE) vs the arms named in ARMS
#                    (each non-stock arm BUILDS its own in-memory store on pass 1,
#                    ~1 h: clean:11 3,556 s). Then the lane cells and the
#                    >= 16-tick stateful session with ALL flags (SESSION_TICKS).
#           keys   : (I7) dump the rotation contract with and without the flags and
#                    diff -- must be byte-identical. No GPU. Run first.
#           lanes  : --lane-test-block with all flags, then the T1c full-pass lane
#                    equality (needs bundle_${TAG}lanes4.bin; prints the export
#                    command if absent).
#           session: the stateful session alone (all flags), SESSION_TICKS ticks.
#           all    : keys, quick, lanes, session (secure T=1 arms are opt-in: SECURE_ARMS=1).
#
# GATES     (written to $DEMO/schedule_ab/SUMMARY.md; every gate names its record)
#   G1 boots      : pass-1 boots within +-10 % of the replay's tick-0 prediction for the
#                   SAME flags (lazy model; eager printed beside it). Outside -> the
#                   replay or the harness is wrong: read the bootTrace site histogram
#                   ($DEMO/schedule_ab/<arm>.sites) against SCHEDULE_DESIGN section 5
#                   and say which.
#   G2 top1Agree  : equal to the stock arm's.
#   G3 relErrRms  : fidelity.relLogitErrRmsMean within 2.73 % (relative) of the stock
#                   arm's (MEASUREMENTS.md R3: rms reproduces to 2.73 %; never max).
#   G4 wvCheck    : layer-0 wvCheck relErr (max over chunks, rms is not emitted there)
#                   no worse than stock's x 1.5 for --schedule parent-first
#                   (operands at 10-16 limbs instead of 8-23, design section 7).
#   G5 rotations  : --share-babies: babyShared == 7 x 24 = 168 per pass (2K-1 reuses per
#                   layer) and babyComputed == 24 + 2 x 24 (u2 once, win, wout) + 4 x 24 (wv)
#                   per pass; syncSplit.babyRotMs lower than stock (same binary).
#   G6 store      : session: reqEncPtMs reaches 0 and storeMisses stops growing by tick
#                   <= 15 (replay: last new key at tick 12-15 under parent-first, 19 stock).
#   G7 flat       : session: reqMsPerToken of the last 4 ticks within +-10 % of their
#                   mean; peakVramGB flat after tick 2; store size flat after warm-up (I1).
#   G8 keys       : contract JSONs identical with and without the flags (I7).
#   G9 lanes      : laneTestBlock pass:true with all flags; full-pass gate PASS.
#   G10 pool      : --pool-levels 4: poolBuilds/pass <= 20 (replay: 10.3 stock, 14.6 pf)
#                   vs ~90 at LRU 2; poolBuildMs reported (its share of the tick was
#                   never measured before this cell).
#   Every arm: no CUDA/fatal text in its log (F74: the exit code is not evidence),
#   summary failed:false.
#
# INPUTS    env per _lib.sh; ARMS (default "pf share nd2 pool all"); SESSION_TICKS=16;
#           SECURE_ARMS=1 to run the 2^17 T=1 arms in `all`; POOL_LEVELS=4.
#           Pod keys are NOT needed for the T=1 arms (in-process keygen, canDecrypt);
#           the session needs $KEYS_POD (60_fidelity_gate.sh keygen-only).
# OUTPUTS   $DEMO/schedule_ab/{keys_*.json, <mode>_<arm>.log, <arm>.sites,
#           session_all/, SUMMARY.md}; one CAMPAIGN_LOG line per arm.
# RULES     R6: one binary, one box, nothing else on the GPU (busy marker + load state
#           recorded); no CPU-heavy rider. F74: verdicts from log contents. T7: no pkill.
#           I5: no flag here moves the ring or the depth; --scale-bits is never set.
set -uo pipefail
SCRIPT=$(basename "$0")
. "$(dirname "$(readlink -f "$0")")/_lib.sh"
: "${ARMS:=pf share nd2 pool all}"; : "${SESSION_TICKS:=16}"; : "${POOL_LEVELS:=4}"; : "${SECURE_ARMS:=0}"
: "${AB:=$DEMO/schedule_ab}"; : "${SEEDS:=$ART/${TAG}_seeds.json}"
MODE=${1:-all}
mkdir -p "$AB"
SUM="$AB/SUMMARY.md"
REPLAY="$POD_SRC/results/theory/tick_level_trace.py"

arm_flags() {   # arm name -> harness flags
  case "$1" in
    stock) echo "";;
    pf)    echo "--schedule parent-first";;
    share) echo "--share-babies";;
    nd2)   echo "--newton-depth2";;
    pool)  echo "--pool-levels $POOL_LEVELS";;
    pfnd2) echo "--schedule parent-first --newton-depth2";;
    all)   echo "--schedule parent-first --share-babies --newton-depth2 --pool-levels $POOL_LEVELS";;
    *) echo "unknown arm $1" >&2; return 2;;
  esac
}
replay_flags() {   # arm name -> replay flags (only the trace-changing ones)
  case "$1" in
    pf) echo "--schedule parent-first";; nd2) echo "--newton-depth2";;
    pfnd2|all) echo "--schedule parent-first --newton-depth2";; pool) echo "--pool-levels $POOL_LEVELS";; *) echo "";;
  esac
}
# replay prediction for a configuration: prints "cold=<tick0> warm=<mean 8..> min max period" for lazy and eager
predict() {   # predict DEPTH LAM ARM TICKS
  local D=$1 LAM=$2 ARM=$3 TICKS=$4 RF; RF=$(replay_flags "$ARM")
  require_file "$SEEDS" "the bundle's Newton sidecar (20_stage_artifacts.sh)"
  for m in "--lazy" ""; do
    # shellcheck disable=SC2086
    local LB="--lanes-block"; [ "${QUICK_H1:-0}" = 1 ] && [ "$D" != "$EXPECT_DEPTH" ] && LB=""   # h = 1 in the quick arms under QUICK_H1
    # REPLAY_D/REPLAY_DFF/REPLAY_L/REPLAY_SHIFT (2026-09-03): the model shape for the
    # prediction (defaults = pbd430a; the anchor agnd768b is 768/3072/12, shift-mix on).
    local SM="--shift-mix"; [ "${REPLAY_SHIFT:-1}" = 0 ] && SM=""
    python3 "$REPLAY" --depth "$D" --lam "$LAM" --boot-floor "$BOOT_FLOOR" --margin 4 --dpad 1024 --d "${REPLAY_D:-1024}" --dff "${REPLAY_DFF:-4096}" \
      --layers "${REPLAY_L:-24}" --seeds "$SEEDS" $LB $SM --ticks "$TICKS" $m $RF --json "$AB/predict_${ARM}_${D}${m:-_eager}.json" > /dev/null
    python3 - "$AB/predict_${ARM}_${D}${m:-_eager}.json" "${m:-eager}" <<'PY'
import json, sys
o = json.load(open(sys.argv[1])); pt = o["per_tick"]; w = pt[8:] if len(pt) > 8 else pt
b = [t["boots"] for t in w]
print("PREDICT %s cold=%d warm=%.1f min=%d max=%d ticks=%s pool_builds=%d rot=%d" % (
    sys.argv[2].lstrip("-"), pt[0]["boots"], sum(b)/len(b), min(b), max(b), [t["boots"] for t in pt[:16]], pt[-1]["pool_builds"], pt[-1]["rot"]))
PY
  done
}
sites_hist() {   # bootTrace stage histogram from a log (site names as in the replay: strip the layer/token prefix)
  grep -a '"bootTrace":true' "$1" | python3 -c '
import sys, json, re, collections
c = collections.Counter()
for l in sys.stdin:
    try: d = json.loads(l)
    except Exception: continue
    s = d.get("stage", "")
    s = re.sub(r"^L\d+\.", "", s); s = re.sub(r"\.t\d+$", "", s); s = re.sub(r"\.ch\d+", ".chX", s)
    c[s] += 1
for k, v in c.most_common(40): print("%6d  %s" % (v, k))
'
}
jnum() { jget "$1" "$2"; }

run_arm() {   # run_arm MODE ARM  (T = 1, --passes 2)
  local M=$1 A=$2 F L; F=$(arm_flags "$A") || return 2; L="$AB/${M}_${A}.log"
  local RING_FLAGS DEP LAM STOREF=""
  if [ "$M" = quick ]; then
    RING_FLAGS="--log-ring 15 --level-budget 3 3 --ms-norm 5.5 --enc-threads $ENC_THREADS --boot-floor $BOOT_FLOOR --matvec-margin 4 --extra-depth 9"
    DEP=$((10 + 19 + 9)); LAM=$((DEP - 19))
    # Store files per LEVEL TRACE (2026-09-03 pod): the store-build pass is the
    # cost of an arm (~10-15 min at 2^15 on 24 encode threads). --share-babies and
    # --pool-levels do not change the trace, so they LOAD the stock arm's file
    # (their stamps are identical by design, I6); trace-changing arms get their own.
    case "$A" in stock|share|pool) STOREF="--store-file $AB/store_${M}_stock.bin";; *) STOREF="--store-file $AB/store_${M}_${A}.bin";; esac
    # QUICK_H1=1 (2026-09-03 pod, 62 GB host, slow encode): drop --lanes-block so the
    # store-build pass encodes 344,064 halves instead of 687,792 (the S3.2 sanity
    # record's shape, h = 1). The replay is run with the matching h.
    if [ "${QUICK_H1:-0}" = 1 ]; then BF=(--pack-tokens 1 --compressed-store); else BF=("${BLOCK_FLAGS[@]}"); fi
  else
    BF=("${BLOCK_FLAGS[@]}")
    RING_FLAGS="${DEMO_FLAGS[*]}"; DEP=$EXPECT_DEPTH; LAM=$((DEP - 19))
    [ "$A" = stock ] && [ -f "$STORE" ] && STOREF="--store-file $STORE"   # non-stock stamps differ: they build their own store
  fi
  { echo "=== ARM $A ($M) $(utc) ==="; record_load_state
    echo "CMD: $BIN ${MODEL_FLAGS[*]} $RING_FLAGS ${BF[*]} --tokens 1 --passes 2 --trace-boots --sync-timers --proc-vram $STOREF $F"; } > "$L"
  gpu_idle_or_warn || echo "LOADSTATE CONTENDED" >> "$L"
  # shellcheck disable=SC2086
  "$BIN" "${MODEL_FLAGS[@]}" $RING_FLAGS "${BF[@]}" --tokens 1 --passes 2 --trace-boots --sync-timers --proc-vram $STOREF $F >> "$L" 2>&1
  echo "RC=$? $(utc)" >> "$L"
  sites_hist "$L" > "$AB/${M}_${A}.sites"
  predict "$DEP" "$LAM" "$A" 8 | tee -a "$L"
  cuda_ok "$L" || log "ARM $A: CUDA/fatal text: $(cuda_problems "$L" | tr '\n' ';')"
  local S; S=$(summary_line "$L")
  # per-pass boots from the passTiming lines (pass 1 = cold tick 0; pass 2 = another cold tick, same count expected)
  echo "PASSBOOTS: $(grep -ao '"passTiming"[^}]*' "$L" | grep -ao '"boots":[0-9]*' | cut -d: -f2 | tr '\n' ' ')" | tee -a "$L"
  echo "SUMMARY: schedule=$(jnum "$S" schedule) pfEntry=$(jnum "$S" pfEntry) shareBabies=$(jnum "$S" shareBabies) newtonDepth2=$(jnum "$S" newtonDepth2) poolLevels=$(jnum "$S" poolLevels) boots=$(jnum "$S" boots) pfBoots=$(jnum "$S" pfBoots) babyShared=$(jnum "$S" babyShared) babyComputed=$(jnum "$S" babyComputed) poolBuilds=$(jnum "$S" poolBuilds) poolBuildMs=$(jnum "$S" poolBuildMs) top1Agree=$(jnum "$S" fidelity.top1Agree) relLogitErrRmsMean=$(jnum "$S" fidelity.relLogitErrRmsMean) failed=$(jnum "$S" failed) peakVramGB=$(jnum "$S" peakVramGB) storeHits=$(jnum "$S" storeHits) storeMisses=$(jnum "$S" storeMisses) msPerToken=$(jnum "$S" msPerToken) wallMsPerToken=$(jnum "$S" wallMsPerToken) syncSplit=$(jnum "$S" syncSplit) harness=$(jnum "$S" harnessSha256)" | tee -a "$L"
  echo "WVCHECK(layer0): $(grep -ao '{"wvCheck":true,"layer":0[^}]*}' "$L" | grep -ao '"relErr":[0-9.e+-]*' | cut -d: -f2 | tr '\n' ' ')" | tee -a "$L"
  campaign_log "arm $A ($M) | boots $(jnum "$S" boots) | top1 $(jnum "$S" fidelity.top1Agree) rms $(jnum "$S" fidelity.relLogitErrRmsMean) | failed $(jnum "$S" failed) | $L"
}

judge() {   # judge MODE  -> SUMMARY.md section, from the arm logs (records only)
  local M=$1
  python3 - "$AB" "$M" "$ARMS" "$POOL_LEVELS" >> "$SUM" <<'PY'
import json, sys, re, os, glob
AB, M, ARMS, POOL = sys.argv[1], sys.argv[2], sys.argv[3].split(), int(sys.argv[4])
def summ(path):
    S = None
    for l in open(path, errors="replace"):
        if l.startswith('{"summary":true'):
            try: S = json.loads(l)
            except Exception: pass
    return S
def passboots(path):
    out = []
    for l in open(path, errors="replace"):
        if '"passTiming"' in l:
            m = re.search(r'"boots":(\d+)', l)
            if m: out.append(int(m.group(1)))
    return out
def pred(path, model):
    for l in open(path, errors="replace"):
        if l.startswith("PREDICT " + model):
            m = re.search(r"cold=(\d+) warm=([\d.]+)", l); return int(m.group(1)), float(m.group(2))
    return None, None
def wv(path):
    v = [float(x) for x in re.findall(r'"wvCheck":true,"layer":0[^}]*"relErr":([0-9.e+-]+)', open(path, errors="replace").read())]
    return max(v) if v else None
def bad(path):
    t = open(path, errors="replace").read()
    return bool(re.search(r'out of memory|Cuda failure|cudaError|CUDA_ERROR|map::at|out_of_range|terminate called|"fatal"', t, re.I))
print(f"\n## {M} arms (T = 1, --passes 2, one binary)\n")
print("| arm | boots pass1/pass2 | replay lazy cold / warm | replay eager cold | G1 (+-10 % of lazy cold) | top1 | relErrRms | G2/G3 vs stock | wvCheck L0 max | G4 | babyShared / computed | poolBuilds / ms | failed | CUDA text | harness |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
stock = summ(f"{AB}/{M}_stock.log") if os.path.exists(f"{AB}/{M}_stock.log") else None
swv = wv(f"{AB}/{M}_stock.log") if stock else None
for a in ["stock"] + ARMS:
    p = f"{AB}/{M}_{a}.log"
    if not os.path.exists(p): print(f"| {a} | (not run) |||||||||||||"); continue
    S = summ(p); pb = passboots(p); lc, lw = pred(p, "lazy"); ec, ew = pred(p, "eager")
    if S is None: print(f"| {a} | NO SUMMARY LINE -- read the log | | | FAIL | | | | | | | | | {bad(p)} | |"); continue
    b1 = pb[0] if pb else S.get("boots")
    g1 = "n/a" if lc is None else ("PASS" if abs(b1 - lc) <= 0.10 * lc else f"FAIL ({b1} vs {lc})")
    fid = S.get("fidelity", {}); t1 = fid.get("top1Agree"); rms = fid.get("relLogitErrRmsMean")
    if stock and a != "stock":
        st1 = stock["fidelity"]["top1Agree"]; srms = stock["fidelity"]["relLogitErrRmsMean"]
        g23 = ("PASS" if t1 == st1 else f"FAIL top1 {t1} vs {st1}") + " / " + ("PASS" if srms and abs(rms - srms) <= 0.0273 * srms else f"FAIL rms {rms} vs {srms}")
    else: g23 = "(stock)"
    w = wv(p); g4 = "(stock)" if a == "stock" else ("n/a" if (w is None or swv is None) else ("PASS" if w <= 1.5 * swv else f"FAIL {w:.3g} vs stock {swv:.3g}"))
    g5 = f"{S.get('babyShared')} / {S.get('babyComputed')}"
    pool = f"{S.get('poolBuilds')} / {S.get('poolBuildMs')}"
    print(f"| {a} | {'/'.join(map(str, pb)) or S.get('boots')} | {lc} / {lw} | {ec} | {g1} | {t1} | {rms} | {g23} | {w} | {g4} | {g5} | {pool} | {S.get('failed')} | {bad(p)} | {S.get('harnessSha256')} |")
print("\nSite histograms: <arm>.sites beside each log; compare with SCHEDULE_DESIGN_20260903.md section 5 / trace_schedule_20260903.txt section B.")
print("R6: milliseconds above are comparable only within this table (one binary, load state in each log's LOADSTATE lines).")
PY
}

mode_keys() {   # I7: the rotation contract must not change
  require_bin; local K0="$AB/keys_stock.json" K1="$AB/keys_all.json" L="$AB/keys.log"
  # KEYS_RING/KEYS_XD (2026-09-03 pod): --dump-rot-indices mints the full key set in HOST
  # RAM; at 2^17 that is ~50 GB and was OOM-killed on a 62 GB box. The invariant
  # (flags add no index) is ring-independent, so it may be checked at 2^15.
  local KF=("${DEMO_FLAGS[@]}")
  if [ -n "${KEYS_RING:-}" ]; then KF=(--log-ring "$KEYS_RING" --level-budget 3 3 --ms-norm 5.5 --enc-threads "$ENC_THREADS" --boot-floor "$BOOT_FLOOR" --matvec-margin 4 --extra-depth "${KEYS_XD:-9}"); fi
  { echo "=== KEYS $(utc) ring flags: ${KF[*]} ==="; } > "$L"
  # shellcheck disable=SC2086
  "$BIN" "${MODEL_FLAGS[@]}" "${KF[@]}" "${BLOCK_FLAGS[@]}" --dump-rot-indices "$K0" >> "$L" 2>&1
  # shellcheck disable=SC2086
  "$BIN" "${MODEL_FLAGS[@]}" "${KF[@]}" "${BLOCK_FLAGS[@]}" $(arm_flags all) --dump-rot-indices "$K1" >> "$L" 2>&1
  if [ -s "$K0" ] && cmp -s "$K0" "$K1"; then log "G8 PASS: rotation contract byte-identical with all flags ($(file_bytes "$K0") B)"; echo "G8 keys: PASS (identical contract, $(file_bytes "$K0") B)" >> "$SUM"; campaign_log "G8 PASS keys identical | $K0"; return 0; fi
  log "G8 FAIL: contract differs or missing -- STOP: a lever needs a new rotation key (invariant I7), report before anything else"; echo "G8 keys: FAIL -- $K0 vs $K1 differ" >> "$SUM"; campaign_log "G8 FAIL keys | $AB/keys.log"; return 1
}

mode_lanes() {
  require_bin; local L="$AB/lanes_test_block_all.log"
  { echo "=== LANE TEST BLOCK (all flags) $(utc) ==="; record_load_state; } > "$L"
  # shellcheck disable=SC2086
  "$BIN" "${MODEL_FLAGS[@]}" "${DEMO_FLAGS[@]}" --lane-test-block --pack-tokens 1 $(arm_flags all) >> "$L" 2>&1
  local P; P=$(grep -ao '{"laneTestBlockSummary"[^}]*}' "$L" | tail -1)
  echo "G9a lane-test-block (all flags): $P" | tee -a "$SUM"
  [ "$(jnum "$P" pass)" = true ] || { log "G9a FAIL"; campaign_log "G9a FAIL lane-test-block | $L"; }
  # full-pass lane equality (T1C_FULLPASS_DRIVER_20260829.md): lanes bundle + 4 broadcast references, same flags
  local LB="$ART/bundle_${TAG}lanes4.bin"
  if [ ! -s "$LB" ]; then
    cat <<EOF | tee -a "$SUM"
G9b full-pass lane equality: SKIPPED -- $LB missing. Export it (needs ${TAG}_weights.npz + config beside the bundle):
  cd $POD_SRC && $VENV/bin/python ml-eval/train_fhe_native_ssm.py export --tag $TAG --tokens 8 --lanes 4 --lane-stride 512 --bundle-suffix lanes4
then re-run: bash $DEMO_TOOLS/$SCRIPT lanes
EOF
    return 0
  fi
  local T=8 LD="$AB/lanes_full_all"; mkdir -p "$LD"
  { echo "=== LANES FULL (all flags) $(utc) ==="; record_load_state; } > "$LD/console.log"
  # shellcheck disable=SC2086
  "$BIN" --tag "${TAG}lanes4" --bundle-dir "$ART" --device "$DEVICE" "${DEMO_FLAGS[@]}" --lanes-full 4 --pack-tokens 1 --tokens $T \
      --dump-logits "$LD/lanes.logits" $(arm_flags all) >> "$LD/console.log" 2>&1
  for r in 0 1 2 3; do
    local RD="$AB/lanes_ref_all_$r"; mkdir -p "$RD"
    # shellcheck disable=SC2086
    "$BIN" --tag "${TAG}lanes4" --bundle-dir "$ART" --device "$DEVICE" "${DEMO_FLAGS[@]}" "${BLOCK_FLAGS[@]}" --tokens $T \
        --bundle-row-offset $((r * T)) --dump-logits "$RD/ref.logits" $(arm_flags all) > "$RD/console.log" 2>&1
  done
  ( cd "$POD_SRC" && python3 campaign/scripts/gate_lanes_fullpass.py --lanes 4 --tokens $T --vocab 50277 \
      --lanes-dir "$LD" --ref-dir-prefix "$AB/lanes_ref_all_" ) > "$AB/lanes_full_gate.log" 2>&1
  echo "G9b full-pass lane gate (all flags): rc=$? $(tail -1 "$AB/lanes_full_gate.log")" | tee -a "$SUM"
}

mode_session() {   # >= SESSION_TICKS stateful ticks with ALL flags, through the pod-side driver (as 60 does)
  require_bin; require_file "$ART/bundle_${TAG}.bin" "20_stage_artifacts.sh"; require_file "$ART/${TAG}_weights.npz" "20_stage_artifacts.sh"
  [ -s "$KEYS_POD/secret.key" ] || { log "pod keys missing: run 60_fidelity_gate.sh keygen-only first"; return 2; }
  local OUT="$AB/session_all" RL="$AB/session_all.log" PR="$AB/session_prompts4.txt"
  [ -e "$OUT" ] && mv "$OUT" "$OUT.$(date -u +%Y%m%dT%H%M%SZ).prev"; mkdir -p "$OUT"
  local SRC="$DEMO/demo_prompts_64.txt"; [ -s "$SRC" ] || SRC="$POD_SRC/results/dense-demo-s31/sessions/demo_prompts_64.txt"; require_file "$SRC" "20_stage_artifacts.sh"
  python3 - "$SRC" "$PR" "$SESSION_TICKS" <<'PY'
import sys
lines = [l.strip() for l in open(sys.argv[1]) if l.strip()]
# 4 lanes x >= SESSION_TICKS tokens: concatenate enough D10 lines per lane (fidelity_tick caps ticks at the shortest prompt)
need = int(sys.argv[3]); per = max(2, (need + 6) // 7 + 1)
out = [" ".join(lines[i*per:(i+1)*per]) for i in range(4)]
open(sys.argv[2], "w").write("\n".join(out) + "\n"); print("\n".join(out))
PY
  # a schedule-specific store file (the stock $STORE's stamp differs and would be REFUSED -- by design, I6)
  local STORE_ALL="$DEMO/store_${TAG}_r${LOG_RING}_xd${EXTRA_DEPTH}_lanesblock_sched_all.bin"
  local HC="$BIN ${MODEL_FLAGS[*]} ${DEMO_FLAGS[*]} ${BLOCK_FLAGS[*]} ${DEVICES_FLAGS:-} --store-file $STORE_ALL --vram-trace --trace-boots $(arm_flags all)"
  { echo "=== SESSION (all flags, $SESSION_TICKS ticks) $(utc) ==="; record_load_state; echo "HARNESS_CMD: $HC"; } > "$RL"
  gpu_idle_or_warn || echo "LOADSTATE CONTENDED" >> "$RL"
  local T0; T0=$(date +%s)
  ( cd "$POD_SRC" && "$VENV/bin/python" spec_decode/fidelity_tick.py --out "$OUT" --tag "$TAG" \
      --prompts-file "$PR" --ticks "$SESSION_TICKS" --device cpu --keys-dir "$KEYS_POD" \
      --bundle-dir "$ART" --art-dir "$ART" --serve-timeout 7200 --poll-interval 2 --harness-cmd "$HC" ) >> "$RL" 2>&1
  echo "RC=$? WALL_S=$(( $(date +%s) - T0 ))" >> "$RL"
  predict "$EXPECT_DEPTH" $((EXPECT_DEPTH - 19)) all "$SESSION_TICKS" | tee -a "$RL"
  local SL="$OUT/logs/server.jsonl"
  [ -s "$SL" ] && sites_hist "$SL" > "$AB/session_all.sites"
  python3 - "$SL" "$RL" "$OUT/fidelity.json" "$SESSION_TICKS" >> "$SUM" <<'PY'
import json, sys, re, os
SL, RL, FJ, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
print(f"\n## stateful session, all flags, {N} ticks\n")
if not os.path.exists(SL): print("NO server.jsonl -- read", RL); sys.exit(0)
served = [json.loads(l) for l in open(SL, errors="replace") if l.startswith('{"serve":"served"')]
boots = [s["reqBoots"] for s in served]; ms = [s["reqMsPerToken"] for s in served]; enc = [s["reqEncPtMs"] for s in served]; vram = [s["peakVramGB"] for s in served]
pred = None
for l in open(RL, errors="replace"):
    if l.startswith("PREDICT lazy"):
        m = re.search(r"ticks=(\[[^\]]*\])", l); pred = json.loads(m.group(1)) if m else None
print("| tick | reqBoots | replay (lazy) | reqEncPtMs | reqMsPerToken | peakVramGB | failed |")
print("|---|---|---|---|---|---|---|")
for i, s in enumerate(served):
    print(f"| {i} | {s['reqBoots']} | {pred[i] if pred and i < len(pred) else ''} | {s['reqEncPtMs']} | {s['reqMsPerToken']:.0f} | {s['peakVramGB']} | {s['failed']} |")
warm = boots[8:] if len(boots) > 8 else boots
g1 = "n/a"
if pred and len(pred) >= len(boots):
    pw = pred[8:len(boots)] if len(boots) > 8 else pred[:len(boots)]
    mb, mp = sum(warm)/len(warm), sum(pw)/len(pw)
    g1 = "PASS" if abs(mb - mp) <= 0.10 * mp else f"FAIL (measured warm mean {mb:.1f} vs replay {mp:.1f})"
last4 = ms[-4:]; flat = (max(last4) - min(last4)) <= 0.20 * (sum(last4)/4) if len(last4) == 4 else False
warmup = next((i for i, e in enumerate(enc) if e == 0), None)
S = None
for l in open(SL, errors="replace"):
    if l.startswith('{"summary":true'): S = json.loads(l)
print(f"\nG1 boots (warm mean vs replay lazy): {g1}")
print(f"G6 store: first tick with reqEncPtMs == 0: {warmup} (replay: last new key at tick 12-15 under parent-first); storeMisses at exit: {S.get('storeMisses') if S else None}, storeBytes {S.get('storeBytes') if S else None}")
print(f"G7 flat: last-4 reqMsPerToken {['%.0f' % x for x in last4]} -> {'PASS' if flat else 'FAIL'}; peakVramGB range {min(vram) if vram else None}-{max(vram) if vram else None}")
fj = json.load(open(FJ)) if os.path.exists(FJ) else None
if fj:
    rows = fj["rows"]; print(f"fidelity: top1 {sum(1 for r in rows if r['top1'])}/{len(rows)}, relErrRms worst {max(r['relErrRms'] for r in rows):.4g} (compare with 60_fidelity_gate.sh's stock record, same keys)")
print(f"failed requests: {sum(1 for s in served if s['failed'])}; CUDA/fatal text: {bool(re.search(r'out of memory|Cuda failure|cudaError|map::at|terminate called|\"fatal\"', open(SL, errors='replace').read(), re.I))}")
PY
  campaign_log "session all flags | $(grep -ac '"serve":"served"' "$SL" 2>/dev/null) served | $RL"
}

case "$MODE" in
  keys)    busy_acquire; { echo "# S3.3 schedule A/B -- $(utc) -- binary $BIN"; } >> "$SUM"; mode_keys; RC=$?; busy_release; exit $RC ;;
  quick)   busy_acquire; { echo "# S3.3 schedule A/B (quick, 2^15 structural xd9) -- $(utc)"; } >> "$SUM"
           run_arm quick stock; for a in $ARMS; do run_arm quick "$a"; done; judge quick; busy_release; log "SUMMARY: $SUM" ;;
  secure)  busy_acquire; { echo "# S3.3 schedule A/B (secure, demo configuration) -- $(utc)"; } >> "$SUM"
           run_arm secure stock; for a in $ARMS; do run_arm secure "$a"; done; judge secure; busy_release; log "SUMMARY: $SUM" ;;
  lanes)   busy_acquire; mode_lanes; busy_release ;;
  session) busy_acquire; mode_session; busy_release ;;
  all)     busy_acquire; { echo "# S3.3 schedule A/B -- $(utc) -- binary $BIN"; } >> "$SUM"
           mode_keys || { busy_release; exit 1; }
           run_arm quick stock; for a in $ARMS; do run_arm quick "$a"; done; judge quick
           if [ "$SECURE_ARMS" = 1 ]; then run_arm secure stock; for a in $ARMS; do run_arm secure "$a"; done; judge secure; fi
           mode_lanes; mode_session; busy_release; log "SUMMARY: $SUM" ;;
  *) echo "usage: $SCRIPT keys|quick|secure|lanes|session|all   (env: ARMS, SESSION_TICKS, POOL_LEVELS, SECURE_ARMS)"; exit 2 ;;
esac
