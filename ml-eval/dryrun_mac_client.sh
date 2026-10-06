#!/bin/bash
# dryrun_mac_client.sh — END-TO-END DRY RUN of the demo client path
# (fhe_client.py generate --mac-tool ...) on this Mac, with NO GPU and NO pod,
# against ml-eval/mock_fhe_server.py. 2026-09-02.
#
# What it exercises for real: Mac keygen, prepare (64 ragged lanes), the
# --remote-ssh local transport, mac_session_preflight (stale-dir + ps-based
# server check), per-tick enc -> copy -> ready marker -> poll -> copy back ->
# dec -> head matvec -> ragged lane bookkeeping -> transcript, then
# spec_decode/audit_ragged.py on the session. What is FAKE: the server's
# arithmetic (a mock with a per-lane carry) and, when `transformers` is not
# importable, the tokenizer (--stub-tokenizer, byte-level). Nothing here is a
# demo record; every stub session is stamped as such (F62 class).
#
# Steps
#   1. keygen with the CURRENT hpc_gpu_port/mac_fhe_client:
#        keygen --out KEYS --log-ring R --secure 0 --extra-depth 9 --lanes-block 1
#      Ladder (env LADDER, default "17:64 16:32 15:16" = log-ring:lanes):
#      ring 2^17 is what 64 lanes need (REP = SLOTS/1024 = 64). A rung is
#      abandoned and the next tried when keygen exceeds KEYGEN_TIME_LIMIT
#      seconds (900) or the machine's swap grows by more than
#      KEYGEN_SWAP_LIMIT_MB (4096) -- "exceeds RAM" -- or exits nonzero.
#      If the binary supports `--minimal 1` (context + key pair + evalmult;
#      the mock needs nothing else) a minimal keygen at the SAME ring is
#      tried before dropping a rung. Timings -> KEYS/keygen_timing.json.
#      Existing keys with a good keygen.log are reused (FORCE_KEYGEN=1 to redo).
#   2. prepare --lanes N --prompts-file (first N of demo_prompts_64.txt)
#      --bundle-dir BUNDLE_DIR --tag pbd430a --dpad 1024, real tokenizer if
#      `python3 -c "import transformers"` works, else --stub-tokenizer.
#   3. mock server in the background: SERVE/gpu_real_model (a symlink to the
#      python binary, so the ps line's first token ends in gpu_real_model)
#      mock_fhe_server.py --serve SERVE --stateful ...
#   4. generate --mac-tool ... --remote-ssh local --remote-reqdir SERVE
#      --steps STEPS --words-per-lane WPL --stop-on-newline --greedy
#      --harness-cmd "x --pack-tokens 1"  (+ --keys-dir, required by argparse)
#   5. spec_decode/audit_ragged.py --checks AB (ABC only if torch imports;
#      C/D need the plaintext model and are meaningless against a mock)
#   6. stop the server via SERVE/stop; per-tick enc/serve/dec timings and the
#      lanes_text.json head; summary.json.
#
# Everything lands under results/dense-demo-s31/dryrun/<RUN>/ (keys beside it
# in keys_rNN_dryrun/). Env knobs: LADDER STEPS WPL MODEL PROMPTS_SRC
# BUNDLE_DIR TAG PY RUN KEYGEN_TIME_LIMIT KEYGEN_SWAP_LIMIT_MB FORCE_KEYGEN.
# Exit 0 only if keygen, prepare, generate, audit and the server all succeeded.
set -u
set -o pipefail

REPO=${REPO:-$REPO}
ROOT=${ROOT:-$REPO/results/dense-demo-s31/dryrun}
MAC=${MAC:-$REPO/hpc_gpu_port/mac_fhe_client}
PY=${PY:-/usr/bin/python3}
CLIENT=$REPO/ml-eval/fhe_client.py
MOCK=$REPO/ml-eval/mock_fhe_server.py
AUDIT=$REPO/spec_decode/audit_ragged.py
PROMPTS_SRC=${PROMPTS_SRC:-$REPO/results/dense-demo-s31/sessions/demo_prompts_64.txt}
BUNDLE_DIR=${BUNDLE_DIR:-$BACKUP/pod_artifacts}
TAG=${TAG:-pbd430a}
DPAD=${DPAD:-1024}
D=${D:-1024}
STEPS=${STEPS:-12}
WPL=${WPL:-3}
MODEL=${MODEL:-linear 1}
LADDER=${LADDER:-17:64 16:32 15:16}
KEYGEN_TIME_LIMIT=${KEYGEN_TIME_LIMIT:-900}
KEYGEN_SWAP_LIMIT_MB=${KEYGEN_SWAP_LIMIT_MB:-4096}
FORCE_KEYGEN=${FORCE_KEYGEN:-0}
SERVE_TIMEOUT=${SERVE_TIMEOUT:-1800}
POLL=${POLL:-0.2}
# S3.7 V3 (2026-09-05): WAIT_MODE=poll|remote -> fhe_client.py --wait-mode; the
# V3 Mac test runs the dry run once per mode and reads the control-traffic
# ledger (FHE_CLIENT_COUNT_CALLS=1 makes the client print it at exit).
WAIT_MODE=${WAIT_MODE:-poll}
# REMOTE_SSH (V3): the client's --remote-ssh; `local` (default) = sh -c and
# plain copies; anything else is an ssh command line -- the V3 test passes a
# local test double (`bash .../fakessh/ssh host`, with a fake `scp` on PATH)
# to exercise the ssh/scp argv branch with an emulated round-trip delay.
REMOTE_SSH=${REMOTE_SSH:-local}
RUN=${RUN:-run_$(date -u +%Y%m%dT%H%M%SZ)}

OUT=$ROOT/$RUN
SESSION=$OUT/session
SERVE=$OUT/serve
LOGS=$OUT/logs
mkdir -p "$ROOT" "$OUT" "$SESSION" "$SERVE" "$LOGS"
# multi-GB keys and ciphertexts must never reach git
cat > "$ROOT/.gitignore" <<'EOF'
# dry-run keys (GBs) and serve dirs; logs/json/transcripts ARE kept
keys_*/
*/serve/
*.f64
*.bin
*.key
EOF
exec > >(tee -a "$LOGS/dryrun.log") 2>&1

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
swap_used_mb() {
    sysctl -n vm.swapusage 2>/dev/null | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1 \
        | grep -E '^[0-9]+$' || echo 0
}
FAILS=""
fail() { FAILS="$FAILS $1"; say "FAIL: $*"; }

say "dry run $RUN  repo=$REPO  py=$PY  mac_tool=$MAC"
say "binary: $(ls -la "$MAC" | awk '{print $5, $6, $7, $8}')  source: $(ls -la "$REPO/hpc_gpu_port/mac_fhe_client.cpp" | awk '{print $6, $7, $8}')"
[ -x "$MAC" ] || { say "fatal: $MAC missing/not executable"; exit 2; }
[ -f "$BUNDLE_DIR/bundle_$TAG.index.txt" ] || { say "fatal: $BUNDLE_DIR/bundle_$TAG.index.txt missing"; exit 2; }
say "binary sha256: $(shasum -a 256 "$MAC" | cut -c1-64)  (R6: timings never cross binaries; this line names the one used)"
# Does this build know `keygen --minimal 1`? Probe it (an older binary ignores
# the unknown flag and runs a full 2^12 keygen -- a few seconds -- printing a
# macKeygen line WITHOUT "mode":"minimal").
PROBE=$OUT/keygen_probe_r12; rm -rf "$PROBE"; mkdir -p "$PROBE"
if "$MAC" keygen --out "$PROBE" --log-ring 12 --secure 0 --extra-depth 9 --lanes-block 1 --minimal 1 \
       > "$LOGS/keygen_probe_r12.log" 2>&1 && grep -q '"mode":"minimal"' "$LOGS/keygen_probe_r12.log"; then
    HAVE_MINIMAL=1
else
    HAVE_MINIMAL=0
fi
rm -rf "$PROBE"
say "binary supports keygen --minimal 1: $HAVE_MINIMAL (probed at ring 2^12; log $LOGS/keygen_probe_r12.log)"
sysctl -n hw.memsize | awk '{printf "[mem] physical RAM %.1f GB\n", $1/1073741824}'

# ---------------------------------------------------------------- 1. keygen
# keygen_try RING KEYS LOG EXTRA_FLAGS -> sets KG_RC KG_SEC KG_MAXRSS_MB KG_SWAPG_MB KG_WHY
keygen_try() {
    local ring=$1 keys=$2 log=$3 extra=$4
    mkdir -p "$keys"
    local swap0 t0 pid rss rssmb now el sw swg
    swap0=$(swap_used_mb); t0=$(date +%s)
    KG_MAXRSS_MB=0; KG_SWAPG_MB=0; KG_WHY=completed
    say "keygen: $MAC keygen --out $keys --log-ring $ring --secure 0 --extra-depth 9 --lanes-block 1 $extra"
    # shellcheck disable=SC2086
    "$MAC" keygen --out "$keys" --log-ring "$ring" --secure 0 --extra-depth 9 --lanes-block 1 $extra > "$log" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        sleep 5
        rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' '); rss=${rss:-0}; rssmb=$((rss / 1024))
        [ "$rssmb" -gt "$KG_MAXRSS_MB" ] && KG_MAXRSS_MB=$rssmb
        now=$(date +%s); el=$((now - t0))
        sw=$(swap_used_mb); swg=$((sw - swap0)); [ "$swg" -gt "$KG_SWAPG_MB" ] && KG_SWAPG_MB=$swg
        if [ $((el % 60)) -lt 5 ]; then say "  keygen ring 2^$ring: ${el}s rss ${rssmb}MB swapGrowth ${swg}MB"; fi
        if [ "$el" -gt "$KEYGEN_TIME_LIMIT" ]; then
            KG_WHY="timeLimit(${KEYGEN_TIME_LIMIT}s)"; kill -9 "$pid" 2>/dev/null; break
        fi
        if [ "$swg" -gt "$KEYGEN_SWAP_LIMIT_MB" ]; then
            KG_WHY="exceedsRam(swapGrowth ${swg}MB > ${KEYGEN_SWAP_LIMIT_MB}MB)"; kill -9 "$pid" 2>/dev/null; break
        fi
    done
    wait "$pid" 2>/dev/null; KG_RC=$?
    now=$(date +%s); KG_SEC=$((now - t0))
    if [ "$KG_RC" -ne 0 ] && [ "$KG_WHY" = completed ]; then KG_WHY="exit$KG_RC"; fi
    if ! grep -q '"macKeygen":true' "$log"; then [ "$KG_WHY" = completed ] && KG_WHY="noMacKeygenLine"; [ "$KG_RC" -eq 0 ] && KG_RC=1; fi
    say "keygen ring 2^$ring: rc=$KG_RC ${KG_SEC}s maxRss ${KG_MAXRSS_MB}MB swapGrowth ${KG_SWAPG_MB}MB why=$KG_WHY"
    say "  log tail: $(tail -c 600 "$log" | tr '\n' ' ')"
    local rec
    rec=$(printf '{"run":"%s","ring":%d,"flags":"--secure 0 --extra-depth 9 --lanes-block 1 %s","rc":%d,"seconds":%d,"maxRssMB":%d,"swapGrowthMB":%d,"why":"%s","log":"%s","binaryMtime":"%s","binarySha256":"%s"}' \
        "$RUN" "$ring" "$extra" "$KG_RC" "$KG_SEC" "$KG_MAXRSS_MB" "$KG_SWAPG_MB" "$KG_WHY" "$log" \
        "$(stat -f %Sm -t %Y-%m-%dT%H:%M:%S "$MAC")" "$(shasum -a 256 "$MAC" | cut -c1-64)")
    echo "$rec" >> "$keys/keygen_timing.json"
    # the keys dir is wiped before a retry, so every attempt is ALSO recorded
    # in one append-only ledger beside the key dirs (a failed 2^17 attempt is
    # a measurement too), and its log is kept under a distinct name.
    echo "$rec" >> "$ROOT/keygen_attempts.jsonl"
    if [ "$KG_RC" -ne 0 ]; then cp "$log" "$ROOT/keygen_r${ring}_attempt_$(date -u +%Y%m%dT%H%M%SZ).log"; fi
}

KEYS=""; LOG_RING=""; LANES=""; KEYGEN_MODE=""
for rung in $LADDER; do
    ring=${rung%%:*}; lanes=${rung##*:}
    keys=$ROOT/keys_r${ring}_dryrun
    if [ "$FORCE_KEYGEN" != 1 ] && [ -f "$keys/cryptocontext.bin" ] && [ -f "$keys/public.key" ] \
       && [ -f "$keys/secret.key" ] && grep -q '"macKeygen":true' "$keys/keygen.log" 2>/dev/null; then
        say "keys for ring 2^$ring already present in $keys (reusing; FORCE_KEYGEN=1 to redo)"
        say "  prior timing: $(tail -1 "$keys/keygen_timing.json" 2>/dev/null)"
        KEYS=$keys; LOG_RING=$ring; LANES=$lanes; KEYGEN_MODE=$(grep -o '"mode":"[a-z]*"' "$keys/keygen.log" | head -1); KEYGEN_MODE=${KEYGEN_MODE:-"\"mode\":\"full\""}
        break
    fi
    rm -rf "$keys"; mkdir -p "$keys"
    keygen_try "$ring" "$keys" "$keys/keygen.log" ""
    if [ "$KG_RC" -eq 0 ]; then KEYS=$keys; LOG_RING=$ring; LANES=$lanes; KEYGEN_MODE='"mode":"full"'; break; fi
    if [ "$HAVE_MINIMAL" = 1 ]; then
        say "full keygen at 2^$ring failed ($KG_WHY); retrying MINIMAL keygen at the same ring (mock needs only ctx+pk+sk)"
        rm -rf "$keys"; mkdir -p "$keys"
        keygen_try "$ring" "$keys" "$keys/keygen.log" "--minimal 1"
        if [ "$KG_RC" -eq 0 ]; then KEYS=$keys; LOG_RING=$ring; LANES=$lanes; KEYGEN_MODE='"mode":"minimal"'; break; fi
    fi
    say "ring 2^$ring abandoned ($KG_WHY); falling back to the next rung of LADDER=[$LADDER]"
done
if [ -z "$KEYS" ]; then say "fatal: no rung of LADDER=[$LADDER] produced keys"; exit 2; fi
say "USING ring 2^$LOG_RING, $LANES lanes, keys $KEYS ($KEYGEN_MODE)"
[ "$LOG_RING" = 17 ] || say "NOTE: NOT the 64-lane/2^17 configuration the spec asked for -- see keygen_timing.json for why"

# ---------------------------------------------------------------- 2. prepare
TOKFLAG=""
if "$PY" -c "import transformers" 2>/dev/null; then
    say "tokenizer: REAL (transformers importable in $PY)"
else
    TOKFLAG="--stub-tokenizer"
    say "tokenizer: STUB byte-level (--stub-tokenizer; transformers NOT importable in $PY) -- protocol-only session"
fi
PROMPTS=$OUT/prompts_${LANES}.txt
head -n "$LANES" "$PROMPTS_SRC" > "$PROMPTS"
say "prepare: $LANES lanes from $PROMPTS_SRC (first $LANES lines) bundle $BUNDLE_DIR tag $TAG"
"$PY" "$CLIENT" prepare --lanes "$LANES" --prompts-file "$PROMPTS" --bundle-dir "$BUNDLE_DIR" \
    --tag "$TAG" --out "$SESSION" --dpad "$DPAD" $TOKFLAG > "$LOGS/prepare.log" 2>&1
PREP_RC=$?
grep -E '"prepare"|fatal|STUB' "$LOGS/prepare.log" | head -5
[ "$PREP_RC" -eq 0 ] || { fail "prepare rc=$PREP_RC"; tail -20 "$LOGS/prepare.log"; exit 2; }

# ---------------------------------------------------------------- 3. mock server
rm -f "$SERVE/stop" "$SERVE"/req.* "$SERVE"/resp.*
# The preflight wants argv[0] to END in gpu_real_model. Apple's python3 and
# its bin/python3.9 are launcher stubs that REWRITE argv[0] on exec; the
# framework's Python.app/Contents/MacOS/Python binary keeps it.
PYREAL=$("$PY" -c 'import os,sys; print(os.path.realpath(sys.executable))')
FW=$(printf '%s' "$PYREAL" | sed -E 's#(.*/Python3?\.framework/Versions/[^/]+)/.*#\1#')
if [ -x "$FW/Resources/Python.app/Contents/MacOS/Python" ]; then
    PYBIN=$FW/Resources/Python.app/Contents/MacOS/Python
else
    PYBIN=$PYREAL
fi
ln -sf "$PYBIN" "$SERVE/gpu_real_model"
say "server: $SERVE/gpu_real_model -> $PYBIN"
# shellcheck disable=SC2086
"$SERVE/gpu_real_model" "$MOCK" --serve "$SERVE" --stateful --keys-dir "$KEYS" --mac-tool "$MAC" \
    --d "$D" --model $MODEL --serve-timeout "$SERVE_TIMEOUT" > "$LOGS/mock_server.log" 2>&1 &
SRV_PID=$!
sleep 2
if ! kill -0 "$SRV_PID" 2>/dev/null; then fail "mock server died at start"; cat "$LOGS/mock_server.log"; exit 2; fi
PSLINE=$(ps -eo pid=,args= | grep -F -- "--serve $SERVE" | grep -v -e grep -e 'ps -eo' | head -1)
say "server ps line: ${PSLINE:0:200}"
ARGV0=$(printf '%s' "$PSLINE" | awk '{print $2}')
case "$ARGV0" in
    *gpu_real_model) say "server argv[0] ends in gpu_real_model: OK" ;;
    *) fail "server argv[0] is '$ARGV0' (preflight will reject it)" ;;
esac
head -2 "$LOGS/mock_server.log"

# ---------------------------------------------------------------- 4. generate
say "binary at generate time: sha256 $(shasum -a 256 "$MAC" | cut -c1-64) mtime $(stat -f %Sm -t %Y-%m-%dT%H:%M:%S "$MAC")"
say "generate: $STEPS steps, words-per-lane $WPL, stop-on-newline, greedy, transport '$REMOTE_SSH', reqdir $SERVE, wait-mode $WAIT_MODE"
T_GEN0=$(date +%s)
FHE_CLIENT_COUNT_CALLS=1 "$PY" "$CLIENT" generate --out "$SESSION" --steps "$STEPS" --keys-dir "$KEYS" \
    --mac-tool "$MAC" --mac-keys "$KEYS" --remote-ssh "$REMOTE_SSH" --remote-reqdir "$SERVE" \
    --words-per-lane "$WPL" --stop-on-newline --greedy --poll-interval "$POLL" --wait-mode "$WAIT_MODE" \
    --harness-cmd "x --pack-tokens 1" $TOKFLAG > "$LOGS/generate.log" 2>&1
GEN_RC=$?
T_GEN=$(( $(date +%s) - T_GEN0 ))
say "generate rc=$GEN_RC wall ${T_GEN}s"
grep -E 'macPreflight|fatal|Traceback|lanesDone|^== ' "$LOGS/generate.log" | head -40
[ "$GEN_RC" -eq 0 ] || { fail "generate rc=$GEN_RC"; tail -30 "$LOGS/generate.log"; }

# ---------------------------------------------------------------- 5. audit
if "$PY" -c "import torch" 2>/dev/null; then CHECKS=ABC; else CHECKS=AB; fi
say "audit: $AUDIT --out $SESSION --tag $TAG --checks $CHECKS (torch importable: $([ $CHECKS = ABC ] && echo yes || echo no))"
say "  exit codes: 0 = every requested check ran and passed; 3 = violation; 4 = a requested check could not run; 5 = auditor refused (no retire_cfg)"
[ "$CHECKS" = AB ] && say "  check C (plaintext greedy replay) and D (shift test, inside C) SKIPPED: torch not importable AND the server is a mock, so a plaintext oracle would be meaningless here anyway"
"$PY" "$AUDIT" --out "$SESSION" --tag "$TAG" --checks "$CHECKS" > "$LOGS/audit.log" 2>&1
AUDIT_RC=$?
say "audit rc=$AUDIT_RC"
cat "$LOGS/audit.log"
[ "$AUDIT_RC" -eq 0 ] || fail "audit rc=$AUDIT_RC"

# ---------------------------------------------------------------- 6. stop + report
echo stop > "$SERVE/stop"
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
    kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1
done
if kill -0 "$SRV_PID" 2>/dev/null; then fail "server ignored stop file; killing"; kill -9 "$SRV_PID"; fi
wait "$SRV_PID" 2>/dev/null; SRV_RC=$?
say "mock server rc=$SRV_RC  last log line: $(tail -1 "$LOGS/mock_server.log")"
[ "$SRV_RC" -eq 0 ] || fail "server rc=$SRV_RC"
grep -c '"serve": "served"' "$LOGS/mock_server.log" | sed 's/^/[server] requests served: /'
LEFT=$(ls "$SERVE"/req.* "$SERVE"/resp.* 2>/dev/null | wc -l | tr -d ' ')
say "serve dir leftovers after session: $LEFT (must be 0)"; [ "$LEFT" = 0 ] || fail "serve dir not clean"

"$PY" - "$SESSION" "$LOGS/mock_server.log" "$KEYS/keygen_timing.json" "$OUT/summary.json" "$LOG_RING" "$LANES" "$STEPS" "$TOKFLAG" "$GEN_RC" "$AUDIT_RC" "$SRV_RC" "$T_GEN" <<'EOF'
import json, os, sys
sess, srvlog, kgt, outp, ring, lanes, steps, tokflag, gen_rc, audit_rc, srv_rc, t_gen = sys.argv[1:13]
tr = json.load(open(os.path.join(sess, "transcript.json"))) if os.path.exists(os.path.join(sess, "transcript.json")) else []
st = json.load(open(os.path.join(sess, "tokens.json")))
print("tick  prompt/gen/retired   encSec  serveSec  decSec   picks  waitMode ctlCalls ctlSec")
tot = [0.0, 0.0, 0.0]
ctl_calls, ctl_sec = [], []
for e in tr:
    ph = e["phases"]; c = (ph.count("prompt"), ph.count("gen"), ph.count("retired"))
    picks = sum(1 for p in e["picked"] if p is not None)
    print("%4d  %3d/%3d/%3d          %6.2f  %8.2f  %6.2f   %d      %-6s %5s %7s" % (
        e["tick"], c[0], c[1], c[2], e["encSec"], e["serveSec"], e["decSec"], picks,
        e.get("waitMode", "-"), e.get("ctlCalls", "-"), e.get("ctlSec", "-")))
    tot[0] += e["encSec"]; tot[1] += e["serveSec"]; tot[2] += e["decSec"]
    if "ctlCalls" in e: ctl_calls.append(e["ctlCalls"]); ctl_sec.append(e["ctlSec"])
n = max(len(tr), 1)
print("mean over %d ticks: enc %.2fs serve %.2fs dec %.2fs; sum %.1fs (client wall incl. bookkeeping %ss)" % (len(tr), tot[0]/n, tot[1]/n, tot[2]/n, sum(tot), t_gen))
if ctl_calls:   # V3 (2026-09-05): control-traffic ledger, additive transcript fields
    print("control traffic (V3): mean ctlCalls %.2f/tick (min %d max %d), mean ctlSec %.3f s/tick" % (
        sum(ctl_calls)/len(ctl_calls), min(ctl_calls), max(ctl_calls), sum(ctl_sec)/len(ctl_sec)))
srv = [json.loads(l) for l in open(srvlog) if l.startswith("{")]
served = [s for s in srv if s.get("serve") == "served"]
if served:
    m = lambda k: sum(s[k] for s in served) / len(served)
    print("server-side mean over %d requests: decMs %.0f modelMs %.0f encMs %.0f totalMs %.0f; laneSpread(last) %.3g; respBytes %d" % (len(served), m("decMs"), m("modelMs"), m("encMs"), m("totalMs"), served[-1]["laneSpread"], served[-1]["respBytes"]))
print("session: lanes %d ragged %s promptLens min/max %d/%d stubTokenizer %s generatedPerLane %s retired %d/%d" % (
    st["lanes"], st.get("ragged"), min(st["prompt_lens"]), max(st["prompt_lens"]), st.get("stubTokenizer", False),
    [len(g) for g in st["generated_lanes"]], sum(1 for r in st.get("retired", []) if r), st["lanes"]))
lt_path = os.path.join(sess, "lanes_text.json")
if os.path.exists(lt_path):
    lt = json.load(open(lt_path))
    print("lanes_text.json head:")
    for k in list(lt)[:6]:
        print("  %s: %s" % (k, json.dumps(lt[k])[:110]))
kg = [json.loads(l) for l in open(kgt)] if os.path.exists(kgt) else []
summary = {"run": os.path.basename(os.path.dirname(sess)), "logRing": int(ring), "lanes": int(lanes), "steps": int(steps),
           "tokenizer": "stub-bytes" if tokflag else "real", "keygen": kg, "ticks": len(tr),
           "meanEncSec": tot[0]/n, "meanServeSec": tot[1]/n, "meanDecSec": tot[2]/n, "generateWallSec": int(t_gen),
           "serverRequests": len(served), "genRc": int(gen_rc), "auditRc": int(audit_rc), "serverRc": int(srv_rc),
           "generatedPerLane": [len(g) for g in st["generated_lanes"]], "retired": st.get("retired"),
           "waitMode": (tr[0].get("waitMode") if tr else None),
           "meanCtlCalls": (sum(ctl_calls)/len(ctl_calls) if ctl_calls else None),
           "meanCtlSec": (sum(ctl_sec)/len(ctl_sec) if ctl_sec else None),
           "mock": True, "demoRecord": False}
json.dump(summary, open(outp, "w"), indent=1)
print("summary ->", outp)
EOF

if [ -n "$FAILS" ]; then say "DRY RUN FAILED:$FAILS"; exit 1; fi
say "DRY RUN OK (mock server, $([ -n "$TOKFLAG" ] && echo stub || echo real) tokenizer, ring 2^$LOG_RING, $LANES lanes, $STEPS steps) -- NOT a demo record"
exit 0
