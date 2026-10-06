#!/bin/bash
# dryrun_freerun_mock.sh -- exercise `fidelity_tick.py --free-run` END TO END on this Mac with NO GPU: the Mac client path
# (mac mode, --remote-ssh local) against ml-eval/mock_fhe_server.py at ring 2^15. The mock's arithmetic is FAKE, so
# top-1 agreement is meaningless here; what is tested is the driver: ragged prompts -> feed back the decrypted argmax ->
# the recurrent plaintext reference on the fed sequence -> rows / state file. Nothing it writes is a record.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=$REPO/.venv/bin/python
KEYS=${KEYS:-$HOME/Documents/fhe-ssm-backup/mac_keys_r15}; MAC=$REPO/hpc_gpu_port/mac_fhe_client; ART=${ART:-$HOME/Documents/fhe-ssm-backup/mac_art}
W=${W:-$SCRATCH/freerun_dry}; LANES=${LANES:-8}; TICKS=${TICKS:-14}
rm -rf "$W"; mkdir -p "$W/serve" "$W/out"; head -$LANES "$REPO/results/dense-demo-s31/sessions/demo_prompts_64.txt" > "$W/prompts.txt"
PYREAL=$("$PY" -c 'import os,sys; print(os.path.realpath(sys.executable))'); FW=$(printf '%s' "$PYREAL" | sed -E 's#(.*/Python3?\.framework/Versions/[^/]+)/.*#\1#')
PYBIN=$PYREAL; [ -x "$FW/Resources/Python.app/Contents/MacOS/Python" ] && PYBIN=$FW/Resources/Python.app/Contents/MacOS/Python
ln -sf "$PYBIN" "$W/serve/gpu_real_model"
PYTHONPATH=$("$PY" -c 'import site; print(site.getsitepackages()[0])') "$W/serve/gpu_real_model" "$REPO/ml-eval/mock_fhe_server.py" --serve "$W/serve" --stateful --keys-dir "$KEYS" --mac-tool "$MAC" --d 1024 --model linear 1 --serve-timeout 600 > "$W/mock_server.log" 2>&1 &
SRV=$!; sleep 2; kill -0 $SRV 2>/dev/null || { echo "mock server died"; cat "$W/mock_server.log"; exit 2; }
"$PY" "$REPO/spec_decode/fidelity_tick.py" --out "$W/out" --tag pbd430a --prompts-file "$W/prompts.txt" --ticks "$TICKS" --free-run ${EXTRA_DRIVER_ARGS:-} \
  --bundle-dir "$ART" --art-dir "$ART" --device cpu --mac-tool "$MAC" --mac-keys "$KEYS" --remote-ssh local --remote-reqdir "$W/serve" > "$W/driver.log" 2>&1
RC=$?; touch "$W/serve/stop"; sleep 1; kill $SRV 2>/dev/null
echo "driver rc=$RC"; grep -v "Warn\|warn" "$W/driver.log" | grep -E "fatal|Traceback|Error|fidelityTick" | head -8
"$PY" - "$W/out" "$W/prompts.txt" <<'PY'
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])) if False else ".", "ml-eval"))
out, pf = sys.argv[1], sys.argv[2]
rows = json.load(open(out + "/fidelity.json")); rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
st = json.load(open(out + "/tokens.json")) if os.path.exists(out + "/tokens.json") else json.load(open(out + "/state.json"))
by = {}
for r in rows: by.setdefault(r["lane"], []).append(r)
ok = True
for lane in sorted(by):
    seq = sorted(by[lane], key=lambda r: r["tick"]); P = len(st["ids_lanes"][lane]); gen = st["generated_lanes"][lane]
    fed = [r["fed"] for r in seq]; src = [r["fedSrc"] for r in seq]
    exp_fed = (st["ids_lanes"][lane] + gen)[:len(seq)]
    # invariants of a free run: prompt tokens first, then exactly the tokens the (mock) encrypted session produced
    c1 = fed == exp_fed; c2 = all(s == "prompt" for s in src[:P]) and all(s in ("fhe", "ref") for s in src[P:])
    # under a decode policy the generated token is the policy's pick (encPick / refPick); greedy rows have no such fields
    c3 = gen == [r.get("encPick", r["encArgmax"]) if not r.get("decFailed") else r.get("refPick", r["refArgmax"]) for r in seq if r["generating"]]
    pol = (st.get("decodePolicy") or {}).get("policy", "greedy"); c4 = True
    if pol == "norepeat":                           # the rule's own guarantee: no 4-gram of the fed history ends in a generated token twice
        n = (st.get("decodePolicy") or {}).get("ngram", 4); h = st["ids_lanes"][lane] + gen; seen = {}
        for i in range(len(h) - n + 1):
            g = tuple(h[i:i + n])
            if g in seen and i + n - 1 >= P: c4 = False
            seen[g] = i
    ok &= c1 and c2 and c3 and c4
    print(f"lane {lane}: prompt {P} tokens, ticks {len(seq)}, generated {len(gen)}, fed==prompt+generated {c1}, sources ok {c2}, generated==picked rows {c3}, policy {pol} no-repeat holds {c4}")
print("FREE-RUN DRIVER DRY RUN:", "PASS" if ok and len(by) > 0 else "FAIL")
PY
