#!/usr/bin/env bash
# 50_sovereignty.sh -- the pod loads the MAC-minted eval material (no secret key
#                      anywhere on the box) and runs the harness selftest.
#
# PURPOSE   Architecture gate: the OpenFHE-written context + FIDESlib .dev
#           sidecar + merged evalrot.part*.bin from the Mac must deserialize,
#           carry every index LoadContext/bootstrap-precompute pulls (the
#           2026-09-01 map::at class, F62_CLASS_SWEEP "Key-sovereignty
#           diagnosis"), and drive the CUDA primitives. Precedent at 2^15:
#           sessions/final/run_sov2.sh (--lanes-block needs --compressed-store)
#           and demo_smoke/sovereignty3.log (hardFail 0 on the patched Mac set).
# INPUTS    $KEYS with cryptocontext.bin(+.dev), public.key, evalmult.bin,
#           evalrot.bin OR evalrot.part*.bin, optional MANIFEST.sha256.
#           secret.key MUST NOT exist there (the script refuses if it does).
# OUTPUTS   $DEMO/sovereignty_r17.log
# PASS      the header line {"harness":"gpu_real_model"...} is present (setup
#           completed -- LoadContext accepted the key set), no map::at /
#           out_of_range / terminate / CUDA text, and a {"selftestSummary"} line
#           exists. WITHOUT a secret key the decrypting probes throw the
#           empty-secret sentinel "bad any_cast" BY DESIGN (demo_smoke/sov_ab.log
#           test A; gpu_real_model.cu:961-964), so hardFail is REPORTED, not
#           gated, and any throw text other than "bad any_cast" fails.
#           NOTE the definitive index-coverage gate is the Mac-mode fidelity
#           probe (60_fidelity_gate.sh mac-cmd) -- it decrypts on the Mac.
# RECORD    sovereignty_r17.log (cite beside the key contract).
# DURATION  keys-load setup at 2^17: setupMs 45,100 (demo_smoke/fid17_run.log:6);
#           selftest itself seconds (keys_r17 console.log:8-16). ~2 min.
set -uo pipefail
SCRIPT=$(basename "$0")
. "$(dirname "$(readlink -f "$0")")/_lib.sh"
require_bin
mkdir -p "$DEMO"
L="$DEMO/sovereignty_r17.log"
if [ -e "$KEYS/secret.key" ]; then
  log "REFUSED: $KEYS/secret.key EXISTS on the pod. The fully-secure demo requires that the secret never be here. Delete it (shred -u) and re-run."
  campaign_log "REFUSED sovereignty | secret.key present in $KEYS"; exit 2
fi
for f in cryptocontext.bin cryptocontext.bin.dev public.key evalmult.bin; do require_file "$KEYS/$f" "upload from the Mac (30_dump_indices.sh step 4)"; done
if ls "$KEYS"/evalrot.part*.bin >/dev/null 2>&1; then
  NP=$(ls "$KEYS"/evalrot.part*.bin | wc -l); RB=$(stat -c %s "$KEYS"/evalrot.part*.bin | awk '{s+=$1} END{print s+0}')
  log "evalrot parts: $NP files, $RB bytes total"
elif [ -s "$KEYS/evalrot.bin" ]; then RB=$(file_bytes "$KEYS/evalrot.bin"); log "evalrot.bin: $RB bytes"
else log "MISSING: evalrot.bin / evalrot.part*.bin in $KEYS"; exit 2; fi
if [ -s "$KEYS/MANIFEST.sha256" ]; then
  if ( cd "$KEYS" && sha256sum -c --quiet MANIFEST.sha256 ); then log "MANIFEST.sha256: all files verified"; else
    log "MANIFEST.sha256 MISMATCH -- the upload is incomplete or corrupt; re-run the rsync (it resumes) before this gate"; campaign_log "FAIL sovereignty | manifest mismatch"; exit 2; fi
else log "WARN: no MANIFEST.sha256 in $KEYS (30_dump_indices.sh step 3 makes it) -- proceeding on sizes only"; fi
grep -q 'RotationIndexes' "$KEYS/cryptocontext.bin.dev" || log "WARN: .dev sidecar has no RotationIndexes line (mac_fhe_client.cpp:155-161 format)"

busy_acquire
{ echo "=== SOVEREIGNTY $(utc) keys=$KEYS ==="; record_load_state; ls -la "$KEYS"
  echo "CMD: $BIN ${MODEL_FLAGS[*]} ${DEMO_FLAGS[*]} ${BLOCK_FLAGS[*]} ${DEVICES_FLAGS:-} --keys-dir $KEYS --keys-load --selftest"; } > "$L"
# DEVICES_FLAGS (2026-09-03): the two-card route (export DEVICES=0,1) -- same words as 60/70.
"$BIN" "${MODEL_FLAGS[@]}" "${DEMO_FLAGS[@]}" "${BLOCK_FLAGS[@]}" ${DEVICES_FLAGS:-} --keys-dir "$KEYS" --keys-load --selftest >> "$L" 2>&1
echo "RC=$? (4 is the designed rmsnorm soft-fail, T6; never the verdict)" >> "$L"

P=()
grep -aq '^{"harness":"gpu_real_model"' "$L" || P+=("no harness header line: setup (deserialize / LoadContext / bootstrap precompute) did not complete")
grep -aqiE 'map::at|out_of_range|terminate called' "$L" && P+=("KEY-SET MISMATCH: $(grep -aoiE 'map::at|out_of_range|terminate called[^\"]{0,60}' "$L" | head -1)")
grep -aqiE 'out of memory|Cuda failure|cudaError|CUDA_ERROR' "$L" && P+=("CUDA: $(grep -aoiE 'Cuda failure[^\"]*|out of memory' "$L" | head -1)")
grep -aq '"fatal"' "$L" && P+=("fatal: $(grep -ao '{"fatal"[^}]*}' "$L" | head -1)")
SUMM=$(grep -a '"selftestSummary"' "$L" | tail -1)
[ -n "$SUMM" ] || P+=("no selftestSummary line")
OTHER=$(grep -ao '"throw":"[^"]*"' "$L" | grep -v 'bad any_cast' | sort -u | head -3)
[ -z "$OTHER" ] || P+=("probe throws other than the no-secret sentinel: $(echo "$OTHER" | tr '\n' ' ')")
echo "selftest lines:"; grep -a '"selftest"' "$L" | cut -c1-160
echo "summary: $SUMM  (hardFail counts bad any_cast sentinels when no secret is loaded -- expected)"
if [ ${#P[@]} -gt 0 ]; then
  for p in "${P[@]}"; do log "SOVEREIGNTY PROBLEM: $p"; done
  echo "=== VERDICT: FAIL -- STOP. Do not start the server on these keys. ===" | tee -a "$L"
  campaign_log "FAIL sovereignty | ${P[*]} | $L"
  cat <<EOF
DIAGNOSE (index diff, the 2026-09-01 method; keycount.cpp == sessions/final/keycount.cpp == hpc_gpu_port/keycount.cpp):
  1. pod key set for comparison:  KEYGEN_ONLY=1 bash $DEMO_TOOLS/60_fidelity_gate.sh     (writes $KEYS_POD, ~2 min)
  2. build the probe against the demo install:
     I=$WORK/install/include/openfhe; g++ -std=c++17 -O1 -o $DEMO/keycount $POD_SRC/hpc_gpu_port/keycount.cpp \\
       -I\$I -I\$I/third-party/include -I\$I/core -I\$I/pke -I\$I/binfhe -L$WORK/install/lib -lOPENFHEpke -lOPENFHEcore -fopenmp
  3. cat $KEYS/evalrot.part*.bin > $DEMO/mac_evalrot_merged.bin   (if parts)
     $DEMO/keycount $KEYS/cryptocontext.bin $DEMO/mac_evalrot_merged.bin > $DEMO/keys_mac.idx
     $DEMO/keycount $KEYS_POD/cryptocontext.bin $KEYS_POD/evalrot.bin > $DEMO/keys_pod.idx
  4. missing = indices in keys_pod.idx not in keys_mac.idx:
     comm -13 <(grep ^indices $DEMO/keys_mac.idx | tr ' ' '\n' | sort -u) <(grep ^indices $DEMO/keys_pod.idx | tr ' ' '\n' | sort -u)
  5. Mac: re-mint ONLY those (mac_fhe_client keygen ... --extra-autos i1,i2,... or a corrected --indices contract),
     upload the new evalrot.part*.bin only (the merge reads every part, sorted), re-run this gate.
  Also check: the contract vs the .dev RotationIndexes (73 rotation keys matched at 2^15; the gap was bootstrap keys).
EOF
  busy_release; exit 1
fi
echo "=== VERDICT: PASS -- Mac eval material loads and drives the CUDA primitives (no secret on the box) ===" | tee -a "$L"
campaign_log "PASS sovereignty | $SUMM | evalrotBytes $RB | $L"
busy_release
