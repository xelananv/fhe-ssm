#!/usr/bin/env bash
# 00_box_preflight.sh -- is this box able to host the fully-secure demo? (read-only)
#
# PURPOSE   Before any upload or build: GPU / RAM / disk / CPU / toolchain
#           inventory, the three hard resource gates, and the INBOUND
#           bandwidth gate that must pass BEFORE the ~50 GB key upload starts
#           (the 2026-09-01 pod's uplink collapsed to ~35 KB/s --
#           results/dense-demo-s31/RECOVERY_STATE_20260901.md:3-4).
# INPUTS    env per _lib.sh; optional BW_SECONDS=<seconds the Mac's 512 MiB scp
#           took> (after the operator ran the printed scp command).
#           Thresholds (env): DISK_MIN_GB=450 RAM_MIN_GB=384 RAM_GO_GB=512 (2026-09-02: store
#           ~312 GB warm at 2^17 + Mac keys ~48 GB host + CPU precompute; file mirrors the store)
#           VRAM_RISK_GIB=93 VRAM_GO_GIB=140 BW_MIN_MBPS=5
#           (2026-09-03: 93, not 94 -- an H100 NVL card reports 95,830 MiB = 93.58 GiB,
#           integer 93, and it is a legal TWO-CARD box: below VRAM_GO_GIB per card the
#           demo config runs only sharded, export DEVICES=0,1 + 41_mgpu_ladder.sh.)
#           P2P_OVERRIDE=1 lifts the peer-capability NO-GO, only after 41 rung 0 passed.
# OUTPUTS   $DEMO/preflight.txt (verbatim tool output + gate table),
#           one CAMPAIGN_LOG.md line "GO" / "NO-GO" / "GO-WITH-RISK".
# PASS      disk >= DISK_MIN_GB (450), RAM >= RAM_MIN_GB (384; 512 = GO), >= 1 GPU,
#           inbound >= 5 MB/s when measured (PENDING otherwise). VRAM and RAM
#           also print an on-record risk class -- see the notes printed at the end.
#           Two or more cards: PEER CAPABILITY is gated from `nvidia-smi topo -p2p w`
#           (GPU0->GPU1 "OK" = GO; NS/CNS/GNS/TNS = NO-GO when the per-card VRAM is
#           below VRAM_GO_GIB, because the demo config fits only sharded; U/unparsed
#           = RISK). The link TYPE (`topo -m`: NV* vs PIX/PXB/PHB) is printed as
#           information only -- it is bandwidth, not viability: FIDESlib printed
#           `GPU P2P? 1` on the PCIe-only 2x RTX PRO 6000 box (demo_smoke/mgpu17.log:5)
#           and on the 4x A100 80 GB PCIe box whose 2-device 2^17 runs completed
#           (a100-secure-20260730/mgpu2_secure_d59.jsonl), while the 2026-09-03 box
#           (PHB, no P2P exposed) failed rung 0 (pod_s33_20260903/demo/mgpu_rung0.log).
#           41 rung 0 (peer_pool_probe) remains the authority for POOL memory.
#           (The task brief's 200 GB / 128 GB floors were raised on 2026-09-02
#           after the store-growth finding below; both stay overridable by env.)
# RECORD    preflight.txt is the box identity every later record cites.
#
# Why 450 GB disk: Mac eval keys ~50.8 GB (see 30_dump_indices.sh arithmetic) +
#   pod keys 46.5 GB (s31_t31_keys_r17/console.log:1 evalRotKeyBytes 46,507,938,432)
#   + the store FILE, which mirrors the host store: 61 GB after a T=1 pass
#   (s31_t1b_store_secure_r17/console.log:18 storeBytes 60,850,962,432) and up to
#   the ~312 GB warm-session class below + bundles 3.45 + 1.31 GB + weights
#   1.82 GB + toolchain/venv.
# Why 384 GB RAM minimum, 512 GB = GO: the single-lane 2^17 store is
#   60.85 GB host (above). The store key carries the ciphertext LEVEL
#   (gpu_real_model.cu:1299-1300 `struct CompKey {... uint32_t lvl;}`), so a
#   STATEFUL session keeps adding (diagonal, level) entries as the carry's level
#   schedule drifts: the 2^15 lanes-block stateful server reached storeBytes
#   275,917,520,896 after 13 ticks (sessions/final/demo_smoke/genRb/logs/
#   server.jsonl summary) against 53,854,863,360 for the single-lane 2^15 store
#   (pod/state/runs/s31_t1b_store_block_r15/console.log summary) = 5.12x.
#   Scaled to the 2^17 single-lane store that is ~312 GB of host RAM. Entries
#   are never evicted (only appended, gpu_real_model.cu:2482-2483).
# Why 140 GiB VRAM recommended: keys-load, stateless, single-lane, 132 keys
#   peaked procPeakVramGB 92.2676 on the 95.6 GiB card
#   (pod/state/runs/s31_oom_keysload_probe/console.log summary); the lanes keys
#   add 143-132 = 11 keys = 48,048 - 44,352 = 3,696 MB device
#   (demo_smoke/lanes17.log:7 vs s31_t1b_store_secure_r17/console.log:5); the
#   stateful carry (2 cts x 24 layers) is unmeasured on top; the 4-lane
#   lanes-block batch run OOMed at xd12 (lanes17.log:19) and at xd10
#   (lanes17_ladder.log:20 -- its "FITS at xd10" line 22 is the F74 false pass).
set -uo pipefail
SCRIPT=$(basename "$0")
. "$(dirname "$(readlink -f "$0")")/_lib.sh"
: "${DISK_MIN_GB:=450}"; : "${RAM_MIN_GB:=384}"; : "${RAM_GO_GB:=512}"
: "${VRAM_RISK_GIB:=93}"; : "${VRAM_GO_GIB:=140}"; : "${BW_MIN_MBPS:=5}"
mkdir -p "$DEMO"
OUT="$DEMO/preflight.txt"
{
echo "=== PREFLIGHT $(utc) host=$(hostname) ==="
echo "--- nvidia-smi (index, name, memory.total, compute_cap) ---"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader 2>&1
echo "--- nvidia-smi topo -p2p w (two-card route: GPU0->GPU1 must be OK; NS/CNS = no peer path, the 2026-09-03 box) ---"
nvidia-smi topo -p2p w 2>&1
echo "--- nvidia-smi topo -m (link TYPE: NV* = NVLink bandwidth, PIX/PXB/PHB = PCIe; information, not the gate) ---"
nvidia-smi topo -m 2>&1
echo "--- nvidia-smi compute apps (must be empty for any timing) ---"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1
echo "--- free -g ---"; free -g 2>&1
echo "--- df -h /root ---"; df -h /root 2>&1
echo "--- nproc ---"; nproc 2>&1
echo "--- toolchain ---"
for c in nvcc cmake ninja g++ git rsync python3 readelf; do
  printf '%-8s %s\n' "$c" "$(command -v $c 2>/dev/null || ls /usr/local/cuda*/bin/$c 2>/dev/null | head -1 || echo MISSING)"
done
[ -x /usr/local/cuda-12.9/bin/nvcc ] && /usr/local/cuda-12.9/bin/nvcc --version | tail -1
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | sed 's/^/driver /'
echo "--- uplink DNS sanity (198.18/15 = reserved benchmark range = tunnelled, RECOVERY_STATE:4-5) ---"
hostname -I 2>/dev/null || true
} | tee "$OUT"

# ---------------------------------------------------------------- gates
NOGO=0; RISK=0
GPUS=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | grep -c . || echo 0)
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
VRAM_GIB=$(( ${VRAM_MIB:-0} / 1024 ))
RAM_GB=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1000000 ))
DISK_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
NPROC=$(nproc)
NVCC_OK=0; { command -v nvcc >/dev/null 2>&1 || [ -x /usr/local/cuda-12.9/bin/nvcc ]; } && NVCC_OK=1

gate() { printf '  %-10s %-34s %s\n' "$1" "$2" "$3"; }
# NOTE: a brace group with a plain redirection runs in THIS shell (a `| tee`
# pipeline would fork it and lose NOGO/RISK/V). Printed afterwards.
G="$OUT.gates"
{
echo "=== GATES ==="
if [ "${GPUS:-0}" -lt 1 ]; then gate NO-GO "GPUs: $GPUS" "need >= 1"; NOGO=1
else gate GO "GPUs: $GPUS x ${VRAM_MIB} MiB (${VRAM_GIB} GiB)" ""; fi
if [ "${GPUS:-0}" -ge 2 ]; then
  # GPU0 DATA row of either matrix, whitespace-split: GPU0  X  <cell for GPU1> ...
  # ($2=="X" skips the header row, whose first token is also GPU0 once the
  #  leading tab is trimmed -- caught by the sample test on 2026-09-03.)
  P2P01=$(nvidia-smi topo -p2p w 2>/dev/null | awk '$1=="GPU0" && $2=="X"{print $3; exit}')
  LINK01=$(nvidia-smi topo -m 2>/dev/null | awk '$1=="GPU0" && $2=="X"{print $3; exit}')
  case "$LINK01" in NV*) LT="NVLink ($LINK01)";; PIX|PXB|PHB|NODE|SYS) LT="PCIe ($LINK01)";; *) LT="link '$LINK01'";; esac
  case "$P2P01" in
    OK)  gate GO "GPU0->GPU1 P2P write OK over $LT" "peer capability exposed (P2P=1 class of mgpu17.log:5 / the A100 PCIe box); 41 rung 0 (peer_pool_probe) still decides POOL access before any two-card store run";;
    NS|CNS|GNS|TNS)
      if [ "$VRAM_GIB" -ge "$VRAM_GO_GIB" ]; then gate RISK "GPU0->GPU1 P2P $P2P01 over $LT" "no peer path (the 2026-09-03 box class); single-card route only"; RISK=1
      elif [ "${P2P_OVERRIDE:-0}" = 1 ]; then gate RISK "GPU0->GPU1 P2P $P2P01, P2P_OVERRIDE=1" "operator asserts 41 rung 0 passed"; RISK=1
      else gate NO-GO "GPU0->GPU1 P2P $P2P01 over $LT and per-card VRAM ${VRAM_GIB} GiB < ${VRAM_GO_GIB}" "the demo config fits only sharded and this box exposes no peer path (2026-09-03: pod_s33_20260903/demo/mgpu_rung0.log no-peer-capability); P2P_OVERRIDE=1 only after rung 0 passes"; NOGO=1; fi;;
    *)   gate RISK "GPU0->GPU1 P2P status unparsed ('$P2P01') over $LT" "read the -p2p matrix above; run 41 rung 0 right after the toolchain, before any upload"; RISK=1;;
  esac
fi
if [ "$VRAM_GIB" -ge "$VRAM_GO_GIB" ]; then gate GO "VRAM ${VRAM_GIB} GiB >= ${VRAM_GO_GIB}" "keys+precomp+store+carry class fits"
elif [ "$VRAM_GIB" -ge "$VRAM_RISK_GIB" ]; then
  if [ "${GPUS:-0}" -ge 2 ]; then gate RISK "VRAM ${VRAM_GIB} GiB per card in [${VRAM_RISK_GIB},${VRAM_GO_GIB}), $GPUS cards" "single-card records: 92.27 GB keys-load stateless, 94.28 GB T=64 pass -- the demo config runs SHARDED here: export DEVICES=0,1, 41_mgpu_ladder.sh rungs 0-3 first (fit model ~72 GB/card today, ~60 with the key patch, MGPU_FIT_MODEL_20260903.md 2)"; RISK=1
  else gate RISK "VRAM ${VRAM_GIB} GiB in [${VRAM_RISK_GIB},${VRAM_GO_GIB}), one card" "on-record: 92.27 GB keys-load stateless + 3.7 GB lanes keys + unmeasured carry; the 4-lane lanes-block batch OOMed on 95.6 GiB -- 40/60 decide, expect a second card to be needed"; RISK=1; fi
else gate NO-GO "VRAM ${VRAM_GIB} GiB < ${VRAM_RISK_GIB}" "single-lane 2^17 store already peaked 90.34 GB"; NOGO=1; fi
if [ "$RAM_GB" -ge "$RAM_GO_GB" ]; then gate GO "RAM ${RAM_GB} GB >= ${RAM_GO_GB}" ""
elif [ "$RAM_GB" -ge "$RAM_MIN_GB" ]; then gate RISK "RAM ${RAM_GB} GB in [${RAM_MIN_GB},${RAM_GO_GB})" "stateful store grew 5.12x at 2^15 (genRb) -> ~312 GB class at 2^17; 60 measures the growth"; RISK=1
else gate NO-GO "RAM ${RAM_GB} GB < ${RAM_MIN_GB}" "T=1 store is 60.85 GB host; a stateful session grew it 5.12x at 2^15 -> ~312 GB class + ~48 GB Mac keys host"; NOGO=1; fi
if [ "${DISK_GB:-0}" -ge "$DISK_MIN_GB" ]; then gate GO "disk /root ${DISK_GB} GB free >= ${DISK_MIN_GB}" ""
else gate NO-GO "disk /root ${DISK_GB} GB free < ${DISK_MIN_GB}" "keys 50.8+46.5 GB + store FILE 61..~312 GB (mirrors the host store) + bundles 4.8 GB + weights 1.8 GB"; NOGO=1; fi
gate INFO "nproc $NPROC" "encode uses --enc-threads $ENC_THREADS (ompMaxThreads in the header line must read $ENC_THREADS)"
[ "$NPROC" -lt "$ENC_THREADS" ] && gate RISK "nproc $NPROC < ENC_THREADS $ENC_THREADS" "export ENC_THREADS=$NPROC (timings then not comparable to the 24-thread records, R6)"
if [ "$NVCC_OK" = 1 ]; then gate GO "nvcc present" ""; else gate INFO "nvcc missing" "10_bringup.sh runs 00_toolchain.sh (apt cuda-toolkit-12-9; untimed on record)"; fi
command -v rsync >/dev/null 2>&1 && gate GO "rsync present" "" || gate INFO "rsync missing" "10_bringup.sh installs it (20_stage needs it on the receiving side)"
echo "=== INBOUND BANDWIDTH GATE (run BEFORE any key upload) ==="
echo "  On the MAC, run (fill KEY/PORT/HOST):"
echo "    dd if=/dev/urandom bs=1m count=512 of=/tmp/bw512.bin"
echo "    time scp -i KEY -P PORT /tmp/bw512.bin root@HOST:/root/bw512.bin"
echo "  MB/s = 512 / real-seconds. Then on the pod: BW_SECONDS=<real> $SCRIPT"
echo "  Need >= ${BW_MIN_MBPS} MB/s (40 Mbit/s): the ~50.8 GB eval-key upload takes 50.8e3/MBps s"
echo "  (68 min at 12.5 MB/s = 100 Mbit; 2.8 h at 5 MB/s; 17 DAYS at the 35 KB/s of 2026-09-01)."
if [ -n "${BW_SECONDS:-}" ]; then
  B=$(file_bytes /root/bw512.bin)
  if [ "$B" -ge 536870912 ]; then
    MBPS=$(python3 -c "print(round(512/float('$BW_SECONDS'),2))")
    UP_MIN=$(python3 -c "print(round(50.8e3/float('$MBPS')/60,1))")
    if python3 -c "import sys; sys.exit(0 if float('$MBPS') >= float('$BW_MIN_MBPS') else 1)"; then
      gate GO "inbound ${MBPS} MB/s (512 MiB in ${BW_SECONDS}s)" "50.8 GB key upload ~${UP_MIN} min"
    else gate NO-GO "inbound ${MBPS} MB/s < ${BW_MIN_MBPS}" "key upload ~${UP_MIN} min -- change box/route first"; NOGO=1; fi
  else gate PENDING "/root/bw512.bin is $B bytes (want 536870912)" "scp not finished or wrong path"; fi
else gate PENDING "inbound bandwidth not measured" "re-run with BW_SECONDS=<seconds>"; fi
echo "=== NOTES (records) ==="
echo "  VRAM: keys-load stateless single-lane 132 keys peaked 92.2676 GB (s31_oom_keysload_probe/console.log);"
echo "        lanes keys +3,696 MB (lanes17.log:7 48,048 MB vs store record :5 44,352 MB); 4-lane batch OOMed"
echo "        at xd12 (lanes17.log:19) and xd10 (lanes17_ladder.log:20); the T=64 stateless pass peaked 94.28 GB"
echo "        card-wide (x_clockv2_dense_t64). TWO-CARD ROUTE (2026-09-03, author): --devices 0,1 --store-multi-gpu,"
echo "        limbs sharded by global id, peer writes need cudaMemPoolSetAccess (peer_pool_probe = 41 rung 0);"
echo "        never yet exercised on hardware (the 2026-09-03 box exposed no P2P; the 2026-09-01 2x RTX PRO 6000 box,"
echo "        PCIe only, printed GPU P2P? 1 and its OOM is unattributed; the 4x A100 PCIe box completed 2-device 2^17"
echo "        runs, mgpu2_secure_d59). PCIe vs NVLink is bandwidth, not viability. Export DEVICES=0,1, run 41 first."
echo "  RAM : store 60,850,962,432 B single-lane (s31_t1b_store_secure_r17/console.log:18); stateful+lanes-block"
echo "        at 2^15 grew to 275,917,520,896 B in 13 ticks (genRb/logs/server.jsonl) = 5.12x the 53,854,863,360 B"
echo "        single-lane 2^15 store (s31_t1b_store_block_r15/console.log). Never evicted (gpu_real_model.cu:2482)."
echo "        S3.3 replay (results/theory/S33_REPORT_20260903.md 4): under the STOCK schedule the 2^17 store converges"
echo "        to ~1,800 distinct keys ~ 637 GB over ~20 ticks (parent-first ~250 GB) -- a ${RAM_GO_GB} GB box covers a"
echo "        short timing cell and a ~12-tick session, not a 24-tick stock-schedule session."
if [ "$NOGO" = 1 ]; then echo "=== VERDICT: NO-GO ==="; V=NO-GO
elif [ "$RISK" = 1 ]; then echo "=== VERDICT: GO-WITH-RISK (see RISK rows) ==="; V=GO-WITH-RISK
else echo "=== VERDICT: GO ==="; V=GO; fi
} > "$G"
cat "$G" | tee -a "$OUT"; rm -f "$G"
campaign_log "$V | gpus=$GPUS vram=${VRAM_GIB}GiB ram=${RAM_GB}GB disk=${DISK_GB}GB nproc=$NPROC nvcc=$NVCC_OK bw=${BW_SECONDS:-unmeasured}s/512MiB | $OUT"
[ "$V" = NO-GO ] && exit 1
exit 0
