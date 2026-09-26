#!/bin/bash
# idle-GPU filler while the artifacts upload: 6 more selftest repeats per bootstrap arm on the BASE binary (the binary of cell 1b),
# interleaved arm by arm so that drift cannot masquerade as an arm effect. Stops early when /root/demo/STOP_MORE exists.
. /root/demo/longrun.env; D=/root/src/results/dense-demo-s31/pod_tools/demo; L=/root/demo/boot_arms_more.log
for r in 1 2 3 4 5 6; do for arm in 0 1 split; do
  [ -e /root/demo/STOP_MORE ] && { echo "$(date -u +%H:%M:%SZ) stopped by STOP_MORE" >> $L; exit 0; }
  C="bootarm2_${arm}_r${r}"
  FIDESLIB_BOOT_UNIFORM_EXT=$arm BINX=/root/fhe-main-demo/build-demo/gpu_real_model_x.base CELL=$C MODE=selftest XBIN=1 FLAGS="--trace-boots" bash $D/90_s37_cell.sh > /root/demo/$C.cell.log 2>&1
  echo "$(date -u +%H:%M:%SZ) $C $(grep -a "\"selftest\":\"bootstrap\"" /root/demo/s37/$C/console.log | tail -1) $(grep -ao "\"tableK\":[0-9]*,\"bootK\":[0-9]*" /root/demo/s37/$C/console.log | tail -1)" >> $L
done; done
echo "$(date -u +%H:%M:%SZ) MORE_DONE" >> $L
