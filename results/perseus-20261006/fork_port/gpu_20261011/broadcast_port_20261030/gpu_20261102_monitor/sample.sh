#!/bin/bash
# one row every ${1:-5} s: GPU util/mem-util/VRAM/power/temp, host CPU busy % (all 20 cores, /proc/stat delta), host RAM, load
F=/root/monitor/usage_$(date -u +%Y%m%d).csv
[ -f "$F" ] || echo "utc,gpu_util_pct,gpu_memutil_pct,vram_used_mib,vram_total_mib,power_w,temp_c,cpu_busy_pct,ram_used_mib,ram_total_mib,load1" > "$F"
read -r _ u n s i w q sq st _ < /proc/stat; pt=$((u+n+s+i+w+q+sq+st)); pi=$((i+w))
while true; do
  sleep "${1:-5}"
  read -r _ u n s i w q sq st _ < /proc/stat; t=$((u+n+s+i+w+q+sq+st)); id=$((i+w))
  cpu=$(awk -v dt=$((t-pt)) -v di=$((id-pi)) "BEGIN{printf \"%.1f\", (dt>0)?100*(dt-di)/dt:0}"); pt=$t; pi=$id
  g=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits | tr -d " ")
  m=$(free -m | awk "/^Mem:/{print \$3\",\"\$2}")
  echo "$(date -u +%FT%TZ),$g,$cpu,$m,$(cut -d" " -f1 /proc/loadavg)" >> "$F"
done
