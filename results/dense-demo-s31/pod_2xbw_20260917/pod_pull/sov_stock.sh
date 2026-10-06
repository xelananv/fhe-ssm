#!/bin/bash
# waits for the STOCK Mac key set (manifest verified, no secret), then sovereignty + the no-eps selftest control,
# queued on the busy marker (slots in between lever arms).
set -u
. /root/demo/s37.env; D=/root/src/results/dense-demo-s31/pod_tools/demo; L=/root/demo/sov_stock.log
u() { echo "$(date -u +%H:%M:%SZ) $*" >> $L; }
keyset_ok() { ( cd "$1" 2>/dev/null && [ -s MANIFEST.sha256 ] && [ ! -e secret.key ] && sha256sum -c --quiet MANIFEST.sha256 >/dev/null 2>&1 ); }
until keyset_ok /root/mac_keys_r17; do sleep 60; done
u "stock set verified on the pod"
u "=== sovereignty STOCK ==="; KEYS=/root/mac_keys_r17 bash $D/50_sovereignty.sh >> $L 2>&1; u "sovereignty stock rc=$?"
u "=== selftest stock keys --no-norm-eps (V1 control: rmsnormSeeded must FAIL ~0.028) ==="; KEYS=/root/mac_keys_r17 CELL=selftest_stock_noeps MODE=selftest KEYS_LOAD=1 FLAGS=--no-norm-eps bash $D/90_s37_cell.sh >> $L 2>&1; u "selftest noeps rc=$?"
u "SOV_STOCK_DONE"
