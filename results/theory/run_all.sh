#!/usr/bin/env bash
# Regenerates every derived table quoted in TICK_SCALING_BOUNDS_20260903.md.
# Pure replay of gpu_real_model.cu's level/operation rules; reads nothing under results/.
set -euo pipefail
cd "$(dirname "$0")"
python3 tick_reports.py probe  > trace_probe_20260903.txt
python3 tick_reports.py grid   > trace_grid_20260903.txt
python3 tick_reports.py sweep  > trace_lamsweep_20260903.txt
python3 tick_reports.py sites  > trace_sites_20260903.txt
python3 tick_reports.py sanity > trace_sanity_cell_20260903.txt
python3 tick_reports.py variants > trace_variants_20260903.txt
python3 tick_reports.py schedule > trace_schedule_20260903.txt   # S3.3 schedule levers (SCHEDULE_DESIGN_20260903.md)
python3 tick_bounds_tables.py --md        --json tick_bounds_tables_eager.json > tick_bounds_tables_eager.md
python3 tick_bounds_tables.py --md --lazy --json tick_bounds_tables_lazy.json  > tick_bounds_tables_lazy.md
echo "done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
