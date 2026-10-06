#!/usr/bin/env python3
"""Paste generated tables into the hand-written report sources, so that no number in a report is typed by hand twice.
PROBE_B_REPORT.src.md + probe_b_tables.md -> PROBE_B_REPORT.md ;  PROBE_C_REPORT.src.md + probe_c_tables*.md -> PROBE_C_REPORT.md"""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
def sections(path):
    out = {}; cur = None
    for line in open(path):
        m = re.match(r"^## (T\w+)\.", line)
        if m: cur = m.group(1); out[cur] = []; continue
        if cur: out[cur].append(line)
    return {k: "".join(v).strip("\n") for k, v in out.items()}
b = sections(os.path.join(HERE, "probe_b_tables.md"))
src = open(os.path.join(HERE, "PROBE_B_REPORT.src.md")).read()
for k, v in b.items(): src = src.replace("<<%s>>" % k, v)
assert "<<T" not in src, "unfilled table marker in PROBE_B_REPORT.src.md"
open(os.path.join(HERE, "PROBE_B_REPORT.md"), "w").write(src)
cs = os.path.join(HERE, "PROBE_C_REPORT.src.md")
if os.path.exists(cs):
    src = open(cs).read()
    for tag, fn in (("<<C_PRIMARY>>", "probe_c_tables.md"), ("<<C_EXT>>", "probe_c_tables_ext.md"), ("<<TAIL>>", "tail_probe_tables.md"), ("<<TAILC>>", "tail_probe_combined.md"), ("<<RATES>>", "demo_rate_from_thresholds.md")):
        p = os.path.join(HERE, fn)
        if tag in src and os.path.exists(p): src = src.replace(tag, open(p).read().strip("\n"))
    if "<<" in src: print("PROBE_C_REPORT.md NOT written: a marker is still unfilled (its table file does not exist yet)")
    else: open(os.path.join(HERE, "PROBE_C_REPORT.md"), "w").write(src)
print("assembled")
