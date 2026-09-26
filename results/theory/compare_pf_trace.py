#!/usr/bin/env python3
"""Compare the CPU validator's one-layer level trace (harness/cpu_real_model.cpp
--cell pf-trace, real OpenFHE bootstraps at 2^16, FLEXIBLEAUTO) with the
replay's (results/theory/tick_level_trace.py --trace-levels, lazy model) at
the SAME configuration, site by site, in program order.

Usage:
  python3 compare_pf_trace.py CPU.jsonl [--schedule parent-first] [--newton-depth2]
The replay is invoked here with depth/lam/F/margin/I/Dpad/d/dff read from the
CPU cell's own summary line ({"cell":"pf-trace",...}), so the two sides are
guaranteed to describe one configuration. Exit 0 iff every refresh site agrees
on (site name, reported level) and every boot fires at the same site.

What a disagreement means: the replay's level rules (Lemma 7 of
TICK_SCALING_BOUNDS_20260903.md) do not reproduce OpenFHE's GetLevel() at that
statement -- the replay, not the harness, is then wrong there, and the S3.3
predictions inherit the discrepancy. Agreement is the CPU half of invariant I8
(stock) and of the parent-first prediction (schedule = parent-first).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_cpu(path):
    lines, summ = [], None
    for l in open(path):
        l = l.strip()
        if not l.startswith("{"):
            continue
        try:
            d = json.loads(l)
        except json.JSONDecodeError:
            continue
        if d.get("cell") == "pf-trace":
            summ = d
        elif "pfTrace" in d and d["pfTrace"] in ("refresh", "boot"):
            lines.append(d)
    if summ is None:
        sys.exit(f"{path}: no pf-trace summary line (did the cell finish?)")
    return lines, summ


def run_replay(summ, schedule, nd2, pf_entry):
    lam = int(summ["lamObserved"])
    cmd = [sys.executable, os.path.join(HERE, "tick_level_trace.py"),
           "--depth", str(summ["depth"]), "--lam", str(lam), "--boot-floor", str(summ["F"]),
           "--margin", str(summ["margin"]), "--dpad", str(summ["Dpad"]), "--d", str(summ["d"]),
           "--dff", str(summ["dff"]), "--layers", "1", "--iters", str(summ["iters"]), "--ticks", "1",
           "--lazy", "--boot-deg2", "--trace-levels", "--shift-mix",   # OpenFHE: boot output deg 2; the CPU cell applies u*mix
           "--schedule", schedule, "--pf-entry", pf_entry]
    if nd2:
        cmd.append("--newton-depth2")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    lines = []
    for l in out.splitlines():
        if l.startswith("{"):
            d = json.loads(l)
            if d.get("pfTrace") in ("refresh", "boot"):
                lines.append(d)
    return lines, cmd


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    path = a[0]
    schedule = a[a.index("--schedule") + 1] if "--schedule" in a else None
    pf_entry = a[a.index("--pf-entry") + 1] if "--pf-entry" in a else None
    nd2 = "--newton-depth2" in a
    cpu, summ = load_cpu(path)
    schedule = schedule or summ.get("schedule", "stock")
    pf_entry = pf_entry or summ.get("pfEntry", "loop")
    nd2 = nd2 or bool(summ.get("newtonDepth2", False))
    rep, cmd = run_replay(summ, schedule, nd2, pf_entry)
    # the CPU cell stops after the layer; the replay's tick adds the output norm -> compare the common prefix
    n = min(len(cpu), len(rep))
    mism = []
    for i in range(n):
        c, r = cpu[i], rep[i]
        if c["pfTrace"] != r["pfTrace"] or c["site"] != r["site"]:
            mism.append((i, "site/kind", c, r)); break
        if c["pfTrace"] == "refresh" and int(c["level"]) != int(r["level"]):
            mism.append((i, "level", c, r))
        if c["pfTrace"] == "boot" and (int(c["lvlPre"]) != int(r["lvlPre"]) or int(c["lvlPost"]) != int(r["lvlPost"])):
            mism.append((i, "bootLevels", c, r))
    cpu_boots = sum(1 for x in cpu if x["pfTrace"] == "boot")
    rep_boots_prefix = sum(1 for x in rep[:n] if x["pfTrace"] == "boot")
    extra = "" if len(rep) <= n + 8 else f"; replay continues with {len(rep) - n} lines (output norm) not present in the CPU cell"
    print(json.dumps({"compare": "pf-trace", "cpuFile": path, "schedule": schedule, "pfEntry": pf_entry, "newtonDepth2": nd2,
                      "depth": summ["depth"], "lamObserved": summ["lamObserved"], "cpuLines": len(cpu), "replayLines": len(rep),
                      "compared": n, "mismatches": len(mism), "cpuBoots": cpu_boots, "replayBootsInPrefix": rep_boots_prefix,
                      "replayCmd": " ".join(cmd)}))
    for i, kind, c, r in mism[:40]:
        print(f"  MISMATCH #{i} [{kind}]: cpu={json.dumps(c)}  replay={json.dumps(r)}")
    if len(cpu) != n:
        print(f"  note: CPU cell has {len(cpu) - n} lines beyond the replay's{extra}")
    sys.exit(0 if not mism and cpu_boots == rep_boots_prefix else 3)


if __name__ == "__main__":
    main()
