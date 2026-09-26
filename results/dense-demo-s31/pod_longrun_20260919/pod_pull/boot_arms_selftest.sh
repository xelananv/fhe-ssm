#!/bin/bash
# Cell 1b of POD_RUN_PLAN.md: the two-card selftest of the NEW binary under the three bootstrap arms, REPS times each,
#   0      stock table            (g_coefficientsUniform,    degree  88, scalar 1/(512 N))
#   1      Ext table, variant A   (g_coefficientsUniformExt, degree 118, scalar 1/(768 N): rounded constant, see K768_CPU_REPORT.md 2.5)
#   split  Ext table, variant B   (same table, scalar 1/(256 N), the 3 inside the CoeffsToSlots plaintexts: needs OpenFHE patch O1)
# and ONE table: which table/K each process really used (the library's own lines), the selftest's bootstrap error (max-abs over
# 1024 channels of ONE bootstrap: a coarse number, R3 -- it catches a broken arm, it does not rank close ones), lvlPost, hardFail.
# It ends with the arm the long run should use, by this rule (written before any of it ran):
#   split if its library line shows scaleEncOddPart 3, every repeat passes (selftest bootstrap ok, hardFail 0, level moved) and its mean err <= 2 x stock's;
#   else 1 under the same conditions; else STOP (the stock table is what failed at tick 42; do not start a long run on it silently).
# The result is written to /root/demo/BOOT_ARM (sourced by longrun.env). ~40 s per process.
set -u
. /root/demo/longrun.env; D=/root/src/results/dense-demo-s31/pod_tools/demo; S=/root/demo; L=$S/boot_arms_selftest.log; REPS=${REPS:-2}
u() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a $L; }
u "=== boot arms selftest: bin $BINX ($(sha256sum $BINX | cut -c1-16)), REPS=$REPS ==="
for arm in 0 1 split; do for r in $(seq 1 $REPS); do
  C="bootarm_${arm}_r${r}"; u "run $C"
  FIDESLIB_BOOT_UNIFORM_EXT=$arm CELL=$C MODE=selftest XBIN=1 FLAGS="--trace-boots" bash $D/90_s37_cell.sh > $S/$C.cell.log 2>&1
done; done
python3 - "$REPS" <<'PY' | tee -a $L
import json, re, sys, glob, os
reps = int(sys.argv[1]); S37 = os.environ.get("S37", "/root/demo/s37"); OUT = os.environ.get("BOOT_ARM_DIR", "/root/demo"); rows = {}
def jl(path, key):
    out = []
    for line in open(path, errors="replace"):
        line = line.strip()
        if line.startswith("{") and ('"%s"' % key) in line:
            try: out.append(json.loads(line))
            except Exception: pass
    return out
for arm in ("0", "1", "split"):
    rows[arm] = []
    for r in range(1, reps + 1):
        p = "%s/bootarm_%s_r%d/console.log" % (S37, arm, r)
        if not os.path.exists(p): rows[arm].append(None); continue
        tab = jl(p, "fideslibBootTable"); o1 = jl(p, "openfheBootScaleEnc"); st = [x for x in jl(p, "selftest") if x.get("selftest") == "bootstrap"]
        summ = jl(p, "selftestSummary"); bt = jl(p, "bootTrace")
        rows[arm].append({"table": tab[-1] if tab else None, "o1Line": bool(o1), "boot": st[-1] if st else None, "summary": summ[-1] if summ else None,
                          "lvlPost": sorted({b.get("lvlPost") for b in bt}), "restored": all(b.get("bootRestored") for b in bt) if bt else None,
                          "moved": all(b.get("lvlPost") != b.get("lvlPre") for b in bt) if bt else None})
print("| arm | rep | table | tableK | bootK (scalar) | scaleEncOddPart | OpenFHE O1 line | chebyDegree | selftest bootstrap | err (max-abs) | lvlPost | bootRestored (lvlPost<lvlPre: false for a shallow input) | hardFail | softFail |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
# 2026-09-19 (first use on the pod): the harness's bootRestored flag is lvlPost < lvlPre. The selftest bootstraps a ciphertext at
# level 3, so a CORRECT bootstrap lands deeper (19) and the flag is false in every arm, stock included: it is not a criterion here.
# What is required instead: the selftest's own value check (result ok), hardFail 0, and a bootTrace whose lvlPost differs from lvlPre.
def ok(x): return x is not None and x["boot"] is not None and x["boot"].get("result") == "ok" and x["summary"] is not None and x["summary"].get("hardFail") == 0 and x["moved"] is not False
mean = {}
for arm in ("0", "1", "split"):
    errs = []
    for r, x in enumerate(rows[arm], 1):
        if x is None: print("| %s | %d | NO LOG |" % (arm, r)); continue
        t = x["table"] or {}; b = x["boot"] or {}; s = x["summary"] or {}
        print("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (arm, r, t.get("fideslibBootTable"), t.get("tableK"), t.get("bootK"), t.get("scaleEncOddPart"),
              x["o1Line"], t.get("chebyDegree"), b.get("result", b.get("throw")), b.get("err"), x["lvlPost"], x["restored"], s.get("hardFail"), s.get("softFail")))
        if ok(x): errs.append(float(b["err"]))
    mean[arm] = (sum(errs) / len(errs)) if len(errs) == reps else None
print("mean err per arm (None = some repeat failed or is missing):", mean)
def tbl(arm, k, v): return all(x is not None and (x["table"] or {}).get(k) == v for x in rows[arm])
choice, why = None, ""
if mean["0"] is None: why = "the STOCK arm did not pass: the binary or the box is broken, nothing to choose yet"
elif mean["split"] is not None and tbl("split", "scaleEncOddPart", 3) and tbl("split", "tableK", 768) and mean["split"] <= 2 * mean["0"]: choice, why = "split", "variant B passes and is within 2x of the stock table's selftest error"
elif mean["1"] is not None and tbl("1", "tableK", 768) and mean["1"] <= 2 * mean["0"]: choice, why = "1", "variant B unavailable or failing; variant A passes and is within 2x of the stock table's selftest error"
else: why = "neither K = 768 arm passes within 2x of the stock table: STOP and read the logs (do not start the long run on the stock table silently)"
print("BOOT ARM DECISION:", choice, "--", why)
open(OUT + "/BOOT_ARM.decision.txt", "w").write("%s -- %s\n" % (choice, why))
if choice: open(OUT + "/BOOT_ARM", "w").write("export FIDESLIB_BOOT_UNIFORM_EXT=%s\n" % choice)
PY
u "BOOT_ARMS_DONE: $(cat $S/BOOT_ARM.decision.txt 2>/dev/null)"
