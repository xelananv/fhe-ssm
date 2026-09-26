#!/usr/bin/env python3
"""ARITHMETIC, NOT A MEASUREMENT: the Gaussian-tail probability that a ring-2^17 bootstrap sees max_j |I_j| > B, for the overflow
levels B at which probe B MEASURED each kind of damage in the real library (planted sweep at F = 7, the correction factor the
library's default formula gives at N = 2^17 / 2^16 slots; the F = 10 sweep is listed for comparison). sigma_I comes from the
Hamming weight probe C read off the Mac secret key (probe_c_stats.json); the A11 session ran on OTHER keys (pod test keys), whose
Hamming weight is not known here -- a uniform ternary key has h = 2N/3 +- 171 (one sd), i.e. sigma_I within +-0.1 %.
Writes demo_rate_from_thresholds.md next to itself."""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
def Q(x): return 0.5 * math.erfc(x / math.sqrt(2))
N = 1 << 17; boots = 480
st = json.load(open(os.path.join(HERE, "probe_c_stats.json")))
rows = [("h measured on the Mac key (probe C)", st["hamming"]), ("h = 2N/3 (expected value)", 2 * N / 3)]
L = ["| sigma_I source | h | sigma_I | K/sigma | B | what the planted sweep measured just above B | P(|I_j| > B) per coefficient | per bootstrap | per tick (480 boots) | mean ticks to first event |",
     "|---|---|---|---|---|---|---|---|---|---|"]
what = {512: "edge of the fitted range (no measurable effect up to 515-517)", 518: "F=7 and F=10: median relErrRms first exceeds 2x baseline at 519",
        521: "F=10: median relErrRms >= 1e-3 from 522", 525: "F=7: median relErrRms >= 1e-3 from 526 (F=10: >= 1e-2 from 526)",
        528: "F=7: >= 1e-2 from 529 (F=10: >= 0.1 from 529)", 531: "F=7: >= 0.1 from 532", 534: "F=10: first decode exception at 535",
        538: "F=7: first decode exception at 539", 545: "F=7: every trial throws from 546", 552: "F=7 and F=10: the WHOLE ciphertext is destroyed from 553"}
for label, h in rows:
    sig = math.sqrt((h + 1) / 12)
    for B in sorted(what):
        p1 = 2 * Q((B + 0.5) / sig); pb = 1 - (1 - p1) ** N; pt = 1 - (1 - pb) ** boots
        L.append(f"| {label} | {h:.0f} | {sig:.4f} | {512 / sig:.4f} | {B} | {what[B]} | {p1:.3e} | {pb:.3e} | {pt:.3e} | {1 / pt:.0f} |")
open(os.path.join(HERE, "demo_rate_from_thresholds.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
