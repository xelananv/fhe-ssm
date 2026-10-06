# Pass cost vs context position

*demo_20260918T074952Z BUILD_IDENTITY 2026-09-17T22:14:49Z (rebuilt after pod-fix 0b8ee68)*

Sources: `$REPO/results/dense-demo-s31/sessions/demo_20260918T074952Z/client` (transcript.json), `$REPO/results/dense-demo-s31/sessions/demo_20260918T074952Z/pod/server.log` (served lines). Client `serveSec` includes the wire both ways; the server-side clock is `reqMsPerToken` (R6: one binary, one load state per record).

| tick | client serveSec | server reqMsPerToken (s) | reqBoots | reqEncPtMs | storeAppends | peakVramGB | per-device usedGB at serve.done |
|---|---|---|---|---|---|---|---|
| 0 | 306.5 | 139.4 | 528 | 0 |  | 57.74 | 0:57.74;1:57.73 |
| 1 | 243.5 | 136.4 | 480 | 0 |  | 64.78 | 0:64.78;1:63.76 |
| 2 | 248.2 | 137.1 | 480 | 0 |  | 64.78 | 0:64.78;1:63.76 |
| 3 | 268.5 | 136.9 | 480 | 0 |  | 64.81 | 0:64.81;1:63.79 |
| 4 | 269.3 | 136.0 | 480 | 0 |  | 64.81 | 0:64.81;1:63.79 |
| 5 | 271.1 | 136.7 | 480 | 0 |  | 64.81 | 0:64.81;1:63.79 |
| 6 | 260.7 | 137.1 | 480 | 0 |  | 64.84 | 0:64.84;1:63.82 |
| 7 | 260.3 | 137.4 | 480 | 0 |  | 64.84 | 0:64.84;1:63.82 |
| 8 | 264.7 | 136.7 | 480 | 0 |  | 64.84 | 0:64.84;1:63.82 |
| 9 | 255.0 | 135.8 | 480 | 0 |  | 64.87 | 0:64.87;1:63.85 |
| 10 | 264.8 | 136.9 | 480 | 0 |  | 64.87 | 0:64.87;1:63.85 |
| 11 | 265.8 | 136.8 | 480 | 0 |  | 64.90 | 0:64.90;1:63.88 |
| 12 | 260.2 | 138.1 | 480 | 0 |  | 64.90 | 0:64.90;1:63.88 |
| 13 | 271.8 | 137.3 | 480 | 0 |  | 64.90 | 0:64.90;1:63.88 |
| 14 | 266.5 | 138.2 | 480 | 0 |  | 64.93 | 0:64.93;1:63.91 |
| 15 | 264.2 | 137.6 | 480 | 0 |  | 64.93 | 0:64.93;1:63.91 |
| 16 | 286.8 | 136.2 | 480 | 0 |  | 64.93 | 0:64.93;1:63.91 |
| 17 | 245.7 | 136.9 | 480 | 0 |  | 64.96 | 0:64.96;1:63.95 |

## Flatness reading (A11 criteria)

- (a) store converged: yes -- reqEncPtMs > 0 at ticks none; storeAppends not recorded
- (b) reqBoots constant: yes -- [480]
- (c) no trend in server reqMsPerToken: yes -- median 136.9 s; median(first 8) 136.8 s vs median(last 8) 137.1 s = +0.2 % (bar +-3.0 %); ticks outside +-10 % of the median: none; min 135.8, max 138.2
- (d) peakVramGB flat: yes -- min 64.78, max 64.96 GB (bar: spread <= 0.5 GB)

**Verdict: partial (17 warm ticks < 64): no trend detected, NOT a flatness claim.**

