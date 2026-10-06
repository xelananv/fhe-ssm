# CTRL512 (arm B) against LONG (arm A): 8 ticks in B, 512 paired lane-ticks of 512 (paired = identical fed history in both arms)

| tick | pairs | median relErrRms A | median relErrRms B | ratio of medians B/A | median of per-pair B/A | pairs with B > A |
|---|---|---|---|---|---|---|
| 0 | 64 | 0.000798 | 0.001114 | 1.396 | 1.453 | 47 of 64 |
| 1 | 64 | 0.000960 | 0.001449 | 1.511 | 1.821 | 44 of 64 |
| 2 | 64 | 0.000805 | 0.001293 | 1.607 | 1.763 | 48 of 64 |
| 3 | 64 | 0.001125 | 0.001613 | 1.434 | 1.462 | 44 of 64 |
| 4 | 64 | 0.001086 | 0.001732 | 1.596 | 1.533 | 45 of 64 |
| 5 | 64 | 0.001279 | 0.002038 | 1.593 | 1.795 | 46 of 64 |
| 6 | 64 | 0.001085 | 0.002077 | 1.914 | 1.739 | 51 of 64 |
| 7 | 64 | 0.000982 | 0.001754 | 1.786 | 1.785 | 46 of 64 |
| all | 512 | 0.000973 | 0.001602 | 1.645 | 1.586 | 371 of 512 |

- sign test over all pairs: B > A in 371 of 512 (z = +10.16); mean of ln(B/A) = +0.4522 +- 0.0365 (s.e.) => geometric-mean ratio B/A = 1.572 [1.461, 1.691] (2 s.e.)
- arm LONG on the paired lane-ticks: raw top-1 512/512, picks 512/512, decode-failed rows 0
- arm CTRL512 on the paired lane-ticks: raw top-1 512/512, picks 512/512, decode-failed rows 0

## arm CTRL512: library line `{"fideslibBootTable": "g_coefficientsUniform", "tableK": 512, "bootK": 512, "scaleEncOddPart": 1, "doubleAngleIts": 6, "chebyDegree": 88, "envBootUniformExt": "0", "splitAskedButOpenfheLacksO1": false`; bootstraps with max overflow > 480 (its table's bound is 512)

| tick | bootstrap n | max overflow | stage | the tick in B: top-1 | median relErrRms | max relErrRms | same tick in A: median | max |
|---|---|---|---|---|---|---|---|---|
| 6 | 3631 | 483 | `L20.cm.hidden.ch0.t0` | 64/64 | 0.002077 | 0.005651 | 0.001085 | 0.005342 |
