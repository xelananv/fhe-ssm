# fidelity trend, arm LONG: 148 compared ticks, 9472 lane-ticks

| ticks | mean of per-tick median relErrRms | mean of per-tick 90th pct | raw top-1 flips (margins) | pick-only disagreements | generating lane-ticks |
|---|---|---|---|---|---|
| 0-9 | 0.001045 | 0.002678 | 0 | 0 | 111/640 |
| 10-19 | 0.001422 | 0.003471 | 0 | 1 | 312/640 |
| 20-29 | 0.001530 | 0.003787 | 4 (0.0058, 0.0082, 0.001, 0.0036) | 2 | 320/640 |
| 30-39 | 0.001522 | 0.004641 | 0 | 2 | 353/640 |
| 40-49 | 0.001713 | 0.004331 | 4 (0.0124, 0.0035, 0.0023, 0.0016) | 1 | 462/640 |
| 50-59 | 0.001940 | 0.004557 | 0 | 1 | 480/640 |
| 60-69 | 0.001981 | 0.005230 | 2 (0.0031, 0.0067) | 1 | 500/640 |
| 70-79 | 0.002051 | 0.005227 | 2 (0.0111, 0.0058) | 4 | 591/640 |
| 80-89 | 0.001970 | 0.005696 | 1 (0.0005) | 1 | 640/640 |
| 90-99 | 0.002111 | 0.005570 | 2 (0.0101, 0.0019) | 2 | 640/640 |
| 100-109 | 0.002280 | 0.005428 | 1 (0.0051) | 0 | 640/640 |
| 110-119 | 0.002146 | 0.005116 | 2 (0.0008, 0.0013) | 0 | 640/640 |
| 120-129 | 0.002119 | 0.005503 | 2 (0.0036, 0.0015) | 1 | 640/640 |
| 130-139 | 0.002540 | 0.006136 | 0 | 0 | 640/640 |
| 140-147 | 0.002373 | 0.006040 | 7 (0.0061, 0.0119, 0.0029, 0.0097, 0.0136, 0.0298, 0.0113) | 0 | 512/512 |

- raw top-1 9445/9472 = 99.715 %; largest plaintext top-two margin among the flips: 0.0298 logits; pick-only disagreements 16
- log-log slope of the block level between block centres 5 and 135: 0.269
- log-log slope of the block level between block centres 15 and 135: 0.264
- least-squares power law through all 14 full blocks: level = 0.000704 * tick^0.242 (description of these records only)
- per lane, median relErrRms of ticks 137-147 over ticks 5-15: lanes with ratio > 1: 59 of 64; median ratio 1.88, quartiles 1.61 / 2.76
- five largest lane-ticks: tick 142 lane 32 0.03206; tick 69 lane 17 0.03141; tick 89 lane 17 0.02666; tick 118 lane 1 0.02614; tick 78 lane 17 0.02598
  - lane 32 ticks 140-146: 0.0158, 0.0048, 0.0321, 0.0147, 0.0102, 0.0107, 0.0026
  - lane 17 ticks 67-73: 0.0072, 0.0046, 0.0314, 0.0026, 0.0027, 0.0027, 0.0027
