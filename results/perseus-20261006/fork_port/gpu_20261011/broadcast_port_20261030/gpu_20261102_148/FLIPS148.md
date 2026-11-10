# 148-tick free run: token flips against the exact model (2026-11-02)

Record `$REPO/results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030/gpu_20261102_148/run_best_free148/record.json`; 148 ticks x 64 lanes = 9472 lane-ticks; decode policy {'policy': 'norepeat', 'penalty': 1.3, 'window': 64, 'ngram': 4, 'ban': [0]}.

- raw top-1 agreement: 9462/9472 = 99.894 % (10 flips)
- decode-rule pick agreement: 9466/9472 = 99.937 % (6 flips)
- exact top-two margin at the raw flips: max 0.0067, median 0.0021 logits

Published, for comparison: FIDESlib 148-tick run raw top-1 9445/9472 (99.715 %, 27 flips, all at margins <= 0.0298),
decode-rule picks 9452/9472 (pod_longrun_20260919/SESSION_REPORT.md:16); Perseus planned 7 flips, eager 73
(ALESSANDRO_REPLY5_20261015.md:21).

## relErrRms and flips by block of ten ticks

- ticks 0-9: median relErrRms 0.00164, raw flips 0
- ticks 10-19: median relErrRms 0.00162, raw flips 0
- ticks 20-29: median relErrRms 0.00158, raw flips 0
- ticks 30-39: median relErrRms 0.00166, raw flips 0
- ticks 40-49: median relErrRms 0.00171, raw flips 2
- ticks 50-59: median relErrRms 0.00176, raw flips 0
- ticks 60-69: median relErrRms 0.00170, raw flips 1
- ticks 70-79: median relErrRms 0.00176, raw flips 0
- ticks 80-89: median relErrRms 0.00183, raw flips 2
- ticks 90-99: median relErrRms 0.00178, raw flips 1
- ticks 100-109: median relErrRms 0.00178, raw flips 0
- ticks 110-119: median relErrRms 0.00178, raw flips 0
- ticks 120-129: median relErrRms 0.00179, raw flips 4
- ticks 130-139: median relErrRms 0.00176, raw flips 0
- ticks 140-147: median relErrRms 0.00181, raw flips 0

## every raw flip (tick, lane, exact top-two margin)

- 45, 22, 0.0035
- 46, 52, 0.0023
- 65, 60, 0.0067
- 84, 51, 0.0012
- 85, 22, 0.0005
- 96, 34, 0.0019
- 122, 44, 0.0027
- 123, 7, 0.0033
- 123, 16, 0.0016
- 127, 11, 0.0015