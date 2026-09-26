# logit-error headroom, arm LONG: 148 ticks, 9472 lane-ticks

## records: the plaintext model's top-two margin (logits) and the flips

- margin percentiles: p1 0.0137, p5 0.0745, p10 0.1394, p25 0.3799, p50 0.9251, p75 1.8820
- raw top-1 flips: 27 of 9472 = 0.285 %; largest margin among them 0.0298; lane-ticks with a margin at or below that: 187
- margin density near zero 0.625 per logit => std of the error of the top-two logit difference 0.0114 logits at the run's median relErrRms 0.001849; the MEDIAN decision margin is 81 of those stds away

| error level | relErrRms | expected raw top-1 flips per lane-tick | top-1 agreement |
|---|---|---|---|
| x1 | 0.0018 | 0.32 % | 99.68 % |
| x2 | 0.0037 | 0.61 % | 99.39 % |
| x3 | 0.0055 | 0.91 % | 99.09 % |
| x5 | 0.0092 | 1.55 % | 98.45 % |
| x10 | 0.0185 | 3.17 % | 96.83 % |
| x30 | 0.0555 | 8.85 % | 91.15 % |
| x100 | 0.1849 | 21.72 % | 78.28 % |

## the fitted power law (a DESCRIPTION of ticks 5..135, not a law): level = 0.000704 * tick^0.242; block levels: 0.001045, 0.001422, 0.001530, 0.001522, 0.001713, 0.001940, 0.001981, 0.002051, 0.001970, 0.002111, 0.002280, 0.002146, 0.002119, 0.002540

| error level | reached at tick (if the power law held) | serving time at 176 s per tick |
|---|---|---|
| x1.5 of the last block (0.0038) | 1,065 | 2.2 days |
| x2 of the last block (0.0051) | 3,493 | 7.1 days |
| x3 of the last block (0.0076) | 18,632 | 38.0 days |
| x5 of the last block (0.0127) | 153,558 | 312.8 days |
| x10 of the last block (0.0254) | 2,686,759 | 5,473.0 days |

## derivation from the weights: the model's memory is bounded (24576 decay channels, a in [0.9, 0.9995] by construction)

- trained decays: min 0.90882, median 0.94811, p99 0.98904, max 0.99175; time constants 1/(1-a): median 19 ticks, p99 91, max 121; channels above 100 ticks: 0.7 %, above 300: 0.0 %
- accumulation of a per-tick state error, std factor sqrt(mean_c (1 - a^2t)/(1 - a^2)): t=1: 1.00, t=5: 2.03, t=15: 2.85, t=25: 3.11, t=50: 3.29, t=100: 3.36, t=150: 3.37, t=300: 3.38, t=1000: 3.38; slowest channel alone saturates at 7.8
- S' = a S + (1 - a) x is a convex average per channel (bounded by the inputs); the only other cross-tick carry is the previous token's normalised u (one tick of memory); the residual stream restarts from the embedding every tick
