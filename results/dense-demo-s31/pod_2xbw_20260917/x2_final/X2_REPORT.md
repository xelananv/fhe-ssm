## arm X2: 4 served ticks, 4 decoded+compared ticks

sources: `results/dense-demo-s31/pod_2xbw_20260917/pod_pull/s37/serve_X2/logs/server.jsonl` (served lines), `results/dense-demo-s31/pod_2xbw_20260917/pod_pull/s37/serve_X2/fidelity.json` (per-lane rows)

| tick | reqMsPerToken s | reqBoots | reqEncPtMs | hostPtEncode n | hostPtEncode s | reqBootMs s | peakVramGB | top-1 | decFailed | relErrRms median | relErrRms worst | dup lanes agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 150.3 | 528 | 0 | 0 | 0.0 | 38.3 | 68.7763 | 64/64 | 0 | 0.00085 | 0.06683 | True |
| 1 | 142.5 | 480 | 0 | 0 | 0.0 | 40.9 | 74.7763 | 64/64 | 0 | 0.00135 | 0.00417 | True |
| 2 | 142.9 | 480 | 0 | 0 | 0.0 | 41.1 | 74.7763 | 64/64 | 0 | 0.00118 | 0.00627 | True |
| 3 | 142.8 | 480 | 0 | 0 | 0.0 | 41.1 | 74.8076 | 62/64 | 0 | 0.00129 | 0.00493 | True |

warm ticks 1..3: median 142.8 s, min 142.5, max 142.9, mean 142.70; first half median 142.5 vs second half 142.8 (+0.26 %); least-squares slope +159.5 ms per tick of context; per lane-token at 64 lanes: 2.23 s
reqBoots warm: [480]; reqEncPtMs warm: [0]; failed requests: 0; peakVramGB: 68.7763–74.8076
fidelity: decode-failed ticks none (0 lane-ticks, every lane of those ticks); on the 256 decoded lane-ticks top-1 254/256 (99.22 %), relErrRms median 0.00119, worst 0.06683; over ALL 256 lane-ticks (failed ticks counted as misses) top-1 254/256 (99.22 %)

top-1 disagreements (the plaintext model's own top-1 margin in logits beside each: a small margin = a near-tie):

| tick | lane | FHE argmax | plaintext argmax | refTop1Margin | relErrRms |
|---|---|---|---|---|---|
| 3 | 16 | 8930 | 6208 | 0.0043 | 0.001785 |
| 3 | 28 | 281 | 368 | 0.003 | 0.000664 |

per-tick median relErrRms: first half of the ticks 0.00109, second half 0.00122
ptCacheTick: {"pass": 0, "ptCalls": 4032, "ptCacheHits": 2892, "ptCacheMisses": 1140, "ptCacheDenseFallback": 0, "ptCacheSkipped": 0, "ptCacheEntries": 1140, "ptCacheLimbs": 20677, "ptCacheResidentGB": 20.1924}
ptCacheTick: {"pass": 0, "ptCalls": 2230, "ptCacheHits": 2226, "ptCacheMisses": 4, "ptCacheDenseFallback": 0, "ptCacheSkipped": 0, "ptCacheEntries": 1144, "ptCacheLimbs": 20695, "ptCacheResidentGB": 20.21}
ptCacheTick: {"pass": 0, "ptCalls": 2230, "ptCacheHits": 2230, "ptCacheMisses": 0, "ptCacheDenseFallback": 0, "ptCacheSkipped": 0, "ptCacheEntries": 1144, "ptCacheLimbs": 20695, "ptCacheResidentGB": 20.21}
ptCacheTick: {"pass": 0, "ptCalls": 2230, "ptCacheHits": 2230, "ptCacheMisses": 0, "ptCacheDenseFallback": 0, "ptCacheSkipped": 0, "ptCacheEntries": 1144, "ptCacheLimbs": 20695, "ptCacheResidentGB": 20.21}
server exit: {"serve": "exit", "reason": "stop", "served": 4}
summary: {"ptCache": true, "ptCacheVerify": true, "ptCacheMaxGb": 24, "ptCalls": 10744, "ptCacheEntries": 1144, "ptCacheHits": 9600, "ptCacheMisses": 1144, "ptCacheMissMs": 26887, "ptCacheDenseFallback": 0, "ptCacheDenseMiss": 0, "ptCacheSkipped": 0, "ptCacheLimbs": 20695, "ptCacheResidentGB": 20.21, "ptCacheVerified": 1144, "ptCacheVerifyFail": 0, "hostPtEncodeMs": 0, "hostPtEncodeCount": 0, "boots": 1968, "peakVramGB": 74.8076}

## warm-tick comparison A11 vs X2 (same binary / box / load state required: R6)

| field (median over warm ticks) | A11 (n=44) | X2 (n=3) | delta |
|---|---|---|---|
| reqMsPerToken | 137476 | 142776 | +5300.5 (+3.9 %) |
| reqEvalMs | 116397 | 101723 | -14674 (-12.6 %) |
| reqBootMs | 21155 | 41052 | +19897 (+94.1 %) |
| reqHostPtEncodeMs | 23576 | 0 | -23576 (-100.0 %) |
| reqHostPtEncodeCount | 2230 | 0 | -2230 (-100.0 %) |
| reqLayerLoopMs | 182190 | 158043 | -24147.5 (-13.3 %) |
| reqLayerLoopUntimedMs | 60272.5 | 46776 | -13496.5 (-22.4 %) |
| reqBoots | 480 | 480 | +0 (+0.0 %) |
| peakVramGB | 65.0263 | 74.7763 | +9.75 (+15.0 %) |

A11 warm range 135.9–139.1 s; X2 warm range 142.5–142.9 s
FHE argmax A11 vs X2 on the ticks both decoded: 254 equal, 2 different
