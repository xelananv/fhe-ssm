# Records

Every number quoted anywhere in this repository is transcribed from a file under this directory (or under `ml-eval/artifacts/`,
`results/estimator-fest/`, `results/theory/`). Directory by directory:

| directory | what it holds |
|---|---|
| [dense-demo-s31/pod_longrun_20260919/](dense-demo-s31/pod_longrun_20260919/) | the long free-running session (148 ticks) and its control arm: reports (`LONG_REPORT.md`, `FIDELITY_TREND.md`, `RECORDER_EVENTS.md`, `ERROR_HEADROOM.md`, `CTRL512_VS_LONG.md`), the scripts that generate them from the raw records, `plots_data/` (CSV per tick / per lane-tick / per bootstrap / host memory per minute), `prompt_select/` (the 640-candidate selection under the decoding rule), `pod_pull/` (the server's `server.jsonl`, the driver's `fidelity.json` and `tokens.json`, the bootstrap-arm selftests, the store build, the memory and power samplers, build identity and logs), [dense-demo-s31/pod_longrun_prep/reports/](dense-demo-s31/pod_longrun_prep/reports/) (the extended-bootstrap-range and leak-fix reports, the recorder's arithmetic check) |
| [dense-demo-s31/pod_2xbw_20260917/](dense-demo-s31/pod_2xbw_20260917/), [dense-demo-s31/sessions/demo_20260918T074952Z/](dense-demo-s31/sessions/demo_20260918T074952Z/) | the recorded client-keyed session (transcript, tokens, per-lane text, the kit manifests) and the same box's arms: `a11_final/` (45 pod-keyed ticks, cost vs context), `x2_final/` (the plaintext cache), `a11_tick42/` (the bootstrap-overflow investigation on the CPU twin: `overflow_probe/`, `k768/`), `pod_pull/` |
| [dense-demo-s31/pod/state/runs/](dense-demo-s31/pod/state/runs/) + [dense-demo-s31/QUALITY_ANCHORS_20260828.md](dense-demo-s31/QUALITY_ANCHORS_20260828.md) | WikiText-103 / holdout / LAMBADA anchors of the served model, its non-annealed checkpoint, and the Pythia-410M harness validation |

Raw-record conventions: the server prints one JSON line per event (`{"serve":"served",…}` per tick, `{"vramTrace":…}` per
phase, `{"bootRecord":true,…}` per bootstrap when the recorder is on); the driver writes one row per lane per tick to
`fidelity.json`. Timer sums (`reqMsPerToken`) are not wall clocks — `reqLayerLoopMs` and the driver's `serveSec` are.
