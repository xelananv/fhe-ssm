# The numbers table (unfenced passes that restaged nothing; every row names its pass)

## The gates

- staging contact tick: PASS under CONTACT_GATE=structure (fidelity not gated)
- staging contact mixed: PASS under CONTACT_GATE=structure (fidelity not gated)


## The PCMM bracket

| run | commit | dirty | _pc .so | load at start (1, 5, 15 min) | driver flags (stc_cache, pcmm_cf) |
|---|---|---|---|---|---|
| run_pcmm24_ctlA | 3afbfdf3 | 0 | 009427e7463beeca | 2.46 1.83 1.00 | 160, 8 |
| run_cfg-pcmm24_tick | 3afbfdf3 | 0 | 009427e7463beeca | 1.10 1.83 1.25 | 160, 8 |
| run_cfg-pcmm24_prefill | 3afbfdf3 | 0 | 009427e7463beeca | 2.40 2.57 1.72 | 160, 8 |
| run_cfg-pcmm24_doc | 3afbfdf3 | 0 | 009427e7463beeca | 3.31 3.09 2.17 | 160, 8 |
| run_cfg-pcmm24_mixed | 3afbfdf3 | 0 | 009427e7463beeca | 2.34 3.00 2.41 | 160, 8 |
| run_cfg-pcmm24-tick_tick | 3afbfdf3 | 0 | 009427e7463beeca | 4.73 3.90 3.11 | 160, 8 |
| run_pcmm24_ctlB | 3afbfdf3 | 0 | 009427e7463beeca | 4.46 3.56 2.79 | 160, 8 |

one build across the bracket

controls pcmm24 ctlA + ctlB: unfenced ticks [13.956, 14.028, 14.087, 13.987, 14.054, 14.098] s: mean 14.035 s, spread (max - min) 0.142 s; build 3afbfdf3 keys insecure-inproc; relErrRms per tick [0.004336851898859028, 0.004895766525194727, 0.005118201144688793, 0.004453726965368053, 0.005148199929904577, 0.005132794303742072]

| run | pass | config | seconds | token-values | conversations | sequences (positions) | tok/s aggregate | tok/s per sequence | tok/s per conversation | relErrRms (worst live lane-slot) | top-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cfg-pcmm24_tick | 1 | tick.1 | 14.004 | 64 | 64 | {} | 4.57 | - | 0.071 | 0.00436 (0.0144) | 64/64 |
| cfg-pcmm24_tick | 3 | tick.3 | 14.077 | 64 | 64 | {} | 4.55 | - | 0.071 | 0.00479 (0.0093) | 64/64 |
| cfg-pcmm24_tick | 5 | tick.5 | 14.117 | 64 | 64 | {} | 4.53 | - | 0.071 | 0.00494 (0.0119) | 62/64 |
| cfg-pcmm24_prefill | 3 | prefill.3 | 14.081 | 64 | 0 | {'L': 64} | 4.55 | L 4.55 | - | 0.00441 (0.00926) | 1/1 |
| cfg-pcmm24_prefill | 7 | generate.2 | 14.17 | 1 | 1 | {} | 0.07 | - | 0.071 | 0.00273 (0.00273) | 1/1 |
| cfg-pcmm24_doc | 3 | doc.3 | 14.094 | 64 | 0 | {'D': 64} | 4.54 | D 4.54 | - | 0.00445 (0.00707) | 1/1 |
| cfg-pcmm24_doc | 7 | readers.2 | 14.162 | 64 | 64 | {} | 4.52 | - | 0.071 | 0.00449 (0.00738) | 64/64 |
| cfg-pcmm24_mixed | 3 | mixed.3 | 14.092 | 64 | 56 | {'B': 8} | 4.54 | B 0.57 | 0.071 | 0.0041 (0.00975) | 56/57 |
| cfg-pcmm24-tick_tick | 1 | tick.1 | 14.001 | 64 | 64 | {} | 4.57 | - | 0.071 | 0.00415 (0.011) | 64/64 |
| cfg-pcmm24-tick_tick | 3 | tick.3 | 14.085 | 64 | 64 | {} | 4.54 | - | 0.071 | 0.0047 (0.0112) | 63/64 |
| cfg-pcmm24-tick_tick | 5 | tick.5 | 14.138 | 64 | 64 | {} | 4.53 | - | 0.071 | 0.00476 (0.00856) | 61/64 |

the modularity overhead (cfg-pcmm24_tick unfenced ticks [14.004, 14.077, 14.117] s): inside the controls' spread (+0.031 s, spread 0.142 s): not quoted

restaging passes of cfg-pcmm24_prefill (pass, staged keys, stage_s = host issue time, seconds, fenced): [(1, 96, 0.0164, 14.021, False), (5, 1794, 0.4816, 14.573, False)]; the restage cost = a restaging pass's seconds minus the next steady pass's in the same configuration

restaging passes of cfg-pcmm24_doc (pass, staged keys, stage_s = host issue time, seconds, fenced): [(1, 96, 0.0165, 14.033, False), (4, 1728, 0.4605, 14.54, True), (5, 258, 0.0434, 14.184, False)]; the restage cost = a restaging pass's seconds minus the next steady pass's in the same configuration

restaging passes of cfg-pcmm24_mixed (pass, staged keys, stage_s = host issue time, seconds, fenced): [(1, 96, 0.0165, 14.024, False), (2, 994, 0.2523, 14.272, True)]; the restage cost = a restaging pass's seconds minus the next steady pass's in the same configuration

## The diagonal (DIAG_SOURCE=generated) bracket

| run | commit | dirty | _pc .so | load at start (1, 5, 15 min) | driver flags (stc_cache, pcmm_cf) |
|---|---|---|---|---|---|
| run_ref_ctlA | 3afbfdf3 | 0 | 009427e7463beeca | 1.93 1.62 2.19 | 160, 8 |
| run_cfg-ref_tick | 3afbfdf3 | 0 | 009427e7463beeca | 3.06 1.73 1.78 | 160, 8 |
| run_ref_ctlB | 3afbfdf3 | 0 | 009427e7463beeca | 1.17 1.37 1.51 | 160, 8 |

one build across the bracket

controls ref ctlA + ctlB: unfenced ticks [31.77, 31.907, 31.948, 31.778, 31.913, 31.96] s: mean 31.879 s, spread (max - min) 0.190 s; build 3afbfdf3 keys insecure-inproc; relErrRms per tick [0.00010068012455967185, 0.00014757969109813597, 0.00019288936674518377, 0.00010113302277363203, 0.00016119545378899823, 0.00018997744838403365]

| run | pass | config | seconds | token-values | conversations | sequences (positions) | tok/s aggregate | tok/s per sequence | tok/s per conversation | relErrRms (worst live lane-slot) | top-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cfg-ref_tick | 1 | tick.1 | 31.867 | 64 | 64 | {} | 2.01 | - | 0.031 | 9.04e-05 (0.000232) | 64/64 |
| cfg-ref_tick | 3 | tick.3 | 31.995 | 64 | 64 | {} | 2.00 | - | 0.031 | 0.00013 (0.000282) | 64/64 |
| cfg-ref_tick | 5 | tick.5 | 32.022 | 64 | 64 | {} | 2.00 | - | 0.031 | 0.00021 (0.000746) | 64/64 |

the modularity overhead (cfg-ref_tick unfenced ticks [31.867, 31.995, 32.022] s): inside the controls' spread (+0.082 s, spread 0.190 s): not quoted

