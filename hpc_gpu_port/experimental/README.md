> The served binary of every recorded session is `gpu_real_model_x` from this directory (`-DFHE_SSM_EXPERIMENTAL=ON`);
> every lever is behind a flag, default off. The status table below is historical.

# hpc_gpu_port/experimental — the served binary

`gpu_real_model_x.cu` is `../gpu_real_model.cu` plus the levers of the recorded configuration, each behind its own flag,
default off (with every flag off the copy behaves exactly as the original). It is built from the same CMake configure with
`-DFHE_SSM_EXPERIMENTAL=ON` (`../CMakeLists.txt`); `cpu_real_model_x.cpp` is the CPU (OpenFHE) twin of the same circuit,
built from `../../harness/`.

Flags used by every recorded session (2026-09-18 demo session and 2026-09-19 long run):

| flag | what it does |
|---|---|
| `--canonical-carry` | at the end of every tick, return every carried ciphertext (the recurrent state and the shift-mix input of each block) to its canonical level and scale, so the next tick is the identical circuit |
| `--periodic-encode` | encode lane-replicated weight plaintexts with the periodic encoder (`../periodic_encode.hpp`) instead of the dense encode |
| `--drop-wv R --drop-wkr R --drop-wout R --drop-win R` | drop the operand to exactly R remaining levels before the corresponding large multiplication (recorded: 7 / 4 / 7 / 4) |
| `--newton-robust B0 N` | the inverse-square-root Newton iteration of every norm: constant seed B0 in the mean-square frame, N steps (recorded: 0.1 12; the 2026-09-18 session 0.3 8) |
| `--pt-cache [--pt-cache-verify] [--pt-cache-max-gb G]` | keep the elementwise constant plaintexts resident on the device across ticks (verified against a fresh encode when asked) |

The remaining flags in the source (`--deferred-scan`, `--u-drop-level`, `--branch-out-boot`, `--pt-scalar`) are experiments that
were not used in any recorded session and are off by default.
