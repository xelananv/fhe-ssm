> Historical operator plan for the recorded sessions, kept as it ran. It cites working notes that are not part of this
> repository; the entry point for a reader is [docs/RUNBOOK.md](../../../../docs/RUNBOOK.md).

# The scripts that ran the recorded sessions on the rented box

Numbered in the order they run; every script takes its paths from the environment (defaults in `_lib.sh`: the tree at
`/root/src`, the work directory `/root/demo`, the artifacts under `/root/src/ml-eval/artifacts`), appends one UTC line to
`$DEMO/CAMPAIGN_LOG.md`, holds the `$DEMO/.busy` marker while it owns the GPU, and decides PASS/FAIL by grepping its own log
(a CUDA out-of-memory can exit 0). The order and the environment for two-card boxes are in
[docs/RUNBOOK.md](../../../../docs/RUNBOOK.md).

| script | what it does |
|---|---|
| `00_box_preflight.sh` | RAM, disk, peer access between the two cards, link bandwidth; GO / GO-WITH-RISK / NO-GO |
| `10_bringup.sh build` | toolchain if needed, then `../s31_bootstrap.sh` (FIDESlib at the pinned commit + patches, the no-LTO harness build, the torch venv); writes `BUILD_IDENTITY.txt` |
| `20_stage_artifacts.sh` | prints the upload commands for the weight bundle and prompts; verifies every byte against the recorded sha256 |
| `30_dump_indices.sh` | dumps the rotation-index contract (`contract_r17.json`) that the client's key generation must cover |
| `40_fit_and_store.sh` | builds the weight store along the canonical trajectory (two cards: `DEVICES=0,1`) |
| `41_mgpu_ladder.sh` | the two-card bring-up ladder (rung 0 = the peer-pool probe, then the selftest, then the store) |
| `50_sovereignty.sh` | asserts that no secret key is on the box and that the server refuses a key directory that holds one |
| `60_fidelity_gate.sh` | teacher-forced probe of the server against the plaintext model (top-1, relErrRms) before recording |
| `70_demo_server.sh start|stop` | the stateful server on the client's public key set; stopped by a `stop` file, never killed mid-tick |
| `71_mac_demo_session.sh` | the client side: prepare 64 lanes, generate tick by tick, audit, archive the verification kit |
| `85_schedule_ab.sh`, `90_s37_cell.sh`, `91_s37_serve.sh`, `92_s37_levers.sh` | the A/B cells and the free-running serve loop the long run used (`91_s37_serve.sh` is called by `../../pod_longrun_prep/pod_scripts/longrun_pod.sh`) |
| `mac_upload_chunked.sh`, `mac_upload_keys.sh` | chunked, resumable uploads (the evaluation keys never include `secret.key`) |
| `_lib.sh` | shared defaults, logging, verdict helpers |
