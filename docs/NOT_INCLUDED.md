# Cited but not included

Comments, docstrings and help strings in the released code cite working notes, earlier campaigns, internal tooling and build products that are not
part of this release. They are listed here so that no reference is a surprise. Nothing needed to train, export, build, serve, replay or check the
released artifacts is among them: every released script's local imports resolve inside this tree (checked when the release is assembled),
and the runbook names the scripts of every step.

| path cited | cited from |
|---|---|
| `estimator/.micromamba` | `estimator/setup_estimator_env_linux64.sh` |
| `harness/build` | `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/PROBE_B_REPORT.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/PROBE_B_REPORT.src.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/PROBE_C_REPORT.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/PROBE_C_REPORT.src.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.src.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/build_probe_k768.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/run_k768.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/run_dose.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/run_planted.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/run_probe_c.sh` |
| `harness/build/boot_overflow_probe` | `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/PROBE_B_REPORT.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/PROBE_B_REPORT.src.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/dose_loadstate.txt`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.src.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/build_probe_k768.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/identity_sha256.txt`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/planted_loadstate.txt`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/run_dose.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/run_planted.sh` |
| `harness/build/boot_overflow_probe_k768` | `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.src.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/build_probe_k768.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/identity_sha256.txt`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/loadstate_extra.txt`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/loadstate_main.txt`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/loadstate_splitB.txt`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/run_k768.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/runlog_extra.txt`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/runlog_main.txt` |
| `harness/build/boot_overflow_probe_k768b` | `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.src.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/build_variantB.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/identity_sha256.txt`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/loadstate_extra.txt`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/loadstate_splitB.txt`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/run_k768.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/runlog_splitB.txt` |
| `harness/build/build.ninja` | `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.src.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/build_probe_k768.sh` |
| `harness/build/overflow_dist_dump` | `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/PROBE_C_REPORT.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/PROBE_C_REPORT.src.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/probe_c_loadstate.txt`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/run_probe_c.sh` |
| `harness/build/overflow_tail_probe` | `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/PROBE_C_REPORT.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/tail_probe_loadstate.txt`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/tail_probe_tables.md` |
| `harness/build/overflow_tail_probe2` | `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/tail_probe2_loadstate.txt`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/tail_probe2_tables.md` |
| `hpc_gpu_port/keyparts_test` | `hpc_gpu_port/mac_build_keyparts_test.sh` |
| `hpc_gpu_port/mac_fhe_client` | `hpc_gpu_port/mac_build_client.sh`, `ml-eval/dryrun_mac_client.sh`, `ml-eval/mock_fhe_server.py`, `results/dense-demo-s31/pod_tools/demo/60_fidelity_gate.sh`, `results/dense-demo-s31/pod_tools/demo/70_demo_server.sh`, `spec_decode/fidelity_tick.py` |
| `recovered/pod-scripts/11_chain_a100.sh` | `provisioning/10_torch.sh` |
| `results/OFFLINE_BATTERY_20260818.md` | `estimator/run_fest_hybrid_r17.py` |
| `results/a100-secure-20260730` | `ml-eval/check_artifact_provenance.py`, `ml-eval/train_fhe_native_ssm.py` |
| `results/a100-secure-20260730/artifacts` | `ml-eval/check_artifact_provenance.py`, `ml-eval/identify_bundle_checkpoint.py` |
| `results/a100-secure-20260730/artifacts/bundle_agnd768b.bin` | `results/dense-demo-s31/pod_tools/demo/20_stage_artifacts.sh` |
| `results/a100-secure-20260730/training/10_torch.log` | `provisioning/10_torch.sh` |
| `results/dense-demo-s31/RECOVERY_STATE_20260901.md` | `results/dense-demo-s31/pod_tools/demo/00_box_preflight.sh`, `results/dense-demo-s31/pod_tools/demo/20_stage_artifacts.sh` |
| `results/dense-demo-s31/dryrun` | `ml-eval/dryrun_mac_client.sh` |
| `results/dense-demo-s31/pod/checkpoints/ckpt_mirror/pbd430a_weights.npz` | `ml-eval/serve_ollama.py` |
| `results/dense-demo-s31/pod/state/CAMPAIGN_LOG.md` | `results/dense-demo-s31/pod_tools/demo/10_bringup.sh` |
| `results/dense-demo-s31/pod/state/runs/s31_t1b_store_secure_r17/console.log` | `results/dense-demo-s31/pod_tools/demo/_lib.sh` |
| `results/dense-demo-s31/pod_2xbw_20260917/MEMORY_RESIDUE_TRACE_20260918.md` | `results/dense-demo-s31/pod_longrun_prep/reports/LEAK_FIX_REPORT.md` |
| `results/dense-demo-s31/pod_rtx6000x5_20260904/pod_pull` | `harness/norm_eps_check.py` |
| `results/dense-demo-s31/sessions/a11_prompts64_long.txt` | `results/dense-demo-s31/pod_2xbw_20260917/arm_report.py`, `tools/token_panel.py` |
| `results/dense-demo-s31/sessions/fid_prompts4_2identical.txt` | `results/dense-demo-s31/pod_tools/demo/91_s37_serve.sh` |
| `results/explore-s29-rtxpro6000bw/pod/state/CAMPAIGN_LOG.md` | `results/dense-demo-s31/pod_tools/demo/10_bringup.sh` |
| `results/gpu-verify-20260728/exp7_amp.jsonl` | `ml-eval/rescale_bundle.py` |
| `results/gpu-verify-20260728/logs/10_torch.log` | `provisioning/10_torch.sh` |
| `results/hpc-gpu.md` | `ml-eval/fhe_circuit_sim.py` |
| `results/openfhe-bootstrap.md` | `ml-eval/remez_ablation.py` |
| `results/quality-ladder` | `ml-eval/train_fhe_native_ssm.py` |
| `results/remez-ablation.md` | `ml-eval/remez_ablation.py` |
| `results/s37-impl-20260905/V1_eps` | `tools/memguard.sh` |
| `results/s37-impl-20260905/V1_eps/INCIDENT_20260910_footprint.txt` | `tools/memlib.sh` |
| `results/theory/S33_REPORT_20260903.md` | `results/dense-demo-s31/pod_2xbw_20260917/pod_pull/preflight.txt`, `results/dense-demo-s31/pod_longrun_20260919/pod_pull/preflight.txt`, `results/dense-demo-s31/pod_tools/demo/00_box_preflight.sh` |
| `results/theory/SCHEDULE_DESIGN_20260903.md` | `results/dense-demo-s31/pod_tools/demo/85_schedule_ab.sh`, `results/theory/tick_level_trace.py` |
| `results/theory/TICK_SCALING_BOUNDS_20260903.md` | `results/theory/tick_level_trace.py` |
| `results/theory/TIERB_SPEC_20260903.md` | `ml-eval/train_fhe_native_ssm.py` |
| `spec_decode/README.md` | `results/dense-demo-s31/pod_tools/demo/71_mac_demo_session.sh` |
| `tools/mem_probe.sh` | `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/PROBE_B_REPORT.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/PROBE_B_REPORT.src.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/K768_CPU_REPORT.src.md`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/build_openfhe_k768.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/build_probe_k768.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/build_variantB.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/run_k768.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/run_dose.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/run_planted.sh`, `results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/run_probe_c.sh` |
| `training/DIAGNOSIS.md` | `ml-eval/staged_token_reader.py`, `ml-eval/train_fhe_native_ssm.py` |
| `training/THROUGHPUT_AUDIT.md` | `ml-eval/train_fhe_native_ssm.py` |
| `training/TRAINER_FIXES_20260822.md` | `ml-eval/train_fhe_native_ssm.py` |
| `training/scan_cost_probe.py` | `ml-eval/train_fhe_native_ssm.py` |
| `training/verify_trainer_fixes.py` | `ml-eval/train_fhe_native_ssm.py` |
| `training/vram_sizing.py` | `ml-eval/train_fhe_native_ssm.py` |
