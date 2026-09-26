# Quality anchors — measured, harness validated, bands FROZEN pre-launch — 2026-08-28

Per the quality plan (S3.0_DECISION_PACKAGE.md §3): the anchor evaluations ran
and are committed BEFORE Phase B trains, so the bars cannot move after the
results exist. All numbers transcribed from the runner-recorded cells
`s31q_pythia{410_final,410_step6000,160_step6000}/anchor.jsonl` on the S31 box
(ml-eval/anchor_eval.py, NeoX tokenizer, WT103 FULL test split ctx 1024 stride
1024 = 285,830 scored tokens, LAMBADA-OpenAI FULL 5,153 items).

## Harness validation — PASSED on both axes

Pythia-410M FINAL (300B tokens) vs the published numbers (Mamba paper tbl 3):

| axis | measured | published | delta |
|---|---|---|---|
| LAMBADA acc | **51.6%** | 51.4% | +0.2 pt |
| LAMBADA ppl (per WORD) | **10.78** | 10.84 | −0.6% |

The per-word derivation (nll summed over the target word tokens, mean over
examples) is the published-comparable one; the per-token variant (5.68) is
also recorded in the jsonl. `anchor_eval.py` now emits both
(`lambadaPplWord`). The evaluation harness is certified; every band below
stands on it.

## The anchors, measured

| model | WT103 token-ppl | WT103 word-ppl (derived) | LAMBADA acc | LAMBADA ppl/word |
|---|---|---|---|---|
| Pythia-410M step6000 (12.58B tok — the dense-arm anchor) | **30.957** | 58.41 | **33.6%** | 37.91 |
| Pythia-160M step6000 (the band16 live-param anchor) | **40.620** | 80.60 | **30.7%** | 55.80 |
| Pythia-410M final (300B, report-only) | 18.170 | 31.07 | 51.6% | 10.78 |
| Mamba-370M final (report-only) | running at freeze time — appended when done; NOT band-setting | | | |

## The bands, FROZEN (the package §3 proposal, adopted at launch)

| our arm | primary anchor | COMPETITIVE iff | HONEST-BUT-WEAKER iff |
|---|---|---|---|
| dense L24 (pbd430, 404.8M, 12.144B tok) | Pythia-410M step6000 | WT103 token-ppl ≤ 1.05× = **≤ 32.50** AND LAMBADA acc ≥ anchor −2pt = **≥ 31.6%** | token-ppl **≤ 38.70** |
| band16 L24 (pbb430, 63.8M live) | Pythia-160M step6000 (also reported vs 410M step6000) | **≤ 42.65** AND acc **≥ 28.7%** | **≤ 50.78** |

Rules that travel (package §3): the NON-annealed model is compared
externally; any WT103 anneal appears side by side with the non-annealed
number; Pythia intermediate checkpoints are mid-schedule (LR not decayed),
which favours us — said here; the Pile contains Wikipedia and FineWeb-Edu
does not — said here; band16 live-parameter count disclosed wherever quoted.
Author may adjust bands only while no arm result exists; arms launch next in
queue order, freezing this file as-is.

## Holdout axis

The FineWeb-Edu 2M-token holdout axis runs for anchors + both arms
post-training (one combined cell) — staging completed after the anchor cells
started; not band-setting, reported alongside.
