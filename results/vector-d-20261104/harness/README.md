# RWKV-7 Polynomial-Nonlinearity Harness

Plaintext PyTorch harness that replaces every non-polynomial operation in a pretrained RWKV-7 (`x070`) checkpoint with the Chebyshev polynomial or constant-seed Newton inverse-square-root iteration evaluated by the CKKS circuit. It supports activation-range calibration (`calibrate.py`), polynomial and Newton fitting (`fit.py`), unclamped activation-range scanning (`scan_ranges.py`), WikiText-103 / LAMBADA evaluation (`eval.py`), and optional block-wise distillation (`heal.py`).

Reference implementation and vocabulary: `../raw/` (`BlinkDL/RWKV-LM`, Apache-2.0).

## Files

| File | Purpose |
|---|---|
| `rwkv7_model.py` | RWKV-7 (`x070`) forward pass in PyTorch. Reads dimensions (layers, width, head size, LoRA ranks, FFN width) directly from the `.pth` checkpoint. Routes every non-polynomial operation through `Sites.__call__(name, x)`. Modes: `exact`, `record`, `poly`. |
| `rwkv_tokenizer.py` | World trie tokenizer (`rwkv_vocab_v20230424.txt`, 65,536 tokens) using `ast.literal_eval`. |
| `common.py` | Text-to-token windowing (cached as `<text>.rwkvtok.*.npy`), padding, SHA-256 hashing, JSONL logging. |
| `calibrate.py` | Runs `record` mode over a calibration text file and writes per-site `min`, `max`, `p0.1`, and `p99.9` as JSON. |
| `fit.py` | Fits Chebyshev polynomials (`cheb`) and inverse-square-root iterations (`newton`, `newton-const`) per site and writes the coefficients JSON consumed by `poly` mode and the encrypted port. |
| `rsqrt_const.py` | Constant-seed (`y_0 = 1`) Newton iteration for $1/\sqrt{x+\epsilon}$ over a $10\times$-widened range (`[min/R, max*R]`) with guaranteed convergence on $(0, 3 \cdot \text{scale})$. |
| `convert_newton_const.py` | Converts `method: newton` sites in an existing coefficients JSON to `method: newton_const`. |
| `scan_ranges.py` | Runs the model unclamped in `poly` or `exact` feed mode over a token corpus and counts out-of-interval and out-of-convergence tokens per site kind. |
| `eval.py` | Evaluates WikiText-103 token perplexity, LAMBADA accuracy and perplexity, and token-level fidelity (`top1_agree`, `kl_mean`) against the exact model. |
| `heal.py` | Block-wise distillation of the polynomial model against the frozen exact teacher with fixed polynomial coefficients. |
| `tests/` | Unit tests (`test_harness.py`, `conftest.py`, `run_without_pytest.py`) verifying exact equivalence to the reference implementation, site inventory, tokenizer round-trip, calibration/fit/eval pipeline, and distillation loss reduction. |

## Polynomial Sites

Each block `blocks.i.` has 10 non-polynomial sites (9 in block 0, which has no value-residual gate):

| Site (per block `blocks.i.`) | Input $x$ | Exact function | Circuit approximation in `coeffs_final.json` |
|---|---|---|---|
| `ln1.rsqrt`, `ln2.rsqrt` | Biased variance over $C$ | $1/\sqrt{x+10^{-5}}$ | `newton_const` ($y_0=1$, $R=10$) or `newton_scaled` |
| `att.w_tanh` | `xw @ w1` | $\tanh(x)$ | Odd Chebyshev polynomial |
| `att.decay` | $z = \text{w0} + \tanh(\dots)\text{w2}$ | $\exp(-e^{-0.5}\sigma(z))$ | Single fused Chebyshev polynomial |
| `att.v_sigmoid` ($i \ge 1$) | `v0 + (xv@v1)@v2` | $\sigma(x)$ | Chebyshev polynomial |
| `att.a_sigmoid` | `a0 + (xa@a1)@a2` | $\sigma(x)$ | Chebyshev polynomial |
| `att.g_sigmoid` | `xg @ g1` | $\sigma(x)$ | Chebyshev polynomial |
| `att.kk_rsqrt` | $\|k \odot k_k\|^2$ per head | $1/\max(\sqrt{x}, 10^{-12})$ | `newton_const` ($y_0=1$, $R=10$) or `newton_scaled` |
| `att.gn_rsqrt` | Per-head readout variance | $1/\sqrt{x+64\times 10^{-5}}$ | `newton_const` ($y_0=1$, $R=10$) or `newton_scaled` |
| `ffn.relu2` | `key(...)` | $\max(x,0)^2$ | Chebyshev polynomial (margin $0.2$ or per-block abs target) |
| `ln0.rsqrt` (block 0), `ln_out.rsqrt` | LayerNorm variance | $1/\sqrt{x+10^{-5}}$ | Client-side exact (`ln0` folded into embedding table; `ln_out` after decrypt) |

For an $L$-layer model there are $10L - 1$ server-side sites (119 for 0.1B $L=12$; 239 for 0.4B and 1.5B $L=24$).

## Measured fp64 Plaintext Polynomial Gate Summary

Evaluated unclamped (`--clamp` off, matching homomorphic evaluation) on WikiText-103 test (`286,849` tokens at `ctx=1024`) and OpenAI LAMBADA (`5,153` documents) using `coeffs_final.json`:

| Model | $L, C$ | Exact WikiText-103 PPL | Unclamped poly PPL ($\Delta$) | WikiText-103 top-1 / KL | Exact LAMBADA acc | Unclamped poly LAMBADA acc | Unclamped full scan (`conv_out` / `nonfinite`) | Gate verdict |
|---|---|---|---|---|---|---|---|---|
| **0.1B** (`v2.8`) | $12, 768$ | $19.404$ | $19.414$ ($+0.05\%$) | $98.17\%$ / $0.00117$ | $48.98\%$ | $48.96\%$ ($-0.02\text{ pp}$) | $0 / 0$ | **PASS** |
| **0.4B** (`v2.9`) | $24, 1024$ | $13.753$ | $13.748$ ($-0.03\%$) | $98.08\%$ / $0.00138$ | $58.49\%$ | $58.33\%$ ($-0.16\text{ pp}$) | $0 / 0$ | **PASS** |
| **1.5B** (`v3`) | $24, 2048$ | $9.618$ | $9.800$ ($+1.89\%$) | $93.07\%$ / $0.01931$ | $69.34\%$ | $69.07\%$ ($-0.27\text{ pp}$) | $0 / 0$ | **FAIL** |

On the 1.5B checkpoint, `scanfull` stays finite (`0` non-finite and `0` `conv_out` tokens across WikiText-103, LAMBADA, and 32 free-running generations), and `coeffs_final.json` equals clamped mode, but five degree-255 Chebyshev sites (`blocks.{6,15,21,22}.att.w_tanh`, `blocks.23.att.a_sigmoid`) miss their `1e-3` target over the widened min-max interval. That drops top-1 agreement to $93.07\%$ (below the $98\%$ gate bar) and raises WikiText-103 perplexity by $+1.89\%$ (above the $0.5\%$ bar), so `size_pipeline.sh` halts the 1.5B model at the plaintext gate before any FHE capture or GPU run.

## Commands

```bash
# 0. Run unit tests on CPU (uses synthetic tiny models; no checkpoint download needed)
pytest tests -q
# Or without pytest:
python tests/run_without_pytest.py

# 1. Calibrate activation ranges on WikiText-103 validation (200k tokens, ctx=4096)
python calibrate.py --model RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
  --text wiki.valid.raw --tokens 200000 --ctx 4096 --batch 2 --out ranges_ctx4096.json

# 2. Fit Chebyshev polynomials + constant-seed Newton rsqrt sites
python fit.py --ranges ranges_ctx4096.json --range minmax --margin 0.5 \
  --target-err 1e-3 --metric rel --max-degree 255 \
  --rsqrt newton-const --rsqrt-range-factor 10 --newton-target 1e-3 \
  --site-target 'ffn\.relu2=1e-4' --out coeffs_wide.json

# 3. Scan unclamped activation ranges over 65,536 tokens
python scan_ranges.py --model RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
  --coeffs coeffs_final.json --text wiki.test.raw --tokens 65536 --ctx 4096 --out scan_poly.json

# 4. Evaluate exact vs. unclamped polynomial quality and fidelity
python eval.py --model RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
  --wikitext wiki.test.raw --lambada lambada_test.jsonl --mode exact --tag exact --out eval.jsonl
python eval.py --model RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
  --wikitext wiki.test.raw --lambada lambada_test.jsonl --mode poly \
  --coeffs coeffs_final.json --tag final_unclamped --out eval.jsonl
```

## License

Any commercial use of these files as part of Perseus (github.com/gladia-research-group/perseus) requires prior written permission from Alessandro Zirilli or Davide Marincione, under Perseus's Business Source License 1.1; on their own the files are MIT. Reference files in `../raw/` are Apache-2.0 (`BlinkDL/RWKV-LM`).
