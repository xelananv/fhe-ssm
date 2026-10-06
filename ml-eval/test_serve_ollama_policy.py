#!/usr/bin/env python3
"""test_serve_ollama_policy.py — serve_ollama.Sampler at temperature 0 must be
the SAME function as spec_decode/decode_policy.py's "norepeat" policy, which is
what the pod's client runs.

The trap this guards against is the one the trainer's Horner note records: a
comparison that passes trivially because the interesting branch never fires. A
random history almost never contains a repeated 3-gram, so the n-gram ban would
never trigger and the test would prove nothing. The cases below therefore plant
repeats on purpose, and the run ASSERTS that the ban actually fired.

  .venv/bin/python ml-eval/test_serve_ollama_policy.py
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "spec_decode"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import decode_policy  # noqa: E402
from serve_ollama import Sampler  # noqa: E402


def planted_history(rng, V, n=4):
    """A history with a genuine repeat: some earlier 3-gram equals the tail, so
    at least one token is banned."""
    h = list(rng.integers(0, V, size=int(rng.integers(8, 40))))
    k = int(rng.integers(0, len(h) - n))
    gram = h[k:k + n - 1]
    return h + gram                      # tail now matches h[k:k+3]


def n_banned(history, n=4):
    if not n or n <= 1 or len(history) < n - 1:
        return 0
    tail = tuple(history[-(n - 1):])
    return sum(1 for i in range(len(history) - n + 1)
               if tuple(history[i:i + n - 1]) == tail)


def main():
    rng = np.random.default_rng(20260923)
    V = 2000
    cases, fired, mismatches = 0, 0, []
    for trial in range(400):
        penalty = float(rng.choice([1.0, 1.1, 1.3, 1.5]))
        ngram = int(rng.choice([0, 2, 3, 4, 5]))
        window = int(rng.choice([8, 64, 256]))
        history = planted_history(rng, V) if trial % 2 else list(rng.integers(0, V, size=30))
        logits = rng.normal(0, 4, size=V)            # both signs: exercises the CTRL rule

        ours = Sampler(temperature=0.0, repeat_penalty=penalty, repeat_last_n=window,
                       no_repeat_ngram=ngram, torch=torch).pick(
            torch.from_numpy(logits), [int(x) for x in history], torch)
        theirs = decode_policy.pick(logits, [int(x) for x in history], policy="norepeat",
                                    penalty=penalty, window=window, ngram=ngram, ban=())
        cases += 1
        if n_banned(list(history), ngram):
            fired += 1
        if ours != theirs:
            mismatches.append({"trial": trial, "penalty": penalty, "ngram": ngram,
                               "window": window, "ours": ours, "theirs": theirs})

    # the pod's own defaults, explicitly
    hist = planted_history(rng, V)
    lg = rng.normal(0, 4, size=V)
    d_ours = Sampler(temperature=0.0, torch=torch).pick(torch.from_numpy(lg),
                                                        [int(x) for x in hist], torch)
    d_theirs = decode_policy.pick(lg, [int(x) for x in hist], policy="norepeat")
    print(f"cases           : {cases}")
    print(f"ngram ban fired : {fired}  (a 0 here would make the test vacuous)")
    print(f"mismatches      : {len(mismatches)}")
    print(f"pod defaults    : ours={d_ours} theirs={d_theirs} "
          f"(penalty 1.3, window 64, ngram 4)")
    for m in mismatches[:5]:
        print("  MISMATCH", m)
    ok = not mismatches and fired > 0 and d_ours == d_theirs
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
