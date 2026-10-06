#!/usr/bin/env python3
"""Client-side decoding policy for free-running sessions (2026-09-19). The CLIENT owns decoding: it decrypts the logits of its lane and
chooses the next token; nothing here touches the encrypted computation or costs the server anything.

greedy   : argmax (the demo's policy so far). A 430M model decoding greedily repeats itself within ~20 words on EVERY prompt tried
           (64 of 64 lanes of the 2026-09-19 selection repeat a 6-word sequence; median first repeat at word 21).
norepeat : DETERMINISTIC anti-repetition, the two standard devices of text generation, applied to the logits before the argmax:
             * repetition penalty p over the last W fed tokens (prompt included): a positive logit is divided by p, a negative one
               multiplied by p (the CTRL / HF `repetition_penalty` rule);
             * no-repeat n-gram: a token that would complete an n-gram already present in the lane's history is banned.
           Deterministic, so the FHE-vs-plaintext comparison stays exact: the same policy is applied to the decrypted logits and to the
           plaintext model's logits on the SAME history, and the two picks are compared (the raw argmax agreement is recorded as well).
"""
import numpy as np


def modified(logits, history, policy="greedy", penalty=1.3, window=64, ngram=4, ban=()):
    """The logits the policy takes its argmax over (a float64 copy; greedy = unchanged)."""
    z = np.array(logits, dtype=np.float64, copy=True)
    if policy == "greedy":
        return z
    if policy != "norepeat":
        raise ValueError(policy)
    for b in ban:                                   # e.g. the end-of-text token: a free-running lane has no stop, so it should not start a new document
        z[int(b)] = -np.inf
    if penalty and penalty != 1.0 and history:
        idx = np.unique(np.asarray(history[-window:], dtype=np.int64))
        v = z[idx]; z[idx] = np.where(v > 0, v / penalty, v * penalty)
    if ngram and ngram > 1 and len(history) >= ngram - 1:
        tail = tuple(history[-(ngram - 1):]); h = history
        for i in range(len(h) - ngram + 1):
            if tuple(h[i:i + ngram - 1]) == tail:
                z[h[i + ngram - 1]] = -np.inf
    return z


def pick(logits, history, policy="greedy", penalty=1.3, window=64, ngram=4, ban=()):
    """logits: 1-D array over the vocabulary; history: list of token ids fed so far for this lane (prompt + generated, the token that
    produced these logits included). Returns the chosen token id (int)."""
    if policy == "greedy":
        return int(np.argmax(logits))
    return int(np.argmax(modified(logits, history, policy, penalty, window, ngram, ban)))
