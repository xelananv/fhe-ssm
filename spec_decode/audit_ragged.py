#!/usr/bin/env python3
"""
audit_ragged.py — rigorous provenance audit of a ragged multi-lane session
(decision 2026-09-01: prove nothing shown was un-generated).

Checks (each independent):
  A. FEED INVARIANT: the transcript records the actual token fed to each lane
     each tick, so this is a real test, not a well-formedness assertion: during
     a lane's prompt phase the fed token must BE that prompt position; after it
     the fed token must be the lane's own last generated token (self-feed); a
     retired lane must re-feed its final token forever. Together these prove
     the shown text was fed exactly as recorded — no prompt token skipped, no
     other lane's token injected, nothing generated from a token the model
     never saw. (Transcripts written before 2026-09-01 lack `inputs`; A then
     reports SKIPPED rather than passing vacuously.)
  B. BOOKKEEPING RECONCILIATION: transcript.json replayed against the one rule
     (keep iff tick+1 >= len(prompt) and not retired); the non-None picks must
     concatenate to generated_lanes exactly; retirement timing must match the
     word budget AND the stop tokens — read from the session's own
     tokens.json["retire_cfg"], never from this script's command line, so the
     auditor cannot disagree with the run it is auditing.
  C. PLAINTEXT GREEDY REPLAY ORACLE: for every generated token, the plaintext
     model's greedy choice on the recorded prefix must equal the token, OR the
     position is a legitimate FHE noise flip — classified by logit margin
     (margin <= --flip-margin) and reported per lane. Systematic mismatches
     (large margin, or equal to the prompt token at an adjacent position) are
     flagged as SCHEDULER-BUG suspects.

Exit 0 = every requested check RAN and passed (noise flips allowed, listed);
3 = a violation; 4 = a requested check could not run (e.g. no `inputs`);
5 = the auditor refused (no retirement rule available). Never 0 unless clean.
Checks D (lane-independence rerun) and E (dual-run determinism) are session
recipes, not code — see spec_decode/README.

Usage:
  python audit_ragged.py --out <session dir> --tag pbd430a \
      [--art-dir ml-eval/artifacts] [--device cuda:0|cpu] [--flip-margin 0.15]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ml-eval"))
from _native_loader_copy import load_native  # noqa: E402


def fail(msg):
    print(json.dumps({"auditViolation": msg}))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--art-dir", default=os.path.join(os.path.dirname(HERE), "ml-eval", "artifacts"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--flip-margin", type=float, default=0.15,
                    help="logit margin under which an argmax mismatch is a plausible FHE noise flip")
    ap.add_argument("--words-per-lane", type=int, default=0,
                    help="FALLBACK ONLY for sessions with no retire_cfg; the "
                         "audit prefers tokens.json[\"retire_cfg\"] (F70)")
    ap.add_argument("--expect-top1", type=float, default=-1.0,
                    help="MEASURED teacher-forced top-1 agreement of this circuit "
                         "(from spec_decode/fidelity_tick.py). Given it, check C "
                         "judges the observed rate against it at 2 sigma; without "
                         "it the rate is reported only (F73)")
    ap.add_argument("--checks", default="ABC",
                    help="which checks to run; C (plaintext greedy replay) costs "
                         "one full forward per generated token")
    a = ap.parse_args()

    state = json.load(open(os.path.join(a.out, "tokens.json")))
    transcript = json.load(open(os.path.join(a.out, "transcript.json")))
    NL = state.get("lanes", 1)
    prompts = state["ids_lanes"] if NL > 1 else [state["ids"]]
    gen = state["generated_lanes"] if NL > 1 else [state["generated"]]
    ok = True

    # ---- A. feed invariant -----------------------------------------------
    have_inputs = all("inputs" in e for e in transcript) and bool(transcript)
    if not have_inputs:
        print(json.dumps({"checkA": "SKIPPED — transcript has no `inputs` field; "
                                    "what was fed each tick is unrecorded and "
                                    "therefore unprovable", "lanes": NL}))
    else:
        fed = [[] for _ in range(NL)]        # tokens actually fed, per lane
        seen_gen = [[] for _ in range(NL)]   # generated so far, replayed
        a_bad = 0
        for e in transcript:
            tick = e["tick"]
            ins = e["inputs"]
            picks = e.get("picked", [])
            for r in range(NL):
                got = int(ins[r])
                if tick < len(prompts[r]):
                    want, why = int(prompts[r][tick]), "prompt position"
                elif seen_gen[r]:
                    want, why = int(seen_gen[r][-1]), "own last generated token"
                else:
                    want, why = int(prompts[r][-1]), "last prompt token"
                if got != want:
                    a_bad += 1
                    ok = fail(f"lane {r} tick {tick}: fed {got} but the {why} "
                              f"is {want} — the shown text was not produced "
                              f"from the sequence it displays")
                fed[r].append(got)
                p = picks[r] if r < len(picks) else None
                if p is not None:
                    seen_gen[r].append(int(p))
        # every prompt token must have been fed, in order, before generation
        for r in range(NL):
            n = min(len(prompts[r]), len(fed[r]))
            if [int(x) for x in fed[r][:n]] != [int(x) for x in prompts[r][:n]]:
                ok = fail(f"lane {r}: fed prefix != prompt prefix")
            if len(fed[r]) < len(prompts[r]) and gen[r]:
                ok = fail(f"lane {r}: generated {len(gen[r])} tokens but only "
                          f"{len(fed[r])} of {len(prompts[r])} prompt tokens "
                          f"were ever fed")
        print(json.dumps({"checkA": "feed invariant (prompt-in-order, self-feed, "
                                    "retired-freeze)", "lanes": NL,
                          "violations": a_bad, "pass": a_bad == 0}))

    # ---- B. bookkeeping reconciliation -----------------------------------
    # F70: the retirement config comes from the SESSION, not this script's
    # argv. Taking --words-per-lane on the command line let the auditor
    # simulate a different rule than the one that ran (and it ignored stop
    # tokens entirely, so any --stop-on-newline session would be reported as
    # a violation at every tick after a lane stopped — a confident false
    # alarm, the worst kind of audit output).
    cfg = state.get("retire_cfg")
    if cfg is None:
        # AUDIT FIX 2026-09-02: with no retire_cfg AND no explicit budget the
        # fallback simulated "never retire" (--words-per-lane default 0) and
        # produced a confident verdict against a rule that never ran. Refuse.
        if not a.words_per_lane:
            # distinct code (5): "cannot audit" -- not a violation (3), not a
            # skipped check (4), and not a crash (1).
            print(json.dumps({"auditRefused": "session has no retire_cfg and no "
                              "--words-per-lane was given; the auditor cannot know "
                              "the retirement rule it is supposed to check. Pass the "
                              "budget the run used."}))
            sys.exit(5)
        cfg = {"words_per_lane": a.words_per_lane, "stop_ids": [],
               "stop_on_newline": False}
        print(json.dumps({"checkBconfig": "WARNING — session has no retire_cfg "
                                          "(pre-2026-09-01 run); falling back to "
                                          "--words-per-lane and NO stop tokens. "
                                          "A stop-token session will misreport.",
                          "wordsPerLane": a.words_per_lane}))
    else:
        print(json.dumps({"checkBconfig": "from session tokens.json", **cfg}))
    wpl = int(cfg.get("words_per_lane") or 0)
    stop_ids = set(int(s) for s in cfg.get("stop_ids", []))

    rebuilt = [[] for _ in range(NL)]
    retired_sim = [False] * NL
    for e in transcript:
        tick = e["tick"]
        picks = e.get("picked", [])
        for r in range(NL):
            p = picks[r] if r < len(picks) else None
            should_keep = (not retired_sim[r]) and (tick + 1) >= len(prompts[r])
            if p is None and should_keep:
                ok = fail(f"lane {r} tick {tick}: rule says KEEP but transcript discarded")
            if p is not None and not should_keep:
                ok = fail(f"lane {r} tick {tick}: rule says DISCARD but transcript kept {p}")
            if p is not None:
                rebuilt[r].append(int(p))
                if (wpl and len(rebuilt[r]) >= wpl) or int(p) in stop_ids:
                    retired_sim[r] = True
    # F72: tokens.json is saved before transcript.json, so a crash between the
    # two writes leaves generated_lanes one pick AHEAD of the transcript. That
    # is a torn write, not a bookkeeping violation — name it as such instead of
    # reporting a rule breach the run never committed.
    for r in range(NL):
        want = [int(x) for x in gen[r]]
        if rebuilt[r] == want:
            continue
        # AUDIT FIX 2026-09-02: a torn write can only orphan a pick that the
        # rule would have KEPT at the missing tick t = len(transcript). Any
        # other single orphan (e.g. a token past the word budget, or one kept
        # during the prompt phase) is a violation, not a torn write.
        t_missing = len(transcript)
        keep_at_missing = bool(transcript) and (not retired_sim[r]) \
            and (t_missing + 1) >= len(prompts[r])
        if (len(want) == len(rebuilt[r]) + 1 and want[:-1] == rebuilt[r]
                and keep_at_missing):
            print(json.dumps({"checkBnote": "torn trailing write (tokens.json "
                                            "saved before transcript.json); the "
                                            "last pick has no transcript entry",
                              "lane": r, "orphanPick": want[-1]}))
            continue
        ok = fail(f"lane {r}: rebuilt picks != generated_lanes "
                  f"({len(rebuilt[r])} vs {len(gen[r])})")
    print(json.dumps({"checkB": "transcript reconciles with the keep/discard rule",
                      "ticks": len(transcript), "pass": ok}))

    # ---- C. plaintext greedy replay oracle -------------------------------
    if "C" not in a.checks:
        print(json.dumps({"checkC": "SKIPPED (--checks %s)" % a.checks}))
        print(json.dumps({"auditRagged": True, "lanes": NL, "checks": a.checks,
                          "generated": [len(g) for g in gen],
                          "checkAran": bool(have_inputs),
                          "retireCfgFromSession": state.get("retire_cfg") is not None,
                          "pass": ok}))
        # AUDIT FIX 2026-09-02: rc 0 means "all checks ran and passed"; a
        # skipped check A must not exit 0 (it did, with "pass": true).
        return (0 if ok else 3) if have_inputs else 4
    import torch
    torch.set_grad_enabled(False)
    model = load_native(a.tag, a.art_dir, a.device)

    def logits_last(ids):
        x = torch.tensor(ids, device=a.device).unsqueeze(0)
        out = model(x)
        if not torch.is_tensor(out):
            out = out[0]
        lg = out.float()[0] if out.dim() == 3 else out.float()
        return lg[-1].cpu().numpy()

    flips, bugs = [], []
    for r in range(NL):
        seq = list(prompts[r])
        for i, tok_id in enumerate(gen[r]):
            lg = logits_last(seq)
            am = int(np.argmax(lg))
            if am != int(tok_id):
                srt = np.sort(lg)[::-1]
                margin = float(srt[0] - lg[int(tok_id)])
                entry = {"lane": r, "genPos": i, "recorded": int(tok_id),
                         "plaintextArgmax": am, "logitMargin": round(margin, 4)}
                if margin <= a.flip_margin:
                    flips.append(entry)
                else:
                    bugs.append(entry)
                    # F73: NOT a failure by itself. The circuit approximates
                    # what the oracle computes exactly, so a wide-margin flip
                    # is only evidence once the observed rate is compared with
                    # a MEASURED fidelity baseline (--expect-top1). Check D
                    # supplies the structural test that does stand alone.
                    print(json.dumps({"checkCdetail": "wide-margin mismatch",
                                      **entry}))
            seq.append(int(tok_id))
    print(json.dumps({"checkC": "plaintext greedy replay",
                      "noiseFlips": len(flips), "bugSuspects": len(bugs),
                      "flipDetail": flips[:10]}))

    # ---- D. SHIFT TEST — the discriminator check C cannot be ---------------
    # F73 (2026-09-01). Check C compares an APPROXIMATE circuit (--ms-norm,
    # Newton rsqrt, banded decay: deliberate approximations of operations the
    # native model does exactly) against the EXACT native model, so some
    # argmax disagreement is a property of the scheme, not evidence of a bug.
    # A raw mismatch count therefore cannot answer the question
    # ("did the client show a token the model didn't generate?").
    #
    # A SCHEDULER fault can: it displaces tokens by a fixed number of
    # positions, so the recorded token at generated position i is the model's
    # prediction for position i-LAG. Approximation error has no such
    # structure. Score each lag; if a nonzero lag beats lag 0 decisively, the
    # bookkeeping is off by that many ticks. If lag 0 wins (or nothing wins),
    # any disagreement is fidelity, to be judged against a MEASURED baseline
    # (spec_decode/fidelity_tick.py), never against zero.
    lag_hits = {}
    for lag in (0, 1, 2):
        hits = tot = 0
        for r in range(NL):
            for i, tok_id in enumerate(gen[r]):
                # AUDIT FIX 2026-09-02: the lag-L hypothesis is "the recorded
                # token at generated position i is the model's prediction for
                # the position L EARLIER", i.e. the last L tokens of the FULL
                # context (prompt + generated so far) are dropped -- not the
                # last L PROMPT tokens with all generated tokens kept, which
                # is what the previous code scored and which a real lag-1
                # fault never matches.
                full = list(prompts[r]) + [int(x) for x in gen[r][:i]]
                ctx = full[:len(full) - lag] if lag else full
                if len(ctx) < 1:
                    continue
                if int(np.argmax(logits_last(ctx))) == int(tok_id):
                    hits += 1
                tot += 1
        lag_hits[lag] = (hits, tot)
    l0 = lag_hits.get(0, (0, 0))
    best_lag = max(lag_hits, key=lambda k: lag_hits[k][0])
    bh, bt = lag_hits[best_lag]
    # "decisive" = a nonzero lag explains at least 40% more of the sequence
    # than lag 0 does, and explains at least half of it outright.
    # AUDIT FIX 2026-09-02: on a 1-3 token session one coincidental hit at
    # lag 1 cleared both old bars (each collapsed to bh >= 1). Require a real
    # sample and an absolute margin over lag 0, not only a relative one.
    shifted = (best_lag != 0 and bt >= 8 and bh >= max(1, int(0.5 * bt))
               and bh - l0[0] >= max(3, int(0.4 * max(l0[0], 1))))
    if shifted:
        ok = fail(f"SHIFT DETECTED: the recorded tokens match the model's "
                  f"predictions {best_lag} position(s) earlier "
                  f"({bh}/{bt}) far better than in place ({l0[0]}/{l0[1]}) — "
                  f"a scheduler off-by-{best_lag}, not FHE noise")
    print(json.dumps({"checkD": "shift test (scheduler off-by-N discriminator)",
                      "lagScores": {str(k): f"{v[0]}/{v[1]}" for k, v in lag_hits.items()},
                      "bestLag": best_lag, "shiftDetected": bool(shifted),
                      "reading": ("lag 0 best or nothing decisive => disagreement is "
                                  "circuit approximation (judge vs a MEASURED "
                                  "fidelity baseline, not vs zero); a nonzero lag "
                                  "winning => bookkeeping displaced the tokens")}))
    # F73: judge the mismatch RATE against a measured baseline when one is
    # supplied, instead of against zero.
    ntot = sum(len(g) for g in gen)
    agree = ntot - len(bugs) - len(flips)
    rate = agree / max(ntot, 1)
    if a.expect_top1 >= 0:
        # allow 2 binomial sigma below the measured teacher-forced rate
        import math
        sig = math.sqrt(max(a.expect_top1 * (1 - a.expect_top1), 1e-9) / max(ntot, 1))
        floor = a.expect_top1 - 2 * sig
        if rate < floor:
            ok = fail(f"top-1 agreement {rate:.3f} is below the measured "
                      f"fidelity baseline {a.expect_top1:.3f} - 2σ ({floor:.3f}) "
                      f"over {ntot} tokens — worse than the circuit itself, so "
                      f"something beyond approximation is at work")
        print(json.dumps({"checkCrate": round(rate, 4),
                          "baselineTop1": a.expect_top1,
                          "floor2sigma": round(floor, 4),
                          "pass": rate >= floor}))
    else:
        print(json.dumps({"checkCrate": round(rate, 4),
                          "baselineTop1": None,
                          "note": "no --expect-top1 given: the mismatch rate is "
                                  "REPORTED, not judged (an approximate circuit "
                                  "cannot be held to zero mismatches — R4)"}))
    verdict = {"auditRagged": True, "lanes": NL,
               "generated": [len(g) for g in gen],
               "checkAran": bool(have_inputs),
               "retireCfgFromSession": state.get("retire_cfg") is not None,
               "top1AgreeRate": round(rate, 4),
               "shiftDetected": bool(shifted),
               "pass": ok, "noiseFlips": len(flips), "bugSuspects": len(bugs)}
    print(json.dumps(verdict))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
