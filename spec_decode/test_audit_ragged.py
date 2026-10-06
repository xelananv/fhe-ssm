#!/usr/bin/env python3
"""test_audit_ragged.py — the auditor audited (2026-09-01).

An audit script that passes everything is worse than no audit, so this builds
synthetic sessions with KNOWN answers and asserts the auditor agrees:

  1. clean ragged session (2 lanes, different prompt lengths)     -> pass
  2. lane 1's shown text contains a token it was never fed        -> FAIL (A)
  3. a prompt token skipped in the feed order                     -> FAIL (A)
  4. transcript kept a pick while the lane was still prompting    -> FAIL (B)
  5. stop-token retirement, config read from the session          -> pass
     (the pre-fix auditor reported this as a violation at every tick after
      the stop — the false alarm F70 was about)
  6. torn trailing write (tokens.json one pick ahead)             -> pass+note

Checks A and B only (C needs the model and a GPU-hour); run:
  python spec_decode/test_audit_ragged.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "audit_ragged.py")


def build(out, prompts, picks_per_tick, retire_cfg, inputs_per_tick=None,
          gen_override=None):
    """picks_per_tick: list over ticks of [pick|None per lane]."""
    NL = len(prompts)
    os.makedirs(out, exist_ok=True)
    gen = [[] for _ in range(NL)]
    transcript = []
    seen = [[] for _ in range(NL)]
    for tick, picks in enumerate(picks_per_tick):
        if inputs_per_tick is not None:
            ins = inputs_per_tick[tick]
        else:
            ins = []
            for r in range(NL):
                if tick < len(prompts[r]):
                    ins.append(prompts[r][tick])
                elif seen[r]:
                    ins.append(seen[r][-1])
                else:
                    ins.append(prompts[r][-1])
        for r in range(NL):
            if picks[r] is not None:
                gen[r].append(picks[r])
                seen[r].append(picks[r])
        transcript.append({"tick": tick, "phases": ["x"] * NL,
                           "inputs": [int(i) for i in ins],
                           "picked": picks, "encSec": 1.0,
                           "serveSec": 1.0, "decSec": 1.0})
    state = {"lanes": NL, "ids_lanes": prompts,
             "generated_lanes": gen_override if gen_override is not None else gen,
             "ids": prompts[0], "generated": gen[0],
             "bundle_dir": "/nonexistent", "tag": "t", "vocab": 100, "d": 8,
             "dpad": 1024, "retire_cfg": retire_cfg}
    json.dump(state, open(os.path.join(out, "tokens.json"), "w"))
    json.dump(transcript, open(os.path.join(out, "transcript.json"), "w"))


def run_audit(out):
    p = subprocess.run([sys.executable, AUDIT, "--out", out, "--tag", "t",
                        "--checks", "AB"], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def expect(name, out, want_pass, want_rc=None, want_substr=None):
    rc, log = run_audit(out)
    # AUDIT FIX 2026-09-02: a FAIL case used to be satisfied by ANY nonzero
    # rc -- an auditor crash (rc 1, traceback) counted as the wanted verdict.
    # Now a verdict is a specific code (3 = violation, 4 = a check did not
    # run) and, when given, a specific violation text.
    if want_rc is None:
        want_rc = 0 if want_pass else 3
    got_pass = (rc == want_rc) and (want_substr is None or want_substr in log)
    mark = "ok " if got_pass else "BAD"
    print(f"[{mark}] {name}: rc={rc} (wanted rc {want_rc}"
          f"{' + ' + repr(want_substr) if want_substr else ''})")
    if not got_pass:
        print("      ---- auditor said ----")
        for ln in log.strip().splitlines():
            print("      " + ln[:180])
    return got_pass


def main():
    tmp = tempfile.mkdtemp(prefix="auditselftest.")
    allok = True
    P = [[10, 11, 12], [20, 21, 22, 23, 24]]        # ragged: 3 and 5 tokens
    NOSTOP = {"words_per_lane": 0, "stop_ids": [], "stop_on_newline": False}

    # 1. clean: lane0 starts generating at tick 2 (tick+1 >= 3), lane1 at tick 4
    clean = [[None, None], [None, None], [90, None], [91, None],
             [92, 80], [93, 81]]
    d = os.path.join(tmp, "clean"); build(d, P, clean, NOSTOP)
    allok &= expect("clean ragged session", d, True)

    # 2. shown text contains a token never fed: corrupt one input
    d = os.path.join(tmp, "unfed"); build(d, P, clean, NOSTOP)
    tr = json.load(open(os.path.join(d, "transcript.json")))
    tr[4]["inputs"][0] = 777          # lane0 should have re-fed its pick 91
    json.dump(tr, open(os.path.join(d, "transcript.json"), "w"))
    allok &= expect("token shown but never fed", d, False, want_substr="auditViolation")

    # 3. a prompt token skipped
    d = os.path.join(tmp, "skipped"); build(d, P, clean, NOSTOP)
    tr = json.load(open(os.path.join(d, "transcript.json")))
    tr[1]["inputs"][1] = 24           # lane1 fed prompt[4] at tick 1
    json.dump(tr, open(os.path.join(d, "transcript.json"), "w"))
    allok &= expect("prompt token skipped in the feed", d, False, want_substr="auditViolation")

    # 4. kept a pick while still prompting (the keep/discard off-by-one)
    early = [[None, None], [None, 70], [90, None], [91, None],
             [92, 80], [93, 81]]
    d = os.path.join(tmp, "early"); build(d, P, early, NOSTOP)
    allok &= expect("pick kept during prompt phase", d, False, want_substr="auditViolation")

    # 5. stop-token retirement — the F70 false alarm
    STOP = {"words_per_lane": 0, "stop_ids": [92], "stop_on_newline": True}
    stopped = [[None, None], [None, None], [90, None], [91, None],
               [92, 80], [None, 81]]      # lane0 retires on 92, discards after
    d = os.path.join(tmp, "stopped"); build(d, P, stopped, STOP)
    allok &= expect("stop-token retirement (F70 false alarm)", d, True)

    # 5b. same session, but the config is NOT in the state (old-format run):
    #     the auditor must warn, and it will misreport — assert the warning.
    d = os.path.join(tmp, "stopped_nocfg"); build(d, P, stopped, STOP)
    st = json.load(open(os.path.join(d, "tokens.json")))
    del st["retire_cfg"]
    json.dump(st, open(os.path.join(d, "tokens.json"), "w"))
    rc, log = run_audit(d)
    # AUDIT FIX 2026-09-02: the old contract was "warn and proceed", which
    # produced a confident verdict against a never-retire rule the run did
    # not use. New contract: with no retire_cfg AND no --words-per-lane the
    # auditor REFUSES with its own code (5) -- it cannot know the rule.
    refused = rc == 5 and "auditRefused" in log and "retire_cfg" in log
    print(f"[{'ok ' if refused else 'BAD'}] missing retire_cfg and no budget -> "
          f"auditor refuses (rc=5), never a silent assumption (rc={rc})")
    allok &= refused

    # 6. torn trailing write
    d = os.path.join(tmp, "torn")
    build(d, P, clean, NOSTOP)
    st = json.load(open(os.path.join(d, "tokens.json")))
    st["generated_lanes"][0].append(999)      # tokens.json one ahead
    json.dump(st, open(os.path.join(d, "tokens.json"), "w"))
    rc, log = run_audit(d)
    noted = rc == 0 and "torn trailing write" in log
    print(f"[{'ok ' if noted else 'BAD'}] torn trailing write is a note, "
          f"not a rule violation")
    allok &= noted

    # 7. a transcript with no `inputs` must SKIP check A, not pass it
    d = os.path.join(tmp, "noinputs"); build(d, P, clean, NOSTOP)
    tr = json.load(open(os.path.join(d, "transcript.json")))
    for e in tr:
        e.pop("inputs")
    json.dump(tr, open(os.path.join(d, "transcript.json"), "w"))
    rc, log = run_audit(d)
    # AUDIT FIX 2026-09-02: this used to assert only that the word SKIPPED was
    # printed -- while the auditor exited 0 with "pass": true. The property in
    # the label ("never a silent pass") is the EXIT CODE, so assert it.
    skipped = rc == 4 and "SKIPPED" in log and '"checkAran": false' in log
    print(f"[{'ok ' if skipped else 'BAD'}] absent `inputs` -> check A SKIPPED and rc=4, "
          f"never a silent pass (rc={rc})")
    allok &= skipped

    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({"auditSelfTest": "pass" if allok else "FAIL"}))
    return 0 if allok else 3


if __name__ == "__main__":
    sys.exit(main())
