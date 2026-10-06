#!/usr/bin/env python3
"""fidelity_tick.py — is the STATEFUL MULTI-LANE serve faithful? (2026-09-01)

Why this exists. The ragged 4-lane session /root/demo_smoke/genRb completed
cleanly and its bookkeeping reconciles (audit_ragged check B passes), but 11 of
its 16 generated tokens disagree with the plaintext model's greedy choice at
margins up to 11.2 logits — far outside any FHE noise band. Two very different
faults produce that symptom:

  (a) SCHEDULER — the circuit is faithful, the client fed/kept the wrong
      positions; or
  (b) FIDELITY  — the client is right and the encrypted logits are simply
      wrong, i.e. the stateful carry does not compose with --lanes-block.

A free-running session cannot tell them apart: once one token diverges, every
later prefix differs and the comparison is meaningless. So this probe is
TEACHER-FORCED — at tick t every lane is fed its own prompt[t] no matter what
came back — which keeps the encrypted prefix and the plaintext prefix
identical at every tick by construction. Then the only thing that can differ
is the circuit.

Per tick per lane it reports top-1 agreement, relErrRms over the logit vector
(R3: rms, never max), and the plaintext top-1 margin, against the plaintext
model on prompt[:t+1]. What was NEVER measured before: --lanes-block was
validated single-pass (the --lanes-full gate), and the stateful carry was
validated single-lane. Their COMPOSITION has no record.

Usage (server flags must match the session under investigation):
  fidelity_tick.py --out DIR --tag pbd430a --prompts-file F --ticks 6 \
      --keys-dir K --harness-cmd "BIN ... --lanes-block --compressed-store" \
      --device cuda:0

MAC-KEYS MODE (2026-09-02, the demo-day go/no-go). The same teacher-forced
probe can run THROUGH the demo client path against an already-running remote
server that holds only the public/eval keys: encrypt on the Mac with
mac_fhe_client, ship the ciphertext, decrypt the reply on the Mac, compare
with the plaintext model on the Mac. That gates the exact server, keys, store
and layout the recording will use, and warms the server's store on the way.
  fidelity_tick.py --out DIR --tag pbd430a --prompts-file F --ticks 8 \
      --bundle-dir B --art-dir A --device cpu \
      --mac-tool hpc_gpu_port/mac_fhe_client --mac-keys KEYS \
      --remote-ssh "ssh -i KEY -p PORT root@HOST" --remote-reqdir /root/demo/serve
(--stateless is refused here: the mac round-trip is T=1 by construction.)
"""
import argparse
import json
import os
import shlex
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ml-eval"))
from _native_loader_copy import load_native            # noqa: E402
import fhe_client as FC                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--keys-dir", default=None)
    ap.add_argument("--harness-cmd", default=None)
    ap.add_argument("--client-harness-cmd", default=None)
    ap.add_argument("--decode-policy", choices=("greedy", "norepeat"), default="greedy",
                    help="--free-run only: how the CLIENT picks the next token from its decrypted logits (spec_decode/decode_policy.py). "
                         "norepeat = deterministic repetition penalty + no-repeat n-gram; the same rule is applied to the plaintext "
                         "reference's logits on the same history, so the comparison stays exact (rows gain encPick / refPick / pickAgree).")
    ap.add_argument("--rep-penalty", type=float, default=1.3); ap.add_argument("--rep-window", type=int, default=64)
    ap.add_argument("--no-repeat-ngram", type=int, default=4); ap.add_argument("--ban-eos", type=int, default=1)
    ap.add_argument("--bundle-dir",
                    default=os.path.join(os.path.dirname(HERE), "ml-eval", "artifacts"))
    ap.add_argument("--art-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--serve-timeout", type=int, default=3600)
    ap.add_argument("--poll-interval", type=float, default=2.0)
    ap.add_argument("--mac-tool", default=None,
                    help="MAC-KEYS MODE: path to mac_fhe_client; enc/dec run here, the "
                         "server is remote and already running (needs --mac-keys, "
                         "--remote-ssh, --remote-reqdir; --keys-dir/--harness-cmd unused)")
    ap.add_argument("--mac-keys", default=None)
    ap.add_argument("--remote-ssh", default=None)
    ap.add_argument("--remote-reqdir", default=None)
    ap.add_argument("--tick-timeout", type=float, default=1800,
                    help="mac-tool: seconds one tick may wait for its reply (fhe_client default 1800)")
    ap.add_argument("--wait-mode", choices=("poll", "remote"), default="poll",
                    help="mac-tool (S3.7 V3, 2026-09-05): `poll` = the pre-V3 wait "
                         "(default); `remote` = one ssh per tick (fhe_client.py --wait-mode)")
    ap.add_argument("--keep-cts", default=None,
                    help="mac-tool: retain every request/response ciphertext with its "
                         "plaintext under this dir (+ MANIFEST.jsonl); see fhe_client.py")
    ap.add_argument("--stateless", action="store_true",
                    help="CONTROL for the carry. Instead of one token per tick "
                         "against an in-memory carry, re-send the whole prefix "
                         "every tick and reset the state each time. The circuit "
                         "and the approximations are identical; only the carry "
                         "is removed. If relErrRms grows with position HERE too, "
                         "the drift is inherent to the approximate scan; if it "
                         "stays flat, the carry is what degrades.")
    ap.add_argument("--free-run", action="store_true",
                    help="AUTOREGRESSIVE mode (2026-09-18): feed each lane its prompt, then feed back the token the "
                         "ENCRYPTED session produced (its decrypted argmax), for --ticks ticks in total, no stop token. "
                         "The plaintext reference is evaluated on the SAME fed sequence (plain_recurrent.RecurrentRef, "
                         "O(1) per tick), so top-1 agreement and relErrRms stay exact per tick even after a near-tie "
                         "flip. If a reply fails to decrypt, the lane is fed the plaintext argmax for that tick (row "
                         "field fallbackFeed) so the server's timing record continues.")
    ap.add_argument("--recurrent-ref", action="store_true",
                    help="teacher-forced mode: use the O(1)-per-tick recurrent plaintext reference instead of a "
                         "full-prefix forward per lane per tick (implied by --free-run)")
    a = ap.parse_args()
    a.art_dir = a.art_dir or a.bundle_dir
    mac = bool(a.mac_tool)
    if a.free_run and a.stateless:
        sys.exit("fatal: --free-run is a stateful one-token-per-tick session; it cannot be combined with --stateless")
    if mac:
        if not (a.mac_keys and a.remote_ssh and a.remote_reqdir):
            sys.exit("fatal: --mac-tool needs --mac-keys, --remote-ssh and --remote-reqdir")
        if a.stateless:
            sys.exit("fatal: --stateless is not available in mac-keys mode: the mac "
                     "round-trip sends exactly one token per lane per request")
        if not os.path.isfile(os.path.join(a.mac_keys, "public.key")):
            sys.exit(f"fatal: {a.mac_keys}/public.key missing")
    elif not (a.keys_dir and a.harness_cmd):
        sys.exit("fatal: pod mode needs --keys-dir and --harness-cmd (or use --mac-tool)")
    os.makedirs(a.out, exist_ok=True)
    logs = os.path.join(a.out, "logs")
    os.makedirs(logs, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    tok = FC.get_tokenizer()
    prompts = [ln.rstrip("\n") for ln in open(a.prompts_file) if ln.strip()]
    ids_lanes = [tok.encode(p) for p in prompts]
    NL = len(ids_lanes)
    L = min(len(x) for x in ids_lanes)
    ticks = a.ticks if a.free_run else min(a.ticks, L)
    if ticks < 2 or L < 1:
        sys.exit("fatal: need prompts of at least 2 tokens")
    head, V, d = FC.head_matrix(a.bundle_dir, a.tag)
    model = load_native(a.tag, a.art_dir, a.device)
    rr = None
    if a.free_run or a.recurrent_ref:
        from plain_recurrent import RecurrentRef
        cfg = json.loads(open(os.path.join(a.art_dir, f"{a.tag}_config.json")).read())
        rr = RecurrentRef(model, cfg, NL, a.device)
    gen_lanes = [[] for _ in range(NL)]          # free-run: the tokens the ENCRYPTED session generated, per lane
    from decode_policy import pick as _pick
    _ban = [int(tok.eos_token_id)] if (a.ban_eos and getattr(tok, "eos_token_id", None) is not None and a.decode_policy != "greedy") else []
    def choose(logits, r, t):                    # the client's decoding rule on lane r's history up to and including tick t's input
        if a.decode_policy == "greedy": return int(np.argmax(logits))
        hist = [int(x) for x in ids_lanes[r]] + [int(x) for x in gen_lanes[r]]
        return _pick(logits, hist[:t + 1], a.decode_policy, a.rep_penalty, a.rep_window, a.no_repeat_ngram, _ban)
    fallback = [[] for _ in range(NL)]           # free-run: ticks whose output had to be replaced by the plaintext argmax

    def tick_inputs(t):
        """Token fed to every lane at tick t and where it came from."""
        toks, src = [], []
        for r in range(NL):
            if t < len(ids_lanes[r]):
                toks.append(int(ids_lanes[r][t])); src.append("prompt")
            elif a.free_run:
                g = t - len(ids_lanes[r])
                toks.append(int(gen_lanes[r][g])); src.append("ref" if (t - 1) in fallback[r] else "fhe")
            else:
                sys.exit(f"fatal: lane {r} has no prompt token for tick {t}")
        return toks, src

    def plain_logits(ids):
        o = model(torch.tensor(ids, device=a.device).unsqueeze(0))
        if not torch.is_tensor(o):
            o = o[0]
        lg = o.float()[0] if o.dim() == 3 else o.float()
        return lg[-1].double().cpu().numpy()

    state = {"lanes": NL, "prompts": prompts,
             "ids_lanes": [[int(i) for i in x] for x in ids_lanes],
             "generated_lanes": [[] for _ in range(NL)],
             "ids": [int(i) for i in ids_lanes[0]], "generated": [],
             "bundle_dir": os.path.abspath(a.bundle_dir), "tag": a.tag,
             "vocab": int(V), "d": int(d), "dpad": 1024}
    paths = FC.state_paths(a.out)
    FC.save_state(state, paths)

    base = shlex.split(a.harness_cmd) if a.harness_cmd else []
    # 2026-09-19 (pod): server-only instrument flags must not reach the CLIENT-crypto processes (--enc-in / --dec-out): the K3
    # bootstrap recorder refuses to run there ("a client-crypto process has no device") and the first long-run launch died on it.
    def _client_cmd(v):
        out, skip = [], 0
        for x in v:
            if skip: skip -= 1; continue
            if x == "--boot-record": continue
            if x == "--boot-record-secret": skip = 1; continue
            out.append(x)
        return out
    cbase = shlex.split(a.client_harness_cmd) if a.client_harness_cmd else _client_cmd(base)
    keys = ["--keys-dir", a.keys_dir, "--keys-load"] if a.keys_dir else []

    class A:                       # start_server reads these attributes
        # AUDIT FIX 2026-09-02: the stateless CONTROL launched the server
        # --stateful and relied on per-request .reset alone; its server log
        # therefore read as a stateful session. Bind the flag to the mode.
        stateful = not a.stateless
        serve_timeout = a.serve_timeout
    req_dir = os.path.join(a.out, "serve")
    if mac:
        # The server is not ours: verify it is the one --stateful server on
        # the remote dir and that the dir is clean (F64/F65), exactly as the
        # demo client does, then reuse its tick round-trip verbatim.
        pre = FC.mac_session_preflight(a, state)
        state["serverPreflight"] = pre
        FC.save_state(state, paths)
        server = srv_log = None
    else:
        server, srv_log = FC.start_server(A, base, keys, paths, req_dir)

    rows = []
    pre = state.get("serverPreflight") if mac else None
    try:
        for t in range(ticks):
            # TEACHER FORCING: every lane is fed its own prompt token for this
            # position regardless of what the server returned last tick.
            if mac:
                inputs, in_src = tick_inputs(t)
                FC.write_tick_rows(state, paths, inputs)
                lg, t_enc, t_srv, t_dec = FC.mac_tick_roundtrip(a, state, paths, t, t)
                enc = np.asarray(lg, dtype=np.float64)[-NL:]
                res = None
            elif a.stateless:
                # CONTROL: whole prefix every tick, carry dropped every tick.
                # Lane-major N*(t+1) x d, one stride because all lanes send the
                # same number of tokens here by construction.
                blocks = [head[np.asarray(ids_lanes[r][:t + 1], dtype=np.int64)]
                          for r in range(NL)]
                np.ascontiguousarray(np.concatenate(blocks, axis=0),
                                     dtype="<f8").tofile(paths["row_tail"])
                ntok = t + 1
            else:
                # TEACHER FORCING: every lane is fed its own prompt token for
                # this position regardless of what the server returned last tick.
                # (--free-run: after the prompt, the token the encrypted session produced.)
                inputs, in_src = tick_inputs(t)
                FC.write_tick_rows(state, paths, inputs)
                ntok = 1
            if not mac:
                FC.run_step(cbase + keys + ["--enc-in", paths["row_tail"],
                                            "--tokens", str(ntok),
                                            "--client-lanes", str(NL)],
                            os.path.join(logs, f"t{t}.enc.jsonl"), ("clientEnc", "fatal"))
                res, t_srv = FC.serve_roundtrip(server, paths, req_dir, t,
                                                a.poll_interval, paths["row_tail"],
                                                reset_state=(a.stateless or t == 0))
                try:
                    FC.run_step(cbase + keys + ["--allow-secret", "--dec-out", res,
                                                "--tokens", str(ntok)],
                                os.path.join(logs, f"t{t}.dec.jsonl"), ("clientDecDone", "fatal"))
                    # rows are tok-major, lane-minor; the LAST position's NL rows are
                    # the prediction for position t in both modes.
                    enc = np.fromfile(res + ".logits.f64", dtype="<f8").reshape(-1, V)[-NL:]
                except SystemExit as ex:
                    # 2026-09-18 (pod_2xbw): a client decode that fails ("approximation error is
                    # too high") is a RESULT of the stateful carry, not a driver fault. Record it
                    # per lane (relErrRms 1.0 sentinel, top1 False, decFailed) and keep ticking:
                    # teacher forcing needs no decoded output, so the server's per-tick record
                    # (timing, boots, encodes) continues to the asked tick count.
                    print(json.dumps({"tick": t, "decFailed": True, "serveSec": round(t_srv, 1),
                                      "what": str(ex).splitlines()[0][:160]}), flush=True)
                    enc = None
            ref_all = None
            if rr is not None and not a.stateless:
                ref_all = rr.step(inputs).double().cpu().numpy()      # plaintext model on the SAME fed tokens, O(1) per tick
            for r in range(NL):
                ref = ref_all[r] if ref_all is not None else plain_logits(ids_lanes[r][:t + 1])
                generating = a.free_run and t >= len(ids_lanes[r]) - 1
                ref_pick = choose(ref, r, t) if a.free_run else int(np.argmax(ref))
                enc_pick = (choose(enc[r], r, t) if a.free_run else int(np.argmax(enc[r]))) if enc is not None else -1
                if generating:                                        # this tick's output is the lane's next fed token
                    if enc is None:
                        gen_lanes[r].append(ref_pick); fallback[r].append(t)
                    else:
                        gen_lanes[r].append(enc_pick)
                if enc is None:
                    srt = np.sort(ref)[::-1]
                    rec = {"tick": t, "lane": r, "encArgmax": -1, "refArgmax": int(np.argmax(ref)),
                           "top1": False, "relErrRms": 1.0,
                           "refTop1Margin": round(float(srt[0] - srt[1]), 4),
                           "serveSec": round(t_srv, 1), "decFailed": True}
                    if a.free_run:
                        rec.update({"fed": inputs[r], "fedSrc": in_src[r], "generating": bool(generating), "fallbackFeed": bool(generating),
                                    "encPick": -1, "refPick": ref_pick, "pickAgree": False})
                    rows.append(rec)
                    print(json.dumps(rec), flush=True)
                    continue
                e = enc[r]
                num = float(np.sqrt(np.mean((e - ref) ** 2)))
                den = float(np.sqrt(np.mean(ref ** 2)))
                srt = np.sort(ref)[::-1]
                rec = {"tick": t, "lane": r,
                       "encArgmax": int(np.argmax(e)),
                       "refArgmax": int(np.argmax(ref)),
                       "top1": int(np.argmax(e)) == int(np.argmax(ref)),
                       "relErrRms": round(num / max(den, 1e-30), 6),
                       "refTop1Margin": round(float(srt[0] - srt[1]), 4),
                       "serveSec": round(t_srv, 1)}
                if a.free_run:
                    rec.update({"fed": inputs[r], "fedSrc": in_src[r], "generating": bool(generating),
                                "encPick": enc_pick, "refPick": ref_pick, "pickAgree": enc_pick == ref_pick})
                rows.append(rec)
                print(json.dumps(rec), flush=True)
            if res is not None:
                for f in __import__("glob").glob(res + ".*"):
                    os.remove(f)
            json.dump(rows, open(os.path.join(a.out, "fidelity.json"), "w"), indent=1)
            if a.free_run:
                state["generated_lanes"] = [[int(x) for x in g] for g in gen_lanes]
                state["freeRun"] = True; state["fallbackTicks"] = fallback
                state["decodePolicy"] = {"policy": a.decode_policy, "penalty": a.rep_penalty, "window": a.rep_window, "ngram": a.no_repeat_ngram, "ban": _ban}
                FC.save_state(state, paths)
    finally:
        if server is not None:
            FC.stop_server(server, srv_log, req_dir)
        if mac:
            FC.mac_session_finish(a)   # V3: remote wait mode defers the last rm

    agree = sum(1 for x in rows if x["top1"])
    rel = [x["relErrRms"] for x in rows]
    verdict = {"fidelityTick": True, "lanes": NL, "ticks": ticks,
               "comparisons": len(rows),
               "top1Agree": agree, "top1AgreeFrac": round(agree / max(len(rows), 1), 4),
               "relErrRmsMedian": round(float(np.median(rel)), 6),
               "relErrRmsWorst": round(float(np.max(rel)), 6),
               "teacherForced": not a.free_run, "freeRun": bool(a.free_run),
               "decFailedTicks": sorted({x["tick"] for x in rows if x.get("decFailed")}),
               # REVIEW 2026-09-03: the mode string used to hardcode "secret
               # never left it" even against the mock (which decrypts with the
               # secret). It is now derived from the preflight record.
               "mode": (("mac-transport against a MOCK server (the mock holds the "
                         "secret): protocol check only, not a fidelity record"
                         if (pre or {}).get("mock") else
                         ("mac-transport, local server on this machine"
                          if (pre or {}).get("transport") == "local" else
                          "mac-keys: remote server, enc/dec on this machine, secret "
                          "never left it")) if mac else
                        ("stateless-control" if a.stateless else "pod-keys")),
               "server": pre,
               "remoteReqDir": a.remote_reqdir if mac else None,
               "reading": ("high top1 + small relErrRms => the CIRCUIT is faithful "
                           "and genRb's mismatches are a SCHEDULER fault; low top1 "
                           "or large relErrRms => the stateful carry does not "
                           "compose with --lanes-block (FIDELITY fault)")}
    print(json.dumps(verdict))
    json.dump({"rows": rows, "verdict": verdict},
              open(os.path.join(a.out, "fidelity.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
