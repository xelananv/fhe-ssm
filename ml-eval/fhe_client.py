#!/usr/bin/env python3
"""fhe_client.py — client driver for the FHE-SSM client/server inference demo
(blocker 6). Pairs with hpc_gpu_port/gpu_real_model.cu's demo modes; the full
runbook is docs/RUNBOOK.md.

ROLES AND HONESTY. This script performs the PLAINTEXT client duties itself
(tokenize, embed, unembed-sample) and drives the harness binary as a
subprocess for the CRYPTOGRAPHIC client duties (encrypt with --enc-in,
decrypt with --dec-out) and for the server forward pass (--ct-in/--ct-out).
There are no OpenFHE Python bindings in play: "client operations executed by
the same binary in client mode" is the honest description. Key isolation is
therefore PROCESS-level, not code-level — the server subprocess never loads
the secret key file (it is started without --allow-secret and the harness
refuses to read secret.key without it), but client and server are the same
executable on the same machine. This demonstrates the PROTOCOL split, not a
deployment boundary.

EMBEDDING/UNEMBEDDING ARE CLIENT-SIDE BY DESIGN (consistent with the paper's
stated model): the bundle's `head` matrix is the tied embedding, so the
client embeds token ids by row lookup into `head` and turns returned hidden
states into logits with the same matrix. The server only ever sees CKKS
ciphertexts of embedded rows.

PREFILL MODE. By DEFAULT this is STATELESS: each generated token reruns the
FULL prefix through the server, so step cost grows with sequence length and
generating N tokens costs N(N+1)/2 scan steps. That cost is reported per step
in transcript.json rather than hidden.

--stateful turns that off. Step 0 prefills the prompt; every later step sends
ONE token and the server resumes its per-layer carry, so cost per step is
flat. The carry is TWO ciphertexts per layer -- the scan accumulator AND the
shift-mix boundary u_{t-1}, whose u_{-1}=0 hardcode is a cold-start-only
condition. It is held in SERVER memory under --serve-mode (no transfer) or
crosses on disk otherwise; the client never holds it, because one carried
ciphertext runs ~1 MB per RNS limb and shipping 2 per layer per step would
dwarf the prefix it replaces. Numerics validated on CPU/OpenFHE by
harness/state_carry_probe.cpp (arms A-D).

Subcommands
  prepare   tokenize a prompt, embed via `head`, write <out>/rows.f64 + tokens.json
  sample    read a logits file, sample the next token, re-embed, update state
  generate  orchestrate N steps of (client-enc -> server-run -> client-dec -> sample)

Typical use (paths per docs/RUNBOOK.md):
  python3 fhe_client.py prepare --prompt "The " \
      --bundle-dir /root/src/ml-eval/artifacts --tag agn --out /root/demo/gen
  python3 fhe_client.py generate --out /root/demo/gen --steps 8 --greedy \
      --keys-dir /root/demo/keys \
      --harness-cmd "/root/fhe-ssm-hpc/build-demo/gpu_real_model --tag agn \
          --bundle-dir /root/src/ml-eval/artifacts --device 0 --log-ring 15 \
          --level-budget 3 3 --ms-norm 5.5 --interleave --enc-threads 32 \
          --pack-tokens 16 --extra-depth 9 --boot-floor 3 --matvec-margin 4"
"""

import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

TOK_MODEL = "RWKV/rwkv-4-169m-pile"   # GPT-NeoX tokenizer, vocab 50277
                                      #

# --stub-tokenizer (2026-09-02, dry-run support). Set ONLY from the CLI flag
# by cmd_prepare/cmd_generate/cmd_sample; get_tokenizer() consults it. A stub
# session is stamped stubTokenizer:true in tokens.json and refused by any
# later subcommand invoked without the flag (and vice versa) -- F62 class: a
# stub session must never be mistaken for a demo record.
_STUB_TOKENIZER = False
_STUB_BANNER_SHOWN = False


class _ByteStubTokenizer:
    """Byte-level stand-in for the GPT-NeoX tokenizer, for protocol dry runs
    on machines without `transformers`. encode = the UTF-8 bytes of the text
    (ids < 256), decode = the bytes back (errors=replace). Model predictions
    are argmaxes over the REAL vocab (0..V-1), so an id >= 256 can legitimately
    arrive here; it is rendered as a visible <<id>> marker rather than dropped
    or crashed on. Nothing this class emits is model text."""
    is_stub = True

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        out, buf = [], bytearray()
        for i in ids:
            i = int(i)
            if 0 <= i < 256:
                buf.append(i)
            else:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
                out.append("<<%d>>" % i)
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)


# --------------------------------------------------------------- bundle access
def parse_index(bundle_dir, tag):
    """Parse bundle_<tag>.index.txt: '<name> <offset_bytes> <nElems> <dims...>'
    (the exact format the C++ loader `Bundle::load` reads)."""
    path = os.path.join(bundle_dir, f"bundle_{tag}.index.txt")
    idx = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            name, off, n = parts[0], int(parts[1]), int(parts[2])
            dims = [int(x) for x in parts[3:]]
            idx[name] = (off, n, dims)
    return idx


def head_matrix(bundle_dir, tag):
    """Memory-map the tied embedding/unembedding matrix `head` (V, d) float64.
    With tied embeddings, head IS the embedding table: embed = head[ids]."""
    idx = parse_index(bundle_dir, tag)
    if "head" not in idx:
        sys.exit(f"fatal: no `head` tensor in bundle_{tag}.index.txt")
    off, n, dims = idx["head"]
    if len(dims) != 2:
        sys.exit(f"fatal: head dims {dims} not 2-D")
    V, d = dims
    mm = np.memmap(os.path.join(bundle_dir, f"bundle_{tag}.bin"),
                   dtype="<f8", mode="r")
    return mm[off // 8: off // 8 + n].reshape(V, d), V, d


def get_tokenizer():
    global _STUB_BANNER_SHOWN
    if _STUB_TOKENIZER:
        if not _STUB_BANNER_SHOWN:
            _STUB_BANNER_SHOWN = True
            banner = ("#" * 72 + "\n"
                      "#  STUB TOKENIZER (--stub-tokenizer): byte-level ids, NOT GPT-NeoX.\n"
                      "#  This session exercises the PROTOCOL only. Its text, ids and\n"
                      "#  lanes_text.json are NOT a demo record and must never be quoted\n"
                      "#  as one. tokens.json is stamped stubTokenizer:true.\n"
                      + "#" * 72)
            print(banner, file=sys.stderr)
            print(banner)
        return _ByteStubTokenizer()
    # No silent fallback: a missing `transformers` is fatal unless the caller
    # asked for the stub explicitly.
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("fatal: `transformers` not installed "
                 "(pip install transformers tokenizers), and --stub-tokenizer "
                 "was not given (the stub is for protocol dry runs only)")
    return AutoTokenizer.from_pretrained(TOK_MODEL)


def set_stub_tokenizer(flag):
    global _STUB_TOKENIZER
    _STUB_TOKENIZER = bool(flag)


def check_stub_consistency(state, args):
    """F62 class: the stamp in tokens.json and the CLI flag must agree, both
    ways. Without this a stub session could be continued/sampled/audited as
    a real one (or a real one decoded with the stub) and nothing would say."""
    stamped = bool(state.get("stubTokenizer", False))
    flag = bool(getattr(args, "stub_tokenizer", False))
    if stamped and not flag:
        sys.exit("fatal: this session was prepared with --stub-tokenizer "
                 "(tokens.json stubTokenizer:true) but the flag was not given "
                 "now. A stub session is a protocol dry run, not a demo record; "
                 "pass --stub-tokenizer to continue it, or re-run `prepare` "
                 "without the stub.")
    if flag and not stamped:
        sys.exit("fatal: --stub-tokenizer given but this session's tokens.json "
                 "has no stubTokenizer:true stamp — it was prepared with the "
                 "real tokenizer. Refusing to mix the two (byte ids would be "
                 "decoded as GPT-NeoX ids and vice versa).")


# --------------------------------------------------------------- state on disk
def state_paths(out_dir):
    return {
        "tokens": os.path.join(out_dir, "tokens.json"),
        "rows": os.path.join(out_dir, "rows.f64"),
        "row_tail": os.path.join(out_dir, "row_tail.f64"),
        "transcript": os.path.join(out_dir, "transcript.json"),
        "logs": os.path.join(out_dir, "logs"),
    }


def load_state(out_dir):
    p = state_paths(out_dir)
    if not os.path.exists(p["tokens"]):
        sys.exit(f"fatal: {p['tokens']} missing — run `prepare` first")
    with open(p["tokens"]) as f:
        return json.load(f), p


def write_rows(state, paths):
    """(Re)write rows.f64 = head[ids] as raw little-endian float64, T x d —
    exactly what the harness --enc-in expects. LANES MODE (state["lanes"]>1):
    LANE-MAJOR N*T x d (lane r block first), matching --client-lanes; all
    lanes hold equal-length sequences by construction (prepare enforces)."""
    head, _, _ = head_matrix(state["bundle_dir"], state["tag"])
    if state.get("lanes", 1) > 1:
        blocks = []
        lens = []
        for r in range(state["lanes"]):
            ids = state["ids_lanes"][r] + state["generated_lanes"][r]
            lens.append(len(ids))
            blocks.append(head[np.asarray(ids, dtype=np.int64)])
        # F69: --client-lanes reads lane r at row offset r*T -- ONE stride for
        # every lane. Unequal lengths have no such stride, so writing the
        # concatenation would hand the reader lane-boundary garbage that
        # decrypts to plausible logits. Refuse loudly instead.
        if len(set(lens)) > 1:
            sys.exit("fatal: write_rows: lanes have unequal lengths %s — the "
                     "lane-major rows.f64 layout is stride-T and cannot "
                     "represent this. The v2 tickwise engine (T=1 per lane per "
                     "request) is the ragged path; rows.f64 is v1-only." % lens)
        rows = np.ascontiguousarray(np.concatenate(blocks, axis=0), dtype="<f8")
        rows.tofile(paths["rows"])
        return lens[0]
    ids = state["ids"] + state["generated"]
    rows = np.ascontiguousarray(head[np.asarray(ids, dtype=np.int64)],
                                dtype="<f8")
    rows.tofile(paths["rows"])
    return len(ids)


def write_row_tail(state, paths, n=1):
    """Write ONLY the last `n` embedded rows to row_tail.f64.

    STATEFUL DECODE. With a stateful server the client sends just the new
    token(s): the server already holds the per-layer carry (scan accumulator +
    shift-mix boundary) from the previous step, so re-sending the prefix would
    recompute what the server already has. rows.f64 still holds the FULL
    history — it is the reference for the stateless path and for --dec-out
    comparisons — and this is a separate tail file rather than an offset into
    it because the harness's --enc-in reads the first T rows of its input and
    has no offset argument.
    """
    head, _, _ = head_matrix(state["bundle_dir"], state["tag"])
    if state.get("lanes", 1) > 1:
        # AUDIT FIX 2026-09-02: with n>1 a lane shorter than n yields fewer
        # rows from [-n:], silently misaligning lane boundaries (the F69
        # shape). Every live caller passes n=1; make that a contract.
        if n != 1:
            sys.exit("fatal: write_row_tail on lanes supports n=1 only (a "
                     "1-row tail is always aligned; n>1 is not on ragged lanes)")
        blocks = []
        for r in range(state["lanes"]):
            ids = (state["ids_lanes"][r] + state["generated_lanes"][r])[-n:]
            blocks.append(head[np.asarray(ids, dtype=np.int64)])
        rows = np.ascontiguousarray(np.concatenate(blocks, axis=0), dtype="<f8")
        rows.tofile(paths["row_tail"])
        return n
    ids = (state["ids"] + state["generated"])[-n:]
    rows = np.ascontiguousarray(head[np.asarray(ids, dtype=np.int64)],
                                dtype="<f8")
    rows.tofile(paths["row_tail"])
    return len(ids)


def save_state(state, paths):
    with open(paths["tokens"], "w") as f:
        json.dump(state, f, indent=1)


# ------------------------------------------------------------------ subcommands
def cmd_prepare(args):
    os.makedirs(args.out, exist_ok=True)
    paths = state_paths(args.out)
    os.makedirs(paths["logs"], exist_ok=True)
    set_stub_tokenizer(args.stub_tokenizer)

    if args.lanes > 1:
        if not args.prompts_file:
            sys.exit("fatal: --lanes N needs --prompts-file (one prompt per line)")
        with open(args.prompts_file) as f:
            prompts = [ln.rstrip("\n") for ln in f if ln.strip()]
        if len(prompts) != args.lanes:
            sys.exit(f"fatal: --lanes {args.lanes} but {len(prompts)} prompts in file")
        tok = get_tokenizer()
        ids_lanes = [tok.encode(pr) for pr in prompts]
        # AUDIT FIX 2026-09-02: the single-prompt path refuses a prompt that
        # tokenizes to nothing; the lanes path did not, and an empty lane
        # crashed the engine at prompts[r][-1] several ticks later.
        empty = [r for r, ids in enumerate(ids_lanes) if len(ids) == 0]
        if empty:
            sys.exit(f"fatal: lane(s) {empty} tokenized to zero tokens")
        # v2 (2026-09-01): RAGGED prompt lengths supported — the generate loop
        # is tick-wise with a per-lane cursor; lanes in the prompt phase feed
        # scripted tokens (predictions discarded), lanes past their prompt
        # feed their own sampled token. No equal-length constraint.
        head, V, d = head_matrix(args.bundle_dir, args.tag)
        for ids in ids_lanes:
            bad = [i for i in ids if i < 0 or i >= V]
            if bad:
                sys.exit(f"fatal: token ids outside head rows: {bad[:5]}")
        lens = [len(ids) for ids in ids_lanes]
        ragged = len(set(lens)) > 1
        state = {"lanes": args.lanes, "prompts": prompts,
                 "ids_lanes": [[int(i) for i in ids] for ids in ids_lanes],
                 "generated_lanes": [[] for _ in range(args.lanes)],
                 "ids": [int(i) for i in ids_lanes[0]], "generated": [],
                 "bundle_dir": os.path.abspath(args.bundle_dir),
                 "tag": args.tag, "vocab": int(V), "d": int(d),
                 # F68: dpad is a LAYOUT constant shared by the mac client, the
                 # server and the decryptor. Stamp it at prepare so no tool has
                 # to hardcode it and every tool can cross-check.
                 "dpad": int(args.dpad),
                 # F69: with ragged prompts the lane-major rows.f64 has no
                 # single stride-T, so the v1 (whole-prefix) path cannot read
                 # it. Mark the state; the v1 path refuses, v2 never uses it.
                 "ragged": bool(ragged), "prompt_lens": lens}
        if args.stub_tokenizer:
            state["stubTokenizer"] = True      # F62 class: never a demo record
        paths2 = state_paths(args.out)
        # F67: a stale transcript.json from a PREVIOUS session in this --out
        # makes the v2 engine start at tick=len(transcript), so every lane is
        # already "past its prompt" and the model never sees a prompt token.
        # Archive rather than delete -- the old session is still evidence.
        for stale_key, stale_path in (("transcript", paths2["transcript"]),
                                      ("lanes_text",
                                       os.path.join(args.out, "lanes_text.json"))):
            if os.path.exists(stale_path):
                bak = stale_path + ".prev"
                os.replace(stale_path, bak)
                print(f"note: archived stale {stale_key} -> {os.path.basename(bak)}")
        save_state(state, paths2)
        if ragged:
            # do not write a rows.f64 that misparses; v2 uses row_tail only.
            for p in (paths2["rows"],):
                if os.path.exists(p):
                    os.remove(p)
            rows_bytes = 0
        else:
            write_rows(state, paths2)
            rows_bytes = os.path.getsize(paths2["rows"])
        print(json.dumps({"prepare": True, "lanes": args.lanes,
                          "promptTokensPerLane": lens, "ragged": ragged,
                          "d": d, "vocab": V, "dpad": int(args.dpad),
                          "rowsBytes": rows_bytes,
                          "stubTokenizer": bool(args.stub_tokenizer),
                          "note": ("ragged: rows.f64 intentionally absent "
                                   "(v2 tickwise uses row_tail.f64)") if ragged
                          else "equal-length: rows.f64 written"}))
        for r, pr in enumerate(prompts):
            print(f"lane {r} ({lens[r]} tok):", json.dumps(pr))
        return

    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        sys.exit("fatal: --prompt or --prompt-file required")

    tok = get_tokenizer()
    ids = tok.encode(prompt)
    if args.max_tokens and len(ids) > args.max_tokens:
        ids = ids[: args.max_tokens]
    if not ids:
        sys.exit("fatal: prompt tokenized to zero tokens")
    if len(ids) > 512:
        print(f"warning: {len(ids)} prompt tokens; the model was trained at "
              f"ctx 512 and per-step cost grows with T", file=sys.stderr)

    head, V, d = head_matrix(args.bundle_dir, args.tag)
    bad = [i for i in ids if i < 0 or i >= V]
    if bad:
        sys.exit(f"fatal: token ids outside head rows 0..{V-1}: {bad[:5]}")

    state = {
        "prompt": prompt,
        "ids": [int(i) for i in ids],
        "generated": [],
        "bundle_dir": os.path.abspath(args.bundle_dir),
        "tag": args.tag,
        "vocab": int(V),
        "d": int(d),
    }
    if args.stub_tokenizer:
        state["stubTokenizer"] = True          # F62 class: never a demo record
    save_state(state, paths)
    T = write_rows(state, paths)
    print(json.dumps({"prepare": True, "out": args.out, "tokens": T,
                      "d": d, "vocab": V,
                      "stubTokenizer": bool(args.stub_tokenizer),
                      "rowsBytes": os.path.getsize(paths["rows"])}))
    print("prompt:", json.dumps(prompt))
    print("ids:", state["ids"])


def sample_from_logits(logits, temp, top_k, greedy, rng):
    if greedy:
        return int(np.argmax(logits))
    l = logits.astype(np.float64) / max(temp, 1e-6)
    if top_k and top_k > 0 and top_k < len(l):
        cand = np.argpartition(l, -top_k)[-top_k:]
    else:
        cand = np.arange(len(l))
    p = np.exp(l[cand] - l[cand].max())
    p /= p.sum()
    return int(rng.choice(cand, p=p))


def do_sample(state, paths, logits_path, args, rng, expect_rows=None):
    """Read the LAST position's logits row, sample the next id, append it,
    re-embed the whole sequence (rows.f64 grows to T+1 rows)."""
    V = state["vocab"]
    raw = np.fromfile(logits_path, dtype="<f8")
    if raw.size == 0 or raw.size % V != 0:
        sys.exit(f"fatal: {logits_path} has {raw.size} doubles, "
                 f"not a multiple of vocab {V}")
    logits = raw.reshape(-1, V)
    NL = state.get("lanes", 1)
    if NL > 1:
        # rows are tok-major, lane-minor: last position = rows[-NL:]
        tok = get_tokenizer()
        nxts = []
        for r in range(NL):
            row = logits[-NL + r]
            nxt = sample_from_logits(row, args.temp, args.top_k, args.greedy, rng)
            state["generated_lanes"][r].append(int(nxt))
            nxts.append(int(nxt))
        state["ids"] = state["ids_lanes"][0]
        state["generated"] = state["generated_lanes"][0]
        save_state(state, paths)
        write_rows(state, paths)
        pieces = [tok.decode([n]) for n in nxts]
        print(json.dumps({"sample": True, "lanes": NL, "next_ids": nxts,
                          "next_pieces": pieces}))
        for r in range(NL):
            txt = tok.decode(state["ids_lanes"][r] + state["generated_lanes"][r])
            print(f"lane {r} text:", json.dumps(txt))
        return nxts[0]
    T = len(state["ids"]) + len(state["generated"])
    # STATEFUL DECODE returns logits for the token(s) actually SENT (1 on a
    # resumed step), not for the whole prefix, so expect_rows is passed in.
    expect = expect_rows if expect_rows is not None else T
    if logits.shape[0] != expect:
        print(f"warning: logits rows {logits.shape[0]} != expected {expect} "
              f"(using the last row regardless)", file=sys.stderr)
    nxt = sample_from_logits(logits[-1], args.temp, args.top_k, args.greedy, rng)
    state["generated"].append(int(nxt))
    save_state(state, paths)
    newT = write_rows(state, paths)
    tok = get_tokenizer()
    text = tok.decode(state["ids"] + state["generated"])
    print(json.dumps({"sample": True, "next_id": int(nxt),
                      "next_piece": tok.decode([nxt]), "tokens_now": newT}))
    print("text so far:", json.dumps(text))
    return nxt


def apply_lane_predictions(state, args, rng, rows, tick, prompts, gen,
                           retired, target, stop_ids):
    """The ONE load-bearing rule (D11): keep a lane's prediction iff its NEXT
    position is past its prompt and it is not retired; else discard."""
    picked = []
    for r in range(state["lanes"]):
        if retired[r] or (tick + 1) < len(prompts[r]):
            picked.append(None)
            continue
        nxt = sample_from_logits(np.asarray(rows[r]), args.temp, args.top_k,
                                 args.greedy, rng)
        gen[r].append(int(nxt))
        picked.append(int(nxt))
        if (target and len(gen[r]) >= target) or nxt in stop_ids:
            retired[r] = True
    state["generated_lanes"] = gen
    return picked


def write_tick_rows(state, paths, tick_inputs):
    """One tick's inputs: N token ids (one per lane) -> row_tail.f64,
    LANE-MAJOR N x 1 x d (what --client-lanes/--enc-in expects at T=1)."""
    head, _, _ = head_matrix(state["bundle_dir"], state["tag"])
    rows = np.ascontiguousarray(
        head[np.asarray(tick_inputs, dtype=np.int64)], dtype="<f8")
    rows.tofile(paths["row_tail"])


def _remote_is_local(args):
    """--remote-ssh 'local' (2026-09-02): the server is a process on THIS
    machine (e.g. ml-eval/mock_fhe_server.py for a no-GPU dry run). Remote
    shell commands then run via `sh -c`, and file transfers are plain copies
    with no host prefix. Every other value is the ssh command as before."""
    return args.remote_ssh == "local"


# ---------------------------------------------------- control-traffic ledger
# S3.7 V3 (2026-09-05; the timer-correction notes N2, T12 §2.9, FINAL_TABLE §3).
# On the author's proxy path every ssh/scp PROCESS is a ~4 s round trip
# (pod_rtx6000x5_20260904/POD_LOG.md:21), and the poll-mode tick spends
# 24–53 s in them: 2 puts + `touch ready` + a poll every --poll-interval +
# liveness every 10 polls + `stat` + 2 gets + `rm`. The client recorded no
# per-call timestamps, so that split stopped at the code (T12 §2.9). Every
# transport launch (_rsh, _rcp, the chunked scp streams, the one-shot remote
# wait) is now counted here, per tick, so the transcript can carry the
# ADDITIVE fields ctlCalls / ctlSec / waitMode beside the untouched
# encSec/serveSec/decSec, and FHE_CLIENT_COUNT_CALLS=1 prints the whole
# ledger as one JSON line at exit (the dry run's Mac test reads it).
#   ctlCalls = number of ssh/scp/sh/copy process invocations in the tick
#   ctlSec   = wall time inside those processes MINUS the server-side wait
#              loop of the remote-mode script (the poll-mode sleep() is not
#              inside any process and is therefore never counted either)
_CTL = {"tick": None, "calls": 0, "sec": 0.0, "wait": 0.0, "byKind": {},
        "ticks": [], "pre": {"calls": 0, "sec": 0.0}, "last": None,
        "atexit": False}


def _ctl_note(kind, seconds, wait_sec=0.0, calls=1):
    """Record `calls` transport launches of kind `kind` that took `seconds`
    of wall, of which `wait_sec` was the remote script's wait loop."""
    if _CTL["tick"] is None:
        _CTL["pre"]["calls"] += calls
        _CTL["pre"]["sec"] += max(seconds - wait_sec, 0.0)
        return
    _CTL["calls"] += calls
    _CTL["sec"] += max(seconds - wait_sec, 0.0)
    _CTL["wait"] += wait_sec
    _CTL["byKind"][kind] = _CTL["byKind"].get(kind, 0) + calls


def _ctl_tick_begin(tick, mode):
    _ctl_tick_end()
    _CTL.update(tick=int(tick), calls=0, sec=0.0, wait=0.0, byKind={}, mode=mode)
    if os.environ.get("FHE_CLIENT_COUNT_CALLS") == "1" and not _CTL["atexit"]:
        import atexit
        atexit.register(_ctl_print_ledger)
        _CTL["atexit"] = True


def _ctl_tick_end():
    if _CTL["tick"] is None:
        return None
    rec = {"tick": _CTL["tick"], "waitMode": _CTL.get("mode"),
           "ctlCalls": int(_CTL["calls"]), "ctlSec": round(_CTL["sec"], 3),
           "waitSec": round(_CTL["wait"], 3), "byKind": dict(_CTL["byKind"])}
    _CTL["ticks"].append(rec)
    _CTL["last"] = rec
    _CTL["tick"] = None
    return rec


def _ctl_last():
    """The ledger record of the most recently finished tick (or None)."""
    return _CTL["last"]


def _ctl_print_ledger():
    _ctl_tick_end()
    t = _CTL["ticks"]
    print(json.dumps({"ctlLedger": True, "ticks": t,
                      "preTickCalls": _CTL["pre"]["calls"],
                      "preTickSec": round(_CTL["pre"]["sec"], 3),
                      "totalTickCalls": sum(x["ctlCalls"] for x in t),
                      "totalTickCtlSec": round(sum(x["ctlSec"] for x in t), 3),
                      "meanCallsPerTick": (round(sum(x["ctlCalls"] for x in t) / len(t), 3)
                                           if t else None)}), flush=True)


def _rsh(args, cmd, check=False, retry=True):
    """Run ONE shell command string on the server host and return the
    CompletedProcess (stdout/stderr captured, text). Transport:
      * local  -> ["sh", "-c", cmd]
      * ssh    -> shlex.split(args.remote_ssh) + [cmd]   (unchanged argv)
    check=True turns a nonzero exit into a fatal with the stderr shown.
    retry=False (V3, 2026-09-05) disables the exit-255 backoff -- used only
    by the session-end cleanup, where a hung transport must not stall exit."""
    if _remote_is_local(args):
        argv = ["sh", "-c", cmd]
    else:
        argv = shlex.split(args.remote_ssh) + [cmd]   # e.g. ssh -i key -p 1868 root@host
    # REVIEW 2026-09-03: a demo session makes ~1,000 ssh connections; one
    # transient transport failure (ssh exit 255) used to abort the recording
    # irrecoverably (F66 forbids resume). Every command here is idempotent
    # (test -f, touch, printf >, rm -f, cat), so retry exit 255 with backoff.
    r = None
    for attempt in range(5 if retry else 1):
        t0 = time.time()
        r = subprocess.run(argv, capture_output=True, text=True)
        _ctl_note("rsh", time.time() - t0)
        if r.returncode != 255 or _remote_is_local(args):
            break
        print(f"warning: ssh transport failure (attempt {attempt + 1}/5): {r.stderr.strip()[:200]}",
              file=sys.stderr, flush=True)
        time.sleep(2 * (2 ** attempt))
    if check and r.returncode != 0:
        sys.exit("fatal: remote command failed (exit %d): %s\n--- stdout ---\n%s"
                 "--- stderr ---\n%s" % (r.returncode, cmd, r.stdout, r.stderr))
    return r


def _rcp(args, src, dst, direction="put", known_size=None):
    """Copy one file between this machine and the server host. src/dst are
    PLAIN paths; the host: prefix is added here (scp) or not at all (local).
      direction "put": src is local, dst is on the host
      direction "get": src is on the host, dst is local
    scp argv is built exactly as before: the ssh options between the `ssh`
    word and the host are reused, -p (ssh port) becomes -P (scp port).
    known_size (V3, 2026-09-05): a "get" whose remote size the caller already
    holds (the remote wait script prints it) skips the `stat` round trip."""
    if direction not in ("put", "get"):
        sys.exit(f"fatal: _rcp direction {direction!r}")
    if _remote_is_local(args):
        t0 = time.time()
        shutil.copy(src, dst)
        _ctl_note("copy", time.time() - t0)
        return
    rssh = shlex.split(args.remote_ssh)
    host = rssh[-1]
    scp_base = ["scp"] + [a for a in rssh[1:-1]] + ["-q"]
    # scp uses -P for port where ssh uses -p
    scp_base = ["-P" if a == "-p" else a for a in scp_base]
    # 2026-09-04 (pod O-2086900 through the author's proxy): one scp stream moves
    # 1.5 MB/s up and 0.6 MB/s down on a ~4 s RTT path, while 6 streams aggregate
    # ~4.7 MB/s. A 2^17 ciphertext is 88 MB per direction per tick, so single-stream
    # wire time would exceed the tick itself. Files >= RCP_CHUNK_MIN go as
    # RCP_STREAMS parallel chunks, reassembled with cat and size-verified; small
    # files (metas) keep the plain scp path. Both paths are idempotent overwrites.
    try:
        size = os.path.getsize(src) if direction == "put" else -1
    except OSError:
        size = -1
    if direction == "get":
        if known_size is not None and int(known_size) >= 0:
            size = int(known_size)
        else:
            q = _rsh(args, f"stat -c %s {shlex.quote(src)} 2>/dev/null || stat -f %z {shlex.quote(src)}")
            try: size = int(q.stdout.strip().split()[-1])
            except Exception: size = -1
    if size >= RCP_CHUNK_MIN:
        _rcp_chunked(args, scp_base, host, src, dst, direction, size)
        return
    if direction == "put":
        argv = scp_base + [src, f"{host}:{dst}"]
    else:
        argv = scp_base + [f"{host}:{src}", dst]
    # idempotent overwrite: retry transport failures with backoff (see _rsh)
    for attempt in range(5):
        t0 = time.time()
        r = subprocess.run(argv)
        _ctl_note("scp", time.time() - t0)
        if r.returncode == 0:
            return
        print(f"warning: scp failed (exit {r.returncode}, attempt {attempt + 1}/5): "
              f"{' '.join(argv)}", file=sys.stderr, flush=True)
        time.sleep(2 * (2 ** attempt))
    sys.exit(f"fatal: scp failed 5 times: {' '.join(argv)}")


RCP_STREAMS = int(os.environ.get("FHE_RCP_STREAMS", "6"))
RCP_CHUNK_MIN = int(os.environ.get("FHE_RCP_CHUNK_MIN", str(8 << 20)))


def _rcp_chunked(args, scp_base, host, src, dst, direction, size):
    """Parallel-stream copy of ONE large file (see _rcp). put: split locally, scp
    the parts concurrently to <dst>.partNN, cat them on the host. get: split on
    the host, scp the parts down concurrently, cat locally. The reassembled size
    must equal the source size or the copy is fatal; parts are removed."""
    n = max(1, min(RCP_STREAMS, (size + (1 << 20) - 1) // (1 << 20)))   # >= 1 MiB per part
    chunk = (size + n - 1) // n
    t0 = time.time()

    def run_parallel(argvs):
        for attempt in range(3):
            tp = time.time()
            procs = [subprocess.Popen(a) for a in argvs]
            bad = [a for p_, a in zip(procs, argvs) if p_.wait() != 0]
            # V3 ledger: N concurrent scp processes = N launches, ONE wall
            _ctl_note("scp-chunk", time.time() - tp, calls=len(argvs))
            if not bad:
                return
            print(f"warning: {len(bad)} of {len(argvs)} chunk transfers failed (attempt {attempt + 1}/3); retrying those",
                  file=sys.stderr, flush=True)
            argvs = bad
            time.sleep(3 * (2 ** attempt))
        sys.exit(f"fatal: chunked transfer failed 3 times: {argvs[0]}")

    if direction == "put":
        parts = []
        with open(src, "rb") as f:
            for i in range(n):
                part = f"{src}.part{i:02d}"
                with open(part, "wb") as pf:
                    pf.write(f.read(chunk))
                parts.append(part)
        run_parallel([scp_base + [p_, f"{host}:{dst}.part{i:02d}"] for i, p_ in enumerate(parts)])
        for p_ in parts:
            try: os.remove(p_)
            except OSError: pass
        qd = shlex.quote(dst)
        r = _rsh(args, f"cat {' '.join(shlex.quote(f'{dst}.part{i:02d}') for i in range(n))} > {qd} && "
                       f"rm -f {qd}.part?? && stat -c %s {qd}", check=True)
        got = r.stdout.strip().split()[-1] if r.stdout.strip() else "?"
        if str(size) != got:
            sys.exit(f"fatal: chunked put of {src} reassembled to {got} bytes, expected {size}")
    else:
        qs = shlex.quote(src)
        _rsh(args, f"split -b {chunk} -d -a 2 {qs} {qs}.part", check=True)
        run_parallel([scp_base + [f"{host}:{src}.part{i:02d}", f"{dst}.part{i:02d}"] for i in range(n)])
        with open(dst, "wb") as out:
            for i in range(n):
                part = f"{dst}.part{i:02d}"
                with open(part, "rb") as pf:
                    shutil.copyfileobj(pf, out, 1 << 22)
                os.remove(part)
        _rsh(args, f"rm -f {qs}.part??")
        got = os.path.getsize(dst)
        if got != size:
            sys.exit(f"fatal: chunked get of {src} reassembled to {got} bytes, expected {size}")
    print(f"  [rcp] {direction} {os.path.basename(src)} {size / 1e6:.1f} MB in {time.time() - t0:.1f} s "
          f"({n} streams, {size / 1e6 / max(time.time() - t0, 1e-3):.2f} MB/s)", file=sys.stderr, flush=True)


# ------------------------------------------------ V3: one-round-trip wait mode
# S3.7 V3 (2026-09-05; S3.6 N2 / T12 §2.9 / FINAL_TABLE §3). In `--wait-mode
# remote` the whole "ready -> wait -> reply metadata" of a tick is ONE ssh
# invocation running the script below on the server host: (i) the previous
# tick's `rm -f resp.<n-1>.*` (folded in; a final rm at session end), (ii)
# the step-0 reset marker and `touch req.N.ready`, (iii) a `sleep 0.2` wait
# loop until resp.N.done exists and req.N.ready is gone (F64 predicate) or
# resp.N.error appears or --tick-timeout elapses, with the liveness test
# `_server_lines` makes (argv[0] ends in gpu_real_model, `--serve <rdir>`)
# ported to ps+awk every ~10 s and a heartbeat on stderr every 60 s so the
# recorded terminal shows life, (iv) the reply files' sizes and the one-line
# resp.N.meta on stdout (so the get is ONE transfer, resp.N.0 only). Exit
# codes: 0 done, 3 resp.N.error, 4 tick timeout, 5 server gone, 2 marker
# write failed; 255 stays ssh's own transport failure.
REMOTE_WAIT_LIVE_EVERY = 10     # s between server-process checks in the script
REMOTE_WAIT_BEAT_EVERY = 60     # s between heartbeat lines on stderr
REMOTE_WAIT_SLEEP = "0.2"       # the script's poll sleep (GNU and BSD sleep take fractions)


def _remote_wait_script(rdir, n, step0, markers, prev_rm, tick_timeout):
    """POSIX sh text for the one-shot wait (see above). markers: "touch"
    (first attempt: write the markers unconditionally), "cond" (a retry after
    an ssh transport failure: write them ONLY if the request was never
    accepted -- req.N.meta present, req.N.ready absent, resp.N.done absent;
    once the server has removed ready, re-touching it would make the F64
    predicate unsatisfiable), or "none". prev_rm: request number whose
    resp.* files are removed first (None = nothing to remove)."""
    R = shlex.quote(os.path.normpath(rdir))
    lines = ["R=%s" % R, "N=%d" % int(n)]
    if prev_rm is not None:
        lines.append('rm -f "$R"/resp.%d.* 2>/dev/null' % int(prev_rm))
    mk = []
    if step0:
        mk.append('printf reset > "$R/req.$N.reset" || exit 2')
    mk.append('touch "$R/req.$N.ready" || exit 2')
    if markers == "touch":
        lines += mk
    elif markers == "cond":
        lines.append('if test -f "$R/req.$N.meta" && ! test -f "$R/req.$N.ready" '
                     '&& ! test -f "$R/resp.$N.done"; then')
        lines += ["  " + x for x in mk]
        lines.append("fi")
    # the awk is the shell port of _server_lines: a process whose argv[0]
    # ends in gpu_real_model and whose `--serve` argument is this dir (the
    # path as normalised by the client, with or without a trailing slash)
    live = ("ps -eo pid=,args= | awk -v d=\"$R\" 'BEGIN{f=0} $2 ~ /gpu_real_model(_x)?$/ "
            "{ for (i=3; i<NF; i++) if ($i==\"--serve\" && ($(i+1)==d || $(i+1)==d\"/\")) f=1 } "
            "END{exit f?0:1}'")
    # WAIT_BEGIN / WAIT_END are timestamped by the CLIENT as they arrive
    # (sub-ms, portable); the script's own WAIT_S is whole seconds (`date
    # +%s`: BSD date has no %N) and is printed for the record only.
    lines += [
        "t0=$(date +%s); tl=$t0; tb=$t0; rc=4",
        'echo "WAIT_BEGIN"',
        "while :; do",
        '  if test -f "$R/resp.$N.error"; then rc=3; break; fi',
        '  if test -f "$R/resp.$N.done" && ! test -f "$R/req.$N.ready"; then rc=0; break; fi',
        "  now=$(date +%s)",
        "  if [ $((now - t0)) -ge %d ]; then rc=4; break; fi" % int(tick_timeout),
        "  if [ $((now - tl)) -ge %d ]; then tl=$now" % REMOTE_WAIT_LIVE_EVERY,
        "    if ! %s; then rc=5; break; fi" % live,
        "  fi",
        "  if [ $((now - tb)) -ge %d ]; then tb=$now; "
        "echo \"  [remote-wait] resp.$N: $((now - t0)) s, server alive\" >&2; fi" % REMOTE_WAIT_BEAT_EVERY,
        "  sleep %s" % REMOTE_WAIT_SLEEP,
        "done",
        "t1=$(date +%s)",
        'echo "WAIT_END"',
        'echo "WAIT_S=$((t1 - t0))"',
        'echo "RC=$rc"',
        'if [ "$rc" = 0 ]; then',
        '  s0=$(stat -c %s "$R/resp.$N.0" 2>/dev/null || stat -f %z "$R/resp.$N.0")',
        '  sm=$(stat -c %s "$R/resp.$N.meta" 2>/dev/null || stat -f %z "$R/resp.$N.meta")',
        '  echo "SIZE0=$s0"; echo "SIZEMETA=$sm"',
        '  echo "META_BEGIN"; cat "$R/resp.$N.meta"; printf "\\nMETA_END\\n"',
        "fi",
        'exit "$rc"',
    ]
    return "\n".join(lines) + "\n"


def _remote_wait(args, rdir, n, step0, prev_rm, tick_timeout, tick):
    """Run the one-shot wait script; stderr streams to the terminal (the
    heartbeat), stdout is parsed. Returns (wait_seconds, size0, meta_bytes)
    where wait_seconds is the client-timestamped WAIT_BEGIN -> WAIT_END span.
    An ssh transport failure (255) is retried with the "cond" marker variant
    (see _remote_wait_script); every other nonzero exit is fatal with the
    code named."""
    markers = "touch"
    for attempt in range(5):
        script = _remote_wait_script(rdir, n, step0, markers, prev_rm, tick_timeout)
        if _remote_is_local(args):
            argv = ["sh", "-c", script]
        else:
            argv = shlex.split(args.remote_ssh) + [script]
        t0 = time.time()
        # stdout is read line by line so WAIT_BEGIN / WAIT_END can be
        # timestamped here (the wait excluded from ctlSec); stderr (the
        # heartbeat) is inherited. A hung transport is killed at
        # --tick-timeout + 300 s by the timer (the script's own deadline
        # is --tick-timeout).
        import threading
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=None)
        killer = threading.Timer(float(tick_timeout) + 300.0, proc.kill)
        killer.daemon = True
        killer.start()
        chunks, t_wb, t_we = [], None, None
        while True:
            ln = proc.stdout.readline()
            if not ln:
                break
            chunks.append(ln)
            if ln == b"WAIT_BEGIN\n":
                t_wb = time.time()
            elif ln == b"WAIT_END\n":
                t_we = time.time()
        rc_ = proc.wait()
        hung = not killer.is_alive()
        killer.cancel()
        wall = time.time() - t0
        out = b"".join(chunks)
        if hung:
            _ctl_note("rsh-wait", wall)
            sys.exit(f"fatal: tick {tick} (req {n}): the remote wait ssh did not return "
                     f"within --tick-timeout {int(tick_timeout)} s + 300 s (transport hang)")
        r = subprocess.CompletedProcess(argv, rc_, out, None)
        fields = {}
        for ln in out.decode("utf-8", errors="replace").splitlines():
            if "=" in ln and ln.split("=", 1)[0] in ("WAIT_S", "RC", "SIZE0", "SIZEMETA"):
                k, v = ln.split("=", 1)
                fields[k] = v.strip()
        if t_wb is not None and t_we is not None and t_we >= t_wb:
            wait_s = t_we - t_wb
        else:
            try:
                wait_s = float(fields.get("WAIT_S", "0") or 0)
            except ValueError:
                wait_s = 0.0
        _ctl_note("rsh-wait", wall, wait_sec=min(wait_s, wall))
        if r.returncode == 255 and not _remote_is_local(args):
            print(f"warning: ssh transport failure during the remote wait (attempt "
                  f"{attempt + 1}/5); retrying with conditional markers", file=sys.stderr, flush=True)
            markers = "cond"
            time.sleep(2 * (2 ** attempt))
            continue
        if r.returncode == 3:
            sys.exit(f"fatal: server reported error for req {n} (tick {tick}) [remote-wait rc 3]")
        if r.returncode == 4:
            sys.exit(f"fatal: tick {tick} (req {n}) exceeded --tick-timeout "
                     f"{int(tick_timeout)} s; server alive but silent [remote-wait rc 4]")
        if r.returncode == 5:
            sys.exit(f"fatal: the server watching {rdir} is gone while serving "
                     f"req {n} (tick {tick}); no resp.{n}.error was written [remote-wait rc 5]")
        if r.returncode != 0:
            sys.exit(f"fatal: remote wait script exited {r.returncode} for req {n} "
                     f"(tick {tick}); stdout:\n{out.decode('utf-8', 'replace')[:800]}")
        b0 = out.find(b"META_BEGIN\n")
        b1 = out.find(b"\nMETA_END", b0 + 11 if b0 >= 0 else 0)
        if b0 < 0 or b1 < 0:
            sys.exit(f"fatal: remote wait returned no resp.{n}.meta block (tick {tick}):\n"
                     f"{out.decode('utf-8', 'replace')[:800]}")
        meta = out[b0 + 11:b1]
        try:
            size0 = int(fields.get("SIZE0", "-1"))
        except ValueError:
            size0 = -1
        return wait_s, size0, meta
    sys.exit(f"fatal: remote wait failed 5 times on ssh transport (tick {tick}, req {n})")


def mac_session_finish(args):
    """V3: in remote wait mode the `rm -f resp.N.*` of a tick is folded into
    the NEXT tick's ssh; the last tick's files are removed here, once, at
    session end (no retry: a dead transport must not stall the exit)."""
    n = getattr(args, "_v3_pending_rm", None)
    if n is None:
        return
    args._v3_pending_rm = None
    rdir = args.remote_reqdir
    _rsh(args, f"rm -f {shlex.quote(rdir)}/resp.{int(n)}.*", retry=False)


def _server_lines(args, rdir):
    """All `gpu_real_model --serve` processes on the host, and the subset whose
    --serve argument is exactly rdir (path-normalised: a server on
    `<rdir>2` or `<rdir>.bak` no longer passes as ours -- review 2026-09-03).
    Returns (all_lines, mine) where each entry is (pid, argv_string)."""
    ps = _rsh(args, "ps -eo pid=,args= | grep -F -- '--serve' "
                    "| grep -v -e grep -e 'ps -eo' | head -16")
    want = os.path.normpath(rdir)
    lines, mine = [], []
    for ln in ps.stdout.splitlines():
        m = re.match(r"\s*(\d+)\s+(\S+)(.*)$", ln)
        if not (m and m.group(2).endswith(("gpu_real_model", "gpu_real_model_x"))):   # 2026-09-18: the experimental copy serves the take
            continue
        pid, argv = int(m.group(1)), (m.group(2) + m.group(3)).strip()
        lines.append((pid, argv))
        try:
            toks = shlex.split(argv)
        except ValueError:
            toks = argv.split()
        served = [toks[i + 1] for i, t in enumerate(toks[:-1]) if t == "--serve"]
        if any(os.path.normpath(d) == want for d in served):
            mine.append((pid, argv))
    return lines, mine


def _server_status(args, rdir):
    """Parse REQDIR/serve.status (written by the server every time it waits):
    {"nextReq":N,"served":k,"pid":p,...}. Required -- an older server without
    it cannot be used safely (its counter is invisible)."""
    q = _rsh(args, f"cat {shlex.quote(rdir)}/serve.status 2>/dev/null")
    txt = q.stdout.strip()
    if not txt:
        sys.exit(f"fatal: {rdir}/serve.status is missing. The server must publish "
                 f"its request counter (gpu_real_model built after 2026-09-03, or the "
                 f"mock); without it a second session against a live server waits "
                 f"forever for resp.0.")
    try:
        return json.loads(txt.splitlines()[-1])
    except Exception:
        sys.exit(f"fatal: unparsable serve.status: {txt[:200]}")


def mac_tick_roundtrip(args, state, paths, tick, step):
    """MAC-TOOL mode: encrypt locally with mac_fhe_client, copy the request to
    the REMOTE serve dir (scp, or a plain copy under --remote-ssh local), poll
    for resp.done, copy the response back, decrypt locally, head-matvec in
    numpy. Returns (logits_rows[NL,V], t_enc, t_srv, t_dec). The secret key
    never leaves this machine."""
    import subprocess as sp
    import time as _t
    NL = state["lanes"]
    K = args.mac_keys
    rdir = args.remote_reqdir
    dpad = int(state.get("dpad", 1024))
    # Request numbers on the wire are server-relative: the preflight read the
    # server's nextReq and this session's step 0 maps onto it.
    base = int(getattr(args, "req_base", 0) or 0)
    n = base + step
    loc = os.path.join(args.out, f"req.{n}")
    t0 = _t.time()
    enc = [args.mac_tool, "enc", "--ctx", f"{K}/cryptocontext.bin",
           "--pub", f"{K}/public.key", "--rows", paths["row_tail"],
           "--lanes", str(NL), "--tokens", "1",
           "--d", str(state["d"]), "--dpad", str(dpad), "--out", loc]
    r = sp.run(enc, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"fatal: mac enc failed: {r.stdout} {r.stderr}")
    t_enc = _t.time() - t0
    # V3 (2026-09-05): --wait-mode. `poll` is the code below, untouched;
    # `remote` is one ssh invocation for markers + wait + reply metadata.
    # getattr: spec_decode/fidelity_tick.py builds its own args object.
    wait_mode = getattr(args, "wait_mode", "poll") or "poll"
    _ctl_tick_begin(tick, wait_mode)
    tick_timeout = float(getattr(args, "tick_timeout", 1800) or 1800)
    t0 = _t.time()
    for suf in ("0", "meta"):
        _rcp(args, f"{loc}.{suf}", f"{rdir}/req.{n}.{suf}", "put")
    if wait_mode == "remote":
        prev_rm = getattr(args, "_v3_pending_rm", None)
        print(f"  [tick {tick}] remote wait: one ssh (markers + wait + meta), "
              f"timeout {int(tick_timeout)} s" + (f", rm resp.{prev_rm}.*" if prev_rm is not None else ""),
              file=sys.stderr, flush=True)
        wait_s, size0, meta = _remote_wait(args, rdir, n, step == 0, prev_rm,
                                           tick_timeout, tick)
        args._v3_pending_rm = None
        with open(f"{loc}.resp.meta", "wb") as mf:
            mf.write(meta)
        _rcp(args, f"{rdir}/resp.{n}.0", f"{loc}.resp.0", "get", known_size=size0)
        if size0 >= 0 and os.path.getsize(f"{loc}.resp.0") != size0:
            sys.exit(f"fatal: resp.{n}.0 fetched as {os.path.getsize(f'{loc}.resp.0')} bytes, "
                     f"the remote wait reported {size0} (tick {tick})")
        args._v3_pending_rm = n        # removed by the next tick's ssh / session end
        t_srv = _t.time() - t0
        _ctl_tick_end()
        return _mac_tick_decrypt(args, state, paths, tick, n, loc, t_enc, t_srv)
    mark = f"touch {shlex.quote(rdir)}/req.{n}.ready"
    if step == 0:
        mark = (f"printf reset > {shlex.quote(rdir)}/req.{n}.reset && " + mark)
    _rsh(args, mark, check=True)
    polls = 0
    t_poll0 = _t.time()
    t_beat = t_poll0
    while True:
        # F64: `resp.N.done` ALONE does not prove the server served THIS
        # request -- a leftover resp.N.* from a crashed session satisfies the
        # poll instantly, and since the server's own request counter persists
        # across client invocations while the client always numbers from 0,
        # a whole session can be replayed from stale files while the pod sits
        # idle. AUDIT FIX 2026-09-02: the server removes req.N.ready in its
        # FINISH tail, *after* touching resp.N.done (gpu_real_model.cu: done
        # at the tail's touchFile, ready removed a few lines later) -- NOT on
        # accept, as this comment used to claim. So "done AND ready" is a
        # normal in-flight state lasting a few ms, not a stale replay, and
        # treating it as fatal aborted healthy runs. Stale replay is already
        # excluded by mac_session_preflight's clean-dir check at session
        # start; within a session our own step N cannot pre-exist.
        q = _rsh(args, f"test -f {shlex.quote(rdir)}/resp.{n}.done && echo D; "
                       f"test -f {shlex.quote(rdir)}/resp.{n}.error && echo E; "
                       f"test -f {shlex.quote(rdir)}/req.{n}.ready && echo R")
        if "E" in q.stdout:
            sys.exit(f"fatal: server reported error for req {n} (tick {tick})")
        if "D" in q.stdout and "R" not in q.stdout:
            break
        # "D" with "R": the server is between touching done and removing
        # ready -- keep polling; it clears within milliseconds.
        polls += 1
        now = _t.time()
        # REVIEW 2026-09-03: the poll used to spin forever when the server died
        # without writing resp.N.error (every `return 2` in the serve loop, or
        # an abort). Liveness every 10 polls, a hard per-tick deadline, and a
        # heartbeat so a recorded terminal shows the wait is alive.
        if polls % 10 == 0:
            _, mine = _server_lines(args, rdir)
            if not mine:
                sys.exit(f"fatal: the server watching {rdir} is gone while serving "
                         f"req {n} (tick {tick}); no resp.{n}.error was written")
        if now - t_beat >= 60:
            print(f"  [tick {tick}] waiting for resp.{n}: {int(now - t_poll0)} s",
                  file=sys.stderr, flush=True)
            t_beat = now
        if now - t_poll0 > tick_timeout:
            sys.exit(f"fatal: tick {tick} (req {n}) exceeded --tick-timeout "
                     f"{int(tick_timeout)} s; server alive but silent")
        _t.sleep(args.poll_interval)
    for suf in ("0", "meta"):
        _rcp(args, f"{rdir}/resp.{n}.{suf}", f"{loc}.resp.{suf}", "get")
    _rsh(args, f"rm -f {shlex.quote(rdir)}/resp.{n}.*", check=True)
    t_srv = _t.time() - t0
    _ctl_tick_end()
    return _mac_tick_decrypt(args, state, paths, tick, n, loc, t_enc, t_srv)


def _mac_tick_decrypt(args, state, paths, tick, n, loc, t_enc, t_srv):
    """The decrypt + head-matvec + retention tail of mac_tick_roundtrip,
    shared by both wait modes (V3, 2026-09-05; the body is the pre-V3 code,
    moved verbatim). Returns (logits_rows[NL,V], t_enc, t_srv, t_dec)."""
    import subprocess as sp
    import time as _t
    NL = state["lanes"]
    K = args.mac_keys
    dpad = int(state.get("dpad", 1024))
    t0 = _t.time()
    # F63: the F62 harness fix stamps `lanes` into the RESPONSE meta, but that
    # only protects the pod-side --dec-out path. The mac decryptor took lanes,
    # count and dpad from its own CLI and never opened the meta at all, so the
    # exact bug F62 fixed was still live on the fully-secure path. Pass the
    # meta and let mac_fhe_client hard-fail on any mismatch.
    hid = f"{loc}.hidden.f64"
    dec = [args.mac_tool, "dec", "--ctx", f"{K}/cryptocontext.bin",
           "--sec", f"{K}/secret.key", "--in", f"{loc}.resp",
           "--count", "1", "--lanes", str(NL),
           "--d", str(state["d"]), "--dpad", str(dpad),
           "--require-meta", f"{loc}.resp.meta", "--out", hid]
    r = sp.run(dec, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"fatal: mac dec failed: {r.stdout} {r.stderr}")
    head, V, d = head_matrix(state["bundle_dir"], state["tag"])
    hv = np.fromfile(hid, dtype="<f8")
    if hv.size != NL * d:
        sys.exit(f"fatal: decrypted {hv.size} doubles for tick {tick}, expected "
                 f"{NL} lanes x {d}")
    hv = hv.reshape(NL, d)
    # DRY-RUN FIX 2026-09-02: a non-finite hidden row (corrupt or wrong-layout
    # decrypt) used to flow straight into np.argmax, where NaN WINS -- every
    # such lane would silently "predict" its first NaN column and the run
    # would complete looking healthy. Refuse before the matvec.
    if not np.isfinite(hv).all():
        bad = [r for r in range(NL) if not np.isfinite(hv[r]).all()]
        sys.exit(f"fatal: non-finite decrypted hidden rows at tick {tick} for "
                 f"lanes {bad[:8]} -- corrupt reply or wrong layout (F62/F63)")
    # numpy 2.0.2 on Apple Accelerate raises SPURIOUS divide/overflow/invalid
    # FP flags inside matmul on perfectly finite inputs (reproduced 2026-09-02
    # on random data; results finite). The explicit finiteness checks around
    # this call are the real guard, so the flags are silenced here only.
    with np.errstate(all="ignore"):
        logits = hv @ np.asarray(head, dtype=np.float64).T  # (NL, V)
    if not np.isfinite(logits).all():
        sys.exit(f"fatal: non-finite logits at tick {tick} after the head "
                 f"matvec (hidden was finite; head or matmul is broken)")
    t_dec = _t.time() - t0
    keep = getattr(args, "keep_cts", None)
    if keep:
        # DECISION 2026-09-04: every ciphertext that left this Mac, and every
        # one that came back, is retained byte-for-byte with the plaintext it
        # encrypts (input rows) / decrypts to (hidden rows), so an independent
        # reviewer holding secret.key can re-decrypt both sides and check them
        # against tokens.json. Layout: <keep>/tick_TTT_reqN/{req.N.0,req.N.meta,
        # resp.N.0,resp.N.meta,input_rows.f64,hidden.f64} + one JSON line per
        # file in <keep>/MANIFEST.jsonl (sha256, bytes, tick, req, lanes).
        import hashlib, shutil
        d = os.path.join(keep, f"tick_{tick:03d}_req{n}")
        os.makedirs(d, exist_ok=True)
        moves = [(f"{loc}.0", f"req.{n}.0"), (f"{loc}.meta", f"req.{n}.meta"),
                 (f"{loc}.resp.0", f"resp.{n}.0"), (f"{loc}.resp.meta", f"resp.{n}.meta"),
                 (hid, "hidden.f64")]
        shutil.copy2(paths["row_tail"], os.path.join(d, "input_rows.f64"))
        for src, name in moves:
            try: os.replace(src, os.path.join(d, name))
            except OSError as e: sys.exit(f"fatal: --keep-cts could not retain {src}: {e}")
        with open(os.path.join(keep, "MANIFEST.jsonl"), "a") as mf:
            for name in [m[1] for m in moves] + ["input_rows.f64"]:
                fp = os.path.join(d, name)
                h = hashlib.sha256()
                with open(fp, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 22), b""):
                        h.update(chunk)
                mf.write(json.dumps({"tick": tick, "req": n, "lanes": NL, "d": int(state["d"]),
                                     "dpad": dpad, "file": os.path.relpath(fp, keep),
                                     "bytes": os.path.getsize(fp), "sha256": h.hexdigest(),
                                     "utc": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())}) + "\n")
    else:
        for f in (f"{loc}.0", f"{loc}.meta", f"{loc}.resp.0", f"{loc}.resp.meta", hid):
            try: os.remove(f)
            except OSError: pass
    return logits, t_enc, t_srv, t_dec


def mac_session_preflight(args, state):
    """F64/F65 preflight for --mac-tool mode, where the client does NOT own the
    server process and therefore cannot assume anything about it.

    Two things are unverifiable from this side unless we go and look:
      * whether the serve dir still holds responses from a previous session
        (which would be consumed as if fresh), and
      * whether the server was started --stateful at all. A stateless server
        answers every T=1 tick from a ZERO carry, so each token is conditioned
        only on its immediate predecessor. The text stays fluent; nothing in
        any log says a word about it.
    """
    rdir = args.remote_reqdir
    q = _rsh(args, f"ls {shlex.quote(rdir)}/req.* {shlex.quote(rdir)}/resp.* "
                   f"2>/dev/null | head -40")
    stale = [ln for ln in q.stdout.splitlines() if ln.strip()]
    if stale:
        sys.exit("fatal: %d stale request/response files in %s (%s ...). A "
                 "leftover resp.N.done is indistinguishable from a fresh reply "
                 "at the wire level, so this run could replay a previous "
                 "session's ciphertexts. Clear the dir AND restart the server "
                 "(its request counter persists across client invocations)."
                 % (len(stale), rdir, ", ".join(os.path.basename(s) for s in stale[:3])))
    # AUDIT FIX 2026-09-02. The old `pgrep -af 'gpu_real_model.*--serve'`
    # (i) matched the ssh login shell that RAN it, whose own command line
    # contains the pattern text, so "no server" was undetectable; and (ii)
    # its result was one concatenated string, so a STATELESS server on our
    # rdir passed whenever any OTHER --stateful gpu_real_model existed --
    # exactly the F65 corruption this check exists to stop. Now: the binary
    # must be the FIRST token (excludes any `sh -c` wrapper), one line per
    # process, and every check is made on the ONE line bound to our rdir.
    #
    # PORTABILITY (2026-09-02): `pgrep -af` is Linux-procps only -- on macOS
    # `pgrep -a` means "include ancestors" and `-f` alone prints pids -- so
    # the listing is now `ps -eo pid=,args=` (POSIX, identical output shape
    # on procps and BSD ps: "<pid> <argv0> <args...>"). The shell filters
    # keep only lines carrying --serve and drop the shell running this very
    # check (its argv contains 'grep' and 'ps -eo'); the first-token test
    # is done here, in Python, on each line's own argv[0].
    lines, mine = _server_lines(args, rdir)
    if not lines:
        sys.exit(f"fatal: no `gpu_real_model --serve` process found on the "
                 f"remote; start the server before the client.")
    if len(mine) != 1:
        sys.exit(f"fatal: expected exactly ONE server whose --serve is {rdir}, found "
                 f"{len(mine)} (of {len(lines)} serve processes). Two servers on "
                 f"one dir would race on req.N; zero means --remote-reqdir is wrong.")
    pid, argv = mine[0]     # every check below is on THIS server's own line
    if "--stateful" not in argv:
        sys.exit("fatal: the remote server was NOT started with --stateful:\n  "
                 + argv.splitlines()[0][:300] +
                 "\nThe tickwise engine sends ONE token per request and relies "
                 "on the server's in-memory carry for all prior context. "
                 "Against a stateless server every tick starts from a zero "
                 "carry — fluent output, no error, no context.")
    status = _server_status(args, rdir)
    args.req_base = int(status.get("nextReq", 0))
    rec = {"macPreflight": "ok", "serveDirClean": True, "serverStateful": True,
           "transport": "local" if _remote_is_local(args) else "ssh",
           "serverPid": pid, "serverArgv": argv.strip()[:300],
           "serverServeDir": rdir, "serverNextReq": args.req_base,
           "serverServed": status.get("served"),
           "mock": ("mock_fhe_server" in argv),
           "secretOnServerHost": ("mock_fhe_server" in argv) or _remote_is_local(args)}
    print(json.dumps(rec), flush=True)
    return rec


def generate_lanes_tickwise(args, state, paths, base, cbase, keys, rng,
                            transcript, server, srv_log, req_dir):
    """v2 multi-lane engine: EVERY request carries exactly one token per lane
    (T=1). Per-lane cursor decides prompt-feed vs self-feed; predictions are
    kept only once a lane is past its prompt and not retired. Retired lanes
    (word budget hit, or stop token) freeze: they re-feed their last token
    and their outputs are discarded — no garbage in any transcript."""
    import glob as _glob
    NL = state["lanes"]
    tok = get_tokenizer()
    prompts = state["ids_lanes"]
    gen = state["generated_lanes"]
    retired = state.setdefault("retired", [False] * NL)
    target = args.words_per_lane
    stop_ids = set()
    if args.stop_on_newline:
        for cand in ("\n", "\n\n"):
            ids = tok.encode(cand)
            if len(ids) == 1:
                stop_ids.add(ids[0])
    # F66: RESUME IS NOT IMPLEMENTED for the tickwise engine and cannot be
    # faked. `tick` would resume at len(transcript) while the server carry is
    # empty (a fresh --serve process, or wiped by the step==0 reset marker this
    # loop still sends), and there is no prefill-replay path: every request is
    # T=1. Each lane would continue from its last token as if that token began
    # the sequence -- fluent, wrong, and invisible.
    if transcript:
        sys.exit("fatal: %s already holds %d ticks. The tickwise lanes engine "
                 "cannot resume: the server carry does not survive the client, "
                 "and every request is T=1 so there is no prefill to replay. "
                 "Re-run `prepare` (it archives the old transcript) to start a "
                 "clean session." % (paths["transcript"], len(transcript)))
    # F70: the AUDITOR replays this rule offline. If it has to be told the
    # retirement config on its own command line it can disagree with what
    # actually ran and return a confident wrong verdict. Persist the config
    # with the session so there is exactly one copy of the truth.
    state["retire_cfg"] = {"words_per_lane": int(target or 0),
                           "stop_ids": sorted(int(s) for s in stop_ids),
                           "stop_on_newline": bool(args.stop_on_newline)}
    save_state(state, paths)
    t0_global = len(transcript)
    try:
        for step in range(args.steps):
            tick = len(transcript)
            inputs, phases = [], []
            done = 0
            for r in range(NL):
                pos = tick                        # global tick == position
                if retired[r]:
                    nxt = (gen[r] or prompts[r])[-1]
                    phases.append("retired"); done += 1
                elif pos < len(prompts[r]):
                    nxt = prompts[r][pos]
                    phases.append("prompt")
                else:
                    nxt = gen[r][-1] if gen[r] else prompts[r][-1]
                    phases.append("gen")
                inputs.append(int(nxt))
            if done == NL:
                print("== all lanes retired; stopping ==")
                break
            write_tick_rows(state, paths, inputs)
            print(f"== tick {tick} ({phases.count('prompt')} prompting, "
                  f"{phases.count('gen')} generating, {done} retired) ==")
            if args.mac_tool:
                lg, t_enc, t_srv, t_dec = mac_tick_roundtrip(
                    args, state, paths, tick, step)
                picked = apply_lane_predictions(
                    state, args, rng, lg, tick, prompts, gen, retired,
                    target, stop_ids)
                # DRY-RUN FIX 2026-09-02: the mac branch never mirrored lane 0
                # into the v1 fields, so tokens.json carried "generated": []
                # beside a populated generated_lanes -- a stale field of the
                # F62 class. The local branch below has always done this.
                state["generated_lanes"] = gen
                state["ids"] = prompts[0]; state["generated"] = gen[0]
                save_state(state, paths)
                tok2 = get_tokenizer()
                for r in range(min(NL, 4)):
                    tx = tok2.decode((prompts[r] + gen[r])[-14:])
                    print(f"  lane {r}: ...{json.dumps(tx)}")
                ent = {"tick": tick, "phases": phases,
                       "inputs": inputs, "picked": picked,
                       "encSec": t_enc,
                       "serveSec": t_srv, "decSec": t_dec}
                # V3 (2026-09-05): ADDITIVE control-traffic fields; the three
                # timing fields above keep their exact meaning (serveSec still
                # spans put -> get incl. the wait, N2).
                ctl = _ctl_last()
                if ctl and ctl.get("tick") == tick:
                    ent["waitMode"] = ctl["waitMode"]
                    ent["ctlCalls"] = ctl["ctlCalls"]
                    ent["ctlSec"] = ctl["ctlSec"]
                transcript.append(ent)
                with open(paths["transcript"], "w") as f:
                    json.dump(transcript, f, indent=1)
                continue
            enc_cmd = cbase + keys + ["--enc-in", paths["row_tail"],
                                      "--tokens", "1",
                                      "--client-lanes", str(NL)]
            t_enc = run_step(enc_cmd,
                             os.path.join(paths["logs"], f"tick{tick}.enc.jsonl"),
                             ("clientEnc", "fatal"))
            res, t_srv = serve_roundtrip(server, paths, req_dir, step,
                                         args.poll_interval,
                                         paths["row_tail"],
                                         reset_state=(step == 0))
            dec_cmd = cbase + keys + ["--allow-secret", "--dec-out", res,
                                      "--tokens", "1"]
            t_dec = run_step(dec_cmd,
                             os.path.join(paths["logs"], f"tick{tick}.dec.jsonl"),
                             ("clientDecDone", "fatal"))
            V = state["vocab"]
            raw = np.fromfile(res + ".logits.f64", dtype="<f8").reshape(-1, V)
            picked = apply_lane_predictions(state, args, rng, raw[-NL:],
                                            tick, prompts, gen, retired,
                                            target, stop_ids)
            state["generated_lanes"] = gen
            state["ids"] = prompts[0]; state["generated"] = gen[0]
            save_state(state, paths)
            for f in _glob.glob(res + ".*"):
                os.remove(f)
            live = [tok.decode((prompts[r] + gen[r])[-14:]) for r in range(min(NL, 4))]
            for r, tx in enumerate(live):
                print(f"  lane {r}: ...{json.dumps(tx)}")
            transcript.append({"tick": tick, "phases": phases,
                               "inputs": inputs, "picked": picked,
                               "encSec": t_enc, "serveSec": t_srv,
                               "decSec": t_dec})
            with open(paths["transcript"], "w") as f:
                json.dump(transcript, f, indent=1)
    finally:
        if server:
            stop_server(server, srv_log, req_dir)
        if args.mac_tool:
            mac_session_finish(args)     # V3: the deferred rm of the last resp.N.*
    full = {str(r): tok.decode(prompts[r] + gen[r]) for r in range(NL)}
    if _STUB_TOKENIZER:
        # F62 class: the text file is what gets read; stamp it too.
        full["_stubTokenizer"] = ("STUB byte-level tokenizer — protocol dry "
                                  "run, NOT a demo record")
    with open(os.path.join(args.out, "lanes_text.json"), "w") as f:
        json.dump(full, f, indent=1, ensure_ascii=False)
    print(json.dumps({"lanesDone": True, "ticks": len(transcript) - t0_global,
                      "generatedPerLane": [len(g) for g in gen],
                      "retired": retired,
                      "stubTokenizer": bool(_STUB_TOKENIZER)}))
    return None


def cmd_sample(args):
    state, paths = load_state(args.out)
    check_stub_consistency(state, args)
    set_stub_tokenizer(args.stub_tokenizer)
    rng = np.random.default_rng(args.seed)
    do_sample(state, paths, args.logits, args, rng)


def run_step(cmd, log_path, echo_keys):
    """Run one harness subprocess, tee its stdout to log_path, echo the
    important JSON lines, and fail hard on a nonzero exit."""
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            lf.write(line)
            if any(k in line for k in echo_keys):
                print("   ", line.rstrip())
        proc.wait()
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = open(log_path).read().splitlines()[-15:]
        sys.exit("fatal: harness step failed (exit %d): %s\n--- log tail ---\n%s"
                 % (proc.returncode, " ".join(map(shlex.quote, cmd)),
                    "\n".join(tail)))
    return dt


def start_server(args, base, keys, paths, req_dir):
    """--serve-mode: launch ONE persistent server (context + rotation keys +
    bootstrap precompute stay resident across steps). Returns (proc, logfile).
    """
    os.makedirs(req_dir, exist_ok=True)
    stop = os.path.join(req_dir, "stop")
    if os.path.exists(stop):
        os.remove(stop)                      # stale stop from a previous run
    stale = glob.glob(os.path.join(req_dir, "req.*")) + \
        glob.glob(os.path.join(req_dir, "resp.*"))
    for f in stale:
        os.remove(f)                         # server numbering restarts at 0
    srv_cmd = base + keys + ["--serve", req_dir,
                             "--serve-timeout", str(args.serve_timeout)]
    if getattr(args, "stateful", False):
        # In-memory carry across requests: the state never leaves the process,
        # so a resumed step costs no serialization and no transfer. This is
        # why the client holds only token ids, never the state blob — one
        # carried ciphertext is ~1 MB per RNS limb, so shipping 2 per layer
        # per step would dwarf the prefix it replaces.
        srv_cmd += ["--stateful"]
    print("  [server-start]", " ".join(map(shlex.quote, srv_cmd)))
    log = open(os.path.join(paths["logs"], "server.jsonl"), "w")
    proc = subprocess.Popen(srv_cmd, stdout=log, stderr=subprocess.STDOUT,
                            text=True)
    return proc, log


def stop_server(proc, log, req_dir):
    try:
        with open(os.path.join(req_dir, "stop"), "w") as f:
            f.write("stop\n")
        try:
            # the exit persist is a journal append now (seconds), but a base
            # save of a fresh store can be minutes: give it room
            proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            print("warning: server did not exit within 900 s of `stop`; terminating",
                  file=sys.stderr, flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        log.close()


def serve_roundtrip(proc, paths, req_dir, reqn, poll_interval, enc_src=None,
                    reset_state=False):
    """Move the freshly encrypted <enc_src>.ct.* into REQDIR as req.<reqn>.*,
    touch the ready marker, and wait for resp.<reqn>.done (or fail on
    resp.<reqn>.error / server death). Returns (resp_prefix, seconds).

    enc_src defaults to rows.f64 (the full prefix). Under --stateful a resumed
    step encrypts row_tail.f64 instead, so the source must follow the encrypt
    step or the server would be handed the whole prefix again.
    """
    src = enc_src or paths["rows"]
    if reset_state:
        # SESSION BOUNDARY. A persistent --stateful server would otherwise
        # carry the previous conversation's state into this one -- silently,
        # since the circuit stays well-formed. Written BEFORE .ready so the
        # server sees it in the same request.
        with open(os.path.join(req_dir, f"req.{reqn}.reset"), "w") as f:
            f.write("reset\n")
    with open(src + ".ct.meta") as f:
        count = json.load(f)["count"]        # the meta is plain one-line JSON
    req = os.path.join(req_dir, f"req.{reqn}")
    resp = os.path.join(req_dir, f"resp.{reqn}")
    for i in range(count):                   # files first, ready marker LAST
        os.replace(src + f".ct.{i}", f"{req}.{i}")
    os.replace(src + ".ct.meta", f"{req}.meta")
    with open(f"{req}.ready", "w") as f:
        f.write("ready\n")
    t0 = time.time()
    srv_log = os.path.join(paths["logs"], "server.jsonl")
    while True:
        if os.path.exists(f"{resp}.done"):
            # F64 (same predicate the mac path uses): the server deletes
            # req.N.ready when it ACCEPTS the request, so a .done sitting
            # beside a still-pending .ready is a response that predates us.
            # The local path sweeps the dir at start_server, so this should be
            # unreachable here -- which is exactly why it is worth asserting.
            if os.path.exists(f"{req}.ready"):
                sys.exit(f"fatal: resp.{reqn}.done exists while req.{reqn}.ready "
                         f"is still pending — that reply was not produced for "
                         f"this request ({req_dir})")
            return resp, time.time() - t0
        if os.path.exists(f"{resp}.error"):
            sys.exit(f"fatal: server reported an error for req {reqn} "
                     f"(see {srv_log})")
        if proc.poll() is not None:
            tail = "\n".join(open(srv_log).read().splitlines()[-15:])
            sys.exit("fatal: server exited (code %s) while serving req %d\n"
                     "--- server log tail ---\n%s"
                     % (proc.returncode, reqn, tail))
        time.sleep(poll_interval)


def cmd_generate(args):
    state, paths = load_state(args.out)
    # F62 class: stub stamp and stub flag must agree before anything runs.
    check_stub_consistency(state, args)
    set_stub_tokenizer(args.stub_tokenizer)
    if args.mac_tool and not (args.mac_keys and args.remote_ssh
                              and args.remote_reqdir):
        sys.exit("fatal: --mac-tool needs --mac-keys, --remote-ssh (an ssh "
                 "command, or the word `local`) and --remote-reqdir")
    if args.stateful:
        # Fail here rather than after spawning a server. The harness refuses
        # too, but the reason is worth stating once, in full: scanState is live
        # ONLY in the packTokens==1 branch of the scan; the blocked scan carries
        # S[b] instead, so a carry taken from scanState would be silently wrong.
        toks = shlex.split(args.harness_cmd)
        pt = None
        for i, a in enumerate(toks):
            if a == "--pack-tokens" and i + 1 < len(toks):
                pt = toks[i + 1]
        if pt != "1":
            sys.exit("fatal: --stateful requires --pack-tokens 1 in --harness-cmd "
                     f"(found {pt!r}). scanState is live only in the packTokens==1 "
                     "scan branch; at packTokens>1 the blocked scan carries S[b] "
                     "and a carry taken from scanState would be silently wrong.")
    os.makedirs(paths["logs"], exist_ok=True)
    rng = np.random.default_rng(args.seed)
    base = shlex.split(args.harness_cmd)
    # --client-harness-cmd: run the CLIENT crypto steps (enc/dec, CPU-only,
    # never timed, never part of a server claim) on a different binary —
    # e.g. a newer build with --client-lanes while the SERVER stays on the
    # anchored binary every record names. Defaults to the server cmd.
    cbase = shlex.split(args.client_harness_cmd) if args.client_harness_cmd else base
    keys = ["--keys-dir", args.keys_dir, "--keys-load"]

    transcript = []
    if os.path.exists(paths["transcript"]):
        with open(paths["transcript"]) as f:
            transcript = json.load(f)

    # F65: the tickwise lanes engine sends ONE token per lane per request and
    # depends ENTIRELY on the server's in-memory carry for prior context. A
    # server without --stateful answers each tick from a zero carry: the run
    # completes, the timings look normal, and every token is conditioned only
    # on its immediate predecessor. The old gate only asked for --serve-mode,
    # which buys a persistent PROCESS, not a persistent CARRY.
    if state.get("lanes", 1) > 1:
        if not args.serve_mode and not args.mac_tool:
            sys.exit("fatal: lanes v2 requires --serve-mode (persistent carry) "
                     "or --mac-tool (remote persistent server)")
        if args.serve_mode and not args.stateful:
            sys.exit("fatal: lanes v2 requires --stateful. Every request carries "
                     "T=1; without an in-memory carry the server recomputes each "
                     "tick from a ZERO state and every token is conditioned only "
                     "on the one before it — fluent output, no context, no error.")
        if args.mac_tool:
            state["serverPreflight"] = mac_session_preflight(args, state)
            save_state(state, state_paths(args.out))

    server = srv_log = None
    req_dir = args.req_dir or os.path.join(args.out, "serve")
    if args.serve_mode:
        server, srv_log = start_server(args, base, keys, paths, req_dir)

    try:
        if state.get("lanes", 1) > 1:
            return generate_lanes_tickwise(args, state, paths, base, cbase,
                                           keys, rng, transcript, server,
                                           srv_log, req_dir)
        for step in range(args.steps):
            if False:
                pass
            else:
                T_all = len(state["ids"]) + len(state["generated"])
            # STATEFUL DECODE. Step 0 is the PREFILL: the server has no carry
            # yet, so it must see the whole prompt. Every later step sends ONE
            # token and the server resumes its per-layer carry. That turns the
            # N(N+1)/2 replayed scan steps of the stateless path into N.
            if args.stateful and step > 0:
                T = 1
                enc_src = paths["row_tail"]
                write_row_tail(state, paths, 1)
                print(f"== step {step + 1}/{args.steps}  (T=1 of {T_all}; "
                      f"STATEFUL — server resumes its carry) ==")
            else:
                T = T_all
                enc_src = paths["rows"]
                mode = ("stateful prefill" if args.stateful
                        else "stateless prefill reruns the whole prefix")
                print(f"== step {step + 1}/{args.steps}  (T={T} tokens; {mode}) ==")

            # 1. client encrypt (CPU-only process; public key only)
            enc_cmd = cbase + keys + ["--enc-in", enc_src,
                                      "--tokens", str(T)]
            if state.get("lanes", 1) > 1:
                enc_cmd += ["--client-lanes", str(state["lanes"])]
            print("  [client-enc]", " ".join(map(shlex.quote, enc_cmd)))
            t_enc = run_step(enc_cmd,
                             os.path.join(paths["logs"], f"step{len(transcript)}.enc.jsonl"),
                             ("clientEnc", "fatal"))

            # 2. server forward (NO --allow-secret => no secret key)
            if args.serve_mode:
                # persistent server: hand the ciphertexts over via REQDIR.
                # step-2+ serverSec should drop to roughly pure compute --
                # context/keys/bootstrap precompute are already resident.
                res, t_srv = serve_roundtrip(server, paths, req_dir, step,
                                             args.poll_interval, enc_src,
                                             reset_state=(args.stateful and step == 0))
                print(f"    [serve] resp ready in {t_srv:.1f}s")
            else:
                res = os.path.join(args.out, f"step{len(transcript)}.result.ct")
                srv_cmd = base + keys + ["--ct-in", enc_src + ".ct",
                                         "--ct-out", res, "--tokens", str(T)]
                if args.stateful:
                    # spawn-per-step server: the carry crosses the process
                    # boundary on disk. Step 0 writes it; later steps resume.
                    st = os.path.join(args.out, "carry")
                    if step > 0:
                        srv_cmd += ["--state-in", st]
                    srv_cmd += ["--state-out", st]
                print("  [server-run]", " ".join(map(shlex.quote, srv_cmd)))
                t_srv = run_step(srv_cmd,
                                 os.path.join(paths["logs"], f"step{len(transcript)}.srv.jsonl"),
                                 ("serverOut", "summary", "error", "fatal"))

            # 3. client decrypt + unembed (CPU-only; secret key allowed)
            dec_cmd = cbase + keys + ["--allow-secret", "--dec-out", res,
                                      "--tokens", str(T)]
            print("  [client-dec]", " ".join(map(shlex.quote, dec_cmd)))
            t_dec = run_step(dec_cmd,
                             os.path.join(paths["logs"], f"step{len(transcript)}.dec.jsonl"),
                             ("clientDecDone", "fatal"))

            # 4. sample next token from the LAST position's logits
            nxt = do_sample(state, paths, res + ".logits.f64", args, rng,
                            expect_rows=T)
            if args.serve_mode:
                for f in glob.glob(res + ".*"):   # resp files are consumed
                    os.remove(f)

            transcript.append({
                "step": len(transcript), "T": T, "next_id": int(nxt),
                "tokensInPrefix": T_all, "stateful": bool(args.stateful),
                "serveMode": bool(args.serve_mode),
                "encSec": round(t_enc, 2), "serverSec": round(t_srv, 2),
                "decSec": round(t_dec, 2),
            })
            with open(paths["transcript"], "w") as f:
                json.dump(transcript, f, indent=1)
    finally:
        if server is not None:
            stop_server(server, srv_log, req_dir)

    tok = get_tokenizer()
    print(json.dumps({
        "generate": True, "steps_done": args.steps,
        "serve_mode": bool(args.serve_mode),
        "total_tokens": len(state["ids"]) + len(state["generated"]),
        "text": tok.decode(state["ids"] + state["generated"]),
        "transcript": paths["transcript"],
    }, indent=1))


# ------------------------------------------------------------------------ main
def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)   # per-tick lines reach `tee` live
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="tokenize + embed a prompt")
    p.add_argument("--prompt")
    p.add_argument("--prompt-file")
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--tag", default="agn")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=0,
                   help="truncate the prompt to this many tokens (0 = no cap)")
    p.add_argument("--lanes", type=int, default=1,
                   help="N conversations of ONE client packed lane-wise (needs --prompts-file)")
    p.add_argument("--prompts-file", default=None,
                   help="lanes mode: N prompts, one per line (v2: any lengths)")
    p.add_argument("--dpad", type=int, default=1024,
                   help="slot stride per replica — MUST equal the server's "
                        "--dpad. Stamped into tokens.json so the mac client "
                        "and the auditor read it instead of hardcoding it.")

    def stub_flag(q):
        q.add_argument("--stub-tokenizer", action="store_true",
                       help="PROTOCOL DRY RUNS ONLY: byte-level stub instead of "
                            "the GPT-NeoX tokenizer (no `transformers` needed). "
                            "prepare stamps stubTokenizer:true into tokens.json; "
                            "generate/sample refuse a mismatch between that "
                            "stamp and this flag, in both directions.")
    stub_flag(p)
    p.set_defaults(fn=cmd_prepare)

    def sampling_flags(q):
        q.add_argument("--temp", type=float, default=1.0)
        q.add_argument("--top-k", type=int, default=40)
        q.add_argument("--greedy", action="store_true")
        q.add_argument("--seed", type=int, default=0)
        stub_flag(q)

    q = sub.add_parser("sample", help="sample the next token from a logits file")
    q.add_argument("--out", required=True)
    q.add_argument("--logits", required=True,
                   help="raw float64 T x vocab file (harness --dec-out output)")
    sampling_flags(q)
    q.set_defaults(fn=cmd_sample)

    g = sub.add_parser("generate", help="run N enc->server->dec->sample steps")
    g.add_argument("--out", required=True)
    g.add_argument("--steps", type=int, default=4)
    g.add_argument("--keys-dir", required=True)
    g.add_argument("--mac-tool", default=None,
                   help="path to mac_fhe_client: enc/dec run LOCALLY (secret never leaves); needs --mac-keys/--remote-ssh/--remote-reqdir")
    g.add_argument("--mac-keys", default=None)
    g.add_argument("--remote-ssh", default=None,
                   help="full ssh command to the server host, e.g. 'ssh -i k -p "
                        "1868 root@host'; or exactly `local` when the server "
                        "runs on THIS machine (commands via sh -c, transfers "
                        "by plain copy) -- e.g. ml-eval/mock_fhe_server.py")
    g.add_argument("--remote-reqdir", default=None)
    g.add_argument("--keep-cts", default=None,
                   help="mac-tool: directory that RETAINS every request/response "
                        "ciphertext with its plaintext (input rows, decrypted hidden "
                        "rows) and a sha256 MANIFEST.jsonl, for independent "
                        "re-decryption with secret.key (decision 2026-09-04); "
                        "default: files are deleted after each tick")
    g.add_argument("--tick-timeout", type=float, default=1800,
                   help="mac-tool: seconds a single tick may wait for its reply "
                        "before the client aborts (server liveness is checked "
                        "every 10 polls regardless)")
    g.add_argument("--wait-mode", choices=("poll", "remote"), default="poll",
                   help="mac-tool (S3.7 V3, 2026-09-05): how a tick waits for its "
                        "reply. `poll` (DEFAULT = the pre-V3 code, untouched): "
                        "`touch ready` ssh, then one `test -f` ssh every "
                        "--poll-interval s, liveness every 10 polls, `stat` + 2 "
                        "gets + `rm` -- 8-9 round trips + poll overshoot per tick "
                        "(24-53 s on a ~4 s RTT path, S3.6 N2). `remote`: ONE ssh "
                        "per tick writes the markers, waits server-side (sleep 0.2, "
                        "liveness every 10 s, heartbeat every 60 s, --tick-timeout) "
                        "and returns the reply sizes + resp.N.meta; then ONE get of "
                        "resp.N.0; the rm is folded into the next tick's ssh. "
                        "Transcript adds waitMode/ctlCalls/ctlSec (additive).")
    g.add_argument("--words-per-lane", type=int, default=0,
                   help="lanes v2: retire a lane after this many generated words (0 = no cap)")
    g.add_argument("--stop-on-newline", action="store_true",
                   help="lanes v2: retire a lane when it emits a newline token")
    g.add_argument("--client-harness-cmd", default=None,
                   help="optional separate binary+flags for the CPU-only client "
                        "enc/dec steps (e.g. a newer build with --client-lanes "
                        "while the server stays on the anchored binary)")
    g.add_argument("--harness-cmd", required=True,
                   help="base harness command INCLUDING the binary path and all "
                        "circuit flags (--tag/--bundle-dir/--log-ring/"
                        "--pack-tokens/--interleave/...); the client appends "
                        "only the per-step mode flags")
    g.add_argument("--serve-mode", action="store_true",
                   help="start ONE persistent server (--serve) instead of "
                        "spawning a fresh server process per step; context, "
                        "rotation keys and bootstrap precompute stay resident, "
                        "so step-2+ server time drops to roughly pure compute. "
                        "Default off (per-step spawn) for A/B comparison.")
    g.add_argument("--stateful", action="store_true",
                   help="STATEFUL DECODE: step 0 prefills the prompt, every "
                        "later step sends ONE token and the server resumes its "
                        "per-layer carry (scan accumulator + shift-mix "
                        "boundary). Turns the stateless path's N(N+1)/2 "
                        "replayed scan steps into N. Requires --pack-tokens 1 "
                        "in --harness-cmd (the carry is only well-defined in "
                        "the packTokens==1 scan branch; the harness refuses "
                        "otherwise). With --serve-mode the carry stays in "
                        "server memory; without it, it crosses on disk via "
                        "--state-in/--state-out. Validated by "
                        "harness/state_carry_probe.cpp.")
    g.add_argument("--req-dir", default=None,
                   help="request directory for --serve-mode "
                        "(default <out>/serve)")
    g.add_argument("--serve-timeout", type=int, default=3600,
                   help="server idle-exit seconds (passed to --serve-timeout)")
    g.add_argument("--poll-interval", type=float, default=1.0,
                   help="seconds between resp.N.done polls in --serve-mode")
    sampling_flags(g)
    g.set_defaults(fn=cmd_generate)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
