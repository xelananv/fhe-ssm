#!/usr/bin/env python3
"""mock_fhe_server.py — a stand-in for `gpu_real_model --serve DIR --stateful`
that speaks the demo FILE PROTOCOL only. No GPU, no FIDESlib, no model: it
decrypts each request with the Mac client tool, applies a trivial mock
"model" with a per-lane carry, re-encrypts, and answers exactly the way the
real server does (gpu_real_model.cu, the FHE_SSM_DEMO_SER serve loop at
~3183-3300 and its FINISH tail at ~4240-4280). It exists so that
`fhe_client.py generate --mac-tool ... --remote-ssh local` can be exercised
end to end on a machine with no GPU and no pod: the preflight, the ragged
multi-lane bookkeeping, the request/response file ordering, the transcript
and the audit all run for real; only the arithmetic inside is fake.

IT IS A MOCK. It holds the SECRET key (the real server never does), its
output is not model text, and nothing it produces is a demo record. Sessions
driven against it should be prepared with `--stub-tokenizer` or otherwise
marked so they can never be mistaken for one (F62 class).

HOW TO LAUNCH — the preflight in fhe_client.py (mac_session_preflight)
accepts a server only if its process line has a FIRST token ending in
`gpu_real_model`, carries `--serve <dir>` and `--stateful`, and exactly one
such line mentions the request dir. A python script's argv[0] is `python3`,
so run the interpreter through a symlink (or copy) named gpu_real_model:

    ln -sf "$(python3 -c 'import sys,os;print(os.path.realpath(sys.executable))')" DIR/gpu_real_model
    DIR/gpu_real_model ml-eval/mock_fhe_server.py --serve DIR --stateful \
        --keys-dir KEYS --mac-tool hpc_gpu_port/mac_fhe_client --d 1024 \
        --model linear 1

macOS note: the Apple/CommandLineTools `python3` and its `bin/python3.9` are
launcher stubs that exec the real binary with argv[0] REWRITTEN (the ps line
then starts with .../Python.app/Contents/MacOS/Python and the preflight
rejects it). Symlink to that `Python.app/Contents/MacOS/Python` binary
instead — it keeps argv[0]. ml-eval/dryrun_mac_client.sh resolves this and
verifies the ps line before starting the client. Linux CPython keeps argv[0]
through a symlink as is.

PROTOCOL (request N, N counts from 0 per server process, client numbers its
own requests from 0 per session — the mock is started fresh per session):
  wait      poll DIR/req.N.ready (200 ms); DIR/stop => exit 0 "stop";
            --serve-timeout idle seconds => exit 0 "idleTimeout"
  request   read DIR/req.N.meta (ONE JSON line: count, packTokens,
            tokensReal, dpad, interleave, lanes, ringDim, level);
            if DIR/req.N.reset exists: zero every lane's carry, remove it,
            log {"serve":"stateReset"} (the client's tick-0 marker)
  decrypt   mac_fhe_client dec --in DIR/req.N --count <count> --lanes <lanes>
            --dpad <dpad> --require-meta DIR/req.N.meta  ->  hidden rows
            (count*lanes x d float64, tok-major lane-minor)
  model     per token t, per lane r:  x = M @ h  (M = identity, or the seeded
            orthogonal matrix of --model linear SEED so lanes visibly differ)
            hidden = 0.5*x + 0.5*carry[r];  carry[r] = hidden
            Without --stateful the carry is zeroed on EVERY request, which is
            what a stateless real server does to a T=1 tick (F65).
  encrypt   mac_fhe_client enc --rows <lane-major rows> --lanes <lanes>
            --tokens <count> --out DIR/resp.N   (writes resp.N.<t> and
            resp.N.meta; the meta is re-read and must stamp the same lanes
            and count — F62/F63)
  finish    touch DIR/resp.N.done; THEN remove req.N.<t>, req.N.meta and
            req.N.ready LAST. The client's poll predicate is
            "done AND NOT ready" (fhe_client.py, F64 audit fix), so this
            order is load-bearing.
  error     any failure: write DIR/resp.N.error, log {"serve":"error"} and
            exit 3. Never exit 0 after an error.
Log lines are JSON, one per event: serve = waiting | request | stateReset |
served (with done:true and per-phase ms) | error | exit.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


def log(**kw):
    print(json.dumps(kw), flush=True)


class MockError(Exception):
    pass


def run_tool(argv, what):
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        raise MockError("%s failed (exit %d): %s %s" % (what, r.returncode,
                                                         r.stdout.strip(),
                                                         r.stderr.strip()))
    return r


def build_model(spec, d):
    kind = spec[0]
    if kind == "identity":
        return None, {"model": "identity"}
    if kind == "linear":
        seed = int(spec[1]) if len(spec) > 1 else 0
        rng = np.random.default_rng(seed)
        q, r = np.linalg.qr(rng.standard_normal((d, d)))
        q = q * np.sign(np.diag(r))          # unique, orthogonal (det-free)
        # numpy on Apple Accelerate raises spurious FP flags inside matmul on
        # finite inputs (2026-09-02); the value is checked explicitly instead.
        with np.errstate(all="ignore"):
            ortho_err = float(np.abs(q.T @ q - np.eye(d)).max())
        if not np.isfinite(ortho_err) or ortho_err > 1e-8:
            raise SystemExit("fatal: QR produced a non-orthogonal matrix (err %g)" % ortho_err)
        return q, {"model": "linear", "seed": seed, "orthoErr": ortho_err}
    raise SystemExit("fatal: --model must be `identity` or `linear SEED`")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", required=True, help="request dir DIR")
    ap.add_argument("--stateful", action="store_true",
                    help="keep the per-lane carry across requests (the real "
                         "flag the preflight demands); without it every "
                         "request starts from a zero carry")
    ap.add_argument("--keys-dir", required=True,
                    help="dir with cryptocontext.bin, public.key, secret.key "
                         "(the mock decrypts: it is a mock)")
    ap.add_argument("--mac-tool", required=True, help="path to mac_fhe_client")
    ap.add_argument("--d", type=int, default=1024, help="hidden width d")
    ap.add_argument("--model", nargs="+", default=["identity"],
                    help="`identity` or `linear SEED`")
    ap.add_argument("--serve-timeout", type=int, default=3600,
                    help="exit after this many idle seconds")
    ap.add_argument("--poll", type=float, default=0.2)
    args = ap.parse_args()

    D = args.serve
    K = args.keys_dir
    ctx, pub, sec = (os.path.join(K, "cryptocontext.bin"),
                     os.path.join(K, "public.key"), os.path.join(K, "secret.key"))
    for p in (ctx, pub, sec):
        if not os.path.exists(p):
            log(serve="fatal", reason="missing key file", path=p)
            return 3
    if not os.path.isdir(D):
        log(serve="fatal", reason="serve dir missing", dir=D)
        return 3
    work = os.path.join(D, ".mock_work")
    os.makedirs(work, exist_ok=True)
    M, mdesc = build_model(args.model, args.d)
    log(serve="start", dir=D, stateful=bool(args.stateful), d=args.d,
        keysDir=K, macTool=args.mac_tool, argv0=sys.argv[0], pid=os.getpid(),
        mock=True, **mdesc)

    carry = None                      # (lanes, d) per-lane state
    n = 0
    served = 0
    t_idle = time.time()
    while True:
        req = os.path.join(D, "req.%d" % n)
        resp = os.path.join(D, "resp.%d" % n)
        log(serve="waiting", req=n, dir=D, timeoutSec=args.serve_timeout,
            served=served)
        # serve.status: the client adopts nextReq as its base (review 2026-09-03)
        with open(os.path.join(D, "serve.status.tmp"), "w") as f:
            f.write(json.dumps({"nextReq": n, "served": served, "pid": os.getpid(),
                                "stateful": bool(args.stateful), "mock": True}) + "\n")
        os.replace(os.path.join(D, "serve.status.tmp"), os.path.join(D, "serve.status"))
        got = None
        while True:
            if os.path.exists(os.path.join(D, "stop")):
                got = "stop"; break
            if os.path.exists(req + ".ready"):
                got = "req"; break
            if time.time() - t_idle > args.serve_timeout:
                got = "idle"; break
            time.sleep(args.poll)
        if got != "req":
            log(serve="exit", reason=("stop" if got == "stop" else "idleTimeout"),
                served=served)
            return 0
        t_req = time.time()
        try:
            with open(req + ".meta") as f:
                meta = json.loads(f.read().strip())
            count = int(meta["count"])
            lanes = int(meta["lanes"])
            dpad = int(meta["dpad"])
            tokens_real = int(meta.get("tokensReal", count))
            if count < 1 or lanes < 1:
                raise MockError("bad meta: count %d lanes %d" % (count, lanes))
            log(serve="request", req=n, tokens=tokens_real, count=count,
                lanes=lanes, dpad=dpad, ringDim=meta.get("ringDim"),
                meta=meta)
            if os.path.exists(req + ".reset"):
                carry = None
                os.remove(req + ".reset")
                log(serve="stateReset", req=n)
            if not args.stateful and carry is not None:
                carry = None
                log(serve="statelessDrop", req=n,
                    note="no --stateful: carry zeroed for this request")
            if carry is not None and carry.shape[0] != lanes:
                log(serve="warn", req=n, note="lane count changed %d -> %d; "
                    "carry re-initialised" % (carry.shape[0], lanes))
                carry = None
            if carry is None:
                carry = np.zeros((lanes, args.d), dtype=np.float64)

            # ---- decrypt (the mock holds the secret key) ------------------
            hid_in = os.path.join(work, "req.%d.hidden.f64" % n)
            t0 = time.time()
            run_tool([args.mac_tool, "dec", "--ctx", ctx, "--sec", sec,
                      "--in", req, "--count", str(count), "--lanes", str(lanes),
                      "--d", str(args.d), "--dpad", str(dpad),
                      "--require-meta", req + ".meta", "--out", hid_in], "dec")
            dec_ms = (time.time() - t0) * 1e3
            hv = np.fromfile(hid_in, dtype="<f8")
            if hv.size != count * lanes * args.d:
                raise MockError("decrypted %d doubles, expected %d x %d x %d"
                                % (hv.size, count, lanes, args.d))
            hv = hv.reshape(count, lanes, args.d)      # tok-major, lane-minor
            if not np.isfinite(hv).all():
                raise MockError("decrypted hidden rows are not finite (corrupt "
                                "request or wrong layout)")

            # ---- mock model + per-lane carry ------------------------------
            t0 = time.time()
            out = np.empty_like(hv)
            with np.errstate(all="ignore"):       # spurious Accelerate flags
                for t in range(count):
                    x = hv[t] @ M.T if M is not None else hv[t]
                    h = 0.5 * x + 0.5 * carry
                    carry = h.copy()
                    out[t] = h
            if not np.isfinite(out).all():
                raise MockError("mock model produced non-finite hidden rows")
            model_ms = (time.time() - t0) * 1e3
            in_rms = float(np.sqrt(np.mean(hv ** 2)))
            out_rms = float(np.sqrt(np.mean(out ** 2)))
            # how different the lanes are (0 => every lane identical)
            lane_spread = float(np.std(out[-1], axis=0).mean()) if lanes > 1 else 0.0

            # ---- re-encrypt: enc wants LANE-MAJOR (lanes, T, d) -------------
            rows_path = os.path.join(work, "resp.%d.rows.f64" % n)
            np.ascontiguousarray(out.transpose(1, 0, 2), dtype="<f8").tofile(rows_path)
            t0 = time.time()
            run_tool([args.mac_tool, "enc", "--ctx", ctx, "--pub", pub,
                      "--rows", rows_path, "--lanes", str(lanes),
                      "--tokens", str(count), "--d", str(args.d),
                      "--dpad", str(dpad), "--out", resp], "enc")
            enc_ms = (time.time() - t0) * 1e3
            with open(resp + ".meta") as f:
                rmeta = json.loads(f.read().strip())
            if int(rmeta.get("lanes", -1)) != lanes or int(rmeta.get("count", -1)) != count:
                raise MockError("response meta does not stamp lanes=%d count=%d: %s"
                                % (lanes, count, rmeta))
            for t in range(count):
                if not os.path.exists("%s.%d" % (resp, t)):
                    raise MockError("enc did not write %s.%d" % (resp, t))
            resp_bytes = sum(os.path.getsize("%s.%d" % (resp, t)) for t in range(count))

            # ---- FINISH: done first, request files last (client relies on it)
            with open(resp + ".done", "w") as f:
                f.write("ok\n")
            for t in range(count):
                try:
                    os.remove("%s.%d" % (req, t))
                except OSError:
                    pass
            for suf in (".meta", ".ready"):
                try:
                    os.remove(req + suf)
                except OSError:
                    pass
            for p in (hid_in, rows_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
            served += 1
            t_idle = time.time()
            tot_ms = (t_idle - t_req) * 1e3
            log(serve="served", done=True, req=n, failed=False, tokens=tokens_real,
                lanes=lanes, decMs=round(dec_ms, 1), modelMs=round(model_ms, 1),
                encMs=round(enc_ms, 1), totalMs=round(tot_ms, 1),
                reqMsPerToken=round(tot_ms / max(tokens_real, 1), 1),
                respBytes=resp_bytes, inRms=in_rms, outRms=out_rms,
                laneSpread=lane_spread, stateful=bool(args.stateful), mock=True)
            n += 1
        except Exception as e:      # noqa: BLE001 - any failure is fatal
            with open(resp + ".error", "w") as f:
                f.write("mock server failed: %s\n" % e)
            log(serve="error", req=n, error=str(e), failed=True)
            log(serve="exit", reason="error", served=served, rc=3)
            return 3


if __name__ == "__main__":
    sys.exit(main())
