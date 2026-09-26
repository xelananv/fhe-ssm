#!/usr/bin/env python3
"""Offline verification for `staged_token_reader.py` (Mac, no torch, no HF).

  python ml-eval/test_staged_token_reader.py

Builds synthetic staged corpora byte-compatible with
`stage_fineweb_tokens.py`'s output (tokens.uint16.bin + provenance.json +
holdout.npy, INCLUDING the ragged tail past the holdout that the stager
leaves when it stops on a whole document), then checks the reader's contract
and each of the four traps its docstring claims to close.

torch is not installed on the Mac, and does not need to be: the base's
`token_stream(torch, tok, data, split)` takes the torch module as a
PARAMETER, so a shim exercises the real code path end to end.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import staged_token_reader as R


# --- shims -----------------------------------------------------------------
class _Torch:
    long = "int64"

    @staticmethod
    def tensor(arr, dtype=None):
        return np.asarray(arr, dtype=np.int64)


class _Tok:
    def __init__(self, eos_token_id=0):
        self.eos_token_id = eos_token_id


TORCH = _Torch()
RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((True, name, ""))
        print(f"  PASS  {name}")
    except AssertionError as e:
        RESULTS.append((False, name, str(e)))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


def raises(fn, needle):
    try:
        fn()
    except Exception as e:
        assert needle.lower() in str(e).lower(), \
            f"expected an error mentioning {needle!r}, got: {e}"
        return
    raise AssertionError(f"expected an error mentioning {needle!r}, none raised")


# --- fixture ---------------------------------------------------------------
def make_corpus(root, train=5000, holdout=800, tail=137, eos=0, vocab=50277,
                bad_bin_bytes=False, drop_provenance=False, short_file=False,
                odd_byte=False):
    """Mimic stage_fineweb_tokens.py's on-disk layout exactly."""
    os.makedirs(root, exist_ok=True)
    total = train + holdout + tail
    if short_file:
        total = train + holdout // 2          # staging killed early
    rng = np.random.default_rng(1234)
    ids = rng.integers(1, vocab, size=total, dtype=np.uint16)
    ids[::97] = eos                            # EOS between "documents"
    bin_path = os.path.join(root, "tokens.uint16.bin")
    with open(bin_path, "wb") as f:
        f.write(ids.tobytes())
        if odd_byte:
            f.write(b"\x00")
    np.save(os.path.join(root, "holdout.npy"), ids[train:train + holdout])
    if not drop_provenance:
        prov = {"repo": "HuggingFaceTB/smollm-corpus", "cfg": "fineweb-edu-dedup",
                "files": ["fineweb-edu-dedup/train-00000.parquet"], "col": "text",
                "first_doc_sha16": "deadbeefdeadbeef", "eos_id": int(eos),
                "train_tokens": int(train), "holdout_tokens": int(holdout),
                "total_staged": int(total), "dtype": "uint16",
                "bin_bytes": (1 if bad_bin_bytes else 0) + os.path.getsize(bin_path)}
        with open(os.path.join(root, "provenance.json"), "w") as f:
            json.dump(prov, f, indent=2)
    return bin_path, ids


def drain(data, tok=None, split="train", base=None):
    """Run the reader's generator to exhaustion, returning concatenated ids."""
    out = [a for a in R.token_stream(TORCH, tok, data, split, _base_stream=base)]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


# --- checks ----------------------------------------------------------------
def main():
    tmp = tempfile.mkdtemp(prefix="staged_reader_test_")
    try:
        root = os.path.join(tmp, "good")
        _, ids = make_corpus(root)
        TRAIN, HOLD = 5000, 800

        print("\nspec parsing")

        def t_spec():
            s = R.parse_spec("staged:/d?offset=10&on_exhaust=continue&chunk=64")
            assert s["dir"] == "/d" and s["offset"] == 10, s
            assert s["on_exhaust"] == "continue" and s["chunk"] == 64, s
            assert R.parse_spec("fineweb") is None
            assert R.parse_spec("wt103") is None
            raises(lambda: R.parse_spec("staged:/d?nope=1"), "unknown")
            raises(lambda: R.parse_spec("staged:/d?on_exhaust=shuffle"), "restart")
            raises(lambda: R.parse_spec("staged:/d?chunk=0"), "positive")
            raises(lambda: R.parse_spec("staged:"), "empty")
        check("parses specs and rejects unknown/invalid keys", t_spec)

        print("\ntrap 1 — holdout barrier")

        def t_barrier():
            got = drain(f"staged:{root}?chunk=512")
            assert len(got) == TRAIN, f"read {len(got)} tokens, expected {TRAIN}"
            assert np.array_equal(got, ids[:TRAIN].astype(np.int64)), \
                "content diverges from the training slice"
        check("reads exactly [0, train_tokens) and nothing past it", t_barrier)

        def t_barrier_odd_chunk():
            # a chunk size that does not divide train_tokens must not overrun
            got = drain(f"staged:{root}?chunk=997")
            assert len(got) == TRAIN, f"read {len(got)}, expected {TRAIN}"
        check("non-dividing chunk size still stops on the barrier",
              t_barrier_odd_chunk)

        def t_holdout_disjoint():
            hold = R.holdout_tokens(root)
            assert len(hold) == HOLD, f"holdout {len(hold)} != {HOLD}"
            assert np.array_equal(hold, ids[TRAIN:TRAIN + HOLD].astype(np.int64))
            saved = np.load(os.path.join(root, "holdout.npy"))
            assert np.array_equal(hold, saved.astype(np.int64)), \
                "reader's holdout != the stager's holdout.npy"
            got = drain(f"staged:{root}?chunk=512")
            # positional disjointness: the stream ends on the last training
            # token and the holdout begins on the next one, so no index is
            # served by both paths.
            assert len(got) == TRAIN, f"stream read {len(got)}, expected {TRAIN}"
            assert got[-1] == ids[TRAIN - 1], "stream did not end on train[-1]"
            assert hold[0] == ids[TRAIN], "holdout did not start at train_tokens"
        check("holdout matches holdout.npy and is disjoint from training",
              t_holdout_disjoint)

        print("\ntrap 2 — provenance is mandatory and must bind")

        def t_no_prov():
            bad = os.path.join(tmp, "noprov")
            make_corpus(bad, drop_provenance=True)
            raises(lambda: R.StagedCorpus(bad), "provenance")
        check("missing provenance is a hard error, not a guess", t_no_prov)

        def t_bad_bytes():
            bad = os.path.join(tmp, "badbytes")
            make_corpus(bad, bad_bin_bytes=True)
            raises(lambda: R.StagedCorpus(bad), "bin_bytes")
        check("provenance/file size mismatch refuses to open", t_bad_bytes)

        def t_short():
            bad = os.path.join(tmp, "short")
            make_corpus(bad, short_file=True)
            raises(lambda: R.StagedCorpus(bad), "incomplete")
        check("incomplete staging is caught before it reaches the holdout",
              t_short)

        def t_odd():
            bad = os.path.join(tmp, "odd")
            make_corpus(bad, odd_byte=True)
            raises(lambda: R.StagedCorpus(bad), "whole number")
        check("half-written trailing token is rejected", t_odd)

        print("\ntrap 3 — rank-agnostic (no double sharding)")

        def t_rank():
            a = drain(f"staged:{root}?chunk=512")
            b = drain(f"staged:{root}?chunk=512")
            assert np.array_equal(a, b), \
                "two opens gave different sequences — the reader is not " \
                "deterministic, so ranks would diverge"
            import inspect
            sig = inspect.signature(R.token_stream).parameters
            assert "rank" not in sig and "world" not in sig, \
                "reader takes rank/world; base.batch_iter already shards"
        check("identical sequence per process; no rank/world parameter", t_rank)

        print("\ntrap 4 — tokenizer skew")

        def t_tok_ok():
            got = drain(f"staged:{root}?chunk=512", tok=_Tok(eos_token_id=0))
            assert len(got) == TRAIN
        check("matching tokenizer passes", t_tok_ok)

        def t_tok_skew():
            raises(lambda: drain(f"staged:{root}", tok=_Tok(eos_token_id=2)),
                   "tokenizer skew")
        check("mismatched eos_id aborts rather than training on opaque ids",
              t_tok_skew)

        def t_vocab():
            bad = os.path.join(tmp, "vocab")
            make_corpus(bad, vocab=60000)
            raises(lambda: R.StagedCorpus(bad), "vocab")
        check("out-of-vocab ids are caught by the strided sample", t_vocab)

        print("\noffset and exhaustion policy")

        def t_offset():
            got = drain(f"staged:{root}?offset=1000&chunk=512")
            assert len(got) == TRAIN - 1000, f"read {len(got)}"
            assert np.array_equal(got, ids[1000:TRAIN].astype(np.int64))
            past = drain(f"staged:{root}?offset={TRAIN + 10}")
            assert len(past) == 0, "offset past the barrier must yield nothing"
        check("offset seeks, and cannot seek into the holdout", t_offset)

        def t_restart():
            spec = f"staged:{root}?chunk=512"
            a, b = drain(spec), drain(spec)
            assert np.array_equal(a, b), \
                "on_exhaust=restart must re-read from the same offset " \
                "(data-matched to the streaming path)"
        check("on_exhaust=restart re-walks, matching the base HF path",
              t_restart)

        def t_continue():
            R._CURSORS.clear()
            spec = f"staged:{root}?on_exhaust=continue&chunk=512"
            a = drain(spec)
            assert len(a) == TRAIN
            b = drain(spec)          # wraps: previous pass consumed everything
            assert len(b) == TRAIN, f"second pass read {len(b)}"
            R._CURSORS.clear()
            half = f"staged:{root}?on_exhaust=continue&chunk=512"
            gen = R.token_stream(TORCH, None, half, "train")
            first = next(gen)
            gen.close()
            resumed = drain(half)
            assert len(resumed) == TRAIN - len(first), \
                f"resume read {len(resumed)}, expected {TRAIN - len(first)}"
            assert np.array_equal(resumed,
                                  ids[len(first):TRAIN].astype(np.int64)), \
                "resumed pass did not continue where the previous one stopped"
            R._CURSORS.clear()
        check("on_exhaust=continue resumes and wraps at the barrier",
              t_continue)

        print("\ninstall() wrapping")

        def t_install():
            calls = []

            class FakeBase:
                @staticmethod
                def token_stream(torch, tok, data, split):
                    calls.append((data, split))
                    yield torch.tensor([7, 7, 7], dtype=torch.long)

            base = FakeBase()
            R.install(base)
            R.install(base)                       # idempotent
            assert base._staged_reader_installed

            out = list(base.token_stream(TORCH, None, "fineweb", "train"))
            assert calls == [("fineweb", "train")], \
                f"non-staged data must fall through untouched; got {calls}"
            assert np.array_equal(out[0], [7, 7, 7])

            calls.clear()
            out = list(base.token_stream(TORCH, None, f"staged:{root}", "test"))
            assert calls == [(f"staged:{root}", "test")], \
                "split=='test' must still fall through to WikiText-103"

            calls.clear()
            got = np.concatenate(list(
                base.token_stream(TORCH, None, f"staged:{root}?chunk=512",
                                  "train")))
            assert not calls, "staged train spec must NOT hit the base"
            assert len(got) == TRAIN
        check("wraps the base, is idempotent, falls through for non-staged "
              "and for split=='test'", t_install)

        print("\nCLI --verify")

        def t_cli():
            argv = sys.argv[:]
            sys.argv = ["x", "--dir", root, "--verify", "--chunks", "2",
                        "--chunk", "512"]
            try:
                import io
                import contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = R.main()
                out = json.loads(buf.getvalue())
            finally:
                sys.argv = argv
            assert rc == 0
            assert out["train_tokens"] == TRAIN and out["holdout_tokens"] == HOLD
            assert out["tail_past_holdout"] == 137, out["tail_past_holdout"]
            assert out["within_barrier"] is True
            assert out["walked_tokens"] == 1024, out["walked_tokens"]
        check("--verify reports the barrier, the tail, and a bounded walk",
              t_cli)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    npass = sum(1 for ok, _, _ in RESULTS if ok)
    nfail = len(RESULTS) - npass
    print(json.dumps({"event": "done", "passed": npass, "failed": nfail,
                      "failures": [n for ok, n, _ in RESULTS if not ok]}))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
