#!/usr/bin/env python3
"""Staged-file token reader — feeds the trainer from `stage_fineweb_tokens.py`
output instead of HuggingFace streaming.

AUTHORED 2026-08-18 (Mac, pre-rental). This is the one remaining S3-authored
piece and the Phase B launch-gate item named in `campaign-sessions/
S3_training_program_v2.md`:29, `ml-eval/stage_fineweb_tokens.py`:21-24 and
`campaign/queues/l40s.json` (A-0 staging manifest notes).

NEW FILE per the immutability rule. `train_fhe_native_ssm.py` is NOT edited;
this module WRAPS `base.token_stream`, the same import-as-module pattern
`train_fhe_native_ssm_masked.py` uses. Because the masked trainer calls
`base.batch_iter` -> `base.token_stream`, installing here reaches the dense
and the masked lanes at once.

  # in a launcher, before cmd_train:
  import train_fhe_native_ssm as base
  import staged_token_reader
  staged_token_reader.install(base)
  base.cmd_train("staged:/root/bundles/staged", steps, resume=False)

  # or verify a staged corpus standalone (no torch, no network):
  python ml-eval/staged_token_reader.py --dir DIR --verify

SPEC STRING (the `--data` value). Anything not starting with `staged:` falls
through to the base's own HF path untouched:

  staged:<dir>[?offset=N&on_exhaust=restart|continue&chunk=N]

  offset      first training token to read (default 0)
  on_exhaust  what a NEW iterator does after the corpus ran out, i.e. what
              the trainer's StopIteration handler
              (`train_fhe_native_ssm.py`:373-379) gets on its next call:
                restart  (default) re-read from `offset` — byte-for-byte the
                         base HF path's behaviour, so a staged arm stays
                         data-matched to a streaming arm
                continue resume where the previous pass stopped, wrapping mod
                         train_tokens. This is the fix for the re-walk
                         `training/DIAGNOSIS.md` measured (~48% of q2's
                         tokens were seen-before). OPT-IN — it changes what
                         "tokens seen" means, so it must be a logged decision,
                         never a silent default.
  chunk       tokens per yielded shard (default 1,000,000, matching the base
              fineweb path's 1M flush at `train_fhe_native_ssm.py`:237)

THE FOUR TRAPS THIS FILE EXISTS TO CLOSE
----------------------------------------
1. HOLDOUT CONTAMINATION. `stage_fineweb_tokens.py` writes ONE flat file:
   [0, train_tokens) is training, [train_tokens, +holdout_tokens) is the T4
   instruments' non-annealed FineWeb axis, and there may be a further ragged
   tail (the stager stops on a whole document, so total_staged >= need). A
   reader that just walks the file to EOF trains on its own eval set and the
   resulting number is worthless. The barrier here is HARD and comes from
   provenance, never from the file size.
2. GUESSED PROVENANCE. train_tokens cannot be recovered from the directory
   alone (the ragged tail makes total - holdout ambiguous), so a missing or
   mismatched provenance JSON is a hard error, not a warning. Same reflex as
   R2: an artifact is not bound to the thing beside it unless something says
   so.
3. DOUBLE SHARDING. `base.batch_iter` already strides the window space by
   rank (`:249`), so this reader is deliberately rank-AGNOSTIC and yields the
   identical shard sequence in every process. Sharding here too would give
   each rank 1/world^2 of the corpus while the step counter kept claiming the
   full budget — the same class of silent invalidation as the 2026-07-18
   iterator bug the base documents at :374-378.
4. TOKENIZER SKEW. Staged ids are opaque uint16; a corpus tokenized with a
   different tokenizer than the run's would train perfectly and mean nothing.
   The eos_id in provenance is checked against the live tokenizer.

NOT DONE HERE, DELIBERATELY: `split == "test"` still falls through to the
base (WikiText-103). `training/DIAGNOSIS.md` found that every lineage eval
was WT103 even for `_fineweb`-named files — silently swapping the test split
to the staged holdout would paper over that instrument gap instead of fixing
it. The holdout is reachable, explicitly, via `holdout_tokens()`.
"""
import json
import os
import sys

CHUNK_DEFAULT = 1_000_000
PREFIX = "staged:"

# on_exhaust=continue state, keyed by spec. Per-process, which is what we want
# under DDP: every rank advances identically and batch_iter still strides by
# rank, so the ranks stay disjoint exactly as in a fresh pass.
_CURSORS = {}


def parse_spec(data):
    """`staged:<dir>?k=v&...` -> dict, or None if `data` is not a staged spec."""
    if not isinstance(data, str) or not data.startswith(PREFIX):
        return None
    body = data[len(PREFIX):]
    path, _, query = body.partition("?")
    spec = {"dir": path, "offset": 0, "on_exhaust": "restart",
            "chunk": CHUNK_DEFAULT, "spec": data}
    for part in filter(None, query.split("&")):
        k, _, v = part.partition("=")
        if k not in ("offset", "on_exhaust", "chunk"):
            raise ValueError(f"unknown staged-spec key {k!r} in {data!r}; "
                             "known: offset, on_exhaust, chunk")
        spec[k] = v
    for k in ("offset", "chunk"):
        spec[k] = int(spec[k])
    if spec["on_exhaust"] not in ("restart", "continue"):
        raise ValueError(f"on_exhaust must be restart|continue, got "
                         f"{spec['on_exhaust']!r}")
    if spec["chunk"] <= 0:
        raise ValueError(f"chunk must be positive, got {spec['chunk']}")
    if spec["offset"] < 0:
        raise ValueError(f"offset must be >= 0, got {spec['offset']}")
    if not spec["dir"]:
        raise ValueError(f"staged spec has an empty directory: {data!r}")
    return spec


def load_provenance(dirpath, provenance=None):
    """Provenance is MANDATORY — it carries the holdout barrier (trap 2)."""
    cand = [provenance] if provenance else [
        os.path.join(dirpath, "provenance.json"),
        os.path.join(dirpath, "staging_provenance.json"),
    ]
    for p in cand:
        if p and os.path.exists(p):
            with open(p) as f:
                prov = json.load(f)
            prov["_path"] = p
            return prov
    raise FileNotFoundError(
        f"no provenance JSON for staged corpus {dirpath!r} (looked at "
        f"{[c for c in cand if c]}). train_tokens CANNOT be inferred from the "
        "directory — the stager writes a ragged tail past the holdout, so "
        "total-minus-holdout is ambiguous and guessing it risks training on "
        "the eval slice. Re-run stage_fineweb_tokens.py with "
        "--provenance <dir>/provenance.json, or pass one explicitly.")


class StagedCorpus:
    """Read-only memmap over `tokens.uint16.bin` with a hard holdout barrier.

    26 GB at the Phase B budget, so the file is memmapped and never read into
    RAM; each yielded chunk is the only copy made.
    """

    def __init__(self, dirpath, provenance=None, vocab=None):
        import numpy as np
        self.np = np
        self.dir = dirpath
        self.bin_path = os.path.join(dirpath, "tokens.uint16.bin")
        if not os.path.exists(self.bin_path):
            raise FileNotFoundError(f"no staged corpus at {self.bin_path}")
        self.prov = load_provenance(dirpath, provenance)

        dtype = self.prov.get("dtype")
        if dtype != "uint16":
            raise ValueError(f"provenance says dtype={dtype!r}; this reader "
                             "only handles the uint16 staging format")
        self.train_tokens = int(self.prov["train_tokens"])
        self.holdout_tokens = int(self.prov.get("holdout_tokens", 0))
        self.eos_id = int(self.prov.get("eos_id", 0))

        size = os.path.getsize(self.bin_path)
        if size % 2:
            raise ValueError(f"{self.bin_path} is {size} bytes — not a whole "
                             "number of uint16 tokens; a staging run was "
                             "killed mid-token. Re-run the stager, which "
                             "truncates to the last whole token on resume.")
        self.total_on_disk = size // 2
        recorded = int(self.prov.get("bin_bytes", size))
        if recorded != size:
            raise ValueError(
                f"provenance records bin_bytes={recorded:,} but "
                f"{self.bin_path} is {size:,} bytes. The corpus and its "
                "provenance describe different files — refusing to guess "
                "which is current (R2).")
        need = self.train_tokens + self.holdout_tokens
        if self.total_on_disk < need:
            raise ValueError(
                f"staged corpus holds {self.total_on_disk:,} tokens but "
                f"provenance claims {self.train_tokens:,} train + "
                f"{self.holdout_tokens:,} holdout = {need:,}. Staging is "
                "INCOMPLETE; training now would run off the end of the "
                "training slice and into the holdout.")

        self.mm = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        self._check_ids(vocab)

    def _check_ids(self, vocab):
        """Strided sample — a full scan of 26 GB is not worth the minutes, but
        gross corruption and tokenizer skew both show up immediately."""
        vocab = vocab or self.prov.get("vocab") or 50277
        n = self.train_tokens
        if n == 0:
            raise ValueError("provenance says train_tokens=0")
        step = max(1, n // 10_000)
        sample = self.np.asarray(self.mm[:n:step])
        hi = int(sample.max())
        if hi >= vocab:
            raise ValueError(
                f"staged token id {hi} >= vocab {vocab} in a {len(sample):,}-"
                "token sample. The corpus was not produced by the "
                f"{self.prov.get('repo', '?')} + RWKV/rwkv-4-169m-pile "
                "pipeline this trainer assumes.")
        self.sampled = int(len(sample))
        self.sample_max_id = hi

    def check_tokenizer(self, tok):
        """Trap 4. Hard error: a tokenizer mismatch trains fine and means
        nothing, so it must not be survivable."""
        if tok is None:
            return
        live = getattr(tok, "eos_token_id", None)
        if live is None or self.eos_id is None:
            return
        if int(live) != int(self.eos_id):
            raise ValueError(
                f"tokenizer skew: staged corpus was built with eos_id="
                f"{self.eos_id} ({self.prov.get('repo', '?')}), the running "
                f"trainer's tokenizer has eos_id={int(live)}. The staged ids "
                "are opaque — training would proceed and be meaningless.")

    def chunks(self, offset=0, chunk=CHUNK_DEFAULT):
        """Yield numpy int64 views of [offset, train_tokens), in order.

        The upper bound is train_tokens and NEVER the file length (trap 1).
        """
        pos = int(offset)
        if pos >= self.train_tokens:
            return
        while pos < self.train_tokens:
            end = min(pos + chunk, self.train_tokens)
            yield pos, self.np.asarray(self.mm[pos:end]).astype(self.np.int64)
            pos = end

    def holdout(self):
        """The reserved eval slice — NOT reachable from chunks() by design."""
        if self.holdout_tokens == 0:
            raise ValueError("this staged corpus reserved no holdout")
        lo = self.train_tokens
        hi = lo + self.holdout_tokens
        return self.np.asarray(self.mm[lo:hi]).astype(self.np.int64)

    def summary(self):
        return {"dir": self.dir, "provenance": self.prov.get("_path"),
                "train_tokens": self.train_tokens,
                "holdout_tokens": self.holdout_tokens,
                "total_on_disk": self.total_on_disk,
                "tail_past_holdout": self.total_on_disk - self.train_tokens
                                     - self.holdout_tokens,
                "eos_id": self.eos_id, "sampled_ids": self.sampled,
                "sample_max_id": self.sample_max_id,
                "repo": self.prov.get("repo"),
                "first_doc_sha16": self.prov.get("first_doc_sha16")}


def holdout_tokens(dirpath, provenance=None):
    """The T4 non-annealed FineWeb axis, as a numpy int64 array.

    Deliberately explicit: `split == "test"` is left on WikiText-103 so the
    instrument gap DIAGNOSIS found stays visible rather than silently patched.
    """
    return StagedCorpus(dirpath, provenance).holdout()


def token_stream(torch, tok, data, split, _base_stream=None):
    """Drop-in for `train_fhe_native_ssm.token_stream`.

    Non-staged `data`, or split == "test", falls through to the base.
    """
    spec = parse_spec(data)
    if spec is None or split == "test":
        if _base_stream is None:
            raise ValueError(
                f"staged_token_reader got data={data!r} split={split!r} with "
                "no base stream to fall through to; call install(base) rather "
                "than using this function directly.")
        yield from _base_stream(torch, tok, data, split)
        return

    corpus = StagedCorpus(spec["dir"])
    corpus.check_tokenizer(tok)

    key = spec["spec"]
    start = spec["offset"]
    if spec["on_exhaust"] == "continue":
        start = _CURSORS.get(key, spec["offset"])
        if start >= corpus.train_tokens:      # wrap
            start = spec["offset"]
    print(json.dumps({"event": "staged_stream_open", **corpus.summary(),
                      "offset": start, "on_exhaust": spec["on_exhaust"],
                      "chunk": spec["chunk"]}), flush=True)

    pos = start
    for pos, arr in corpus.chunks(start, spec["chunk"]):
        _CURSORS[key] = pos + len(arr)
        yield torch.tensor(arr, dtype=torch.long)
    print(json.dumps({"event": "staged_stream_exhausted",
                      "from": start, "to": _CURSORS.get(key, start),
                      "train_tokens": corpus.train_tokens}), flush=True)


def with_continue(data):
    """Return `data` with on_exhaust=continue, unless it already says
    otherwise. Used by the trainer under --resume (ADDED 2026-08-22, S2.8).

    Rewriting rather than defaulting keeps the property this module was built
    around: on_exhaust is a LOGGED decision, never a silent default. The
    rewritten spec goes into the run's start event and into the cursor key."""
    spec = parse_spec(data)
    if spec is None:
        return data
    body = data[len(PREFIX):]
    _, _, query = body.partition("?")
    if "on_exhaust=" in query:
        return data
    return data + ("&" if query else "?") + "on_exhaust=continue"


def rekey_cursors(state, spec):
    """Move a saved cursor onto the spec the resuming run will ACTUALLY use.

    ADDED 2026-08-22 (S2.8) after the end-to-end resume check caught the bug:
    `_CURSORS` is keyed by the full spec string, and `--resume` rewrites the
    spec (`with_continue` appends `on_exhaust=continue`). Restoring the saved
    dict verbatim therefore filed the cursor under the PRE-rewrite key, the
    reader looked up the POST-rewrite key, found nothing, and silently restarted
    from `offset` — i.e. re-walked the corpus, which is the exact failure this
    whole path exists to prevent, wearing a "resumed" log line.

    Matching is by staged DIRECTORY, not by spec: the directory identifies the
    corpus, while offset/chunk/on_exhaust are read parameters that may legally
    differ between the killed run and the resumed one. If several saved entries
    point at the same directory the furthest one wins, so a resume never
    re-reads.
    """
    target = parse_spec(spec)
    if target is None or not state:
        return {}
    best = None
    for k, v in state.items():
        src = parse_spec(k)
        if src is not None and os.path.abspath(src["dir"]) == \
                os.path.abspath(target["dir"]):
            best = int(v) if best is None else max(best, int(v))
    return {spec: best} if best is not None else {}


def cursor_state():
    """The read cursors, for persistence into {TAG}_resume.pt."""
    return dict(_CURSORS)


def restore_cursors(state):
    """Re-seed the cursors from a resume checkpoint, so a killed run does not
    re-walk the corpus. training/DIAGNOSIS.md §3 traced ~48% of q2's tokens to
    exactly that re-walk."""
    if state:
        _CURSORS.update({k: int(v) for k, v in state.items()})
    return dict(_CURSORS)


def install(base):
    """Wrap `base.token_stream` so `staged:` specs read from disk.

    Idempotent; returns the module for chaining.
    """
    if getattr(base, "_staged_reader_installed", False):
        return base
    original = base.token_stream

    def wrapped(torch, tok, data, split):
        return token_stream(torch, tok, data, split, _base_stream=original)

    wrapped.__name__ = "token_stream"
    wrapped.__doc__ = (original.__doc__ or "") + \
        "\n\nWRAPPED by staged_token_reader.install: `staged:<dir>` specs " \
        "read the staged uint16 corpus; everything else is unchanged."
    base.token_stream = wrapped
    base._staged_reader_installed = True
    base._staged_reader_original_token_stream = original
    return base


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--provenance")
    ap.add_argument("--verify", action="store_true",
                    help="open, validate, and report — reads no more than a "
                         "10k-token sample")
    ap.add_argument("--chunks", type=int, default=0,
                    help="additionally walk N chunks and report token counts")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=CHUNK_DEFAULT)
    args = ap.parse_args()

    corpus = StagedCorpus(args.dir, args.provenance)
    out = corpus.summary()
    if args.chunks:
        seen, last = 0, args.offset
        for i, (pos, arr) in enumerate(corpus.chunks(args.offset, args.chunk)):
            if i >= args.chunks:
                break
            seen += len(arr)
            last = pos + len(arr)
        out["walked_chunks"] = min(args.chunks, i + 1)
        out["walked_tokens"] = seen
        out["walked_to"] = last
        out["within_barrier"] = last <= corpus.train_tokens
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
