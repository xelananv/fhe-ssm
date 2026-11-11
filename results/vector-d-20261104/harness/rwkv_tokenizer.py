# SPDX-License-Identifier: MIT
"""The RWKV World trie tokenizer, as the reference implements it (`raw/rwkv_v7_demo.py`:53-127, greedy longest match
from a (first byte, second byte) table). One change: the vocabulary literals are parsed with `ast.literal_eval`
instead of `eval` (demo.py:63) — same result for the str/bytes literals in the file, no code execution."""
from __future__ import annotations

import ast
import os

DEFAULT_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw", "rwkv_vocab_v20230424.txt")


class RWKVTokenizer:
    def __init__(self, file_name: str = DEFAULT_VOCAB):
        self.idx2token: dict[int, bytes] = {}
        ordered = []  # the file is already sorted by id
        with open(file_name, "r", encoding="utf-8") as f:
            for l in f:
                if not l.strip():
                    continue
                idx = int(l[: l.index(" ")])
                x = ast.literal_eval(l[l.index(" "): l.rindex(" ")].strip())
                x = x.encode("utf-8") if isinstance(x, str) else x
                assert isinstance(x, bytes)
                assert len(x) == int(l[l.rindex(" "):])
                ordered.append(x)
                self.idx2token[idx] = x
        self.token2idx = {v: k for k, v in self.idx2token.items()}

        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]
        for s in reversed(ordered):  # reverse order: longer tokens first (demo.py:79)
            if len(s) >= 2:
                s0, s1 = s[0], s[1]
                self.table[s0][s1].append(s)
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encode_bytes(self, src: bytes) -> list[int]:
        n = len(src)
        tokens = []
        i = 0
        while i < n:
            s = src[i: i + 1]
            if i < n - 1:
                s0, s1 = src[i], src[i + 1]
                if s1 in self.good[s0]:
                    sss = src[i: i + self.wlen[s0]]
                    s = next(filter(sss.startswith, self.table[s0][s1]), s)
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def encode(self, text: str) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def decode_bytes(self, tokens) -> bytes:
        return b"".join(self.idx2token[i] for i in tokens)

    def decode(self, tokens) -> str:
        return self.decode_bytes(tokens).decode("utf-8")
