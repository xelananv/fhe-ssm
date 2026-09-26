#!/usr/bin/env python3
"""Sidecars for tools/token_panel.py from a prompt_select_longrun.py output prefix: <prefix>.txt.classes.json (lane -> class label) and
<prefix>.txt.outputs.json (lane -> the plaintext model's greedy continuation from the emulation). usage: prompt_select_sidecars.py <prefix>"""
import json, sys
pre = sys.argv[1]; by = {}
for l in open(pre + ".scores.jsonl"):
    r = json.loads(l); by[r["prompt"]] = r
sel = [l.rstrip("\n") for l in open(pre + ".txt") if l.strip()]
lab = {"A": "A long reading", "B": "B medium reading", "C": "C short reading", "D": "D standard short"}
json.dump({str(i): lab[by[p]["class"]] for i, p in enumerate(sel)}, open(pre + ".txt.classes.json", "w"))
json.dump({str(i): by[p]["text"] for i, p in enumerate(sel)}, open(pre + ".txt.outputs.json", "w"))
print("sidecars for", len(sel), "lanes")
