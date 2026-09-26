#!/usr/bin/env python3
"""
demo_ui.py — live product-style dashboard for the FHE-SSM serve session
(S3.1, 2026-09-01). Tails the v2 client's state files (tokens.json,
transcript.json) in --out and renders a lane grid: every conversation as a
chat card with a phase badge (reading / writing / done), plus a header with
the tick clock, security parameters, and honest wire/served counters.

Read-only observer: it never touches the protocol — safe to run beside the
client (same machine) and safe to record. All numbers shown are read from
the client's own records (R5 on screen).

Usage:  python demo_ui.py --out <client out dir> [--lanes-show 8]
        [--title "..."] [--refresh 2]
"""
import argparse
import json
import os
import time

from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def lane_phase(state, transcript, r):
    prompts = state.get("ids_lanes") or [state.get("ids", [])]
    gen = state.get("generated_lanes") or [state.get("generated", [])]
    retired = state.get("retired") or [False] * len(prompts)
    tick = len(transcript)
    if retired[r]:
        return "done", len(gen[r])
    if tick < len(prompts[r]):
        return "reading", len(gen[r])
    return "writing", len(gen[r])


def render(args, tok):
    out = args.out
    state = load_json(os.path.join(out, "tokens.json"), {})
    transcript = load_json(os.path.join(out, "transcript.json"), [])
    lanes = state.get("lanes", 1)
    prompts = state.get("ids_lanes") or [state.get("ids", [])]
    gen = state.get("generated_lanes") or [state.get("generated", [])]

    layout = Layout()
    layout.split_column(Layout(name="head", size=6), Layout(name="grid"))

    tick = len(transcript)
    last = transcript[-1] if transcript else {}
    serve_s = last.get("serveSec", 0.0)
    total_words = sum(len(g) for g in gen)
    phases = [lane_phase(state, transcript, r)[0] for r in range(lanes)]
    head = Table.grid(expand=True)
    head.add_column(justify="left")
    head.add_column(justify="right")
    head.add_row(
        Text(args.title, style="bold"),
        Text(f"tick {tick}   last server pass {serve_s:.1f}s", style="cyan"))
    head.add_row(
        Text(f"{lanes} conversations · one ciphertext · one key",
             style="dim"),
        Text(f"{phases.count('reading')} reading · "
             f"{phases.count('writing')} writing · "
             f"{phases.count('done')} done · {total_words} words generated",
             style="green"))
    head.add_row(
        Text("server sees ciphertext only — no plaintext, no secret key",
             style="yellow"),
        Text(time.strftime("%H:%M:%SZ", time.gmtime()), style="dim"))
    layout["head"].update(Panel(head, box=box.HEAVY, title="FHE-SSM encrypted serving",
                                subtitle="CKKS 128-bit · compressed store · stateful lanes"))

    n_show = min(args.lanes_show, lanes)
    cols = 2 if n_show <= 8 else 4
    grid = Table.grid(expand=True)
    for _ in range(cols):
        grid.add_column(ratio=1)
    cards = []
    for r in range(n_show):
        phase, words = lane_phase(state, transcript, r)
        style = {"reading": "blue", "writing": "green", "done": "dim"}[phase]
        text = tok.decode(prompts[r] + gen[r]) if prompts[r] else ""
        if len(text) > 220:
            text = "…" + text[-220:]
        body = Text(text)
        if phase == "writing":
            body.append(" ▌", style="blink green")
        cards.append(Panel(body, title=f"chat {r} · {phase} · {words}w",
                           border_style=style, box=box.ROUNDED, height=args.card_h))
    for i in range(0, len(cards), cols):
        grid.add_row(*cards[i:i + cols])
    hidden = lanes - n_show
    layout["grid"].update(Panel(grid, box=box.SIMPLE,
                                subtitle=(f"+{hidden} more lanes live (all recorded)"
                                          if hidden > 0 else "all lanes shown")))
    return layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lanes-show", type=int, default=8)
    ap.add_argument("--card-h", type=int, default=7)
    ap.add_argument("--refresh", type=float, default=2.0)
    ap.add_argument("--title", default="Private chat — the server never reads you")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    with Live(render(a, tok), refresh_per_second=2, screen=True) as live:
        while True:
            time.sleep(a.refresh)
            live.update(render(a, tok))


if __name__ == "__main__":
    main()
