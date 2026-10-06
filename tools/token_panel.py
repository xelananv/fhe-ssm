#!/usr/bin/env python3
"""Live panel: the pod-side arms (A11, X2: per-tick server timing curve + per-lane teacher-forced tokens) on top, then the
generated tokens of every Mac-side session (takes), refreshed every 10 s. The pod is read by a background thread (one ssh
round trip per 30 s, aggregated on the pod so only a few KB cross the link); the page never waits on the pod.
Run: .venv/bin/python tools/token_panel.py --port 8765   then open http://localhost:8765"""
import argparse, glob, json, os, subprocess, sys, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "ml-eval")); import fhe_client as FC   # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--sessions", default=os.path.expanduser("~/demo_sessions"))
ap.add_argument("--pod", default="root@<pod-host>"); ap.add_argument("--ssh-key", default=os.path.expanduser("~/.ssh/<pod-key>")); ap.add_argument("--port-ssh", default="2248")
ap.add_argument("--host-key-alias", default="", help="ssh HostKeyAlias when --pod is an IP address (e.g. '[<pod-host>]:2520')")
ap.add_argument("--arms", default="LONG,X2,A11", help="pod-side arm names under /root/demo/s37/serve_<ARM>, shown in this order when present")
ap.add_argument("--arm-prompts", default=os.path.join(REPO, "results/dense-demo-s31/sessions/a11_prompts64_long.txt"))
ap.add_argument("--note-file", default=os.path.join(HERE, "token_panel_note.txt"), help="free text shown at the top (the run plan); re-read on every refresh")
a = ap.parse_args(); tok = FC.get_tokenizer()
EXP = {}
for f in glob.glob(os.path.join(REPO, "results/dense-demo-s31/sessions/plain_greedy_*.jsonl")):
    for l in open(f):
        try: r = json.loads(l); EXP.setdefault(r["prompt"], r["text"])
        except Exception: pass
try: ARM_PROMPTS = [l.rstrip("\n") for l in open(a.arm_prompts) if l.strip()]
except Exception: ARM_PROMPTS = []
ARM_IDS = [tok.encode(p) for p in ARM_PROMPTS]   # the driver's own tokenisation (spec_decode/fidelity_tick.py: ids_lanes = tok.encode(prompt))
def session_view(d):
    try: tk = json.load(open(os.path.join(d, "tokens.json")))
    except Exception:
        try: tk = json.load(open(os.path.join(d, "state.json")))
        except Exception: return None
    st = tk
    prompts = st.get("prompts") or []; gen = tk.get("generated_lanes") or st.get("generated_lanes") or [[] for _ in prompts]
    tick = None
    try:
        for line in open(os.path.join(d, "generate.log"), errors="ignore"):
            if line.startswith("== tick"): tick = line.strip()
    except Exception: pass
    fatal = ""
    try:
        for line in open(os.path.join(d, "generate.log"), errors="ignore"):
            if "dec failed" in line or "fatal" in line: fatal = line.strip()[:140]
    except Exception: pass
    lanes = []; ok = n = 0
    for i, p in enumerate(prompts):
        g = gen[i] if i < len(gen) else []
        got = tok.decode(g) if g else ""; want_ids = tok.encode(EXP[p])[:len(g)] if p in EXP and g else []
        want = tok.decode(want_ids) if want_ids else (EXP.get(p, "")[:40] if p in EXP else "")
        status = "-" if not g else ("OK" if g == want_ids else ("DIFF" if want_ids else "?"))
        if g and want_ids: n += 1; ok += (g == want_ids)
        lanes.append({"lane": i, "prompt": p, "fhe": got, "plain": want, "status": status})
    return {"session": os.path.basename(d), "lanes_n": len(prompts), "tick": tick, "fatal": fatal, "exact": f"{ok}/{n}", "lanes": lanes,
            "mtime": time.strftime("%H:%M:%SZ", time.gmtime(os.path.getmtime(d)))}
# Runs ON THE POD (python3 - ARM ...): the served-tick records + the fidelity rows of each arm, aggregated to a few KB.
POD_PY = r'''
import json, sys, time, os
out = {"now": time.strftime("%H:%M:%SZ", time.gmtime()), "arms": []}
for arm in sys.argv[1:]:
    d = "/root/demo/s37/serve_%s" % arm
    if not os.path.isdir(d): continue
    A = {"arm": arm, "served": [], "ticks": [], "lanes": [], "pt": []}
    try:
        for l in open(d + "/logs/server.jsonl", errors="ignore"):
            if l.startswith('{"serve":"served"'):
                r = json.loads(l)
                # 2026-09-19: the tick time shown is the WALL clock of the layer loop (reqLayerLoopMs). reqMsPerToken is a SUM OF TIMERS: it jumped
                # 137 -> 203 s when the bootstrap recorder made every bootstrap synchronous (the boot timer now sees the real 84 s instead of
                # the 20 s of asynchronous dispatch) while the wall went 182 -> 176 s. Shown as a secondary column only.
                A["served"].append({"s": round(r.get("reqLayerLoopMs", r["reqMsPerToken"]) / 1000.0, 1), "sum": round(r["reqMsPerToken"] / 1000.0, 1), "boots": r.get("reqBoots"), "enc": r.get("reqEncPtMs", 0),
                                    "hn": r.get("reqHostPtEncodeCount"), "hs": round((r.get("reqHostPtEncodeMs") or 0) / 1000.0, 1),
                                    "vram": r.get("peakVramGB"), "failed": bool(r.get("failed"))})
            elif l.startswith('{"ptCacheTick"'):
                r = json.loads(l); A["pt"].append({k: r.get(k) for k in ("pass", "ptCalls", "ptCacheHits", "ptCacheMisses", "ptCacheSkipped", "ptCacheDenseFallback", "ptCacheEntries", "ptCacheResidentGB")})
            elif l.startswith('{"fatal"'): A["fatal"] = l.strip()[:300]
    except Exception as ex: A["err"] = str(ex)
    try:
        dd = json.load(open(d + "/fidelity.json")); rr = dd.get("rows", dd) if isinstance(dd, dict) else dd
        by = {}; bl = {}
        for r in rr:
            by.setdefault(r["tick"], []).append(r); bl.setdefault(r["lane"], []).append(r)
        for t in sorted(by):
            b = by[t]; e = sorted(r["relErrRms"] for r in b)
            A["ticks"].append({"t": t, "top1": sum(1 for r in b if r["top1"]), "n": len(b), "decFailed": sum(1 for r in b if r.get("decFailed")),
                               "pick": (sum(1 for r in b if r.get("pickAgree")) if any("pickAgree" in r for r in b) else None),
                               "relMed": e[len(e) // 2], "relWorst": e[-1], "dup": b[0]["encArgmax"] == b[-1]["encArgmax"]})
        for lane in sorted(bl):
            seq = sorted(bl[lane], key=lambda r: r["tick"])
            # 2026-09-19: under a client-side decode policy (fidelity_tick --decode-policy norepeat) the token a lane GENERATES is the
            # policy's pick (encPick / refPick), not the raw argmax; rows without those fields (greedy) fall back to the argmax.
            A["lanes"].append({"lane": lane, "enc": [r.get("encPick", r["encArgmax"]) for r in seq], "ref": [r.get("refPick", r["refArgmax"]) for r in seq],
                               "fed": [r.get("fed") for r in seq], "gen": [bool(r.get("generating")) for r in seq], "free": any("fed" in r for r in seq)})
    except Exception as ex: A["fidErr"] = str(ex)[:120]
    try: A["mtime"] = time.strftime("%H:%M:%SZ", time.gmtime(os.path.getmtime(d + "/logs/server.jsonl")))
    except Exception: pass
    out["arms"].append(A)
print(json.dumps(out, separators=(",", ":")))
'''
POD = {"data": {"status": "first pod read pending…", "arms": []}}
def t1(i):
    try: return tok.decode([i]) if i is not None and i >= 0 else "∅"
    except Exception: return "?"
def pod_loop():
    while True:
        try:
            r = subprocess.run(["ssh", "-i", a.ssh_key, "-p", a.port_ssh, "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=20"] + (["-o", "HostKeyAlias=" + a.host_key_alias] if a.host_key_alias else []) + [a.pod,
                                "python3 - " + " ".join(x for x in a.arms.split(",") if x.isalnum())], input=POD_PY, capture_output=True, text=True, timeout=90)
            d = json.loads(r.stdout.strip().splitlines()[-1]); arms = []
            for A in d["arms"]:
                w = [x["s"] for x in A["served"][1:]]; ws = sorted(w)
                head = {"arm": A["arm"], "servedN": len(A["served"]), "mtime": A.get("mtime", ""), "fatal": A.get("fatal", ""), "fidErr": A.get("fidErr", "")}
                if w: head.update({"warmMedian": ws[len(ws) // 2], "warmMin": ws[0], "warmMax": ws[-1], "lanesN": len(A["lanes"]) or 64,
                                   "perLaneToken": round(ws[len(ws) // 2] / max(len(A["lanes"]), 1), 2) if A["lanes"] else None})
                ft = {x["t"]: x for x in A["ticks"]}; rows = []
                for i, s in enumerate(A["served"]):
                    f = ft.get(i, {}); c = A["pt"][i] if i < len(A["pt"]) else {}
                    # "dup lanes agree" compares the FIRST and the LAST lane: it only means something when those two lanes carry the SAME prompt
                    # (the A11 / fidelity-gate prompt files did that on purpose); with 64 different prompts it is not applicable.
                    dup_ok = len(ARM_PROMPTS) > 1 and ARM_PROMPTS[0] == ARM_PROMPTS[-1]
                    rows.append(dict(s, t=i, top1=f.get("top1"), n=f.get("n"), decFailed=f.get("decFailed"), relMed=f.get("relMed"), relWorst=f.get("relWorst"),
                                     dup=(f.get("dup") if dup_ok else "n/a"), ptCalls=c.get("ptCalls"), ptMiss=c.get("ptCacheMisses"), pick=f.get("pick")))
                lanes = []; hitF = hitP = npos = 0
                for L in A["lanes"]:
                    if L.get("free"):   # AUTOREGRESSIVE arm: the fed tokens are in the rows; the generated text is the FHE argmax where the lane generates
                        P = next((i for i, g in enumerate(L["gen"]) if g), len(L["gen"]))          # first generating tick = prompt length - 1
                        prompt = tok.decode([x for x in L["fed"][:P + 1] if x is not None])
                        gp = [[t1(e), t1(rf), e == rf] for e, rf, g in zip(L["enc"], L["ref"], L["gen"]) if g]
                        lanes.append({"lane": L["lane"], "prompt": prompt, "free": True, "gen": gp, "top1": f"{sum(1 for e, rf in zip(L['enc'], L['ref']) if e == rf)}/{len(L['enc'])}",
                                      "genN": len(gp)})
                        hitF += 0; npos += 0
                        continue
                    pr = ARM_PROMPTS[L["lane"]] if L["lane"] < len(ARM_PROMPTS) else ""
                    ids = ARM_IDS[L["lane"]] if L["lane"] < len(ARM_IDS) else []
                    pieces = [[t1(ids[t]) if t < len(ids) else "?", t1(e), t1(rf), e == rf, (t + 1 < len(ids) and e == ids[t + 1]), (t + 1 < len(ids) and rf == ids[t + 1])]
                              for t, (e, rf) in enumerate(zip(L["enc"], L["ref"]))]
                    hitF += sum(1 for p in pieces if p[4]); hitP += sum(1 for p in pieces if p[5]); npos += len(pieces)
                    lanes.append({"lane": L["lane"], "prompt": pr[:70], "pieces": pieces, "top1": f"{sum(1 for p in pieces if p[3])}/{len(pieces)}"})
                tot = sum(x["top1"] or 0 for x in A["ticks"]); cnt = sum(x["n"] or 0 for x in A["ticks"])
                head["top1Total"] = f"{tot}/{cnt}"; head["nextTok"] = f"FHE {hitF}/{npos}, plaintext model {hitP}/{npos}" if npos else ""
                head["decFailedTicks"] = [x["t"] for x in A["ticks"] if x["decFailed"]]
                arms.append({"head": head, "rows": rows, "lanes": lanes, "pt": A["pt"]})
            POD["data"] = {"status": "pod read " + d["now"], "arms": arms}
        except Exception as e:
            POD["data"] = dict(POD["data"], status=f"pod read failed at {time.strftime('%H:%M:%SZ', time.gmtime())}: {str(e)[:120]} (showing the last good read)")
        time.sleep(30)
PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>FHE-SSM live tokens</title>
<style>body{font:13px/1.35 -apple-system,Menlo,monospace;background:#111;color:#ddd;margin:12px}h2{margin:10px 0 4px;font-size:15px}h3{margin:8px 0 2px;font-size:13px;color:#bbb}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #333;padding:2px 6px;text-align:left;vertical-align:top}
.OK{color:#7f7}.DIFF{color:#f77}.dash{color:#777}.fhe{color:#fff}.small{color:#999;font-size:12px}.sess{background:#1b1b1b;padding:6px 8px;margin:8px 0;border-radius:6px}
.arm{background:#16202a;padding:6px 8px;margin:8px 0;border-radius:6px;border:1px solid #2c4a66}.bad{background:#611;color:#fff;border-radius:3px}.note{white-space:pre-wrap;color:#fd8;margin:6px 0}
.col{display:inline-block;text-align:center;margin:1px 0;padding:0 3px;border-left:1px solid #2a2a2a}.in{display:block;color:#888;font-size:11px}.pr{display:block;color:#fff}.hit{color:#7f7}
.num td,.num th{text-align:right;font-variant-numeric:tabular-nums}summary{cursor:pointer}a{color:#8cf}.kpi{color:#fff;font-size:14px}</style></head>
<body><div id=hdr class=small>loading…</div><div id=root></div>
<script>
const OPEN={};function keep(id,def){return (id in OPEN)?OPEN[id]:def;}
function det(id,def,sum,body){return '<details id="'+id+'" '+(keep(id,def)?'open':'')+' ontoggle="OPEN[this.id]=this.open"><summary>'+sum+'</summary>'+body+'</details>';}
function spark(rows){const w=rows.slice(1).map(r=>r.s);if(w.length<2)return '';const lo=Math.min(...w)-1,hi=Math.max(...w)+1,W=Math.max(300,w.length*12),H=70;
const pts=w.map((v,i)=>((i/(w.length-1))*(W-50)+40).toFixed(1)+','+(H-8-(v-lo)/(hi-lo)*(H-16)).toFixed(1)).join(' ');
return '<svg width="'+W+'" height="'+H+'" style="background:#0c1218;border-radius:4px"><text x="2" y="12" fill="#999" font-size="10">'+hi.toFixed(0)+' s</text><text x="2" y="'+(H-4)+'" fill="#999" font-size="10">'+lo.toFixed(0)+' s</text><polyline fill="none" stroke="#7cf" stroke-width="1.5" points="'+pts+'"/></svg><div class=small>server seconds per warm tick (tick 1 → '+(rows.length-1)+'), y range '+lo.toFixed(0)+'–'+hi.toFixed(0)+' s</div>';}
function armBlock(A){const hd=A.head;let h='<div class=arm id="arm_'+hd.arm+'"><h2>'+hd.arm+' — pod-side, teacher-forced, 64 lanes, pod test keys <span class=small>(server log updated '+esc(hd.mtime)+')</span></h2>';
h+='<div class=kpi>served ticks: '+hd.servedN+(hd.warmMedian?' · warm median '+hd.warmMedian+' s/tick (min '+hd.warmMin+', max '+hd.warmMax+')'+(hd.perLaneToken?' · '+hd.perLaneToken+' s per lane-token':''):'')+' · top-1 '+hd.top1Total+(hd.decFailedTicks.length?' · <span class=DIFF>decode failed at ticks '+hd.decFailedTicks.join(',')+'</span>':' · decode failures 0')+'</div>';
const FREE=A.lanes.length&&A.lanes[0].free;
if(FREE)h+='<div class=small>AUTOREGRESSIVE (free-running): each lane is fed its prompt, then the token the ENCRYPTED session itself produced; no stop token. The text below is what the encrypted session generated (decrypted with the pod test key). Red = the plaintext model, run on the SAME fed sequence, would have chosen another token there (hover for it): top-1 and relErr stay exact per tick because the reference follows the fed sequence. A small model decoding greedily will loop; this arm demonstrates cost, memory and decryptability over a long generation, not text quality.</div>';
else h+='<div class=small>TEACHER-FORCED: at every tick each lane is fed the TRUE next token of its own prompt, whatever the model predicted. The lane rows below are therefore NOT generated text: each column is one tick = the input token (grey) and the model\'s one-step prediction of the token after it (white; green when it equals the prompt\'s actual next token). What this arm measures: FHE prediction = plaintext-model prediction at every position (top-1, fidelity) and the per-tick server time. Generated text is in the takes below (take 9). Next-token hits on the prompt text (quality, not fidelity): '+esc(hd.nextTok||'')+'</div>';
if(hd.fatal)h+='<div class=DIFF>'+esc(hd.fatal)+'</div>';
h+=spark(A.rows);
let tb='<table class=num><tr><th>tick</th><th>server s (wall clock of the layer loop)</th><th>timer sum s (not a wall clock)</th><th>boots</th><th title="time spent encoding WEIGHT plaintexts that were missing from the pre-built store; 0 = everything came from the store">weights encoded during the tick, ms (0 = all from the store)</th><th title="small per-tick plaintexts (constants, masks): calls / misses of the GPU plaintext cache X2. A miss is encoded on the host and cached; 0 misses = nothing was encoded during the tick. The harness counters hostPtEncode* do not see the cache path, which is why they read 0 even on tick 0.">small plaintexts: calls / cache misses</th><th>uncached host encodes (count, s)</th><th>peak VRAM GB</th><th>top-1 (raw argmax)</th><th>decode rule pick = plaintext</th><th>decode failed</th><th>relErr rms median</th><th>relErr rms worst</th><th title="only meaningful when the first and the last lane are fed the SAME prompt (a consistency check used in earlier arms)">duplicate lanes agree</th></tr>';
for(const r of A.rows.slice().reverse()){tb+='<tr><td>'+r.t+'</td><td>'+r.s+'</td><td class=small>'+(r.sum??'')+'</td><td>'+r.boots+'</td><td>'+r.enc+'</td><td>'+(r.ptCalls!=null?r.ptCalls+' / '+r.ptMiss:'')+'</td><td class=small>'+(r.hn??'')+', '+(r.hs??'')+'</td><td>'+(r.vram??'')+'</td><td class='+((r.top1!=null&&r.top1<r.n)?'DIFF':'OK')+'>'+(r.top1!=null?r.top1+'/'+r.n:'…')+'</td><td class='+((r.pick!=null&&r.pick<r.n)?'DIFF':'OK')+'>'+(r.pick!=null?r.pick+'/'+r.n:'')+'</td><td>'+(r.decFailed??'')+'</td><td>'+(r.relMed!=null?r.relMed.toFixed(5):'')+'</td><td>'+(r.relWorst!=null?r.relWorst.toFixed(5):'')+'</td><td>'+(r.dup==null?'':(r.dup==='n/a'?'<span class=dash title="the first and the last lane carry different prompts in this run, so there is nothing to compare">n/a</span>':(r.dup?'yes':'<span class=DIFF>NO</span>')))+'</td></tr>';}
tb+='</table>';h+=det('d_ticks_'+hd.arm,true,'per-tick records (newest first)',tb);
if(A.pt&&A.pt.length){let pt='<table class=num><tr><th>pass</th><th>mkPt calls</th><th>cache hits</th><th>misses</th><th>skipped (cap)</th><th>dense fallback</th><th>entries</th><th>resident GB</th></tr>';
for(const p of A.pt){pt+='<tr><td>'+p.pass+'</td><td>'+p.ptCalls+'</td><td>'+p.ptCacheHits+'</td><td>'+p.ptCacheMisses+'</td><td>'+p.ptCacheSkipped+'</td><td>'+p.ptCacheDenseFallback+'</td><td>'+p.ptCacheEntries+'</td><td>'+(p.ptCacheResidentGB!=null?p.ptCacheResidentGB.toFixed(2):'')+'</td></tr>';}pt+='</table>';h+=det('d_pt_'+hd.arm,true,'plaintext cache census (X2)',pt);}
let lt='';
if(FREE){lt='<table><tr><th>lane</th><th>prompt</th><th>generated by the encrypted session</th><th>tokens</th><th>FHE = plaintext (all ticks)</th></tr>';
for(const l of A.lanes){lt+='<tr><td>'+l.lane+'</td><td class=small>'+esc(l.prompt)+'</td><td class=fhe>'+l.gen.map(p=>p[2]?esc(p[0]):'<span class=bad title="plaintext model: '+esc(p[1])+'">'+esc(p[0])+'</span>').join('')+'</td><td>'+l.genN+'</td><td class='+(l.gen.every(p=>p[2])?'OK':'DIFF')+'>'+l.top1+'</td></tr>';}
lt+='</table>';}
else{lt='<table><tr><th>lane</th><th>per tick: input token (grey) → FHE one-step prediction (green = the prompt\'s actual next token; red = differs from the plaintext model, hover for its token)</th><th>FHE = plaintext</th></tr>';
for(const l of A.lanes){lt+='<tr><td>'+l.lane+'</td><td>'+l.pieces.map(p=>'<span class=col><span class=in>'+vis(p[0])+'</span><span class="pr'+(p[3]?(p[4]?' hit':''):' bad')+'"'+(p[3]?'':' title="plaintext model: '+esc(p[2])+'"')+'>'+vis(p[1])+'</span></span>').join('')+'</td><td class='+(l.pieces.every(p=>p[3])?'OK':'DIFF')+'>'+l.top1+'</td></tr>';}
lt+='</table>';}h+=det('d_lanes_'+hd.arm,true,'per-lane tokens ('+A.lanes.length+' lanes)',lt);return h+'</div>';}
async function tick(){try{const r=await fetch('/data.json',{cache:'no-store'});const d=await r.json();
document.getElementById('hdr').innerHTML='FHE-SSM live panel — '+d.now+' — refresh 10 s — '+esc(d.pod.status)+' — jump: '+d.pod.arms.map(A=>'<a href="#arm_'+A.head.arm+'">'+A.head.arm+'</a>').join(' · ')+' · <a href="#sessions">Mac sessions ('+d.sessions.length+')</a>';
let h='';if(d.note)h+='<div class=note>'+esc(d.note)+'</div>';
if(d.prompts&&d.prompts.rows.length){const R=d.prompts.rows;const tk=R.map(r=>r.tokens);let pt='<table><tr><th>lane</th><th>class</th><th>prompt tokens</th><th>prompt</th><th>continuation by the PLAINTEXT model on this Mac (greedy emulation, not the encrypted run; the encrypted session\'s own text appears in the LONG arm below)</th></tr>';
for(const r of R)pt+='<tr><td>'+r.lane+'</td><td>'+esc(r.cls)+'</td><td>'+r.tokens+'</td><td class=small style="color:#ddd">'+esc(r.text)+'</td><td class=small style="color:#9c9">'+esc(r.out)+'</td></tr>';pt+='</table>';
h+='<div class=arm>'+det('d_prompts',false,'<b>Prompts of the long run</b> <span class=small>('+R.length+' lanes; '+Math.min(...tk)+'–'+Math.max(...tk)+' tokens each; every lane is fed its prompt one token per tick, then generates; file '+esc(d.prompts.file)+')</span>',pt)+'</div>';}
for(const A of d.pod.arms)h+=armBlock(A);if(!d.pod.arms.length)h+='<div class=arm>no pod-side arm read yet ('+esc(d.pod.status)+')</div>';
h+='<div id=sessions></div>';let first=true;
for(const s of d.sessions){let b='<table><tr><th>#</th><th>prompt</th><th>FHE tokens (decrypted on this Mac)</th><th>plaintext greedy</th><th></th></tr>';
for(const l of s.lanes){b+='<tr><td>'+l.lane+'</td><td>'+esc(l.prompt)+'</td><td class=fhe>'+esc(l.fhe)+'</td><td class=small>'+esc(l.plain)+'</td><td class='+(l.status==='OK'?'OK':(l.status==='DIFF'?'DIFF':'dash'))+'>'+l.status+'</td></tr>';}b+='</table>';
h+='<div class=sess>'+det('d_s_'+s.session,first,'<b>'+s.session+'</b> <span class=small>('+s.lanes_n+' lanes, '+(s.tick||'no tick yet')+', token-exact '+s.exact+(s.fatal?' — <span class=DIFF>'+esc(s.fatal)+'</span>':'')+', updated '+s.mtime+')</span>',b)+'</div>';first=false;}
const y=window.scrollY;document.getElementById('root').innerHTML=h;window.scrollTo(0,y);}catch(e){document.getElementById('hdr').textContent='fetch failed: '+e;}}
function vis(s){const t=(s||'').replace(/\n/g,'⏎').trim();return esc(t===''?'·':t);}
function esc(s){return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
tick();setInterval(tick,10000);</script></body></html>"""
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_GET(self):
        if self.path.startswith("/data.json"):
            dirs = sorted(glob.glob(os.path.join(a.sessions, "demo_2026*")), key=os.path.getmtime, reverse=True)[:8]
            views = [v for v in (session_view(d) for d in dirs) if v]
            try: note = open(a.note_file).read().strip()
            except Exception: note = ""
            try:   # the long run's prompt file, re-read on every refresh (author, 2026-09-19: "display our prompts"); optional class sidecar
                pl = [l.rstrip("\n") for l in open(a.arm_prompts) if l.strip() and not l.startswith("# type:")]
                try: cls = json.load(open(a.arm_prompts + ".classes.json"))
                except Exception: cls = {}
                try: outs = json.load(open(a.arm_prompts + ".outputs.json"))   # lane -> the PLAINTEXT model's greedy continuation (Mac emulation)
                except Exception: outs = {}
                prompts = {"file": os.path.relpath(a.arm_prompts, REPO), "rows": [{"lane": i, "tokens": len(tok.encode(x)), "cls": cls.get(str(i), ""), "text": x, "out": outs.get(str(i), "")} for i, x in enumerate(pl)]}
            except Exception as e: prompts = {"file": str(a.arm_prompts), "rows": [], "err": str(e)[:100]}
            body = json.dumps({"now": time.strftime("%H:%M:%SZ", time.gmtime()), "note": note, "sessions": views, "pod": POD["data"], "prompts": prompts}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            body = PAGE.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
threading.Thread(target=pod_loop, daemon=True).start()
print(f"token panel on http://localhost:{a.port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
