import json, os, sys
sys.path.insert(0,'ml-eval'); import fhe_client as FC; tok=FC.get_tokenizer()
P='results/dense-demo-s31/pod_longrun_20260919/prompt_select/'; D='results/dense-demo-s31/pod_longrun_20260919/pod_pull/s37/serve_LONG/'
prompts=[l.rstrip('\n') for l in open(P+'longrun_prompts64_v4.txt') if l.strip()]
out=json.load(open(P+'longrun_prompts64_v4.txt.outputs.json'))
rows=json.load(open(D+'fidelity.json'))
by={}
for r in rows: by.setdefault(r['lane'],[]).append(r)
lanes=[int(x) for x in sys.argv[1:]]
res=[]
for lane in lanes:
    L=sorted(by[lane],key=lambda r:r['tick'])
    nprompt=sum(1 for r in L if r['fedSrc']=='prompt'); gen=[r['encPick'] for r in L if r.get('generating')]
    enc_text=tok.decode(gen); dis=[r['tick'] for r in L if r.get('pickAgree') is False]
    o=out[str(lane)]; mac=o if isinstance(o,str) else (o.get('text') or o.get('continuation') or json.dumps(o)[:300])
    # how far do the Mac text and the encrypted text agree token-wise? (the Mac rollout is greedy-under-rule on the plaintext model)
    mac_ids=tok.encode(mac); k=0
    while k<min(len(gen),len(mac_ids)) and gen[k]==mac_ids[k]: k+=1
    res.append({'lane':lane,'prompt':prompts[lane],'prompt_tokens':nprompt,'generated_tokens':len(gen),'pick_disagreements_at_ticks':dis,'mac_plaintext':mac,'encrypted_run':enc_text,'identical_leading_tokens':k})
    print(f"=== lane {lane} | prompt tokens {nprompt} | generated {len(gen)} | pick disagreements at ticks {dis} | identical leading tokens vs the Mac rollout: {k}")
    print("PROMPT:", prompts[lane]); print("MAC   :", repr(mac[:600])); print("POD   :", repr(enc_text[:600])); print()
json.dump(res, open(sys.argv[0].replace('longrun_examples.py','longrun_examples.json'),'w'), indent=1, ensure_ascii=False)
