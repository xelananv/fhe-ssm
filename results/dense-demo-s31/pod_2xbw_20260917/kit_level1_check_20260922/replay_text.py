import json, os, sys, numpy as np
sys.path.insert(0, 'ml-eval'); import fhe_client as FC
K=os.path.expanduser('~/Documents/fhe-ssm-backup/demo_verification_demo_20260918T074952Z'); O=sys.argv[1]
ART=os.path.expanduser('~/Documents/fhe-ssm-backup/mac_art'); tok=FC.get_tokenizer()
head,V,d=FC.head_matrix(ART,'pbd430a'); head=np.asarray(head)
tr=json.load(open(K+'/records/transcript.json')); tk=json.load(open(K+'/records/tokens.json'))
gen=[[] for _ in range(64)]; rec=[[] for _ in range(64)]
for t in range(len(tr)):
    rd=np.fromfile(f'{O}/tick_{t:03d}_req{t}_resp.f64','<f8').reshape(64,1024); am=np.argmax(rd@head.T,axis=1)
    for l in range(64):
        if tr[t]['picked'][l] is not None: gen[l].append(int(am[l])); rec[l].append(int(tr[t]['picked'][l]))
rows=[]
for l in range(64):
    rows.append({'lane':l,'prompt':tk['prompts'][l],'replayed_ids':gen[l],'recorded_ids':rec[l],'recorded_generated_lanes':tk['generated_lanes'][l],
                 'replayed_text':tok.decode(gen[l]),'identical_to_recorded':gen[l]==rec[l]==tk['generated_lanes'][l]})
json.dump(rows,open(f'{O}/replayed_lanes.json','w'),indent=1)
print('lanes identical to the record:', sum(r['identical_to_recorded'] for r in rows), '/ 64; generated tokens:', sum(len(r['replayed_ids']) for r in rows))
for r in rows: print(f"{r['lane']:2d} | {r['prompt']} | {r['replayed_text']!r}")
