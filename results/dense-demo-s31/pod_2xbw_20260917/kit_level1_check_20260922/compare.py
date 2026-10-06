import json, os, sys, numpy as np
sys.path.insert(0, 'ml-eval'); import fhe_client as FC
K=os.path.expanduser('~/Documents/fhe-ssm-backup/demo_verification_demo_20260918T074952Z'); O=sys.argv[1]
ART=os.path.expanduser('~/Documents/fhe-ssm-backup/mac_art')
head,V,d=FC.head_matrix(ART,'pbd430a'); head=np.asarray(head)
tr=json.load(open(K+'/records/transcript.json'))
res={'ticks':[], 'resp_max_rel':0, 'req_max_rel':0, 'token_matches':0, 'token_total':0, 'emb_max_rel':0}
for t in range(len(tr)):
    T=f'tick_{t:03d}_req{t}'; D=f'{K}/session_cts/{T}'
    hid=np.fromfile(f'{D}/hidden.f64','<f8').reshape(64,1024); inp=np.fromfile(f'{D}/input_rows.f64','<f8').reshape(64,1024)
    rd=np.fromfile(f'{O}/{T}_resp.f64','<f8').reshape(64,1024); qd=np.fromfile(f'{O}/{T}_req.f64','<f8').reshape(64,1024)
    r1=float(np.max(np.abs(rd-hid))/np.max(np.abs(hid))); r2=float(np.max(np.abs(qd-inp))/np.max(np.abs(inp)))
    ids=tr[t]['inputs']; emb=head[np.array(ids)]; r3=float(np.max(np.abs(inp-emb))/np.max(np.abs(emb)))
    logits=rd@head.T; am=np.argmax(logits,axis=1)
    picked=tr[t]['picked']; m=n=0
    for l in range(64):
        if picked[l] is not None: n+=1; m+= int(am[l]==picked[l])
    res['ticks'].append({'tick':t,'resp_max_rel_vs_recorded':r1,'req_max_rel_vs_recorded':r2,'req_vs_embedding_max_rel':r3,'picks_matched':m,'picks_recorded':n})
    res['resp_max_rel']=max(res['resp_max_rel'],r1); res['req_max_rel']=max(res['req_max_rel'],r2); res['emb_max_rel']=max(res['emb_max_rel'],r3); res['token_matches']+=m; res['token_total']+=n
print(json.dumps({k:v for k,v in res.items() if k!='ticks'}))
for x in res['ticks'][:3]+res['ticks'][-1:]: print(x)
json.dump(res, open(f'{O}/level1_compare.json','w'), indent=1)
