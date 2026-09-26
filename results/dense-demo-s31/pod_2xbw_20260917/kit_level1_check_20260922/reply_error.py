"""Per-tick error statistics of the level-1 replay (re-decrypted rows vs the rows the client recorded during the session).

Inputs: the kit's hidden.f64 / input_rows.f64 (read-only) and the re-decrypted files run_level1.sh wrote to $OUT
(tick_TTT_reqN_resp.f64, tick_TTT_reqN_req.f64; 64 x 1024 float64, lane-major).
Definitions (a = re-decrypted, b = recorded, per tick):
  rel_rms        = ||a-b||_F / ||b||_F                    over the 64 x 1024 matrix
  rel_max_global = max|a-b| / max|b|                      (compare.py's figure)
  row_rel_max    = max over lanes of max|a_l-b_l| / max|b_l|
  row_rel_rms    = min / max over lanes of ||a_l-b_l|| / ||b_l||
Usage: python3 reply_error.py <OUT> [level1_reply_error.json]
"""
import json, os, sys
import numpy as np

K = os.path.expanduser('~/Documents/fhe-ssm-backup/demo_verification_demo_20260918T074952Z')
O = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'level1_reply_error.json')


def stats(a, b):
    d = a - b
    row_rms = np.linalg.norm(d, axis=1) / np.linalg.norm(b, axis=1)
    row_max = np.max(np.abs(d), axis=1) / np.max(np.abs(b), axis=1)
    return {
        'rel_rms': float(np.linalg.norm(d) / np.linalg.norm(b)),
        'rel_max_global': float(np.max(np.abs(d)) / np.max(np.abs(b))),
        'row_rel_max': float(np.max(row_max)),
        'row_rel_rms_min': float(np.min(row_rms)),
        'row_rel_rms_max': float(np.max(row_rms)),
        'nan': int(np.isnan(a).sum()),
    }


res = {'definitions': __doc__.strip().splitlines()[3:8], 'ticks': []}
for t in range(18):
    T = f'tick_{t:03d}_req{t}'
    D = f'{K}/session_cts/{T}'
    hid = np.fromfile(f'{D}/hidden.f64', '<f8').reshape(64, 1024)
    inp = np.fromfile(f'{D}/input_rows.f64', '<f8').reshape(64, 1024)
    rd = np.fromfile(f'{O}/{T}_resp.f64', '<f8').reshape(64, 1024)
    qd = np.fromfile(f'{O}/{T}_req.f64', '<f8').reshape(64, 1024)
    res['ticks'].append({'tick': t, 'reply': stats(rd, hid), 'request': stats(qd, inp)})

for side in ('reply', 'request'):
    xs = [x[side] for x in res['ticks']]
    res[side + '_summary'] = {
        'rel_rms_min': min(x['rel_rms'] for x in xs), 'rel_rms_max': max(x['rel_rms'] for x in xs),
        'rel_max_global_max': max(x['rel_max_global'] for x in xs),
        'row_rel_max_max': max(x['row_rel_max'] for x in xs),
        'row_rel_rms_min': min(x['row_rel_rms_min'] for x in xs), 'row_rel_rms_max': max(x['row_rel_rms_max'] for x in xs),
        'nan_total': sum(x['nan'] for x in xs),
    }
json.dump(res, open(dst, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k.endswith('_summary')}, indent=1))
