import numpy as np
import pickle
import sys

import pandas as pd

if __name__ == '__main__':
    emb_path = sys.argv[1]
    lb_path = sys.argv[2]
    out_path = sys.argv[3]

    voc_size = 152092

    with open(emb_path, 'rb') as f:
        emb = pickle.load(f)
        emb = [np.array(lst, np.float32) for lst in emb.tolist()]
    with open(lb_path, 'rb') as f:
        lb = pickle.load(f)
    df1 = pd.DataFrame({'wid':lb, 'emb': emb})
    df1['wid'] = df1['wid'].astype('int32')
    wids = set(df1['wid'].tolist())

    other_wids = []
    other_embs = []
    for i in range(voc_size+1):
        if i not in wids:
            other_wids.append(i)
            other_embs.append(np.random.normal(loc=0.0, scale=0.01, size=[96]))
    df2 = pd.DataFrame({'wid':other_wids, 'emb': other_embs})
    df = pd.concat([df1, df2], sort=False, ignore_index=True).sort_values(['wid'])
    print(df)
    df.to_pickle(out_path)
