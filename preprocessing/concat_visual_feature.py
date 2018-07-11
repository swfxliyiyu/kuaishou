import sys
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # tr_visual = pd.read_pickle('./visual_feature_train.pkl')
    # te_visual = pd.read_pickle('./visual_feature_test.pkl')
    path = sys.argv[1]
    outpath = sys.argv[2]
    dfs = []
    for f in os.listdir(path):
        file_path = os.path.join(path, f)
        dfs.append(pd.read_pickle(file_path))
        print('loaded file:', file_path)

    visual = pd.concat(dfs, ignore_index=True)
    visual.columns = ['pid', 'visual']
    v1, v2 = np.array_split(visual, 2)
    print('concated.')
    print(visual.describe())
    v1.to_pickle(outpath+'_1.pkl')
    v2.to_pickle(outpath+'_2.pkl')
