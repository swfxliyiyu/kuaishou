import sys

import numpy as np
import pandas as pd


def sigmoid(x):
    res = 1 / (1 + np.e ** (-x))
    return res


def sigmoid_ver(x):
    res = np.log(x / (1 - x))
    return res


def merge(df1, df2, df3):
    df1['ver'] = df1['click'].apply(sigmoid_ver)
    df2['ver'] = df2['click'].apply(sigmoid_ver)
    # df3['ver'] = df3['click'].apply(sigmoid_ver)
    # df1['res'] = (df1['ver'] + df2['ver'] + df3['ver']) / 3
    df1['res'] = (df1['ver'] + df2['ver']) / 2
    print(df1)
    df1['res'] = df1['res'].apply(sigmoid)
    print(df1[['uid', 'pid', 'res']])
    print(df1[['uid', 'pid', 'res']].describe())
    df1[['uid', 'pid', 'res']].to_csv('merge_nn8104_lgb.txt', sep='\t', header=None, index=False,
                                      float_format='%.6f')

if __name__ == '__main__':
    pkl_path = sys.argv[1]
    txt_path = sys.argv[2]
    df1 = pd.read_pickle(pkl_path)
    df2 = pd.read_csv(txt_path, sep='\t', header=None)
    for df in [df1, df2]:
        df.columns = ['uid', 'pid', 'click']
    merge(df1, df2, None)