import numpy as np
import pandas as pd


def read_bine_data(bine_path):
    dic = {'uid': [], 'user_emb': []}
    with open(bine_path, 'r') as fin:
        for line in fin:
            fields = line.strip().split(' ')
            uid = fields[0][1:]
            emb = np.array([float(x) for x in fields[1:]])
            dic['uid'].append(uid)
            dic['user_emb'].append(emb)
    return pd.DataFrame(dic)

if __name__ == '__main__':
    bine_path = '../data/vector_u.dat'
    df_bine = read_bine_data(bine_path)
    df_test = pd.read_pickle('../data/test_data.pkl')
    df = pd.merge(df_bine, df_test, 'left', 'uid')
    df[['uid', 'user_indices', 'user_emb']].to_pickle('../model/bine_emb.pkl')