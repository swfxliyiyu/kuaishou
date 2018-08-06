import sys

import pandas as pd

if __name__ == '__main__':
    fin = sys.argv[1]
    fout = sys.argv[2]
    preds = pd.read_pickle(fin)
    test = pd.read_pickle('../data/test_interaction.pkl')
    test.columns = ['uid', 'pid', 0, 1]
    pd.merge(test, preds, 'left', ['uid', 'pid'])[['uid', 'pid', 'preds']].to_csv(
        fout, header=None, index=None, sep='\t', float_format='%.6f')


