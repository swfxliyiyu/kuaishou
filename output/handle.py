import pandas as pd

if __name__ == '__main__':
    preds = pd.read_pickle('../output/facenum_reg04_usremb_512_0771.pkl')
    test = pd.read_pickle('../data/test_interaction.pkl')
    test.columns = ['uid', 'pid', 0, 1]
    pd.merge(test, preds, 'left', ['uid', 'pid'])[['uid', 'pid', 'preds']].to_csv('../output/facenum_reg04_usremb_512_0771.txt', header=None, index=None, sep='\t', float_format='%.6f')

