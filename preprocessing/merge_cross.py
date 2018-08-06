import pandas as pd
import numpy as np
if __name__ == '__main__':
    df = pd.read_pickle('./cross_stacking_0.pkl')
    for i in range(1, 6):
        temp = pd.read_pickle('./cross_stacking_{}.pkl'.format(i))
        df = pd.merge(df, temp, 'outer', ['uid', 'pid'])
    print(df)
