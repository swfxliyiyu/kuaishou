import os

import pandas as pd

dir_path = '../data'

if __name__ == '__main__':
    file_lst = os.listdir(dir_path)
    for file_name in file_lst:
        prefix, suffix = file_name.split('.')
        if suffix == 'txt':
            pkl_name = '.'.join([prefix, 'pkl'])
            if pkl_name not in file_lst:
                df = pd.read_csv(os.path.join(dir_path, file_name), sep='\t', header=None)
                print(df)
                print(pkl_name)
                df.to_pickle(os.path.join(dir_path, pkl_name))
