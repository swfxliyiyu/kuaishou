import os
import pandas as pd

if __name__ == '__main__':
    time_redc_level = pd.read_pickle('../data/ctx_stacking_rf.pkl')
    knn_stacking = pd.read_pickle('../data/ctx_stacking_knn.pkl')
    # time_redc_level['uid'] = time_redc_level['user_id']
    # time_redc_level['pid'] = time_redc_level['photo_id']
    for file in ['train_data.pkl', 'test_data.pkl', 'val_data.pkl']:
        path = os.path.join('../data', file)
        data = pd.read_pickle(path)
        df = pd.merge(data, time_redc_level, how='left', on=['pid', 'uid'])
        df = pd.merge(df, knn_stacking, how='left', on=['pid', 'uid'])
        print(df)
        print(path, 'merged...')
        df.to_pickle(path)
        print(path, 'pickled...')
