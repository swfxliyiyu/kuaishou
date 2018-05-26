import multiprocessing

import pandas as pd
import time

from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
import numpy as np

N_JOBS = multiprocessing.cpu_count()
print('N_JOBS:', N_JOBS)


def dec_series_func(ser, fuc, arg=None):
    if arg is None:
        return ser.apply(fuc).tolist()
    else:
        return ser.apply(fuc, args=(arg,)).tolist()


def yield_data(ser, num):
    count = 0
    for s in np.array_split(ser, num):
        print(count / num)
        count += 1
        yield s


def series_map(ser, func, arg=None):
    lst = Parallel(n_jobs=-2, backend='multiprocessing')(
        delayed(dec_series_func)(ele, func, arg) for ele in yield_data(ser, N_JOBS * 100))
    return np.concatenate(lst)


def func(lst):
    return np.unique(lst, axis=0)


def func2(lst):
    return np.unique([int(x) for x in lst])


def func3(lst):
    return np.asarray(sorted(lst.tolist()))


if __name__ == '__main__':
    df = pd.read_pickle('../data/interaction_features_1.pkl')
    print('loaded data...')
    print(df.columns)

    # 添加连续特征
    time_redc = pd.read_pickle('../data/time_redc.pkl')
    time_redc['time_redc'] = time_redc['time_redc'].replace({-1: time_redc['time_redc'].mean()})
    pre_5min = pd.read_pickle('../data/user_pre5min_cnt.pkl')
    photo_cnt = pd.read_pickle('../data/batch_photo_cnt.pkl')
    for df2 in [time_redc, pre_5min, photo_cnt]:
        df2['uid'] = df2['user_id']
        df2['pid'] = df2['photo_id']
        df2 = df2.drop(columns=['instance_id', 'user_id', 'photo_id'])
        for col in df2:
            if col not in ['uid', 'pid']:
                df2[col + '_N'] = df2[col]
                df2 = df2.drop(columns=[col])
        df = pd.merge(df, df2, how='left', on=['uid', 'pid'])

    del df['pids']
    encoder = LabelEncoder()
    df['user_indices'] = encoder.fit_transform(df['uid'])
    # face_cols = ['face_num', 'face_max_percent', 'face_whole_percent', 'face_male_num', 'face_famale_num',
    #              'face_gender_mix', 'face_ave_age', 'face_max_appear', 'face_min_appear', 'face_ave_appear']
    df['hour_01'] = df['time'].apply(lambda t: time.strftime('%H', time.localtime(t // 1000))).astype('int')
    # # df['recent_words'] = df['recent_words'].apply(lambda lst: np.array(sorted(lst.tolist())))
    # df['recent_words'] = series_map(df['recent_words'], func)
    for col in df:
        if '_01' in col:
            df[col] = encoder.fit_transform(df[col])
    tr_df = df[(df['is_test'] == False) & (df['is_val'] == False)]
    val_df = df[df['is_val'] == True]
    te_df = df[df['is_test'] == True]
    for d in [tr_df, te_df, val_df]:
        del d['is_test'], d['is_val']
    # del tr_df['uid']
    tr_df.to_pickle('../data/train_data.pkl')
    print(tr_df)
    print(tr_df.columns)
    val_df.to_pickle('../data/val_data.pkl')
    print(val_df)
    print(val_df.columns)
    te_df.to_pickle('../data/test_data.pkl')
    print(te_df)
    print(te_df.columns)

    # tr_df = pd.read_pickle('../data/train_data.pkl')
    # val_df = pd.read_pickle('../data/val_data.pkl')
    # te_df = pd.read_pickle('../data/test_data.pkl')
    #
    # tr_df.to_pickle('../data/train_data_1.pkl')
    # print(tr_df)
    # print(tr_df.columns)
    # val_df.to_pickle('../data/val_data_1.pkl')
    # print(val_df)
    # print(val_df.columns)
    # te_df.to_pickle('../data/test_data_1.pkl')
    # print(te_df)
    # print(te_df.columns)
