# coding=utf-8
from __future__ import print_function, division
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

def get_user_like(data):
    features = [np.array(feat.tolist()) for feat in [data['context'], data['face_cols_num'], data['topics']]]
    data['concated_feature'] = [np.array(lst) for lst in np.concatenate(features, axis=1).tolist()]
    user_like_mean = [[uid, np.array(value['concated_feature'].tolist()).mean(axis=0)] for uid, value in data.groupby(['uid'])]
    user_like_mean = pd.DataFrame(user_like_mean, columns=['uid', 'user_like_mean'])
    del data['concated_feature']
    data = pd.merge(data, user_like_mean, 'left', ['uid'])
    return data

if __name__ == '__main__':
    df = pd.read_pickle('../data/interaction_features_1.pkl')
    print('loaded data...')
    print(df.columns)

    # 添加连续特征

    df_ctx = pd.read_pickle('../data/context_feature.pkl')
    df = pd.merge(df, df_ctx, 'left', left_on=['uid','pid'], right_on=['user_id', 'photo_id'])
    print('context feature concated...')

    # 添加text_lda
    df_text_lda = pd.read_pickle('../data/text_lda_6.pkl')
    df = pd.merge(df, df_text_lda, 'left', on=['pid'])
    empty = np.zeros(shape=[6])
    df['topics'] = df['topics'].apply(lambda lst: empty if pd.isna(lst) is True else lst)
    print('text lda concated...')

    # 求用户平均偏好
    df = get_user_like(df)
    print(df['user_like_mean'])
    print('get user_like_mean...')


    encoder = LabelEncoder()
    df['user_indices'] = encoder.fit_transform(df['uid'])
    df['photo_indices'] = encoder.fit_transform(df['pid'])
    # df['hour_01'] = df['time'].apply(lambda t: time.strftime('%H', time.localtime(t // 1000))).astype('int')
    # for col in df:
    #     if 'hour_01' in col:
    #         df[col] = encoder.fit_transform(df[col])
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


    # text_df = pd.read_pickle('../data/text_lda_6.pkl')
    # tr_df = pd.read_pickle('../data/train_data.pkl')
    # val_df = pd.read_pickle('../data/val_data.pkl')
    # te_df = pd.read_pickle('../data/test_data.pkl')
    # empty = np.zeros(shape=[6])
    # for df in [text_df, tr_df, val_df, te_df]:
    #     df['topics'] = df['topics'].apply(lambda lst: empty if pd.isna(lst) is True else lst)
    #
    # text_df.to_pickle('../data/text_lda_6.pkl')
    # print(text_df)
    # print(text_df.columns)
    # tr_df.to_pickle('../data/train_data.pkl')
    # print(tr_df)
    # print(tr_df.columns)
    # val_df.to_pickle('../data/val_data.pkl')
    # print(val_df)
    # print(val_df.columns)
    # te_df.to_pickle('../data/test_data.pkl')
    # print(te_df)
    # print(te_df.columns)
