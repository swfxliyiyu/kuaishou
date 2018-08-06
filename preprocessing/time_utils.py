# -*- coding: utf-8 -*-

import time
from collections import defaultdict

import pandas as pd
import numpy as np
from joblib import Parallel, delayed


def _timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value // 1000)
    dt = time.strftime(format, value)
    return dt


def trans_time(df_data):
    df_data['realtime'] = df_data['time'].apply(_timestamp_datetime)
    df_data['realtime'] = pd.to_datetime(df_data['realtime'])
    df_data['day'] = df_data['realtime'].dt.day
    df_data['hour'] = df_data['realtime'].dt.hour
    df_data['minute'] = df_data['realtime'].dt.minute
    return df_data


def yield_uid(df_data):
    total = df_data.shape[0]
    temp = df_data[['user_id', 'photo_id', 'realtime']]
    temp = temp.sort_values(['user_id', 'realtime'], ascending=[1, 1])
    temp.index = range(temp.shape[0])
    index = 0
    last_uid = temp.loc[0, 'user_id']
    for i in temp.index:
        if temp.iloc[i, 0] != last_uid:
            print(temp.iloc[i])
            print('===========', index / total, last_uid, index, i)
            yield temp.iloc[index:i, :]
            print('yield', index, i - 1)
            index = i
            last_uid = temp.iloc[i, 0]
    yield temp.iloc[index:, :]
    print('finished yield')

def handle_uid(df_uid):
    df_uid['start_time'] = df_uid['realtime'] - pd.Timedelta(minutes=8)
    df_uid['end_time'] = df_uid['realtime'] + pd.Timedelta(minutes=8)
    df_uid['start_index'] = df_uid[['start_time']].apply(
        lambda x: np.searchsorted(df_uid['realtime'], x['start_time'])[0], axis=1, result_type='reduce')
    df_uid['now_index'] = range(df_uid.shape[0])
    df_uid['end_index'] = df_uid[['end_time']].apply(
        lambda x: np.searchsorted(df_uid['realtime'], x['end_time'])[0], axis=1, result_type='reduce')
    df_uid['user_pre8min_cnt'] = df_uid['now_index'] - df_uid['start_index']
    df_uid['user_aft8min_cnt'] = df_uid['end_index'] - df_uid['now_index']
    return df_uid[['user_id', 'photo_id', 'user_pre8min_cnt', 'user_aft8min_cnt']]


def parallel_user_cnt_pre8min(df_data):
    lst = Parallel(n_jobs=-2, backend='multiprocessing')(
        delayed(handle_uid)(df_u) for df_u in yield_uid(df_data))
    result = pd.concat(lst)
    print('user_num: %s' % result['user_id'].nunique())
    return result


def user_cnt_pre8min(df_data):
    # def get_index(x):
    #     index = x[0] + np.searchsorted(temp.realtime[x[0]:x[1]], x[2])
    #     return index
    temp = df_data[['user_id', 'photo_id', 'realtime']]
    temp = temp.sort_values(['user_id', 'realtime'], ascending=[1, 1])
    temp['start_time'] = temp['realtime'] - pd.Timedelta(minutes=8)
    temp['end_time'] = temp['realtime'] + pd.Timedelta(minutes=8)
    left, right = np.searchsorted(temp.user_id, temp.user_id, side='left'), np.searchsorted(temp['user_id'],
                                                                                            temp['user_id'],
                                                                                            side='right')
    temp['left_index'] = left
    temp['right_index'] = right

    temp['start_index'] = temp[['left_index', 'right_index', 'start_time']].apply(
        lambda x: x['left_index'] + np.searchsorted(temp.realtime[x['left_index']:x['right_index']], x['start_time'])[0],
        axis=1, result_type='reduce')
    temp['now_index'] = range(temp.shape[0])
    temp['user_pre8min_cnt'] = temp['now_index'] - temp['start_index']
    temp['end_index'] = temp[['left_index', 'right_index', 'end_time']].apply(
        lambda x: x['left_index'] + np.searchsorted(temp.realtime[x['left_index']:x['right_index']], x['end_time'])[0],
        axis=1, result_type='reduce')
    temp['user_aft8min_cnt'] = temp['end_index'] - temp['now_index']
    temp = temp.drop(columns=['start_time', 'left_index', 'right_index', 'start_index', 'now_index', 'end_index'])
    df_data = pd.merge(df_data, temp, 'left', on=['user_id', 'photo_id', 'realtime'])
    # df['A'].iloc[row['start_index']:row['end_index'] + 1].sum()
    return df_data


def user_batch_cnt_pre8min(df_data):
    def get_index(x):
        try:
            index = x['left_index'] + np.searchsorted(temp.realtime[x['left_index']:x['right_index']], x['start_time'])[0]
        except Exception as e:
            print(e)
            print(x)
            print(temp.realtime[x['left_index']:x['right_index']])
            return int(x['left_index'] + (x['left_index'] + x['right_index']) / 2)
        return index

    temp = df_data.drop_duplicates(['user_id', 'realtime']).sort_values(['user_id', 'realtime'], ascending=[1, 1])
    temp['start_time'] = temp['realtime'] - pd.Timedelta(minutes=8)
    temp['end_time'] = temp['realtime'] + pd.Timedelta(minutes=8)
    temp['left_index'], temp['right_index'] = np.searchsorted(temp['user_id'], temp['user_id'],
                                                              side='left'), np.searchsorted(temp.user_id, temp.user_id,
                                                                                            side='right')
    # temp['start_index'] = temp[['left_index', 'right_index', 'start_time']].apply(
    #     lambda x: x['left_index'] + np.searchsorted(temp.realtime[x['left_index']:x['right_index']], x['start_time']), axis=1,
    #     result_type='reduce')
    temp['start_index'] = temp[['left_index', 'right_index', 'start_time']].apply(get_index, axis=1, result_type='reduce')
    temp['now_index'] = range(temp.shape[0])
    temp['user_batch_cnt_pre8min'] = temp['now_index'] - temp['start_index']
    temp['end_index'] = temp[['left_index', 'right_index', 'end_time']].apply(
        lambda x: x['left_index'] + np.searchsorted(temp.realtime[x['left_index']:x['right_index']], x['end_time'])[0],
        axis=1, result_type='reduce')
    temp['user_batch_cnt_aft8min'] = temp['end_index'] - temp['now_index']
    temp = temp.drop(columns=['start_time', 'left_index', 'right_index', 'start_index', 'end_index', 'now_index'])
    del temp['photo_id']
    df_data = pd.merge(df_data, temp, 'left', on=['user_id', 'realtime'])
    return df_data


def yield_pid(df_data):
    total = df_data.shape[0]
    temp = df_data[['user_id', 'photo_id', 'realtime']]
    temp = temp.sort_values(['photo_id', 'realtime'], ascending=[1, 1])
    temp.index = range(temp.shape[0])
    index = 0
    cnt = 0
    last_pid = temp.iloc[0, 1]
    for i in temp.index:
        if temp.iloc[i, 1] != last_pid:
            cnt += 1
            if cnt % 2000 == 0:
                print('===========', index / total, last_pid)
            if i - 1 > index:
                yield temp.iloc[index:i, :]
                if cnt % 2000 == 0:
                    print('yield', index, i - 1)
            index = i
            last_pid = temp.iloc[i, 1]
    yield temp.iloc[index:, :]
    print('finished yield')


def handle_pid(df_p):
    df_p['start_time'] = df_p['realtime'] - pd.Timedelta(days=0.5)
    df_p['end_time'] = df_p['realtime'] + pd.Timedelta(days=0.5)
    df_p['start_index'] = df_p[['start_time']].apply(
        lambda x: np.searchsorted(df_p['realtime'], x['start_time'])[0], axis=1, result_type='reduce')
    df_p['now_index'] = range(df_p.shape[0])
    df_p['end_index'] = df_p[['end_time']].apply(
        lambda x: np.searchsorted(df_p['realtime'], x['end_time'])[0], axis=1, result_type='reduce')
    df_p['photo_batch_cnt_pre1day'] = df_p['now_index'] - df_p['start_index']
    df_p['photo_batch_cnt_aft1day'] = df_p['end_index'] - df_p['now_index']
    return df_p[['user_id', 'photo_id', 'photo_batch_cnt_pre1day', 'photo_batch_cnt_aft1day']]


def parallel_photo_batch_cnt_pre1day(df_data):
    lst = Parallel(n_jobs=-2, backend='multiprocessing')(
        delayed(handle_pid)(df_p) for df_p in yield_pid(df_data))
    result = pd.concat(lst)
    result = pd.merge(df_data, result, how='left', on=['user_id', 'photo_id'])
    result = result.fillna(0)
    print(result)
    print('photo_num: %s' % result['photo_id'].nunique())
    return result


def photo_batch_cnt_pre8min(df_data):
    temp = df_data.sort_values(['photo_id', 'realtime'], ascending=[1, 1])
    temp['start_time'] = temp['realtime'] - pd.Timedelta(minutes=8)
    temp['end_time'] = temp['realtime'] + pd.Timedelta(minutes=8)
    temp['left_index'], temp['right_index'] = np.searchsorted(temp['photo_id'], temp['photo_id'],
                                                              side='left'), np.searchsorted(temp['photo_id'],
                                                                                            temp['photo_id'],
                                                                                            side='right')

    temp['start_index'] = temp[['left_index', 'right_index', 'start_time']].apply(
        lambda x: x['left_index'] + np.searchsorted(temp.realtime[x['left_index']:x['right_index']], x['start_time'])[0],
        axis=1, result_type='reduce')
    temp['now_index'] = range(temp.shape[0])
    temp['photo_batch_cnt_pre8min'] = temp['now_index'] - temp['start_index']
    temp['end_index'] = temp[['left_index', 'right_index', 'end_time']].apply(
        lambda x: x['left_index'] + np.searchsorted(temp.realtime[x['left_index']:x['right_index']], x['end_time'])[0],
        axis=1, result_type='reduce')
    temp['photo_batch_cnt_aft8min'] = temp['end_index'] - temp['now_index']
    temp = temp.drop(columns=['start_time', 'left_index', 'right_index', 'start_index', 'end_index', 'now_index'])
    df_data = pd.merge(df_data, temp, 'left', on=['user_id', 'photo_id', 'realtime'])
    return df_data


# def cal_time_reduc_user_cnt(df_data):
#     train_origin = df_data
#     train1 = train_origin[['time', 'user_id', 'instance_id', 'photo_id']]
#     train1 = train1.sort_values(['user_id', 'photo_id', 'time'], ascending=[1, 1, 1])

#     rnColumn = train1.groupby(['user_id', 'photo_id']).rank(method='min')
#     train1['rnnn'] = rnColumn['time']
#     train1['rnnn_1'] = rnColumn['time'] - 1
#     train2 = train1.merge(train1, how='left', left_on=['user_id', 'photo_id', 'rnnn_1'],
#                           right_on=['user_id', 'photo_id', 'rnnn'])

#     train2['time_redc_user_cnt'] = train2['time_x'] - train2['time_y']
#     train2 = train2.fillna(-1).astype('int64')
#     train2 = train2.rename(columns={'instance_id_x': 'instance_id'})
#     train2 = train2.rename(columns={'photo_id_x': 'photo_id'})
#     train2 = train2.rename(columns={'time_x': 'time'})

#     train2 = train2.drop([  # 'rnnn_x','rn_y','rn_1_x','rn_1_y',
#         'time_y', 'instance_id_y'], axis=1)
#     df_data = pd.merge(train_origin, train2, on=['instance_id', 'photo_id', 'user_id', 'time'], how='left')
#     return df_data


def cal_time_reduc(df_data):
    train1 = df_data[['time', 'user_id', 'photo_id']]
    train1 = train1.sort_values(['user_id', 'time'], ascending=[1, 1])

    rnColumn = train1.groupby('user_id').rank(method='min')
    train1['rn'] = rnColumn['time']
    batch_photo_cnt = train1.groupby(['rn', 'user_id'], as_index=False)['time'].agg({'batch_photo_cnt': 'count'})
    train1 = pd.merge(train1, batch_photo_cnt, on=['rn', 'user_id'], how='left')
    train1['rn_1'] = train1['rn'] + train1['batch_photo_cnt']
    train2 = train1.merge(train1, how='left', left_on=['user_id', 'rn_1'], right_on=['user_id', 'rn'])
    train2['time_redc'] = train2['time_y'] - train2['time_x']
    train2['time_redc'] = train2['time_redc'].apply(lambda tr: tr / 1000)
    train2 = train2.fillna(-1).astype('int64')
    train2 = train2.rename(columns={'photo_id_x': 'photo_id'})
    train2 = train2.rename(columns={'time_x': 'time'})

    user_cnt_max = train2.groupby(['user_id']).max()['rn_x'].reset_index().rename(columns={'rn_x': 'user_cnt_max'})
    train2 = pd.merge(train2, user_cnt_max, 'left', on=['user_id'])
    train2['user_remain_cnt'] = train2['user_cnt_max'] - train2['rn_x']
    train2.drop(['user_cnt_max'], inplace=True, axis=1)
    df_data = pd.merge(df_data, train2, on=['photo_id', 'user_id'], how='left')
    print(df_data)
    return df_data


def next_time_diff(df_data):
    train_origin = df_data
    train1 = train_origin[['time', 'user_id', 'photo_id']]
    train1 = train1.drop_duplicates(['user_id', 'time'])
    print(train1)
    train1 = train1.sort_values(['user_id', 'time'], ascending=[1, 1])

    train1['rn'] = train1.groupby(['user_id'])['time'].rank()
    print('rnColumn: ', train1['rn'])
    # batch_photo_cnt = train1.groupby(['rn', 'user_id'], as_index=False)['time'].agg({'batch_photo_cnt': 'count'})
    # train1 = pd.merge(train1, batch_photo_cnt, on=['user_id', 'rn'], how='left')      # 这里就不需要再算当前批次的视频数量了
    train1['rn_1'] = train1['rn'] - 1
    train2 = pd.merge(train1, train1, how='left', left_on=['user_id', 'rn'], right_on=['user_id', 'rn_1'])
    print(train2)
    print('打印 instance_id是否有重复', train2.shape[0], train2[['user_id', 'photo_id_x']].nunique())
    train2.drop_duplicates(['user_id', 'photo_id_x'], keep='first', inplace=True)
    train2['next_time_diff'] = train2['time_y'] - train2['time_x']
    train2['next_time_diff'] = train2['next_time_diff'].apply(lambda tr: tr / 1000)
    print('打印时间差这一列', train2['next_time_diff'])
    train2 = train2.fillna(2)
    train2 = train2.rename(columns={'time_x': 'time'})
    df_data = pd.merge(train_origin, train2, on=['user_id', 'time'], how='left')
    # df_data = df_data.rename(columns={ 'photo_id': 'photo_id', 'time_x': 'time'})
    df_data['next_time_diff'] = df_data['next_time_diff'].astype('float32')
    print(df_data)
    return df_data

def pre_time_diff(df_data):
    train_origin = df_data
    train1 = train_origin[['time', 'user_id', 'photo_id']]
    train1 = train1.drop_duplicates(['user_id', 'time'])
    print(train1)
    train1 = train1.sort_values(['user_id', 'time'], ascending=[1, 1])

    train1['rn'] = train1.groupby(['user_id'])['time'].rank()
    print('rnColumn: ', train1['rn'])
    # batch_photo_cnt = train1.groupby(['rn', 'user_id'], as_index=False)['time'].agg({'batch_photo_cnt': 'count'})
    # train1 = pd.merge(train1, batch_photo_cnt, on=['user_id', 'rn'], how='left')      # 这里就不需要再算当前批次的视频数量了
    train1['rn_1'] = train1['rn'] + 1
    train2 = pd.merge(train1, train1, how='left', left_on=['user_id', 'rn'], right_on=['user_id', 'rn_1'])
    print(train2)
    print('打印 instance_id是否有重复', train2.shape[0], train2[['user_id', 'photo_id_x']].nunique())
    train2.drop_duplicates(['user_id', 'photo_id_x'], keep='first', inplace=True)
    train2['pre_time_diff'] = train2['time_x'] - train2['time_y']
    train2['pre_time_diff'] = train2['pre_time_diff'].apply(lambda tr: tr / 1000)
    print('打印时间差这一列', train2['pre_time_diff'])
    train2 = train2.fillna(2)
    train2 = train2.rename(columns={'time_x': 'time'})
    df_data = pd.merge(train_origin, train2, on=['user_id', 'time'], how='left')
    # df_data = df_data.rename(columns={ 'photo_id': 'photo_id', 'time_x': 'time'})
    print(df_data)
    return df_data


def cal_batch_photo_cnt(df_data):
    train = df_data[['time', 'user_id', 'photo_id']]
    train = train.sort_values(['user_id', 'time'], ascending=[1, 1])
    rnColumn = train.groupby('user_id').rank(method='min')
    train['rn'] = rnColumn['time']
    batch_photo_cnt = train.groupby(['rn', 'user_id'], as_index=False)['time'].agg({'batch_photo_cnt': 'count'})
    train = pd.merge(train, batch_photo_cnt, on=['rn', 'user_id'], how='left')
    df_data = pd.merge(df_data, train[['user_id', 'photo_id', 'batch_photo_cnt']], on=['user_id', 'photo_id'],
                       how='left')
    return df_data



def user_photo_total_cnt(df_data):
    photo_cnt = df_data['photo_id'].value_counts()
    photo_cnt = pd.DataFrame({'photo_total_cnt': photo_cnt, 'photo_id': photo_cnt.index})
    print('photo_cnt:\n', photo_cnt)
    df_data = pd.merge(df_data, photo_cnt, how='left', on=['photo_id'])
    print('photo_total_cnt:\n', df_data['photo_total_cnt'])
    user_batch_cnt = df_data.drop_duplicates(['user_id', 'time'])['user_id'].value_counts()
    user_batch_cnt = pd.DataFrame({'user_id': user_batch_cnt.index, 'user_total_batch_cnt': user_batch_cnt})
    df_data = pd.merge(df_data, user_batch_cnt, 'left', ['user_id'])
    print('user_total_batch_cnt:\n', df_data['user_total_batch_cnt'])
    user_cnt = df_data['user_id'].value_counts()
    user_cnt = pd.DataFrame({'user_id': user_cnt.index, 'user_total_cnt': user_cnt})
    df_data = pd.merge(df_data, user_cnt, 'left', ['user_id'])
    print(df_data)
    return df_data


if __name__ == '__main__':
    train_inter = pd.read_pickle('../data/train_interaction.pkl')
    train_inter.columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
    test_inter = pd.read_pickle('../data/test_interaction.pkl')
    test_inter.columns = ['user_id', 'photo_id', 'time', 'duration_time']
    inter = pd.concat([train_inter, test_inter], ignore_index=True, sort=False)
    # inter = inter.drop(columns=['like', 'follow', 'playing_time', 'duration_time'])
    inter = trans_time(inter)

    # inter = parallel_user_cnt_pre8min(inter)
    # inter[['user_id','photo_id','user_pre8min_cnt','user_aft8min_cnt']].to_pickle('../data/context_feature/user_cnt_pre8min.pkl')
    #
    # inter = user_batch_cnt_pre8min(inter)
    # inter[['user_id','photo_id','user_batch_cnt_pre8min','user_batch_cnt_aft8min']].to_pickle('../data/context_feature/user_batch_cnt_pre8min.pkl')
    #
    # inter = parallel_photo_batch_cnt_pre1day(inter)
    # inter[['user_id','photo_id','photo_batch_cnt_pre1day','photo_batch_cnt_aft1day']].to_pickle('../data/context_feature/photo_batch_cnt_pre1day.pkl')
    #
    inter = next_time_diff(inter)
    inter[['user_id', 'photo_id', 'next_time_diff', 'duration_time']].to_pickle(
        '../data/context_feature/next_time_diff.pkl')

    # inter = pre_time_diff(inter)
    # inter[['user_id', 'photo_id', 'pre_time_diff']].to_pickle(
    #     '../data/context_feature/pre_time_diff.pkl')
    #
    # inter = user_photo_total_cnt(inter)
    # inter[['user_id', 'photo_id', 'photo_total_cnt', 'user_total_cnt', 'user_total_batch_cnt']].to_pickle(
    #     '../data/context_feature/user_photo_total_cnt.pkl')
    #
    # inter = cal_batch_photo_cnt(inter)
    # inter[['user_id', 'photo_id', 'batch_photo_cnt']].to_pickle('../data/context_feature/batch_photo_cnt.pkl')

