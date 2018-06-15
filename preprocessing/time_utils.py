# -*- coding: utf-8 -*-
'''
 @Time    : 18-5-24 下午12:17
 @Author  : sunhongru
 @Email   : sunhongru@sensetime.com
 @File    : time_utils.py
 @Software: PyCharm
'''

import time
import pandas as pd
import numpy as np

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

def user_cnt_pre5min(df_data):
    def get_index(x):
        index = x[0] + np.searchsorted(temp.realtime[x[0]:x[1]], x[2])
        return index

    temp = df_data[['user_id', 'photo_id', 'instance_id', 'realtime']]
    temp = temp.sort_values(['user_id', 'realtime'], ascending=[1, 1])
    temp['start_dates'] = temp['realtime'] - pd.Timedelta(minutes=5)
    left, right = np.searchsorted(temp.user_id, temp.user_id, side='left'), np.searchsorted(temp.user_id, temp.user_id,
                                                                                            side='right')
    temp['left_index'] = left
    temp['right_index'] = right

    a = temp[['left_index', 'right_index', 'start_dates']].apply(lambda x: get_index(tuple(x)), axis=1)
    temp['start_index'] = a['left_index']
    temp['end_index'] = range(temp.shape[0])
    temp['user_pre5min_cnt'] = temp[['start_index', 'end_index']].apply(lambda x: x[1] - x[0], axis=1)
    temp = temp.drop(['start_dates', 'left_index', 'right_index', 'start_index', 'end_index'], axis=1)
    df_data = pd.merge(df_data, temp, 'left', on=['user_id', 'photo_id', 'instance_id', 'realtime'])
    # df['A'].iloc[row['start_index']:row['end_index'] + 1].sum()
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
    train_origin = df_data
    train1 = train_origin[['time', 'user_id', 'instance_id', 'photo_id']]
    train1 = train1.sort_values(['user_id', 'time'], ascending=[1, 1])

    rnColumn = train1.groupby('user_id').rank(method='min')
    train1['rn'] = rnColumn['time']
    batch_photo_cnt = train1.groupby(['rn', 'user_id'], as_index=False)['time'].agg({'batch_photo_cnt': 'count'})
    train1 = pd.merge(train1, batch_photo_cnt, on=['rn', 'user_id'], how='left')
    train1['rn_1'] = train1['rn'] + train1['batch_photo_cnt']
    train2 = train1.merge(train1, how='left', left_on=['user_id', 'rn_1'], right_on=['user_id', 'rn'])
    train2.drop_duplicates('instance_id_x', keep='first', inplace=True)
    train2['time_redc'] = train2['time_y'] - train2['time_x']
    train2['time_redc'] = train2['time_redc'].apply(lambda tr: tr / 1000)
    train2 = train2.fillna(-1).astype('int64')
    train2 = train2.rename(columns={'instance_id_x': 'instance_id'})
    train2 = train2.rename(columns={'photo_id_x': 'photo_id'})
    train2 = train2.rename(columns={'time_x': 'time'})

    user_cnt_max = train2.groupby(['user_id']).max()['rn_x'].reset_index().rename(columns={'rn_x': 'user_cnt_max'})
    train2 = pd.merge(train2, user_cnt_max, 'left', on=['user_id'])
    train2['user_remain_cnt'] = train2['user_cnt_max'] - train2['rn_x']
    train2.drop(['user_cnt_max'], inplace=True, axis=1)
    df_data = pd.merge(train_origin, train2, on=['instance_id', 'photo_id', 'user_id'], how='left')
    print(df_data)
    return df_data

def cal_batch_photo_cnt(df_data):
    train = df_data[['time', 'user_id', 'instance_id', 'photo_id']]
    train = train.sort_values(['user_id', 'time'], ascending=[1, 1])
    rnColumn = train.groupby('user_id').rank(method='min')
    train['rn'] = rnColumn['time']
    batch_photo_cnt = train.groupby(['rn', 'user_id'], as_index=False)['time'].agg({'batch_photo_cnt':'count'})
    train = pd.merge(train, batch_photo_cnt, on=['rn', 'user_id'], how='left')
    df_data = pd.merge(df_data, train[['instance_id', 'batch_photo_cnt']], on='instance_id', how='left')
    return df_data