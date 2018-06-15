# -*- coding: utf-8 -*-
'''
 @Time    : 18-5-18 下午5:22
 @Author  : sunhongru
 @Email   : sunhongru@sensetime.com
 @File    : data_process.py
 @Software: PyCharm
'''

import pandas as pd
import numpy as np
import json
import collections
import time
import sklearn.preprocessing as preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation

import time_utils as time_utils


def load_image_vec(photo_id):
    visual_test_data = np.load('../../data/test/visual_test.zip')
    return visual_test_data['preliminary_visual_test/'+str(photo_id)]

def trans_face_vec():
    df_train_face = pd.read_csv('../../data/train/train_face.txt', header=None,names=['photo_id', 'face_vec'], sep='\t')
    df_test_face = pd.read_csv('../../data/test/test_face.txt', header=None, names=['photo_id', 'face_vec'], sep='\t')
    df_face = pd.concat([df_train_face, df_test_face], axis=0, ignore_index=True)

    df_face['face_vec'] = df_face['face_vec'].apply(lambda vec: np.array(json.loads(vec)))
    df_face['face_num'] = df_face['face_vec'].apply(len)
    df_face['female_num'] = df_face['face_vec'].apply(lambda vec: sum(vec[:, 1]))
    df_face['male_num'] = df_face['face_num'] - df_face['female_num']

    df_face['max_age'] = df_face['face_vec'].apply(lambda vec: max(vec[:, 2]))
    df_face['min_age'] = df_face['face_vec'].apply(lambda vec: min(vec[:, 2]))

    # df_face['face_proportion'] = df_face['face_vec'].apply(lambda vec: sum(vec[:, 0]))
    df_face['male_face_proportion'] = df_face['face_vec'].apply(
        lambda vec: int(100*sum([v[0] for v in vec if v[1] == 1])))
    df_face['female_face_proportion'] = df_face['face_vec'].apply(
        lambda vec: int(100*sum([v[0] for v in vec if v[1] == 0])))

    # df_face['max_face_score'] = df_face['face_vec'].apply(lambda vec: max(vec[:, 3]))
    df_face['min_face_score'] = df_face['face_vec'].apply(lambda vec: min(vec[:, 3]))
    df_face['mean_face_score'] = df_face['face_vec'].apply(lambda vec: sum(vec[:, 3])/len(vec))
    df_face['male_max_face_score'] = df_face['face_vec'].apply(lambda vec: max([v[3] for v in vec if v[1] == 1] + [0]))
    df_face['female_max_face_score'] = df_face['face_vec'].apply(lambda vec: max([v[3] for v in vec if v[1] == 0] + [0]))


    del df_face['face_vec']
    del df_face['face_num']

    print(df_face)
    return df_face

def trans_interaction():
    df_train_interaction = pd.read_csv('../../data/train/train_interaction.txt',
                                       names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time',
                                              'playing_time', 'duration_time'], header=None, sep='\t')
    df_test_interaction = pd.read_csv('../../data/test/test_interaction.txt',
                                      names=['user_id', 'photo_id', 'time', 'duration_time'], header=None, sep='\t')
    df_data = pd.concat([df_train_interaction, df_test_interaction], axis=0, ignore_index=True)
    df_data['instance_id'] = np.arange(df_data.shape[0])

    # print('photo visual merge')
    # df_visual = pd.read_pickle('../../data/visual_vae_sum.pkl').rename(columns={'pid': 'photo_id', 'mid_layer': 'visual'})
    # print(df_visual.columns)
    # print(df_visual.head())
    # df_visual['photo_id'] = df_visual['photo_id'].apply(lambda p: int(p))
    # df_data = pd.merge(df_data, df_visual, on='photo_id', how='left')
    # print(df_data)

    print('merge text topics')
    df_topics = pd.read_pickle('../../data/text_lda_10.pkl')
    print(df_data.dtypes)
    print(df_topics.dtypes)
    print(df_topics)
    df_data = pd.merge(df_data, df_topics, on=['photo_id'], how='left')
    print(df_data)

    print('cal play proportion')
    df_data['play_proportion'] = df_data['playing_time'] / df_data['duration_time']
    df_data['play_proportion'] = df_data['play_proportion'].fillna(0).apply(lambda p: 10 if p > 10 else p)
    print(df_data[df_data['click']==1].describe())

    print('transfer time')
    df_data = time_utils.trans_time(df_data)
    print(df_data)

    df_train_interaction = df_data[df_data['click'].notnull()]
    # df_test_interaction = df_data[df_data['click'].isnull()]

    print('photo id cnt')
    # 该视频被推了多少次(用户)
    photo_id_cnt = df_data.groupby(['photo_id'], as_index=False)['instance_id'].agg({'photo_id_cnt': 'count'})
    df_data = pd.merge(df_data, photo_id_cnt, on=['photo_id'], how='left')
    print(df_data)

    # 该样本所在批次 距离 下一批次开始的时间（刷新频率）
    print('cal_time_reduc')
    df_time_redc = pd.read_pickle('../../data/time_redc.pkl')[['instance_id', 'time_redc', 'user_remain_cnt']]
    df_data = pd.merge(df_data, df_time_redc, on='instance_id', how='left')
    # df_data = time_utils.cal_time_reduc(df_data)
    # df_data[['instance_id', 'user_id', 'photo_id', 'time_redc', 'user_remain_cnt']].to_pickle('../../data/time_pre_redc.pkl')
    print(df_data)

    print('cal_batch_photo_cnt')
    df_batch_photo_cnt = pd.read_pickle('../../data/batch_photo_cnt.pkl')[['instance_id', 'batch_photo_cnt']]
    df_data = pd.merge(df_data, df_batch_photo_cnt, on='instance_id', how='left')
    print(df_data)
    # df_data = time_utils.cal_batch_photo_cnt(df_data)
    # df_data[['instance_id', 'user_id', 'photo_id', 'batch_photo_cnt']].to_pickle('../../data/batch_photo_cnt.pkl')


    print('user cnt pre5min')
    df_user_pre5min_cnt = pd.read_pickle('../../data/user_pre5min_cnt.pkl')[['instance_id', 'user_pre5min_cnt']]
    df_data = pd.merge(df_data, df_user_pre5min_cnt, on='instance_id', how='left')
    print(df_data)
    # df_data = time_utils.user_cnt_pre5min(df_data)
    # df_data[['instance_id', 'user_id', 'photo_id', 'user_pre5min_cnt']].to_pickle('../../data/user_pre5min_cnt.pkl')

    print('count user like, follow')
    user_cnt = df_train_interaction.groupby(['user_id'], as_index=False)['click'].agg({'user_cnt': 'count'})
    user_click_cnt = df_train_interaction.groupby(['user_id'], as_index=False)['click'].agg({'user_click_cnt': 'sum'})
    user_like_cnt = df_train_interaction.groupby(['user_id'], as_index=False)['like'].agg({'user_like_cnt': 'sum'})
    user_follow_cnt = df_train_interaction.groupby(['user_id'], as_index=False)['follow'].agg({'user_follow_cnt': 'sum'})

    '''
    # print('user hour cnt')
    # user_hour_click_cnt = df_train_interaction.groupby(['hour', 'user_id'], as_index=False)['click'].agg({'user_hour_click_cnt': 'sum'})
    # user_hour_cnt = df_train_interaction.groupby(['user_id', 'hour'], as_index=False)['photo_id'].agg({'user_hour_cnt': 'count'})
    # print(df_data)


    # print('user minute cnt')
    # user_minute_click_cnt = df_train_interaction.groupby(['minute', 'user_id'], as_index=False)['click'].agg({'user_minute_click_cnt': 'sum'})
    # user_minute_cnt = df_train_interaction.groupby(['user_id', 'day', 'hour', 'minute'], as_index=False)['photo_id'].agg({'user_minute_cnt': 'count'})
    '''
    print(df_data)

    print('user_merge')
    df_data = pd.merge(df_data, user_cnt, on=['user_id'], how='left')
    df_data = pd.merge(df_data, user_click_cnt, on=['user_id'], how='left')
    df_data = pd.merge(df_data, user_like_cnt, on=['user_id'], how='left')
    df_data = pd.merge(df_data, user_follow_cnt, on=['user_id'], how='left')


    df_data['user_ctr'] = df_data['user_click_cnt'] / df_data['user_cnt']
    df_data['user_like_rate'] = df_data['user_like_cnt'] / df_data['user_click_cnt']
    df_data['user_follow_rate'] = df_data['user_follow_cnt'] / df_data['user_click_cnt']

    '''
    # df_data = pd.merge(df_data, user_hour_cnt, on=['user_id', 'hour'], how='left')
    # df_data = pd.merge(df_data, user_minute_cnt, on=['user_id', 'day', 'hour', 'minute'], how='left')
    # df_data = pd.merge(df_data, user_hour_click_cnt, on=['hour', 'user_id'], how='left')
    # df_data['user_hour_ctr'] = df_data['user_hour_click_cnt'] / df_data['user_hour_cnt']
    '''
    return df_data

def cnt_user_like_feats(df_data):
    print('cnt_user_like_face_feats')
    df_train_click = df_data[df_data['click']==1].fillna(0)
    df_train_true_like = df_data[(df_data['click'] == 1) & (df_data['play_proportion'] > 0.1)].fillna(0)
    use_face_feats = ['female_num', 'male_num', 'max_age', 'min_age', 'male_face_proportion', 'female_face_proportion',
                      'male_max_face_score', 'female_max_face_score', 'min_face_score', 'mean_face_score']
    for feat in use_face_feats:
        user_like_ave_face_feat = df_train_true_like.groupby(['user_id'], as_index=False)[feat].agg({'user_like_ave_{}'.format(feat): 'mean'})
        df_data = pd.merge(df_data, user_like_ave_face_feat, on=['user_id'], how='left')
        print(user_like_ave_face_feat)

    print('cnt_user_like_redc_feats')
    use_redc_feats = ['user_remain_cnt', 'time_redc', 'user_pre5min_cnt', 'batch_photo_cnt']
    for feat in use_redc_feats:
        user_like_ave_face_feat = df_train_click.groupby(['user_id'], as_index=False)[feat].agg({'user_like_ave_{}'.format(feat): 'mean'})
        df_data = pd.merge(df_data, user_like_ave_face_feat, on=['user_id'], how='left')


    return df_data


def split_play_proportion(proportion):
    if proportion <= 0.5: return proportion
    if 0.5 < proportion < 10:
        return np.floor(proportion)
    if proportion >= 10: return 10
    return proportion

def get_playing_time():
    df_train_interaction = pd.read_csv('../../data/train/train_interaction.txt',
                                       names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time',
                                              'playing_time', 'duration_time'], header=None, sep='\t')

    print('user mean playing_time/duration_time')
    df_train_interaction['playing_proportion'] = df_train_interaction['playing_time'] / df_train_interaction[
        'duration_time']
    df_train_interaction['playing_proportion'] = df_train_interaction['playing_proportion'].apply(split_play_proportion)
    print(df_train_interaction['playing_proportion'].value_counts())
    return df_train_interaction['playing_proportion']

if __name__ == '__main__':

    df_interaction = trans_interaction()
    df_face = trans_face_vec()
    df_data = pd.merge(df_interaction, df_face, on='photo_id', how='left')
    df_data = cnt_user_like_feats(df_data)

    print(df_data)
    df_data.to_pickle('../../data/interaction_face_true_like_topic_10_data.pkl')
    print('data process finish')

