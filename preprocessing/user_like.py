# coding=utf-8
import pandas as pd
import numpy as np


# def cnt_user_like_feats(df_data):
#     print('cnt_user_like_face_feats')
#     df_train_click = df_data[df_data['click']==1].fillna(0)
#     df_train_true_like = df_data[(df_data['click'] == 1) & (df_data['play_proportion'] > 0.1)].fillna(0)
#     use_face_feats = ['female_num', 'male_num', 'max_age', 'min_age', 'male_face_proportion', 'female_face_proportion',
#                       'male_max_face_score', 'female_max_face_score', 'min_face_score', 'mean_face_score']
#     for feat in use_face_feats:
#         user_like_ave_face_feat = df_train_true_like.groupby(['user_id'], as_index=False)[feat].agg({'user_like_ave_{}'.format(feat): 'mean'})
#         df_data = pd.merge(df_data, user_like_ave_face_feat, on=['user_id'], how='left')
#
#         print(user_like_ave_face_feat)
#
#     print('cnt_user_like_redc_feats')
#     use_redc_feats = ['user_remain_cnt', 'time_redc', 'user_pre5min_cnt', 'batch_photo_cnt']
#     for feat in use_redc_feats:
#         user_like_ave_face_feat = df_train_click.groupby(['user_id'], as_index=False)[feat].agg({'user_like_ave_{}'.format(feat): 'mean'})
#         df_data = pd.merge(df_data, user_like_ave_face_feat, on=['user_id'], how='left')
#
#     return df_data

def get_user_like(data, test_data):
    features = [np.array(feat.tolist()) for feat in [data['context'], data['face_cols_num'], data['topics']]]
    data['concated_feature'] = [np.array(lst) for lst in np.concatenate(features, axis=1).tolist()]
    user_like_mean = [[uid, np.array(value['concated_feature'].tolist()).mean(axis=0)] for uid, value in data.groupby(['uid'])]
    user_like_mean = pd.DataFrame(user_like_mean, columns=['uid', 'user_like_mean'])
    data = pd.merge(user_like_mean, test_data[['uid', 'user_indices']], 'left', ['uid'])
    return data


if __name__ == '__main__':
    df = pd.read_pickle('../data/interaction_features_1.pkl')
    print('loaded data...')
    print(df.columns)
    # 添加连续特征
    df = df[df['click'] == 1]


    df_ctx = pd.read_pickle('../data/context_feature.pkl')
    df = pd.merge(df, df_ctx, 'left', left_on=['uid', 'pid'], right_on=['user_id', 'photo_id'])
    print('context feature concated...')

    # 添加text_lda
    df_text_lda = pd.read_pickle('../data/text_lda_6.pkl')
    df = pd.merge(df, df_text_lda, 'left', on=['pid'])
    empty = np.zeros(shape=[6])
    df['topics'] = df['topics'].apply(lambda lst: empty if pd.isna(lst) is True else lst)
    print('text lda concated...')

    # 求用户平均偏好
    test_data = pd.read_pickle('../data/test_data.pkl')
    user_like = get_user_like(df, test_data)
    user_like.to_pickle('../model/user_like_mean_2.pkl')
    print(user_like['user_like_mean'])
    print('get user_like_mean...')
