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
    df = test_data[['uid', 'user_indices']].drop_duplicates(['uid', 'user_indices'])
    for col, res_name in zip(['visual', 'topics', 'face_cols_num'], ['user_like_visual', 'user_like_lda', 'user_like_face']):
        user_like = [[uid, np.mean(value[col].tolist(), axis=0).astype(np.float32)] for uid, value in data.groupby(['uid'])]
        user_like = pd.DataFrame(user_like, columns=['uid', res_name])
        df = pd.merge(df, user_like, 'left', ['uid'])
    return df

if __name__ == '__main__':
    df = pd.read_pickle('../data/interaction_features_1.pkl')
    print('loaded data...')
    print(df.columns)
    # 添加连续特征
    # df = df[df['click'] == 1]

    # df_ctx = pd.read_pickle('../data/context_feature.pkl')
    # df = pd.merge(df, df_ctx, 'left', left_on=['uid', 'pid'], right_on=['user_id', 'photo_id'])
    # print('context feature concated...')

    # 添加text_lda
    df_text_lda = pd.read_pickle('../data/text_lda_6.pkl')
    df = pd.merge(df, df_text_lda, 'left', on=['pid'])
    empty = np.zeros(shape=[6])
    df['topics'] = df['topics'].apply(lambda lst: empty if pd.isna(lst) is True else lst)
    print('text lda concated...')

    # 添加visual
    visual_train1 = pd.read_pickle('../data/visual/visual_feature_train_1.pkl')
    visual_train2 = pd.read_pickle('../data/visual/visual_feature_train_2.pkl')
    visual_test = pd.read_pickle('../data/visual/visual_feature_test.pkl')
    visual = pd.concat([visual_train1, visual_train2, visual_test], ignore_index=True, sort=False)
    df = pd.merge(df, visual, 'left', on=['pid'])
    print('visual concated...')

    # 求用户平均偏好
    test_data = pd.read_pickle('../data/test_data.pkl')
    user_like = get_user_like(df, test_data)
    user_like.to_pickle('../model/user_like_visual_lda_face2.pkl')
    print(user_like)
    print('get user_like_mean...')
