from __future__ import division, print_function
import pandas as pd
import numpy as np
from network_text_lda import Model
import tensorflow as tf


def yield_uid(train_data, val_data, test_data):
    total = train_data.shape[0]
    train_data, val_data, test_data = [df.sort_values(['uid']) for df in [train_data, val_data, test_data]]
    print('sorted data...')
    tr_len, val_len, te_len = [df.shape[0] for df in [train_data, val_data, test_data]]
    train_data.index, val_data.index, test_data.index = [range(df.shape[0]) for df in [train_data, val_data, test_data]]
    tr_last, val_last, te_last, tr_idx, val_idx, te_idx = [0] * 6
    # tr_last, val_last, te_last= [0] * 3
    last_uid = train_data.loc[0, 'uid']
    while tr_idx != tr_len - 1:
        tr_temp, te_temp, val_temp = None, None, None
        for tr_idx in xrange(tr_last + 1, tr_len):
            if train_data.loc[tr_idx, 'uid'] != last_uid:
                print('===========train uid: {:.2f} {} {}-{}'.format(tr_idx / total, train_data.loc[tr_idx, 'uid'],
                                                                     tr_last, tr_idx))
                tr_temp = train_data.loc[tr_last:tr_idx - 1, :]
                tr_last = tr_idx
                break
        for te_idx in xrange(te_last + 1, te_len):
            if test_data.loc[te_idx, 'uid'] != last_uid:
                if test_data.loc[te_idx, 'uid'] != train_data.loc[tr_idx, 'uid']:
                    print('[WORNING] train and test uid are not same!!!')
                print(
                    '===========test uid: {:.2f} {} {}-{}'.format(tr_idx / total, test_data.loc[te_idx, 'uid'], te_last,
                                                                  te_idx))
                te_temp = test_data.loc[te_last:te_idx - 1, :]
                te_last = te_idx
                break
        for val_idx in xrange(val_last + 1, val_len):
            if val_data.loc[val_idx, 'uid'] != last_uid:
                if val_data.loc[val_idx, 'uid'] == train_data.loc[tr_idx, 'uid']:
                    print('===========valid uid: {:.2f} {} {}-{}'.format(tr_idx / total, val_data.loc[val_idx, 'uid'],
                                                                         val_last, val_idx))
                    val_temp = val_data.loc[val_last:val_idx - 1, :]
                    val_last = val_idx
                    break
                else:
                    val_idx -= 1
                    val_temp = None
                    break
        last_uid = train_data.loc[tr_idx, 'uid']
        if tr_temp is not None:
            yield tr_temp, te_temp, val_temp
    yield train_data.loc[tr_last:, :], test_data.loc[te_last:, :], val_data.loc[val_last:, :]
    print('finished yield')


if __name__ == '__main__':
    user_embs = pd.read_pickle('../model/user_emb.pkl')
    user_embs = user_embs.sort_values(['user_indices'])
    user_embs = np.array(user_embs['user_emb'].tolist())
    # visual_embs = pd.read_pickle('../data/visual/visual_feature.pkl')

    # val_data = pd.read_pickle('../data/val_data.pkl')
    # train_data = pd.read_pickle('../data/train_data.pkl')
    # test_data = pd.read_pickle('../data/test_data.pkl')
    # empty = np.zeros(shape=[6])
    # for df in [train_data, test_data, val_data]:
    #     df['topics'] = df['topics'].apply(lambda lst: empty if pd.isna(lst) is True else lst)
    #
    # [train_data, test_data, val_data] = [pd.merge(df, visual_embs, 'left', 'pid') for df in
    #                                      [train_data, test_data, val_data]]

    # train_data.to_pickle('../data/train_data_merged.pkl')
    # test_data.to_pickle('../data/test_data_merged.pkl')
    # val_data.to_pickle('../data/val_data_merged.pkl')

    train_data = pd.read_pickle('../data/train_data_merged.pkl')
    val_data = pd.read_pickle('../data/val_data_merged.pkl')
    test_data = pd.read_pickle('../data/test_data_merged.pkl')
    train_data = pd.concat([train_data, val_data])

    one_hots_dims = []
    face_cols = np.array(train_data['face_cols_01'].tolist())
    one_hots_dims.extend((face_cols.max(axis=0) + 1))
    print('one_hot_dims:', one_hots_dims)

    dim_num_feat = val_data.ix[0, 'context'].shape[0]
    print('dim_num_feat:', dim_num_feat)

    model_params = {
        'num_user': 15141,
        'num_recent_item': 30,
        'num_words': 119637,
        'dim_num_feat': dim_num_feat,
        'one_hots_dims': one_hots_dims,
        'dim_k': 96,
        'att_dim_k': 16,
        'dim_hidden_out': (256, 128, 64, 32),
        'reg': 0.00005,
        'att_reg': 0.005,
        'user_emb_feat': user_embs,
        'dim_lda': 6,
        'lr': 0.0005,
        'prefix': None,
        'seed': 1024,
        'use_deep': True,
        'deep_dims': (1024, 512, 256)
    }
    model_ori = Model(**model_params)
    model_ori.compile(optimizer='adam')
    val_preds = []
    val_labels = []
    # vars = []
    # with model_ori.graph.as_default():
    #     for var in tf.global_variables():
    #         vars.append(var.eval(session=model_ori.sess))
    # model_ori.sess.close()
    # model = Model(**model_params)
    # model.compile(optimizer='adam')
    model = model_ori
    te_uids = []
    te_pids = []
    te_preds = []
    # cnt = 1
    for tr_df, te_df, val_df in yield_uid(train_data, val_data, test_data):
        model.load_model('../model/lda_modelbest.ckpt-14487.meta', ckpt_path='../model/lda_modelbest.ckpt-14487')
        # with model.graph.as_default():
        #     for vf, vt in zip(vars, tf.global_variables()):
        #         tf.assign(vt, vf)
        model.drop_out_deep_on_train = 0.5
        model.drop_out_emb_on_train = 0.5
        model.train_on_batch(tr_df, 10)
        if val_df is not None:
            try:
                val_score = model.evaluate(val_df, split=1, cache=False)
                val_pred = model.pred_prob(val_df, split=1, cache=False)
                val_label = val_df['click'].as_matrix()
                val_preds.extend(val_pred)
                val_labels.extend(val_label)
                print('val_score:', val_score, '...')
            except Exception as e:
                print(e)
        te_pred = model.pred_prob(te_df, split=1, cache=False)
        te_preds.extend(te_pred)
        te_uids.extend(te_df['uid'].tolist())
        te_pids.extend(te_df['pid'].tolist())
        del tr_df, te_df, val_df
        # if cnt % 1000 == 0:
        #     pd.DataFrame({'uid': te_uids, 'pid': te_pids, 'preds': te_preds}).to_pickle(
        #         '../model/tmp/peruser_reg001_lr001_{}.pkl'.format(tr_df.iloc[0, :]['uid']))
        #     te_uids, te_pids, te_preds = list(), list(), list()
        # cnt += 1
    pd.DataFrame({'uid': te_uids, 'pid': te_pids, 'preds': te_preds}).to_pickle(
        '../model/peruser_reg001_lr001.pkl')
    print('total_val_score:', model.scoreAUC(val_labels, val_preds))
