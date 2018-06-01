import pandas as pd
import numpy as np
from network_norecent import Model

if __name__ == '__main__':
    val_data = pd.read_pickle('../data/val_data.pkl')
    train_data = pd.read_pickle('../data/train_data.pkl')
    test_data = pd.read_pickle('../data/test_data.pkl')
    one_hots_dims = []
    face_cols = np.array(train_data['face_cols'].tolist())
    one_hots_dims.extend((face_cols.max(axis=0) + 1))
    # for col in val_data:
    #     if '_01' in col:
    #         one_hots_dims.append(train_data[col].max() + 1)
    dim_num_feat = 0
    for col in val_data:
        if '_N' in col:
            dim_num_feat += 1

    print(one_hots_dims)
    model_params = {
        'num_user': 15141,
        'num_recent_item': 30,
        'num_words': 119637,
        'dim_num_feat': dim_num_feat,
        'one_hots_dims': one_hots_dims,
        'dim_k': 64,
        'att_dim_k': 16,
        'reg': 0.002,
        'att_reg': 0.6,
        'lr': 0.002,
        'prefix': None,
        'seed': 1024,
        'use_deep': True,
        'deep_dims': (512, 256, 64, 32)
    }
    model = Model(**model_params)
    model.compile(optimizer='adam')
    # indices = pd.np.array([[1, 500], [1, 508]])
    #
    # feed_dict_ = {
    #     model.user_indices: [1]*2048, model.item_words_indices_a: indices,
    #     model.item_words_values_a: [1, 1],
    #     model.recent_words_indices_a: [[1,2,1]],
    #     model.recent_words_values_a: [1],
    #     model.labels: [1]*2048,
    #     model.one_hots_a: [[1,23,3]]*2048,
    #     model.batch_size: 2048
    # }
    # a = model.sess.run([model.I_Wds_a], feed_dict=feed_dict_)
    # print(pd.np.array(a)[0, 1:2, 507:509])
    #
    # print(a)
    fit_params = {
        'input_data': train_data,
        'test_data': test_data,
        'batch_size': 32768,
        'epochs': 50,
        'drop_out_deep': 0.5,
        'validation_data': val_data, 'shuffle': True,
        'initial_epoch': 0,
        'min_display': 10,
        'max_iter': -1,
        'save_path': '../model/'
    }
    model.fit(**fit_params)
