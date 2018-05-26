import pandas as pd

from network_nopair import Model

if __name__ == '__main__':
    val_data = pd.read_pickle('../data/val_data.pkl')
    one_hots_dims = []
    for col in val_data:
        if '_01' in col:
            one_hots_dims.append(val_data[col].max() + 1)

    model_params = {
        'num_user': 15141,
        'num_recent_item': 30,
        'num_words': 233244,
        'one_hots_dims': one_hots_dims,
        'dim_k': 64,
        'reg': 0.002,
        'lr': 0.01,
        'prefix': None,
        'seed': 1024
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
    trian_data = pd.read_pickle('../data/train_data.pkl')
    fit_params = {
        'input_data': trian_data,
        'batch_size': 2048,
        'epochs': 50,
        'validation_data': val_data, 'shuffle': True,
        'initial_epoch': 0,
        'min_display': 50,
        'max_iter': -1,
        'save_path': '../model/model'
    }
    model.fit(**fit_params)
