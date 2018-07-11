import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# for iter,(key, value) in enumerate(train_visual.iteritems()):
#     lst.append([key[train_str_length:], value])
#     if iter % 427868 == 0:
#         pd.DataFrame(lst, columns=['pid', 'visual']).to_pickle('../data/train_visual_feature_{}.pkl'.format(iter+1))
#         lst = []
# if lst:
#     pd.DataFrame(lst, columns=['pid', 'visual']).to_pickle('../data/train_visual_feature_11.pkl')

# lst_test = []
#
# for iter, (key, value) in enumerate(test_visual.iteritems()):
#     lst_test.append([key[test_str_length:], value])
# pd.DataFrame(lst_test, columns=['pid', 'visual']).to_pickle('../data/test_visual_feature.pkl')
#
# df = pd.DataFrame()
#
# for i in range(1, 12):
#     train_visual_sample = pd.read_pickle('../data/train_visual_feature_{}.pkl'.format(i))
#     df = pd.concat(df, train_visual_sample)

def yield_keys1(visual):
    keys = []
    cnt = 0
    visual = np.load(visual)
    lenth = len(visual.keys())

    for iter, key in enumerate(visual.keys()[1:]):
        keys.append(key)
        if iter % 20000 == 0 or iter == lenth - 1:
            print(iter/7560366)
            if 340 < cnt <= 360:
                yield keys[:]
            keys = []
            cnt += 1
    # if keys:
    #     yield keys[:]

def yield_keys2(visual):
    keys = []
    cnt = 0
    visual = np.load(visual)
    for iter, key in enumerate(visual.keys()[1:]):
        keys.append(key)
        if cnt % 20000 == 0:
            print(cnt/895846)
            yield keys[:]
            keys = []
        cnt += 1
    if keys:
        yield keys[:]

def to_pickle1(visual, it, keys, length):
    print(it)
    visual = np.load(visual)
    lst = []
    for key in keys:
        lst.append([int(key[length:]), visual[key][0]])
    pd.DataFrame(lst, columns=['pid', 'visual']).to_pickle('../data/visual/train_visual_feature_17_{}.pkl'.format(it))
    return None

def to_pickle2(visual, it, keys, length):
    visual = np.load(visual)
    lst = []
    for key in keys:
        lst.append([int(key[length:]), visual[key][0]])
    pd.DataFrame(lst, columns=['pid', 'visual']).to_pickle('../data/visual/test_visual_feature_{}.pkl'.format(it))
    # return pd.DataFrame(lst, columns=['key', 'visual'])

# def fun1(visual,lst, length):
#     for iter, key in enumerate(visual.keys()):
#         lst.append([key[length:], visual[key]])
#         cnt = 0
#         if iter % 427868 == 0:
#             pd.DataFrame(lst, columns=['pid', 'visual']).to_pickle('../data/train_visual_feature_{}.pkl'.format(cnt))
#             cnt += 1
#             lst = []
#     if lst:
#         pd.DataFrame(lst, columns=['pid', 'visual']).to_pickle('../data/train_visual_feature_11.pkl')
#         lst = []

def proc_visual(visual, length, yield_keys, to_pickle, name):
    lst = Parallel(n_jobs=15, backend='multiprocessing')(
        delayed(to_pickle)(visual, it, keys, length) for it, keys in enumerate(yield_keys(visual)))
    # lst = pd.concat(lst, ignore_index=True)
    # lst['key'] = lst['key'].astype('int')
    # lst.to_pickle(
    #     '../data/visual/visual_feature_{}.pkl'.format(name))
    print(name, 'finished...')

if __name__== "__main__":
    # train_visual = np.load('../data/train_visual.zip')
    # test_visual = np.load('../data/test_visual.zip')
    train_visual = '../data/visual/train_visual.zip'
    test_visual = '../data/visual/test_visual.zip'

    train_str_length = len('final_visual_train/')
    test_str_length = len('final_visual_test/')

    proc_visual(train_visual, train_str_length, yield_keys1, to_pickle1, 'train')
    print('*******train_completed!**********')
    # proc_visual(test_visual, test_str_length, yield_keys2, to_pickle2, 'test')
