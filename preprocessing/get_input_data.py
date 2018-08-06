import pandas as pd
import json
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from itertools import chain
from collections import defaultdict

cnt = 0
def handle_text(text):
    dic = {}
    dic_count = defaultdict(lambda: 0)
    for row in text['text']:
        lst = row.strip().split(',')
        lst = [int(x) for x in lst]
        for ele in lst:
            dic_count[ele] += 1
    dic_count = pd.Series(dic_count)
    print('before filtering', len(dic_count))
    dic_count = dic_count[dic_count < 2]
    print('after filtering', len(dic_count))
    dic_count = set(dic_count.index)
    print(dic_count)
    empty = np.array([], dtype=np.int32)

    def func(x):
        x = int(x)
        if x in dic:
            return dic[x]
        else:
            global cnt
            dic[x] = cnt
            cnt += 1
            return dic[x]

    def split_text(x):
        lst = x.strip().split(',')
        lst = [func(x) for x in lst if (x != '0' and int(x) not in dic_count)]
        if not lst:
            return empty
        else:
            return np.asarray(np.sort(lst), dtype=np.int32)

    text['words'] = text['text'].apply(split_text)
    del text['text']
    return text


def words_to_vec(text):
    W2V_PATH = '../data/word2vec/word2vec.model'
    model = Word2Vec.load(W2V_PATH)
    empty = np.zeros([128])

    def sentence2vec(sentence):
        sentence = sentence.strip().split(',')
        words_vecters = []
        for word in sentence:
            try:
                words_vecters.append(model.wv[word])
            except KeyError:
                print(word, "not in vocabulary")
                continue
        if not words_vecters:
            sentence_vec = empty
        else:
            sentence_vec = np.mean(words_vecters, axis=0)
        return sentence_vec

    text['words_vec'] = text['text'].apply(sentence2vec)
    return text

count = 0


def handle_face(face):
    columns = ['face_num_01', 'face_max_percent_01', 'face_whole_percent_01',
               'face_male_num_01', 'face_famale_num_01', 'face_gender_mix_01',
               'face_ave_age_01', 'face_max_age', 'face_min_age',
               'face_max_appear_01', 'face_min_appear_01', 'face_ave_appear_01',
               'famale_ave_appear', 'famale_max_appear', 'famale_min_appear',
               'male_ave_appear', 'male_max_appear', 'male_min_appear',
               'famale_max_percent', 'male_max_percent', 'famle_whole_percent', 'male_whole_percent',
               'max_percent_appear', 'max_famale_percent_appear', 'max_male_percent_appear',
               'famale_max_age', 'famale_min_age', 'famale_ave_age',
               'male_max_age', 'male_min_age', 'male_ave_age',
               ]
    total = face.shape[0]

    def get_face_property(x):
        faces = json.loads(x)
        male_faces = [lst for lst in faces if lst[1] == 0]
        famale_faces = [lst for lst in faces if lst[1] == 1]
        has_male = len(male_faces) > 0
        has_famale = len(famale_faces) > 0
        male_percents = [lst[0] for lst in male_faces]
        famale_percents = [lst[0] for lst in famale_faces]
        percents = [lst[0] for lst in faces]
        male_ages = [lst[2] for lst in male_faces]
        famale_ages = [lst[2] for lst in famale_faces]
        ages = [lst[2] for lst in faces]
        male_appears = [lst[3] for lst in male_faces]
        famale_appears = [lst[3] for lst in famale_faces]
        appears = [lst[3] for lst in faces]

        res = [len(faces), np.max(percents), np.sum(percents),
               len(male_faces), len(famale_faces), 1 if (len(male_faces) > 0 and len(famale_faces) > 0) else 0,
               np.mean(ages), np.max(ages), np.min(ages),
               np.max(appears), np.min(appears), np.mean(appears),
               np.max(famale_appears) if has_famale else 0,
               np.min(famale_appears) if has_famale else 0,
               np.mean(famale_appears) if has_famale else 0,
               np.max(male_appears) if has_male else 0,
               np.min(male_appears) if has_male else 0,
               np.mean(male_appears) if has_male else 0,
               np.max(famale_percents) if has_famale else 0,
               np.max(male_percents) if has_male else 0,
               np.sum(famale_percents) if has_famale else 0,
               np.sum(male_percents) if has_male else 0,
               appears[np.argmax(percents)],
               famale_appears[np.argmax(famale_percents)] if has_famale else 0,
               male_appears[np.argmin(male_percents)] if has_male else 0,
               np.max(famale_ages) if has_famale else np.nan,
               np.min(famale_ages) if has_famale else np.nan,
               np.mean(famale_ages) if has_famale else np.nan,
               np.max(male_ages) if has_male else np.nan,
               np.min(male_ages) if has_male else np.nan,
               np.mean(male_ages) if has_male else np.nan,

               ]
        global count
        if count % 10000 == 0:
            print(count / total)
        count += 1
        return res

    new_col = [get_face_property(x) for x in face['faces']]
    new_col = np.array(new_col)
    new_col_01 = np.empty(new_col.shape)

    inf_cols = set(np.arange(25, 31))
    for col in range(new_col.shape[1]):
        print(columns[col])
        data = new_col[:, col]
        if col in inf_cols:
            data[np.isnan(data)] = np.nanmean(data)
        new_col[:, col] = data
        print(pd.Series(data).drop_duplicates())
        new_col_01[:, col] = pd.cut(data, 10, labels=np.arange(10))
    face = pd.DataFrame({'pid': face['pid'].tolist(), 'face_cols_01': new_col_01.tolist(), 'face_cols_num': new_col.tolist()})
    face['face_cols_01'] = face['face_cols_01'].apply(lambda lst: np.asarray(lst, np.int8))
    face['face_cols_num'] = face['face_cols_num'].apply(lambda lst: np.asarray(lst, np.float32))
    print(face)
    return face


def yield_udf(inter):
    uids = set(inter['uid'])
    inter = inter.sort_values(['time'], ascending=True)
    total = len(uids)
    count = 0
    for uid in uids:
        df_u = inter[inter['uid'] == uid]
        if count % 100 == 0:
            print(count / total)
        count += 1
        yield df_u


def func(pids, recent_num, i):
    lst = pids[max(0, i - recent_num): i]
    if len(lst) < recent_num:
        lst = lst + [-1] * (recent_num - len(lst))
    return lst


def get_recent_pids(df_u, recent_num):
    pids = df_u['pid'].tolist()
    pid_lst = [func(pids, recent_num, i) for i in range(df_u.shape[0])]
    df_u['pids'] = pid_lst
    return df_u


def parallel_handle_inter(inter, recent_num):
    lst = Parallel(n_jobs=12)(
        delayed(get_recent_pids)(df_u, recent_num) for df_u in yield_udf(inter))
    result = pd.concat(lst)
    return result


def handle_inter(inter, recent_num=50):
    uids = set(inter['uid'])
    inter = inter.sort_values(['time'], ascending=True)
    df = pd.DataFrame()
    count = 0
    total = len(uids)
    for uid in uids:
        df_u = inter[inter['uid'] == uid]
        pids = df_u['pid'].tolist()

        def func(i):
            lst = pids[max(0, i - recent_num): i]
            if len(lst) < recent_num:
                lst = lst + [-1] * (recent_num - len(lst))
            return lst

        pid_lst = [func(i) for i in range(df_u.shape[0])]
        df = pd.concat([df, pd.DataFrame({'uid': df_u['uid'].tolist(), 'pids': pid_lst})])
        if count % 100 == 0:
            print(count / total)
        count += 1
    inter = pd.merge(inter, df, 'left', 'uid')
    return inter


# def _item_words_indices_and_values(recents):
#     def func():
#         for ix, words in enumerate(recents):
#             for word in words:
#                 yield [ix, word]
#     indices = list(func())
#     values = [1] * len(indices)
#     return indices, values


def yield_rows(inter):
    count = 0
    total = inter.shape[0]
    for rows in np.array_split(inter, 12 * 5):
        count += rows.shape[0]
        print(count / total)
        yield rows


def get_words(rows, dic):
    def func(pids):
        for ix, pid in enumerate(pids):
            for word in dic.get(pid, [0]):
                yield [ix, word]

    lst = [list(func(pids)) for pids in rows['pids']]
    return lst


def add_new_columns(inter):
    # train_text = pd.read_pickle('../data/train_text.pkl')
    # test_text = pd.read_pickle('../data/test_text.pkl')
    text = pd.read_pickle('../data/text_features.pkl')
    # text = pd.concat([train_text, test_text], ignore_index=True)
    # text.columns = ['pid', 'text']
    dic = text['text']
    dic.index = text['pid'].tolist()
    dic = dic.to_dict()
    lst = Parallel(n_jobs=-2, backend='multiprocessing')(
        delayed(get_words)(rows, dic) for rows in yield_rows(inter))
    # for pids in inter['pids']:
    #     df = pd.DataFrame()
    #     df['pid'] = pids
    #     # df = pd.merge(df, text, 'left', on=['pid'])
    #     df['text'] = df['pid'].apply(lambda x:dic.get(x, [0]))
    #     new_column.append(df['text'].tolist())
    #     if count % 1000 == 0:
    #         print(count / total, df['text'].tolist())
    #     count += 1
    lst = list(chain(*lst))
    inter['recent_words'] = lst
    return inter


def yeild_udf(inter):
    uids = set(inter['uid'])
    total = len(uids)
    count = 0
    for uid in uids:
        df_u = inter[inter['uid'] == uid]
        if count % 100 == 0:
            print(count / total)
        count += 1
        yield df_u


def get_val_set(df, pids):
    df['index'] = df.index
    result = df['index'][[(x in pids) for x in df['pid']]]
    return result


# def parallel_get_val_set(data, val_rate):
#     val_pids = set(data['pid'].drop_duplicates().sample(frac=val_rate))
#     lst = Parallel(n_jobs=-2, backend='multiprocessing')(
#         delayed(get_val_set)(df_u, val_pids) for df_u in yeild_udf(data))
#     result = pd.np.concatenate(lst)
#     return result

def yeild_df(data):
    total = data.shape[0]
    count = 0
    for df in np.array_split(data, 100):
        count += df.shape[0]
        print(count / total)
        yield df

def parallel_get_val_set(data, val_rate):
    val_pids = set(data['pid'].drop_duplicates().sample(frac=val_rate))
    lst = Parallel(n_jobs=-2, backend='multiprocessing')(
        delayed(get_val_set)(df, val_pids) for df in yeild_df(data))
    result = pd.np.concatenate(lst)
    return result


def cast_inter(data):
    data['uid'] = data['uid'].astype('int32')
    data['pid'] = data['pid'].astype('int32')
    for col in ['click', 'like', 'follow', 'playing_time', 'duration_time']:
        data[col] = data[col].astype('float16')
    return data


if __name__ == '__main__':

    # "================================加入历史pid================================="
    # train_inter = pd.read_pickle('../data/train_interaction.pkl')
    # train_inter.columns = ['uid', 'pid', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
    # test_inter = pd.read_pickle('../data/test_interaction.pkl')
    # test_inter.columns = ['uid', 'pid', 'time', 'duration_time']
    # train_inter['is_test'] = False
    # test_inter['is_test'] = True
    # inter = pd.concat([train_inter, test_inter])
    # # inter = handle_inter(inter, recent_num=30)
    # # inter = parallel_handle_inter(inter, recent_num=30)
    # inter.to_pickle('../data/recent_inter.pkl')

    # train_text = pd.read_pickle('../data/train_text.pkl')
    # test_text = pd.read_pickle('../data/test_text.pkl')
    # text = pd.concat([train_text, test_text])
    # text.columns = ['pid', 'text']
    # text = handle_text(text)
    # s = set()
    # for t in text['words']:
    #     s.update(t.tolist())
    # print(text)
    # print(len(s))
    # print(max(list(s)))
    # text.to_pickle('../data/text_features.pkl')
    #
    # train_face = pd.read_pickle('../data/train_face.pkl')
    # test_face = pd.read_pickle('../data/test_face.pkl')
    # face = pd.concat([train_face, test_face])
    # face.columns = ['pid', 'faces']
    # face = handle_face(face)
    # face.to_pickle('../data/face_features.pkl')

    # train_text = pd.read_pickle('../data/train_text.pkl')
    # test_text = pd.read_pickle('../data/test_text.pkl')
    # text = pd.concat([train_text, test_text])
    # text.columns = ['pid', 'text']
    # text = words_to_vec(text)
    # print(text)
    # text.to_pickle('../data/words_vec_features.pkl')

    face = pd.read_pickle('../data/face_features.pkl')
    text = pd.read_pickle('../data/text_features.pkl')
    # text_vec = pd.read_pickle('../data/words_vec_features.pkl')
    # text = pd.merge(text, text_vec, on=['pid'])
    df = pd.merge(face, text, how='outer', on=['pid'])
    df['pid'] = df['pid'].astype('int32')
    # for col in ['face_num', 'face_max_percent', 'face_whole_percent', 'face_male_num', 'face_famale_num',
    #             'face_gender_mix', 'face_ave_age', 'face_max_appear', 'face_min_appear', 'face_ave_appear']:
    #     df[col] = df[col].fillna(0)
    empty = np.array([])
    df['words'] = df['words'].apply(lambda lst: empty if pd.isna(lst) is True else lst)
    empty = np.zeros([31], np.int8)
    empty[28:31] = 8
    df['face_cols_01'] = df['face_cols_01'].apply(lambda lst: empty if pd.isna(lst) is True else lst)
    empty = np.zeros([31], np.float32)
    empty[28:31] = 28
    df['face_cols_num'] = df['face_cols_num'].apply(lambda lst: empty if pd.isna(lst) is True else lst)
    # empty = np.zeros([128])
    # df['words_vec'] = df['words_vec'].apply(lambda lst: empty if pd.isna(lst) is True else lst)
    print(df.sort_values(['pid']))
    df.to_pickle('../data/item_features.pkl')

    train_inter = pd.read_pickle('../data/train_interaction.pkl')
    train_inter.columns = ['uid', 'pid', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
    test_inter = pd.read_pickle('../data/test_interaction.pkl')
    test_inter.columns = ['uid', 'pid', 'time', 'duration_time']
    train_inter['is_test'] = False
    test_inter['is_test'] = True
    recent = pd.concat([train_inter, test_inter], sort=False)
    recent = cast_inter(recent)
    recent.index = range(recent.shape[0])
    print(recent.columns)
    item = pd.read_pickle('../data/item_features.pkl')
    print(item.columns)
    inter = pd.merge(recent, item, how='left', on=['pid'])
    print(inter)
    print(inter.columns)
    # inter.to_pickle('../data/interaction_features.pkl')

    # df = pd.read_pickle('../data/interaction_features.pkl')
    df = inter
    df.index = range(df.shape[0])
    print(df)
    val_set = parallel_get_val_set(df[df['is_test'] == False], 0.05)
    # df = df.drop(index=val_set.index)
    df['is_val'] = False
    # df['is_val'] = False
    # val_set['is_val'] = True
    # df = pd.concat([df, val_set])
    df.loc[val_set, 'is_val'] = True
    print(df)
    df.to_pickle('../data/interaction_features_1.pkl')
