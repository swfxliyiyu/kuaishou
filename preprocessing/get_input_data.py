import pandas as pd
import json
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import MultiLabelBinarizer


def handle_text(text):
    def split_text(x):
        lst = x.strip().split(',')
        return lst

    text['text'] = text['text'].apply(split_text)
    print(text)
    return text


count = 0


def handle_face(face):
    columns = ['pid', 'face_num', 'face_max_percent', 'face_whole_percent', 'face_male_num', 'face_famale_num',
               'face_gender_mix', 'face_ave_age', 'face_max_appear', 'face_min_appear', 'face_ave_appear']
    total = face.shape[0]

    def get_face_property(x):
        faces = json.loads(x)
        genders = [lst[1] for lst in faces]
        percents = [lst[0] for lst in faces]
        ages = [lst[2] for lst in faces]
        appears = [lst[3] for lst in faces]
        # res = {'face_num': len(faces), 'face_max_percent': int(np.max(percents) * 10),
        #        'face_whole_percent': int(np.sum(percents) * 10),
        #        'face_male_num': np.sum([g for g in genders if g == 0]),
        #        'face_famale_num': np.sum([g for g in genders if g == 1]),
        #        'face_gender_mix': 1 if (0 in genders and 1 in genders) else 0,
        #        'face_ave_age': np.mean(ages) // 2,
        #        'face_max_appear': np.max(appears) // 3,
        #        'face_min_appear': np.min(appears) // 3,
        #        'face_ave_appear': np.min(appears) // 3,
        #        }
        res = [len(faces), int(np.max(percents) * 10), int(np.sum(percents) * 10),
               np.sum([g for g in genders if g == 0]), np.sum([g for g in genders if g == 1]),
               1 if (0 in genders and 1 in genders) else 0, np.mean(ages) // 2, np.max(appears) // 3,
               np.min(appears) // 3, np.min(appears) // 3, ]
        global count
        if count % 10000 == 0:
            print(count / total)
        count += 1
        return res

    new_df = [get_face_property(x) for x in face['faces']]
    face = np.concatenate([face[['pid']], new_df], axis=1)
    face = pd.DataFrame(face, columns=columns)
    # face = pd.merge(face, new_df, how='outter', left_index=True, right_index=True)
    print(face)
    return face


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
        delayed(get_recent_pids)(df_u, recent_num) for df_u in yeild_udf(inter))
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


if __name__ == '__main__':
    # train_inter = pd.read_pickle('../data/train_interaction.pkl')
    # train_inter.columns = ['uid', 'pid', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
    # test_inter = pd.read_pickle('../data/test_interaction.pkl')
    # test_inter.columns = ['uid', 'pid', 'time', 'duration_time']
    # train_inter['is_test'] = False
    # test_inter['is_test'] = True
    # inter = pd.concat([train_inter, test_inter])
    # # inter = handle_inter(inter, recent_num=30)
    # inter = parallel_handle_inter(inter, recent_num=30)
    # inter.to_pickle('../data/recent_inter.pkl')

    # train_text = pd.read_pickle('../data/train_text.pkl')
    # test_text = pd.read_pickle('../data/test_text.pkl')
    # text = pd.concat([train_text, test_text])
    # text.columns = ['pid', 'text']
    # text = handle_text(text)
    # text.to_pickle('../data/text_features.pkl')

    # train_face = pd.read_pickle('../data/train_face.pkl')
    # test_face = pd.read_pickle('../data/test_face.pkl')
    # face = pd.concat([train_face, test_face])
    # face.columns = ['pid', 'faces']
    # face = handle_face(face)
    # face.to_pickle('../data/face_features.pkl')

    face = pd.read_pickle('../data/face_features.pkl')
    text = pd.read_pickle('../data/text_features.pkl')
    df = pd.merge(face, text, how='outer', on=['pid'])
    for col in ['face_num', 'face_max_percent', 'face_whole_percent', 'face_male_num', 'face_famale_num',
               'face_gender_mix', 'face_ave_age', 'face_max_appear', 'face_min_appear', 'face_ave_appear']:
        df[col] = df[col].fillna(0)
    df['text'] = df['text'].replace({np.nan: [0]})
    print(df.sort_values(['pid']))
    df.to_pickle('../data/item_features.pkl')