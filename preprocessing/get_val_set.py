import pandas as pd
from joblib import Parallel, delayed


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


def get_val_set(df_u, val_rate):
    pids = df_u['pid'].drop_duplicates().tolist()
    pids = pids[int(len(pids) * (1 - val_rate)):]
    pids = set(pids)
    df_u['index'] = df_u.index
    result = df_u['index'][[(x in pids) for x in df_u['pid']]].tolist()
    return result


def parallel_get_val_set(data, val_rate):
    lst = Parallel(n_jobs=12)(
        delayed(get_val_set, val_rate)(df_u, val_rate) for df_u in yeild_udf(data))
    result = pd.np.concatenate(lst)
    return result


if __name__ == '__main__':
    df = pd.read_pickle('../data/inter.pkl')
    df.index = range(df.shape[0])
    print(df)
    val_set = parallel_get_val_set(df[df['is_test'] == False], 0.15)
    # df = df.drop(index=val_set.index)
    df['is_val'] = False
    # df['is_val'] = False
    # val_set['is_val'] = True
    # df = pd.concat([df, val_set])
    df.loc[val_set, 'is_val'] = True
    df.to_pickle('../data/data.pkl')
