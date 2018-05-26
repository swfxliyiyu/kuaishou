import multiprocessing
import numpy as np

import pandas as pd
from joblib import Parallel, delayed

N_JOBS = multiprocessing.cpu_count()
"""=============================================================="""

def get_kv(name_group):
    name, group = name_group
    return name, group.count()


def count_dict(grouped, count_col):
    lst = Parallel(n_jobs=N_JOBS)(
        delayed(get_kv)((name, group[count_col])) for name, group in grouped)
    return dict(lst)


"""=============================================================="""


def dec_series_func(ser, fuc, arg=None):
    if arg is None:
        return ser.apply(fuc).tolist()
    else:
        return ser.apply(fuc, args=(arg,)).tolist()


def split_str(x):
    return len(str(x).split(';'))


def split_str1(s, i):
    return str(str(s).split(';')[i]) if len(str(s).split(';')) > i else ''


def split_str2(s, i):
    return split_str1(s, i)


def temp_fuc1(x):
    return 1 if x == -1 else 2


def temp_fuc2(x):
    return 1 if (x == 1004) | (x == 1005) | (x == 1006) | (x == 1007) else 2


def temp_fuc3(x):
    return 1 if (x == -1) | (x == 2003) else 2


def temp_fuc4(x):
    return 1 if (x == -1) | (x == 3000) | (x == 3001) else 2


def temp_fuc5(x):
    return 1 if x == 4001 or x == 4002 or x == 4003 or x == 4004 or x == 4007 else 2


def temp_fuc6(x):
    return 0 if x <= 0.98 and x >= 0.96 else 1


def temp_fuc7(x):
    return 1 if (x.deliver_map == 3) & (x.service_map == 3) & (x.de_map == 3) & (x.review_map == 3) else 0


def temp_fuc8(x):
    return 0 if x == -1 else x


def get_cnt(x, dic):
    return dic.get(x, 0)



def series_map(ser, func, arg=None):
    lst = Parallel(n_jobs=N_JOBS)(
        delayed(dec_series_func)(ele, func, arg) for ele in np.array_split(ser, N_JOBS*100))
    return np.concatenate(lst)

def series_map_2(ser, func, arg=None):
    if arg is None:
        lst = Parallel(n_jobs=N_JOBS)(
            delayed(func)(ele,) for ele in ser)
    else:
        lst = Parallel(n_jobs=N_JOBS)(
            delayed(func)(ele, arg) for ele in ser)
    return np.concatenate(lst)

"""=============================================================="""


def dec_frame_func(df, fuc, arg=None):
    if arg is None:
        return df.apply(fuc, axis=1, reduce=True).tolist()
    else:
        return df.apply(fuc, axis=1, reduce=True, args=(arg, )).tolist()

def get_df_cnt(x, args):
    col1, col2, dic = args
    return dic.get((x[col1], x[col2]), 0)

def div(x, args):
    return x[args[0]] / x[args[1]]


def join_func(df, cnt, on):
    return pd.merge(df, cnt, 'left', on=on)


def left_join(df, cnt, on):
    lst = Parallel(n_jobs=N_JOBS)(
        delayed(join_func)(rows, cnt, on) for rows in np.array_split(df, N_JOBS))
    return pd.concat(lst)

def dataframe_map(df, func, arg=None):
    lst = Parallel(n_jobs=N_JOBS)(
        delayed(dec_frame_func)(rows, func, arg) for rows in np.array_split(df, N_JOBS))
    return np.concatenate(lst)


"""=============================================================="""


def comb_str(s1, s2):
    return s1 + s2

def combline_series(ser1, ser2):
    lst = Parallel(n_jobs=N_JOBS)(
        delayed(comb_str)(i, j) for i, j in zip(ser1, ser2))
    return lst


"""=============================================================="""


def get_row(args):
    name, group, col = args
    cnt = group[col].count()
    name = [name] if isinstance(name, int) else name
    return list(name) + [cnt]


def group_count(df, by, col, cnt_name):
    lst = Parallel(n_jobs=N_JOBS)(
        delayed(get_row)((name, group, col)) for name, group in df.groupby(by=by))
    res_df = pd.DataFrame(lst, columns=by + [cnt_name])
    return res_df

