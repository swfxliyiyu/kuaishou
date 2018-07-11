from __future__ import print_function, division

import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    files = ['next_time_diff.pkl', 'pre_time_diff.pkl', 'user_batch_cnt_pre8min.pkl', 'user_cnt_pre8min.pkl',
             'user_photo_total_cnt.pkl', 'batch_photo_cnt.pkl', 'photo_batch_cnt_pre1day.pkl']
    files = [os.path.join('../data/context_feature', f) for f in files]
    df_ctx = pd.read_pickle(files[0])
    for f in files[1:]:
        df_temp = pd.read_pickle(f)
        df_temp = df_temp.drop(columns=[col for col in ['instance_id'] if col in df_temp])
        df_ctx = pd.merge(df_ctx, df_temp, 'left', ['user_id', 'photo_id'])
        print(f, 'concated')
    print(df_ctx)
    ctx_cols = df_ctx[[col for col in df_ctx.columns if col not in ['user_id', 'photo_id', 'instance_id']]]
    ctx_cols_01 = pd.DataFrame()
    for it, col in enumerate(ctx_cols.columns):
        ctx_cols_01['ctx_01_{}'.format(it)] = pd.cut(ctx_cols[col], 20, labels=range(20))
        ctx_cols_01['ctx_01_{}'.format(it)] = ctx_cols_01['ctx_01_{}'.format(it)].astype(np.uint8)
        ctx_cols_01['ctx_{}'.format(it)] = ctx_cols[col].astype(np.float16)
    print(ctx_cols_01)
    df_ctx = df_ctx.drop(columns=[col for col in df_ctx if col not in ['user_id', 'photo_id']])
    # df_ctx['context'] = [np.asarray(arr, dtype=np.float16) for arr in ctx_cols.values.tolist()]
    # df_ctx['context_01'] = [np.asarray(arr, dtype=np.uint8) for arr in ctx_cols_01.values.tolist()]
    ctx_cols_01.index = df_ctx.index
    df_ctx = pd.concat([df_ctx, ctx_cols_01], axis=1)
    print(df_ctx)
    df_ctx.to_pickle('../data/context_feature.pkl')
