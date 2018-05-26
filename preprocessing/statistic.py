import pandas as pd

if __name__ == '__main__':
    # tr_df = pd.read_pickle('../data/train_interaction.pkl')
    # tr_df.columns = ['uid', 'pid', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
    # te_df = pd.read_pickle('../data/test_interaction.pkl')
    # te_df.columns = ['uid', 'pid', 'time', 'duration_time']
    # df = pd.concat([tr_df, te_df])
    # uids = df['uid'].drop_duplicates()
    # print(uids.max())
    # print(len(uids))

    # tr_photo = tr_df[1]
    # te_photo = te_df[1]
    # tr_user = tr_df[0]
    # te_user = te_df[0]
    # # in_train = te_photo.apply(lambda x: True if x in tr_photo else False)
    # in_train = te_user.apply(lambda x: True if x in set(tr_user) else False)
    # print(in_train.value_counts())
    # print(tr_photo.drop_duplicates().shape[0])
    # print(te_photo.drop_duplicates().shape[0])
    ser_text1 = pd.read_pickle('../data/train_text.pkl')[1]
    ser_text2 = pd.read_pickle('../data/test_text.pkl')[1]
    s = set()
    for text in ser_text1:
        s.update([int(x) for x in text.split(',')])
    for text in ser_text2:
        s.update([int(x) for x in text.split(',')])
    print(len(s))
    print(max(list(s)))