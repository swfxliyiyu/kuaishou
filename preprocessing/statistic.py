import pandas as pd

if __name__ == '__main__':
    tr_df = pd.read_pickle('../data/train_interaction.pkl')
    te_df = pd.read_pickle('../data/test_interaction.pkl')
    tr_photo = tr_df[1]
    te_photo = te_df[1]
    tr_user = tr_df[0]
    te_user = te_df[0]
    # in_train = te_photo.apply(lambda x: True if x in tr_photo else False)
    in_train = te_user.apply(lambda x: True if x in set(tr_user) else False)
    print(in_train.value_counts())
    print(tr_photo.drop_duplicates().shape[0])
    print(te_photo.drop_duplicates().shape[0])
