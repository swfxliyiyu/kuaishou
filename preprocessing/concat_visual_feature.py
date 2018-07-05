import pandas as pd

if __name__ == '__main__':
    tr_visual = pd.read_pickle('./visual_feature_train.pkl')
    te_visual = pd.read_pickle('./visual_feature_test.pkl')
    visual = pd.concat([tr_visual, te_visual], ignore_index=True)
    visual.columns = ['pid', 'visual']
    visual.to_pickle('./visual_feature.pkl')