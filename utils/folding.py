import pandas as pd
from sklearn import model_selection


def create_folds(datapath, output_path, nb_folds, method = "KF", target=None):
    '''
    Creating folds for cross validation
    method must be one of KF, GKF, SKF
    target if SKF is the stratify variable which distribution must remain constant accross folds
    target if GKF is the group which must be non-overlapping
    '''
    df = pd.read_csv(datapath)
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = None
    if method == "KF":
        kf = model_selection.KFold(n_splits=nb_folds)
    if method == "GKF":
        kf = model_selection.GroupKFold(n_splits=nb_folds)
        y = df[target]
    if method == "SKF":
        kf = model_selection.StratifiedKFold(n_splits=nb_folds)
        y = df[target]

    if method == "KF":
        for fold, (t_, v_) in enumerate(kf.split(X=df)):
            df.loc[v_, 'kfold'] = fold
    else:
        for fold, (t_, v_) in enumerate(kf.split(X=df, y=y)):
            df.loc[v_, 'kfold'] = fold

    df.to_csv(output_path, index=False)