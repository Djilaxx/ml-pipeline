import pandas as pd
from sklearn import model_selection

def create_splits(input_path, output_path, n_folds=5, split_size = 0.2, target=None):
    '''
    Creating splits for cross validation
    '''
    df = pd.read_csv(input_path)
    df["split"] = 0
    df = df.sample(frac=1).reset_index(drop=True)
    y = df[target]
    
    # FOLDING
    if n_folds >= 2:
        kf = model_selection.StratifiedKFold(n_splits=n_folds)
        for fold, (t_, v_) in enumerate(kf.split(X=df, y=y)):
            df.loc[v_, 'split'] = fold

    # TRAIN VALID SPLIT
    elif n_folds < 2:
        train, valid = model_selection.train_test_split(df, test_size=split_size, stratify=y)
        for i in valid.index:
            df.loc[i, "split"] = 1
    else:
        print("Please select a valid number of splits")

    df.to_csv(output_path, index=False)