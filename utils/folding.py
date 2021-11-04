import pandas as pd
import numpy as np
from sklearn import model_selection

def create_splits(
    df, 
    task, 
    n_folds=5, 
    split=False, 
    split_size=0.2, 
    target=None
):
    """
    Custom dataset splitting function
    Either create a simple train - validation split
    or multiple folds in the dataset

    Parameters
    ----------
    df: pandas dataframe
        the dataframe you wish to split
    task: str
        CLASSIFICATION or REGRESSION - this parameter is specified in the config.py file under config.main.TASK
    n_folds: int
        number of folds to create in the dataset - only used if split is False
    split: Boolean - True or False
        if true, will split the dataset into a training and validation set
        if false, will split in the specified number of n_folds
    split_size: float [0, 1]
        the size of the validation set for the non-folding case
    target: str
        the column name of the target variable in your dataset

    Returns
    -------
    df: pandas dataframe
        the dataframe provided with an additional column [splits] indicating to which split each row belong.
        """
    
    df["splits"] = 0
    df = df.sample(frac=1).reset_index(drop=True)
    
    # BINNING THE TARGET VALUE IF TASK IS REGRESSION
    if task == "REGRESSION":
        num_bins = int(np.floor(1 + np.log2(len(df))))
        if num_bins > 10:
            num_bins = 10
        df["bins"] = pd.cut(df[target].values.ravel(), bins=num_bins, labels=False)
        y = df["bins"].values
    elif task == "CLASSIFICATION":
        y = df[target].values
    else:
        raise Exception("task not supported")
    
    # SPLITTING THE DATASET IN TWO OR CREATING MULTIPLE FOLDS
    if split is True:
        _, valid = model_selection.train_test_split(df, test_size=split_size, stratify=y)
        for i in valid.index:
            df.loc[i, "splits"] = 1
    elif split is False:
        kf = model_selection.StratifiedKFold(n_splits=n_folds)
        for fold, (t_, v_) in enumerate(kf.split(X=df, y=y)):
            df.loc[v_, 'splits'] = fold
    else:
        raise Exception("split must be true or false")

    return df
