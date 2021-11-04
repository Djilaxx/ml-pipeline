import pandas as pd 
from sklearn.preprocessing import LabelEncoder 

def cat_encoding(dataframe):
    cat = dataframe.columns[1:20]
    for feature in cat:
        le = LabelEncoder()
        le.fit(dataframe[feature])
        dataframe[feature] = le.transform(dataframe[feature])
    return dataframe

def feature_engineering(dataframe, train=False):
    dataframe = cat_encoding(dataframe)
    drop_variables = ["cat5", "cat3", "cat9", "cat12", "cont10", "cont7", "cat10", "cont0", "cat8", "cat13"]
    dataframe = dataframe.drop(drop_variables, axis=1)

    features = dataframe.columns[1:21]
    return dataframe, features