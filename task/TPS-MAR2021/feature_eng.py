import pandas as pd 
from sklearn.preprocessing import LabelEncoder 

def cat_encoding(dataframe):
    cat = dataframe.columns[1:20]
    for feature in cat:
        le = LabelEncoder()
        le.fit(dataframe[feature])
        dataframe[feature] = le.transform(dataframe[feature])
    return dataframe

def feature_engineering(dataframe):
    dataframe = cat_encoding(dataframe)
    features = dataframe.columns[1:31]
    return dataframe, features