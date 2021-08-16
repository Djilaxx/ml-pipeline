import pandas as pd
from sklearn import preprocessing
from utils import feature_selection

#def standardize(dataframe):
#    scaler = preprocessing.StandardScaler()
#    features = dataframe.columns[1:101]
#    dataframe[features] = scaler.fit_transform(dataframe[features])
#    return dataframe

def feature_engineering(dataframe, train=False):
    #dataframe = standardize(dataframe)
    features = dataframe.columns[1:101]
    return dataframe, features