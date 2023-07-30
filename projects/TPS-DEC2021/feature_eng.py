import pandas as pd

def feature_engineering(dataframe, train=True):
    dataframe = dataframe.drop(["Soil_Type7", "Soil_Type15"], axis=1)
    features = dataframe.columns[1:53]
    return dataframe, features
