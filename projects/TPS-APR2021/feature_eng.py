import datetime
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from utils import feature_selection

def missing_values(dataframe):
    # Droping features that have too many MV
    drop_features = ["Cabin", "Name", "Ticket"]
    dataframe = dataframe.drop(drop_features, axis=1)
    # DIVIDE INTO NUMERICAL FEATURES AND CATEGORICAL FEATURES
    integer_features = [col for col in dataframe.columns if is_integer_dtype(dataframe[col])]
    float_features = [col for col in dataframe.columns if is_float_dtype(dataframe[col])]
    object_features = [col for col in dataframe.columns if is_object_dtype(dataframe[col])]

    # WE REPLACE MISSING VALUES IN INTEGER  & FLOAT FEATURES WITH MEAN AND MODE FOR CATEGORICAL FEATURES
    dataframe[integer_features] = dataframe[integer_features].apply(lambda x: x.fillna(value=x.mean().astype(int)))
    dataframe[float_features] = dataframe[float_features].apply(lambda x: x.fillna(value=x.mean()))
    dataframe[object_features] = dataframe[object_features].apply(lambda x: x.fillna(value=x.mode()[0]))

    # ASSERT WE DON'T HAVE ANY MISSING VALUES IN THE DATASET
    assert dataframe.columns[dataframe.isnull().any()].empty, 'We still have some missing values in the dataset!'
    return dataframe

def get_cat_features(dataframe):
    categorical_features = [col for col in dataframe.columns if is_object_dtype(dataframe[col])]
    return categorical_features

def cat_encoding(dataframe, features):
    le = LabelEncoder()
    dataframe[features] = dataframe[features].apply(lambda x: le.fit_transform(x))
    return dataframe

def feature_engineering(dataframe, train=True):
    # FEATURE ENG
    dataframe = missing_values(dataframe)
    features_cat = get_cat_features(dataframe)
    dataframe = cat_encoding(dataframe, features_cat)
    features = dataframe.columns.difference(["Survived", "PassengerId", "kfold"])
    # RETURN DATAFRAME & ALL FEATURES NEEDED FOR TRAINING OR PREDICTION
    return dataframe, features