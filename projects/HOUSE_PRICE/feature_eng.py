import datetime
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from utils import feature_selection

def missing_values(dataframe):
    # Droping features that have too many MV
    drop_features = ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]
    dataframe = dataframe.drop(drop_features, axis=1)
    # Garage features
    garage_features = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
    dataframe[garage_features] = dataframe[garage_features].fillna(value="No Garage")
    # Basement features
    basement_features = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
    dataframe[basement_features] = dataframe[basement_features].fillna(value="No Basement")
    # Lot Frontage
    dataframe["LotFrontage"] = dataframe["LotFrontage"].fillna(value=0)

    # OTHER VARIABLES
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

def year_var(dataframe):
    dataframe["GarageAge"] = datetime.datetime.now().year - dataframe["GarageYrBlt"]
    dataframe["SoldAge"] = datetime.datetime.now().year - dataframe["YrSold"]
    dataframe = dataframe.drop(["GarageYrBlt", "YrSold"], axis=1)
    return dataframe

def get_cat_features(dataframe):
    categorical_features = [col for col in dataframe.columns if is_object_dtype(dataframe[col])]
    return categorical_features

def cat_encoding(dataframe, features):
    le = LabelEncoder()
    dataframe[features] = dataframe[features].apply(lambda x: le.fit_transform(x))
    return dataframe

def feature_select(dataframe):
    ufs = feature_selection.UnivariateFeatureSelction(
        n_features=0.9,
        problem_type="regression",
        scoring="f_regression"
    )
    features = dataframe.columns.difference(["Id", "SalePrice", "kfold"]).values.tolist()
    ufs.fit(dataframe[features], dataframe["SalePrice"].values.ravel())
    selected_features = ufs.return_cols(dataframe[features])
    return selected_features

def feature_engineering(dataframe, train=True):
    # FEATURE ENG
    dataframe = missing_values(dataframe)
    dataframe = year_var(dataframe)
    features_cat = get_cat_features(dataframe)
    dataframe = cat_encoding(dataframe, features_cat)
    if train:
        features = dataframe.columns.difference(["Id", "SalePrice", 'kfold'])
    else:
        features = dataframe.columns.difference(["Id"])
    # RETURN DATAFRAME & ALL FEATURES NEEDED FOR TRAINING OR PREDICTION
    return dataframe, features
