from catboost import CatBoostRegressor, CatBoostClassifier

def CATBOOST_REGRESSION(**params):
    return CatBoostRegressor(**params)

def CATBOOST_CLASSIFICATION(**params):
    return CatBoostClassifier(**params)
