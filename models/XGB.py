import xgboost as xgb 

def XGB_REGRESSION(**params):
    return xgb.XGBRegressor(**params)

def XGB_CLASSIFICATION(**params):
    return xgb.XGBClassifier(**params)