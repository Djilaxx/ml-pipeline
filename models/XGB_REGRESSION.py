import xgboost as xgb 

def XGB_REGRESSION(**params):
    return xgb.XGBRegressor(**params)