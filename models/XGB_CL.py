import xgboost as xgb 

def XGB_CL(**params):
    return xgb.XGBClassifier(**params)