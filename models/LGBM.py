import lightgbm as lgb 

def LGBM_REGRESSION(**params):
    return lgb.LGBMRegressor(**params)

def LGBM_CLASSIFICATION(**params):
    return lgb.LGBMClassifier(**params)