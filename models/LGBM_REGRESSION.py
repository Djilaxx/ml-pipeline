import lightgbm as lgb 

def LGBM_REGRESSION(**params):
    return lgb.LGBMRegressor(**params)