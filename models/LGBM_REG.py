import lightgbm as lgb 

def LGBM_REG(**params):
    return lgb.LGBMRegressor(**params)