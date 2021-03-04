import lightgbm as lgb 

def LGBM_CL(**params):
    return lgb.LGBMClassifier(**params)