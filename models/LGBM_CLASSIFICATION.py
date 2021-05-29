import lightgbm as lgb 

def LGBM_CLASSIFICATION(**params):
    return lgb.LGBMClassifier(**params)