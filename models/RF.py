from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def RF_REGRESSION(**params):
    return RandomForestRegressor(**params)

def RF_CLASSIFICATION(*params):
    return RandomForestClassifier(**params)