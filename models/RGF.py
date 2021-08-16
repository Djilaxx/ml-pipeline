from rgf.sklearn import RGFRegressor, RGFClassifier

def RGF_REGRESSION(**params):
    return RGFRegressor(**params)

def RGF_CLASSIFICATION(**params):
    return RGFClassifier(**params)