from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

def LINEAR_REGRESSION(**params):
    return LinearRegression(**params)

def LINEAR_CLASSIFICATION(**params):
    return LogisticRegression(**params)