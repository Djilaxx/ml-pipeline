import numpy as np
class Trainer:
    def __init__(self, model, train_x, train_features, train_y, valid_x=None, valid_y=None):
        self.model = model
        self.train_x = train_x[train_features]
        self.train_y = train_y
        self.valid_x = valid_x[train_features]
        self.valid_y = valid_y
        self.model_name = self.model.__class__.__name__
    
    def fit(self, es = None, verbose = None):
        GBM_model = ["LGBMRegressor", "LGBMClassifier", "XGBRegressor", "XGBClassifier"]

        if self.model_name in GBM_model:
            self.model.fit(self.train_x, self.train_y, eval_set=[(self.valid_x, self.valid_y)], early_stopping_rounds=es, verbose=verbose)
        else:
            self.model.fit(self.train_x, self.train_y)

    def validate(self, metric = None, predict_proba = False):
        if metric is None:
            return print("No Validation metric selected...")
        else:
            if predict_proba is True:
                temp_valid = self.model.predict_proba(self.valid_x)[:, 1]
            else:
                temp_valid = self.model.predict(self.valid_x)
            metric_score = metric(self.valid_y, temp_valid)
        print(f"Validating using metric : {metric.__name__}")
        if metric.__name__ == "mean_squared_error":
            print(f"Validation score {np.sqrt(metric_score)}")
        else:
            print(f"Validation score {metric_score}")
        return metric_score