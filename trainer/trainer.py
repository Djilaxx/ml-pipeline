import numpy as np
import wandb

class Trainer:
    """
    Trainer class to fit a model and validate it on a given dataset.

    Parameters
    ----------
    model: object
        the model object that is to be fitted
    train_x: list
        list containing training observations
    features: list
        list of names of features we want to use for training
    train_y: list
        list of values for the target variable in the training set
    valid_x: list
        list containing validation observations
    valid_y: list
        list of values for the target variable in the validation set
    metric: object
        function for the metric selected
    proba: boolean
        if True the trainer will keep predictions as probability scores (useful for metrics like AUC which need proba-like values)

    """
    def __init__(
        self, 
        model, 
        train_x, 
        features, 
        train_y, 
        valid_x=None, 
        valid_y=None, 
        metric=None, 
        proba=False
    ):

        self.model = model
        self.train_x = train_x[features]
        self.train_y = train_y
        self.valid_x = valid_x[features]
        self.valid_y = valid_y
        self.model_name = self.model.__class__.__name__
        self.metric = metric
        self.proba = proba
    
    def fit(self, es = None):
        """
        Fit the given model using the training set train_x, train_y

        Parameters
        ----------
        es: int
            number of early stopping round without validation loss improving before we stop the training
            this parameter work only for gradient boosted trees (eg. XGB, LGBM, Catboost models)

        Returns
        -------
        self: object
            fitted model
        dict: dictionnary
            return a dictionnary containing:
                - preds: ndarray
                    predictions of the model on the training set
                - preds_proba: ndarray
                    when training a classifier & if proba=True, return the predictions as probabilities
                - score: float
                    model score on the training set using provided metric
        """

        GBM_model = ["LGBMRegressor", "LGBMClassifier", "XGBRegressor", "XGBClassifier", "CatBoostRegressor", "CatBoostClassifier"]
        # fit method is different for Boosted trees
        if self.model_name in GBM_model:
            # Adding wandb_callback if we train with xgboost or lgbm
            self.model.fit(self.train_x, self.train_y, eval_set=[(self.valid_x, self.valid_y)], early_stopping_rounds=es, verbose=200)
        else:
            self.model.fit(self.train_x, self.train_y)
        
    # Prediction on the training set
        if self.proba is True:
            preds_proba_full = self.model.predict_proba(self.train_x)
            preds_proba = preds_proba_full[:, 1]
            preds_int = preds_proba_full.argmax(axis=1)
            metric_score = self.metric(self.train_y, preds_proba)
            return {
                "preds": preds_int,
                "preds_proba_full": preds_proba_full,
                "preds_proba": preds_proba,
                "score" : metric_score
            }
        else:
            predictions = self.model.predict(self.train_x)
            metric_score = self.metric(self.train_y, predictions)
    # SQUARED ERROR MUST BE SQUARE ROOTED TO MAKE SENSE
            if self.metric.__name__ == "mean_squared_error":
                metric_score = np.sqrt(metric_score)

            return {
                "preds": predictions,
                "score": metric_score
            }

    def validate(self):
        """
        Validate the fitted model on the validation set (valid_x, valid_y)

        Parameters
        ----------

        Returns
        -------
        dict: dictionnary
            return a dictionnary containing:
                - preds: ndarray
                    predictions of the model on the validation set
                - preds_proba: ndarray
                    when validating a classifier & if proba=True, return the predictions as probabilities
                - score: float
                    model score on the validation set using provided metric
        """

        if self.proba is True:
            preds_proba_full = self.model.predict_proba(self.valid_x)
            preds_proba = preds_proba_full[:, 1]
            preds_int = preds_proba_full.argmax(axis=1)
            metric_score = self.metric(self.valid_y, preds_proba)
            print(f"Validation score {metric_score}")
            return {
                "preds": preds_int,
                "preds_proba_full": preds_proba_full,
                "preds_proba": preds_proba,
                "score": metric_score
            }

        else:
            predictions = self.model.predict(self.valid_x)
            metric_score = self.metric(self.valid_y, predictions)
            # SQUARED ERROR MUST BE SQUARE ROOTED TO MAKE SENSE
            if self.metric.__name__ == "mean_squared_error":
                metric_score = np.sqrt(metric_score)
            print(f"Validation score {metric_score}")
            return {
                "preds": predictions,
                "score": metric_score
            }
