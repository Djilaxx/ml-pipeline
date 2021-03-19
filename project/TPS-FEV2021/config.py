from easydict import EasyDict as edict
import optuna

config = edict()
########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "project/TPS-FEV2021/"
config.main.TRAIN_FILE = "data/TPS-FEV2021/train.csv"
config.main.TEST_FILE = "data/TPS-FEV2021/test.csv"
config.main.SUBMISSION = "data/TPS-FEV2021/sample_submission.csv"
config.main.FOLD_FILE = "data/TPS-FEV2021/train_folds.csv"
config.main.FOLD_NUMBER = 10
config.main.FOLD_METHOD = "KF"
config.main.TARGET_VAR = "target"
#######################
# TRAINING PARAMETERS #
#######################
config.train = edict()
config.train.ES = 200
config.train.VERBOSE = 1000
config.train.METRIC = "MSE"
config.train.PREDICT_PROBA = False
####################
# MODEL PARAMETERS #
####################
###################
# HYPERPARAMETERS #
###################
config.model = edict()

config.model.LIN_REG = {}
config.model.LGBM_REG = {
    'reg_alpha': 6.147694913504962,
    'reg_lambda': 0.002457826062076097,
    'colsample_bytree': 0.3,
    'subsample': 0.8,
    'learning_rate': 0.0005,
    'max_depth': 20,
    'num_leaves': 111,
    'min_child_samples': 285,
    'random_state': 95,
    'n_estimators': 5000,
    'metric': 'rmse',
    'cat_smooth': 39
}

config.model.XGB_REG = {
    "objective": "reg:squarederror",
    "n_estimators" : 4000,
    "max_depth": 6,
    "learning_rate": 0.01,
    "colsample_bytree": 0.4,
    "subsample": 0.6,
    "reg_alpha" : 6,
    "min_child_weight": 100,
    "n_jobs": 2,
    "seed": 95,
    'tree_method': "gpu_hist",
    "gpu_id": 0,
    'predictor': 'gpu_predictor'
}
