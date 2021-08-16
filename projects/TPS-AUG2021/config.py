from easydict import EasyDict as edict

config = edict()
########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "projects/TPS-AUG2021/"
config.main.TRAIN_FILE = "data/TPS-AUG2021/train.csv"
config.main.TEST_FILE = "data/TPS-AUG2021/test.csv"
config.main.SUBMISSION = "data/TPS-AUG2021/sample_submission.csv"
config.main.FOLD_FILE = "data/TPS-AUG2021/train_folds.csv"
config.main.TASK = "REGRESSION"
config.main.FOLD_NUMBER = 10
config.main.SPLIT_SIZE = 0.2
config.main.TARGET_VAR = "loss"
config.main.TARGET_CONTINUOUS = False
#######################
# TRAINING PARAMETERS #
#######################
config.train = edict()
config.train.ES = 20
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

config.model.LIN_REGRESSION = {}
config.model.LGBM_REGRESSION = {
    'reg_alpha': 0.04616889056961417,
    'reg_lambda': 1.826326408952523,
    'colsample_bytree': 0.1,
    'subsample': 0.8,
    'learning_rate': 0.006,
    'max_depth': 10,
    'num_leaves': 111,
    'min_child_samples': 147,
    'random_state': 95,
    'n_estimators': 5000,
    'metric': 'rmse',
    'cat_smooth': 84
}

config.model.XGB_REGRESSION = {
    "objective": "reg:squarederror",
    "n_estimators" : 10000,
    "max_depth": 10,
    "learning_rate": 0.006,
    "colsample_bytree": 0.5,
    "subsample": 0.6,
    "reg_alpha" : 0.006221417528979453,
    "reg_lambda": 3.178956727410822e-07,
    "min_child_weight": 123,
    "n_jobs": 2,
    "seed": 95,
    'tree_method': "gpu_hist",
    "gpu_id": 0,
    'predictor': 'gpu_predictor'
}

config.model.CATBOOST_REGRESSION = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "task_type" : "GPU",
    "n_estimators": 21541,
    "random_state" : 95,
    "od_wait" : 2000,
    "random_strength": 12.157840392437768,
    "learning_rate": 0.02125056331011215,
    "reg_lambda": 24.506871406961594,
    "subsample": 0.6295671971146753,
    "max_depth": 6,
    "min_child_samples": 10,
    "leaf_estimation_iterations":1,
}

config.model.RGF_REGRESSION = {
    "loss": "LS",
    "algorithm": "RGF_Sib",
    "learning_rate": 0.01
}