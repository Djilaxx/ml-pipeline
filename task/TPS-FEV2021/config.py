from easydict import EasyDict as edict
import optuna

config = edict()
########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "task/TPS-FEV2021/"
config.main.TRAIN_FILE = "data/TPS-FEV2021/train.csv"
config.main.TEST_FILE = "data/TPS-FEV2021/test.csv"
config.main.SUBMISSION = "data/TPS-FEV2021/sample_submission.csv"
config.main.FOLD_FILE = "data/TPS-FEV2021/train_folds.csv"
config.main.FOLD_NUMBER = 10
config.main.FOLD_METHOD = "KF"
config.main.TARGET_VAR = "target"

###################
# HYPERPARAMETERS #
###################
config.hyper = edict()
config.hyper.LGBM_REG = {
    'reg_alpha': 6.147694913504962,
    'reg_lambda': 0.002457826062076097,
    'colsample_bytree': 0.3,
    'subsample': 0.8,
    'learning_rate': 0.0005,
    'max_depth': 20,
    'num_leaves': 111,
    'min_child_samples': 285,
    'random_state': 48,
    'n_estimators': 100000,
    'metric': 'rmse',
    'cat_smooth': 39
}
config.hyper.XGB_REG = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.01,
    "colsample_bytree": 0.4,
    "subsample": 0.6,
    "reg_alpha" : 6,
    "min_child_weight": 100,
    "n_jobs": 2,
    "seed": 2001,
    'tree_method': "gpu_hist",
    "gpu_id": 0,
    'predictor': 'gpu_predictor'
}

config.hyper.LGBM_REG_OPT = {
    'random_state': 95,
    'metric': 'rmse',
    'n_estimators': 30000,
    'n_jobs': -1,
    'bagging_seed': 95,
    'feature_fraction_seed': 95,
    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2),
    'max_depth': trial.suggest_int('max_depth', 6, 127),
    'num_leaves': trial.suggest_int('num_leaves', 31, 128),
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.9),
    'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
    'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
    'subsample': trial.suggest_float('subsample', 0.3, 0.9),
    'max_bin': trial.suggest_int('max_bin', 128, 1024),
    'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200),
    'cat_smooth': trial.suggest_int('cat_smooth', 10, 100),
    'cat_l2': trial.suggest_int('cat_l2', 1, 20)
}