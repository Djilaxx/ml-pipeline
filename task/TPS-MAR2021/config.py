from easydict import EasyDict as edict

config = edict()
########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "task/TPS-MAR2021/"
config.main.TRAIN_FILE = "data/TPS-MAR2021/train.csv"
config.main.TEST_FILE = "data/TPS-MAR2021/test.csv"
config.main.SUBMISSION = "data/TPS-MAR2021/sample_submission.csv"
config.main.FOLD_FILE = "data/TPS-MAR2021/train_folds.csv"
config.main.FOLD_NUMBER = 10
config.main.FOLD_METHOD = "SKF"
config.main.TARGET_VAR = "target"

###################
# HYPERPARAMETERS #
###################
config.hyper = edict()
config.hyper.LGBM_CL = {
    'objective': 'binary',
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
    'metric': 'auc',
    'cat_smooth': 39
}
