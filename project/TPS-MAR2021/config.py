from easydict import EasyDict as edict

config = edict()
########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "project/TPS-MAR2021/"
config.main.TRAIN_FILE = "data/TPS-MAR2021/train.csv"
config.main.TEST_FILE = "data/TPS-MAR2021/test.csv"
config.main.SUBMISSION = "data/TPS-MAR2021/sample_submission.csv"
config.main.FOLD_FILE = "data/TPS-MAR2021/train_folds.csv"
config.main.FOLD_NUMBER = 10
config.main.FOLD_METHOD = "SKF"
config.main.TARGET_VAR = "target"
#######################
# TRAINING PARAMETERS #
#######################
config.train = edict()
config.train.ES = 200
config.train.VERBOSE = 1000
config.train.METRIC = "AUC"
config.train.PREDICT_PROBA = False

####################
# MODEL PARAMETERS #
####################
###################
# HYPERPARAMETERS #
###################
config.model = edict()

config.model.LOGIT_CL = {
    "penalty" : "l2",
    "random_state" : 95,
    "max_iter" : 100,
}

config.model.LGBM_CL = {
    'objective': 'binary',
    'metric' : 'auc',
    'n_estimators' : 100000,
    'random_state' : 95,
    'cat_smooth' : 30,
    'reg_alpha': 0.0031293275223408185,
    'reg_lambda': 0.04787145507141445,
    'colsample_bytree': 0.3,
    'subsample': 0.6,
    'learning_rate': 0.008,
    'max_depth': 100,
    'num_leaves': 584,
    'min_child_samples': 173
}

config.model.XGB_CL = {
    "objective": "binary:logistic",
    "eval_metric" : "auc",
    "seed": 95,
    'tree_method': "gpu_hist",
    'predictor': 'gpu_predictor',
    'use_label_encoder' : False,
    "n_estimators" : 20000,
    'max_bin' : 64,
    "max_depth": 12, #Max should correspond to max number of features (probably ?),
    'alpha' : 11.607239831188968,
    'gamma' : 2.1593805822598444,
    "learning_rate": 0.02,
    "colsample_bytree": 0.8016656211574054,
    "subsample": 0.983461992112787,
    "reg_alpha" : 1.7306711078859136,
    "min_child_weight": 9.417969426623086,
    "n_jobs": 2
}
