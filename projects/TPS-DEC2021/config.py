from easydict import EasyDict as edict

config = edict()
########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "projects/TPS-DEC2021/"
config.main.TRAIN_FILE = "data/TPS-DEC2021/train.csv"
config.main.TEST_FILE = "data/TPS-DEC2021/test.csv"
config.main.SUBMISSION = "data/TPS-DEC2021/sample_submission.csv"
config.main.FOLD_FILE = "data/TPS-DEC2021/train_folds.csv"
config.main.TASK = "CLASSIFICATION"
config.main.SPLIT = True
config.main.FOLD_NUMBER = 10
config.main.SPLIT_SIZE = 0.10
config.main.TARGET_VAR = "Cover_Type"
#######################
# TRAINING PARAMETERS #
#######################
config.train = edict()
config.train.ES = 50
config.train.VERBOSE = 100
config.train.METRIC = "ACCURACY"
config.train.PREDICT_PROBA = False

####################
# MODEL PARAMETERS #
####################
###################
# HYPERPARAMETERS #
###################
config.model = edict()

config.model.LINEAR_CLASSIFICATION = {
    "penalty": "l2",
    "random_state": 95,
    "max_iter": 100,
}

config.model.LGBM_CLASSIFICATION = {
    'objective': 'binary',
    'metric': 'accuracy',
    'colsample_bytree': 0.3,
    'learning_rate': 0.01,
    'max_depth': 100,
    'min_child_samples': 295,
    'cat_smooth': 99,
    'n_estimators': 6126,
    'num_leaves': 30,
    'reg_alpha': 0.623563042492307,
    'reg_lambda': 0.5894900631467016,
    'subsample': 0.4
}

config.model.XGB_CLASSIFICATION = {
    "eval_metric": "merror",
    "seed": 95,
    'tree_method': "gpu_hist",
    'predictor': 'gpu_predictor',
    'use_label_encoder': True,
    "n_estimators": 4989,
    # Max should correspond to max number of features (probably ?),
    "max_depth": 25,
    "learning_rate": 0.006,
    "colsample_bytree": 0.7,
    "subsample": 0.8,
    "reg_lambda": 0.0009399371977626809,
    "reg_alpha": 1.2158348997921837e-06,
    "min_child_weight": 269}

config.model.CATBOOST_CLASSIFICATION = {
    'loss_function': 'CrossEntropy',
    'task_type': "GPU",
    'eval_metric': 'ACCURACY',
    'leaf_estimation_method': 'Newton',
    'bootstrap_type': 'Bernoulli',
    'depth': 6,
    'iterations': 14988,
    'leaf_estimation_iterations': 15,
    'learning_rate': 0.020255100952810502,
    'min_data_in_leaf': 8,
    'od_wait': 616,
    'random_strength': 30.028725416323482,
    'reg_lambda': 87.1683592964532,
    'subsample': 0.472136846097046
}
