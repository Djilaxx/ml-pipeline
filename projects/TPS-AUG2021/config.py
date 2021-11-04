from easydict import EasyDict as edict

config = edict()
########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "projects/TPS-AUG2021/"                  # PATH TO PROJECT
config.main.TRAIN_FILE = "data/TPS-AUG2021/train.csv"               # PATH TO TRAINING CSV FILE
config.main.TEST_FILE = "data/TPS-AUG2021/test.csv"                 # PATH TO TEST CSV FILE
config.main.SUBMISSION = "data/TPS-AUG2021/sample_submission.csv"   # PATH TO SUBMISSION FILE IF PROVIDED
config.main.FOLD_FILE = "data/TPS-AUG2021/train_folds.csv"
config.main.TASK = "REGRESSION"                                     # TASK - REGRESSION OR CLASSIFICATION
config.main.SPLIT = False                                           # IF SET TO TRUE - DATASET WILL BE SPLIT IN TWO (TRAIN AND VALIDATION) AND TRAINED ONE TIME - IF FALSE WILL CREATE FOLD_NUMBER OF SPLITS IN THE DATASET AND TRAIN MULTIPLE TIMES
config.main.FOLD_NUMBER = 5                                        # NOT IMPORTANT IF SPLIT = True - ONLY USED IF CREATING MULTIPLE SPLITS
config.main.SPLIT_SIZE = 0.2                                        # SPLIT_SIZE OF VALIDATION SET 
config.main.TARGET_VAR = "loss"                                     # NAME OF THE TARGET VAR

#######################
# TRAINING PARAMETERS #
#######################
config.train = edict()
config.train.ES = 20                                                # NUMBER OF ITERATION WITHOUT VALIDATION LOSS IMPROVING BEFORE WE STOP TRAINING
config.train.VERBOSE = 1000 
config.train.METRIC = "MSE"                                         # TRAINING METRIC
config.train.PREDICT_PROBA = False                                  # SHOULD PREDICTIONS BE INT OR PROBABILITY-LIKE VALUES (ONLY FOR CLASSIFICATION TAKS)
####################
# MODEL PARAMETERS #
####################
###################
# HYPERPARAMETERS #
###################
config.model = edict()                                              # HYPERPARAMETERS FOR MODELS

config.model.LINEAR_REGRESSION = {}
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