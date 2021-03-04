##################
# IMPORT MODULES #
##################
# SYS IMPORT
import os, inspect, importlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import pandas as pd
import numpy as np
from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import folding

def objective(trial, X=train_df, y=target):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=SEED)
    obj_params = {'random_state': SEED,
                  'metric': 'rmse',
                  'n_estimators': N_ESTIMATORS,
                  'n_jobs': -1,
                  'cat_feature': [x for x in range(len(cat_features))],
                  'bagging_seed': SEED,
                  'feature_fraction_seed': SEED,
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

    obj_model = LGBMRegressor(**obj_params)
    obj_model.fit(train_x, train_y, eval_set=(test_x, test_y), early_stopping_rounds=100, verbose=False)
    obj_preds = obj_model.predict(test_x, num_iteration=obj_model.best_iteration_)
    obj_rmse = mean_squared_error(test_y, obj_preds, squared=False)
    return obj_rmse