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
import joblib


import matplotlib.pyplot as plt
import lightgbm as lgb 
from utils import folding

def predict_for_fold(project, complete_name, test_data, fold=1, predict_proba=False):
    model = joblib.load(f"project/{project}/model_saved/{complete_name}_model_{fold}.joblib.dat")
    #model = lgb.Booster(model_file=f"task/{task}/model_saved/model_{fold}.txt")
    if predict_proba is True:
        temp_test = model.predict_proba(test_data)[:, 1]
    else:
        temp_test = model.predict(test_data)
    return temp_test

def predict(project="TPS-FEV2021", model_name = "LGBM", get_int_preds=False):
    print(f"Predictions on project : {project}")
    config = getattr(importlib.import_module(f"project.{project}.config"), "config")
    complete_name = f"{model_name}_{config.main.TASK}"

    # LOADING DATA FILE
    df = pd.read_csv(config.main.TEST_FILE)

    # FEATURE ENGINEERING
    feature_eng = getattr(importlib.import_module(f"project.{project}.feature_eng"), "feature_engineering")
    df, features = feature_eng(df, train=False)
    
    # PREDICTIONS
    final_preds = np.zeros(len(df))
    for i in range(config.main.FOLD_NUMBER):
        print(f"Starting prediction with model {i+1}")
        preds = predict_for_fold(project, complete_name, df[features], fold=i+1, predict_proba=config.train.PREDICT_PROBA)
        final_preds += preds
    
    final_preds /= config.main.FOLD_NUMBER
    if get_int_preds is True:
        final_int_preds = []
        threshold=0.5
        for proba in final_preds:
            if proba >= threshold:
                pred_int = 1
            else:
                pred_int = 0
            final_int_preds.append(pred_int)
        submission = pd.read_csv(config.main.SUBMISSION)
        submission[config.main.TARGET_VAR] = final_int_preds
        submission.to_csv(f'project/{project}/submission_{complete_name}.csv', index=False)
    else:
        submission = pd.read_csv(config.main.SUBMISSION)
        submission[config.main.TARGET_VAR] = final_preds
        submission.to_csv(f'project/{project}/submission_{complete_name}.csv', index=False)

##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="TPS-FEV2021")
parser.add_argument("--model_name", type=str, default="LGBM")
parser.add_argument("--get_int_preds", type=bool, default=False)

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Prediction start...")
    predict(
        project=args.project,
        model_name=args.model_name,
        get_int_preds=args.get_int_preds
    )