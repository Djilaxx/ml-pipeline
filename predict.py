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
import lightgbm as lgb 
from utils import folding

def predict_for_fold(task, test_data, columns, fold=1):
    model_path = f"model_{fold}.txt"
    model = lgb.Booster(f"{task}/model_saved/model_{fold}.txt")
    temp_test = model.predict(test_data[columns])
    return temp_test

def predict(task="TPS-FEV2021"):
    print(f"Predictions on task : {task}")
    config = getattr(importlib.import_module(f"task.{task}.config"), "config")

    # LOADING DATA FILE & TOKENIZER
    df = pd.read_csv(config.main.TEST_FILE)

    # FEATURE ENGINEERING
    feature_eng = getattr(importlib.import_module(f"task.{task}.feature_eng"), "feature_engineering")
    df, columns = feature_eng(df)
    
    # PREDICTIONS
    final_preds = 0
    for i in range(config.main.FOLD_NUMBER):
        preds = predict_for_fold(task, df, columns, fold=i)
        final_preds += preds
    
    final_preds /= config.main.FOLD_NUMBER

    submission = pd.read_csv(config.main.SUBMISSION)
    submission[config.main.TARGET_VAR] = final_preds

    sub.to_csv(f'{task}/submission.csv', index=False)

##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="TPS-FEV2021")

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Training start...")
    predict(
        task=args.task
    )