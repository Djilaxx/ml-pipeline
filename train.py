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

from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import folding
import xgboost as xgb

def train(folds=10, task="TPS-FEV2021", lib="LGBM", model_type="REG"):
    model_name = f"{lib}_{model_type}"
    print(f"Training on task : {task} for {folds} folds with {model_name} model")
    config = getattr(importlib.import_module(f"task.{task}.config"), "config")

    #CREATING FOLDS
    folding.create_folds(datapath=config.main.TRAIN_FILE,
                        output_path=config.main.FOLD_FILE,
                        nb_folds = folds,
                        method=config.main.FOLD_METHOD,
                        target=config.main.TARGET_VAR)
                
    # LOADING DATA FILE & TOKENIZER
    df = pd.read_csv(config.main.FOLD_FILE)

    # FEATURE ENGINEERING
    feature_eng = getattr(importlib.import_module(f"task.{task}.feature_eng"), "feature_engineering")
    df, features = feature_eng(df)

    # MODEL
    for name, func in inspect.getmembers(importlib.import_module(f"models.{model_name}"), inspect.isfunction):
        if name == model_name:
            model = func(**config.hyper[model_name])

    # START FOLD LOOP
    Path(os.path.join(config.main.PROJECT_PATH, "model_saved/")).mkdir(parents=True, exist_ok=True)
    for fold in range(folds):
        print(f"Starting training for fold : {fold}")
        
        # CREATING TRAINING AND VALIDATION SETS
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        target_train = df_train[config.main.TARGET_VAR].values
        target_valid = df_valid[config.main.TARGET_VAR].values
        
        # TRAINING A LGBM MODEL
        model.fit(
            df_train[features], 
            target_train, 
            eval_set=[(df_valid[features], target_valid)], 
            early_stopping_rounds=config.hyper.es, 
            verbose = 1000
        )

        # MODEL SAVING
        joblib.dump(model, f"{config.main.PROJECT_PATH}/model_saved/{model_name}_model_{fold+1}.joblib.dat")

##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--folds", type=int, default=10)
parser.add_argument("--task", type=str, default="TPS-FEV2021")
parser.add_argument("--lib", type=str, default="LGBM")
parser.add_argument("--model_type", type=str, default="REG")

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Training start...")
    train(
        folds=args.folds,
        task=args.task,
        lib=args.lib,
        model_type=args.model_type
    )