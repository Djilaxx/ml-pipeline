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


def train(folds=10, task="TPS-FEV2021", model="LGBM_REG", loss="MSE", metric="MSE"):
    print(f"Training on task : {task} for {folds} folds with {model} model")
    print(f"{loss} loss & {metric} metric")
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
    df, columns = feature_eng(df)

    
    # MODEL
    for name, func in inspect.getmembers(importlib.import_module("models." + model), inspect.isfunction):
        if name == model:
            model = func(**config.hyper[model])

    # START FOLD LOOP

    for fold in range(folds):
        print(f"Starting training for fold : {fold + 1}")
        
        # CREATING TRAINING AND VALIDATION SETS
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        target_train = df_train[config.main.TARGET_VAR].values
        target_valid = df_valid[config.main.TARGET_VAR].values

        df_train = df_train.drop([config.main.TARGET_VAR], axis = 1)
        df_valid = df_valid.drop([config.main.TARGET_VAR], axis = 1)

        # MODEL TRAINING
        model.fit(df_train[columns], target_train, eval_set=[(df_valid[columns], target_valid)], early_stopping_rounds=1600, verbose = 1000)

        # MODEL SAVING
        model.booster_.save_model(f"task/{task}/model_{fold + 1}.txt") # MODEL CAN BE LOADED AFTERWARDS USING lgb.Booster(model_file=model.txt)

##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--folds", type=int, default=10)
parser.add_argument("--task", type=str, default="TPS-FEV2021")
parser.add_argument("--model", type=str, default="LGBM_REG")
parser.add_argument("--loss", type=str, default="MSE")
parser.add_argument("--metric", type=str, default="MSE")

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Training start...")
    train(
        folds=args.folds,
        task=args.task,
        model=args.model,
        loss=args.loss,
        metric=args.metric
    )