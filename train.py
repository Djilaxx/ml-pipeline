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
from trainer.trainer import Trainer
from utils.metrics import metrics_dict
from utils import folding

def train(run_number, folds=10, project="TPS-FEV2021", model_name="LGBM"):
    print(f"Starting run number {run_number}, training on project : {project} for {folds} folds with {model_name} model")
    config = getattr(importlib.import_module(f"projects.{project}.config"), "config")
    complete_name = f"{model_name}_{config.main.TASK}"
    #CREATING FOLDS
    folding.create_folds(datapath=config.main.TRAIN_FILE,
                        output_path=config.main.FOLD_FILE,
                        nb_folds = folds,
                        method=config.main.FOLD_METHOD,
                        target=config.main.TARGET_VAR)
                
    # LOADING DATA FILE & TOKENIZER
    df = pd.read_csv(config.main.FOLD_FILE)

    # FEATURE ENGINEERING
    feature_eng = getattr(importlib.import_module(f"projects.{project}.feature_eng"), "feature_engineering")
    df, features = feature_eng(df)

    # MODEL
    for name, func in inspect.getmembers(importlib.import_module(f"models.{complete_name}"), inspect.isfunction):
        if name == complete_name:
            model = func(**config.model[complete_name])
    
    # METRIC USED
    metric_selected = metrics_dict[config.train.METRIC]

    # START FOLD LOOP
    Path(os.path.join(config.main.PROJECT_PATH, "model_saved/")).mkdir(parents=True, exist_ok=True)
    for fold in range(folds):
        print(f"Starting training for fold : {fold}")
        
        # CREATING TRAINING AND VALIDATION SETS
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        target_train = df_train[config.main.TARGET_VAR].values
        target_valid = df_valid[config.main.TARGET_VAR].values
        
        # STARTING THE TRAINER
        trainer = Trainer(
            model = model, 
            train_x = df_train, 
            train_features = features,
            train_y = target_train,
            valid_x = df_valid,
            valid_y = target_valid,
        )
        # TRAIN THE MODEL
        trainer.fit(es = config.train.ES, verbose=config.train.VERBOSE)
        # VALIDATION STEP
        trainer.validate(metric = metric_selected, predict_proba = config.train.PREDICT_PROBA)

        #model.fit(
        #    df_train[features], 
        #    target_train, 
        #    eval_set=[(df_valid[features], target_valid)], 
        #    early_stopping_rounds=config.hyper.es, 
        #    verbose = 1000
        #)

        # SAVING THE MODEL
        joblib.dump(model, f"{config.main.PROJECT_PATH}/model_saved/{complete_name}_model_{fold+1}_{run_number}.joblib.dat")

##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--run_number", type=int)
parser.add_argument("--folds", type=int, default=10)
parser.add_argument("--project", type=str, default="TPS-FEV2021")
parser.add_argument("--model_name", type=str, default="LGBM")

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Training start...")
    train(
        run_number=args.run_number,
        folds=args.folds,
        project=args.project,
        model_name=args.model_name    
        )