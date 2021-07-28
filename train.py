##################
# IMPORT MODULES #
##################
# SYS IMPORT
import os, inspect, importlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from pathlib import Path
import joblib
import wandb

from trainer.trainer import Trainer
from utils.metrics import metrics_dict
from utils import folding

def train(run_number, folds=10, project="TPS-FEV2021", model_name="LGBM"):
    print(f"Starting run number {run_number}, training on project : {project} for {folds} folds with {model_name} model")
    # LOADING PROJECT CONFIG
    config = getattr(importlib.import_module(f"projects.{project}.config"), "config")
    complete_name = f"{model_name}_{config.main.TASK}"
    # RECORD RUNS USING WANDB TOOL
    wandb.init(config = config, project = project, name = complete_name + "_" + str(run_number))
    #CREATING FOLDS
    folding.create_splits(datapath=config.main.TRAIN_FILE,
                        output_path=config.main.FOLD_FILE,
                        n_folds = folds,
                        split_size=config.main.SPLIT_SIZE,
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

    for fold in range(max(folds, 1)):
        print(f"Starting training for fold : {fold}")
        
        # CREATING TRAINING AND VALIDATION SETS
        df_train = df[df.split != fold].reset_index(drop=True)
        df_valid = df[df.split == fold].reset_index(drop=True)

        target_train = df_train[config.main.TARGET_VAR].values
        target_valid = df_valid[config.main.TARGET_VAR].values
        
        # STARTING THE TRAINER
        trainer = Trainer(
            model = model, 
            train_x = df_train, 
            train_features = features,
            train_y = target_train,
            valid_x = df_valid,
            valid_y = target_valid
        )
        # TRAIN THE MODEL
        training_score = trainer.fit(metric = metric_selected, 
                                    es = config.train.ES, 
                                    verbose=config.train.VERBOSE, 
                                    predict_proba = config.train.PREDICT_PROBA)
        # VALIDATION STEP
        valid_score = trainer.validate(metric = metric_selected, 
                                    predict_proba = config.train.PREDICT_PROBA)

        wandb.log({f"Training score for fold : {fold}": training_score, f"Validation score for fold : {fold}": valid_score})
        # SAVING THE MODEL
        joblib.dump(model, f"{config.main.PROJECT_PATH}/model_saved/{complete_name}_model_{fold+1}_{run_number}.joblib.dat")
        
        # IF WE GO FOR A TRAIN - VALID SPLIT WE TRAIN ONE MODEL ONLY (folds=0 or 1)
        if folds < 2:
            break
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