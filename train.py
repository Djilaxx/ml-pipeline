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
from utils.memory_usage import reduce_memory_usage


def train(project="TPS-FEV2021", model_name="LINEAR", run_note="test"):
    """
    Train, validate, and log results of a model on a specified dataset

    Parameters
    ----------
    project: str
        the name of the project you wish to work on - must be the name of the project folder under projects/
    model_name: str
        the name of the model you wish to train - name of the python file under models/
    run_note: str
        An string note for your current run

    Returns
    -------
    save trained model under projects/model_saved/
    print training results and log to wandb
    """
    
    # LOADING PROJECT CONFIG
    config = getattr(importlib.import_module(f"projects.{project}.config"), "config")
    if config.main.SPLIT is True:
        print(f"Starting run {run_note}, training on project : {project} with {model_name} model")
    else:
        print(
            f"Starting run {run_note}, training on project : {project} for {config.main.FOLD_NUMBER} folds with {model_name} model")
    complete_name = f"{model_name}_{config.main.TASK}"
    # RECORD RUNS USING WANDB TOOL
    wandb.init(config=config, project=project, name=complete_name + "_" + str(run_note))

    # CREATING FOLDS
    df = pd.read_csv(config.main.TRAIN_FILE)
    df = reduce_memory_usage(df, verbose=True)
    df = folding.create_splits(df=df,
                            task=config.main.TASK,
                            n_folds = config.main.FOLD_NUMBER, 
                            split = config.main.SPLIT, 
                            split_size=config.main.SPLIT_SIZE, 
                            target=config.main.TARGET_VAR)
                
    # FEATURE ENGINEERING
    feature_eng = getattr(importlib.import_module(f"projects.{project}.feature_eng"), "feature_engineering")

    # MODEL
    for name, func in inspect.getmembers(importlib.import_module(f"models.{model_name}"), inspect.isfunction):
        if name == complete_name:
            model = func(**config.model[complete_name])
    
    # METRIC USED
    metric_selected = metrics_dict[config.train.METRIC]

    # SETUP WANDB LOGS
    columns = ["fold", "training_score", "validation_score"]
    result_table = wandb.Table(columns=columns)

    # START FOLD LOOP
    Path(os.path.join(config.main.PROJECT_PATH, "model_saved/")).mkdir(parents=True, exist_ok=True)
    for fold in range(max(config.main.FOLD_NUMBER, 1)):
        print("Starting training...") if config.main.SPLIT is True else print(f"Starting training for fold {fold}")

        # CREATING TRAINING AND VALIDATION SETS
        df_train = df[df.splits != fold].reset_index(drop=True)
        df_train, features = feature_eng(df_train, train=True)
        df_valid = df[df.splits == fold].reset_index(drop=True)
        df_valid, _ = feature_eng(df_valid, train=False)

        target_train = df_train[config.main.TARGET_VAR].values
        target_valid = df_valid[config.main.TARGET_VAR].values

        # STARTING THE TRAINER
        trainer = Trainer(
            model = model, 
            train_x = df_train, 
            features = features,
            train_y = target_train,
            valid_x = df_valid,
            valid_y = target_valid,
            metric = metric_selected,
            proba = config.train.PREDICT_PROBA
        )

        # TRAIN THE MODEL
        training_results = trainer.fit(es = config.train.ES)
        # VALIDATION STEP
        validation_results = trainer.validate()

        result_table.add_data(fold, training_results["score"], validation_results["score"])
        wandb.log({f"Training score for fold : {fold}":training_results["score"], f"Validation score for fold : {fold}": validation_results["score"]})
        # SAVING THE MODEL
        joblib.dump(model, f"{config.main.PROJECT_PATH}/model_saved/{complete_name}_model_{run_note}_{fold+1}.joblib.dat")
        
        if config.main.TASK == "REGRESSION":
            wandb.sklearn.plot_summary_metrics(model, df_train[features], target_train, df_valid[features], target_valid)
            wandb.sklearn.plot_residuals(model, df_valid[features], target_valid)
        elif config.main.TASK == "CLASSIFICATION":
            wandb.sklearn.plot_summary_metrics(model, df_train[features], target_train, df_valid[features], target_valid)
            wandb.sklearn.plot_feature_importances(model)
            wandb.sklearn.plot_confusion_matrix(target_valid, validation_results["preds"], labels=None)
        else:
            raise Exception("Task not supported")

        # IF WE GO FOR A TRAIN - VALID SPLIT WE TRAIN ONE MODEL ONLY
        if config.main.SPLIT is True:
            break
    
    wandb.log({"result_table": result_table})
##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="TPS-FEV2021")
parser.add_argument("--model_name", type=str, default="LGBM")
parser.add_argument("--run_note", type=str, default="test")

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Training start...")
    train(
        project=args.project,
        model_name=args.model_name,
        run_note=args.run_note
    )
