##################
# IMPORT MODULES #
##################
# SYS IMPORT
import os, importlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import joblib

def predict_for_fold(run_number, project, complete_name, test_data, fold=1, predict_proba=False):
    model = joblib.load(f"projects/{project}/model_saved/{complete_name}_model_{fold}_{run_number}.joblib.dat")
    if predict_proba is True:
        temp_test = model.predict_proba(test_data)[:, 1]
    else:
        temp_test = model.predict(test_data)
    return temp_test

def integer_predictions(preds, threshold=0.5):
    '''
    Return the predictions as integers
    '''
    final_int_preds = []
    for proba in preds:
        if proba >= threshold:
            pred_int = 1
        else:
            pred_int = 0
        final_int_preds.append(pred_int)
    return final_int_preds


def predict(run_number, project="TPS-FEV2021", model_name = "LGBM", get_int_preds=False):
    print(f"Predictions on project : {project}")
    config = getattr(importlib.import_module(f"projects.{project}.config"), "config")
    complete_name = f"{model_name}_{config.main.TASK}"

    # LOADING DATA FILE
    df = pd.read_csv(config.main.TEST_FILE)

    # FEATURE ENGINEERING
    feature_eng = getattr(importlib.import_module(f"projects.{project}.feature_eng"), "feature_engineering")
    df, features = feature_eng(df, train=False)
    
    # PREDICTIONS
    final_preds = []
    for i in range(config.main.FOLD_NUMBER):
        print(f"Starting prediction with model {i+1}")
        preds = predict_for_fold(run_number, project, complete_name, df[features], fold=i+1, predict_proba=config.train.PREDICT_PROBA)
        final_preds.append(preds)
    
    final_preds = np.mean(np.column_stack(final_preds), axis=1)

    #Check if we need integer preds
    if get_int_preds is True:
        final_preds = integer_predictions(final_preds)
    
    submission = pd.read_csv(config.main.SUBMISSION)
    submission[config.main.TARGET_VAR] = final_preds
    submission.to_csv(f'projects/{project}/submission_{complete_name}_{run_number}.csv', index=False)

##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--run_number", type=int)
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
        run_number=args.run_number,
        project=args.project, 
        model_name=args.model_name, 
        get_int_preds=args.get_int_preds
    )
