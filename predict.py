##################
# IMPORT MODULES #
##################
# SYS IMPORT
import os, importlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import joblib


def predict_for_fold(name, test_data, fold=1, proba=False):
    """
    Use a trained model to create predictions

    Parameters
    ----------
    name: str
        the complete name of the model you wish to predict with
    test_data: list
        the test data you wish to use to create predictions
    fold: int
        fold your model was trained on
    proba: boolean
        if true will predict_proba and produce probability-like predictions

    Returns
    -------
    temp_test: ndarray
        array of predictions from the model
    """

    model = joblib.load(f"{name}_{fold}.joblib.dat")
    if proba is True:
        temp_test = model.predict_proba(test_data)[:, 1]
    else:
        temp_test = model.predict(test_data)
    return temp_test


def integer_predictions(preds, threshold=0.5):
    """
    transform your predictions into integers equivalent

    Parameters
    ----------
    preds: ndarray
        array of predictions coming from your model
    threshold: float in [0 , 1]
        Value over which prediction is set to 1 - 0 otherwise
    
    Returns
    -------
    Create a csv file containing the model predictions
    """

    final_int_preds = []
    for proba in preds:
        if proba >= threshold:
            pred_int = 1
        else:
            pred_int = 0
        final_int_preds.append(pred_int)
    return final_int_preds


def predict(project="TPS-FEV2021", model_name = "LGBM", run_note="test", get_int_preds=False):
    """
    Predict on a test dataset using a trained model

    Parameters
    ----------
    project: str
        the name of the project you wish to work on - must be the name of the project folder under projects/
    model_name: str
        the name of the model you wish to use to predict - name of the python file under models/
    run_note: str
        An string note for your current run
    get_int_preds: boolean
        if true will return predictions as integers instead of float probabilities
    Returns
    -------
    Create a csv file containing the model predictions
    """
    
    print(f"Predictions on project : {project}")
    config = getattr(importlib.import_module(f"projects.{project}.config"), "config")
    complete_name = f"projects/{project}/model_saved/{model_name}_{config.main.TASK}_model_{run_note}"

    # LOADING DATA FILE
    df = pd.read_csv(config.main.TEST_FILE)

    # FEATURE ENGINEERING
    feature_eng = getattr(importlib.import_module(f"projects.{project}.feature_eng"), "feature_engineering")
    df, features = feature_eng(df, train=False)
    
    # PREDICTIONS
    final_preds = []
    for fold in range(max(config.main.FOLD_NUMBER, 1)):
        print(f"Starting prediction with model {fold+1}")
        preds = predict_for_fold(complete_name, df[features], fold=fold+1, proba=config.train.PREDICT_PROBA)
        final_preds.append(preds)

        # IF TRUE WE ONLY TRAINED ONE MODEL SO WE STOP THE LOOP
        if config.main.SPLIT is True:
            break
    
    final_preds = np.mean(np.column_stack(final_preds), axis=1)

    #Check if we need integer preds
    if get_int_preds is True:
        final_preds = integer_predictions(final_preds)
    
    submission = pd.read_csv(config.main.SUBMISSION)
    submission[config.main.TARGET_VAR] = final_preds
    submission.to_csv(f'projects/{project}/submission_{model_name}_{run_note}.csv', index=False)

##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="TPS-FEV2021")
parser.add_argument("--model_name", type=str, default="LGBM")
parser.add_argument("--run_note", type=str)
parser.add_argument("--get_int_preds", type=bool, default=False)


args = parser.parse_args()
#####################
# START PREDICTIONS #
#####################
if __name__ == "__main__":
    print("Prediction start...")
    predict(
        project=args.project, 
        model_name=args.model_name, 
        run_note=args.run_note,
        get_int_preds=args.get_int_preds
    )
