# MACHINE LEARNING PIPELINE

This project is a machine learning pipeline that i created to be able to iterate fast when i start a working on a new project. \

## Project
The pipeline is articulated around the project folder. Each subfolder in project/ will contain the necessary informations for a model to be trained.
* **config.py** : Should contain all the necessary information needed to train a model, things like data path, hyperparameters, training parameters etc..
* **feature_eng.py** : All the preprocessing functions needed for your data to be training ready.

## Code
<hr />

### **Prerequisites**
```
python -m pip install -r requirements.txt
```

### **Train**

```
python -m train 
--folds=10 
--project=TPS-FEV2021 
--model_name=LGBM 
--model_task=REG
```

* **folds** : int, number of folds - if folds=5 we'll divide the dataset in five part and train five models (each model will be trained on four part and validated on the last)
* **project** : the project name you want to train a model on. 
* **model_name** : the name of the model you want to train 
* **model_task** : **REG** or **CL** Regression or Classification task

### **Predict**

```
python -m predict 
--project=TPS-FEV2021 
--model_name=LGBM 
--model_task=REG
```

### Notebooks
the notebooks/ folder contain many Jupyter notebooks with baseline code for different task that can help when building a model for a project, things like model selection, feature selection, basic EDA, hyperparameter optimization, model evaluation etc...