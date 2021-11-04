# MACHINE LEARNING PIPELINE

This project is a machine learning pipeline that i created to be able to iterate fast when i start a working on a new machine learning project. \

## Project
The pipeline is articulated around the project folder. Each subfolder in project/ will contain the necessary informations for a model to be trained on specific dataset.
* **config.py** : Should contain all the necessary information needed to train a model, things like data path, hyperparameters, training parameters etc..
* **feature_eng.py** : All the preprocessing functions needed for your data to be training ready.

## Code
<hr />

### **Prerequisites**

I use miniconda3 to manage my python environnements and packages. \
You can download the package manager here : https://docs.conda.io/en/latest/miniconda.html

You can use the following command to create the environnement I used:
```
conda env create --name ML-37 --file=ML-37.yml
```

### **Train**

```
python -m train
--project=TPS-FEV2021 
--model_name=LGBM 
--run_note=test
```

* **project** : str - the project name you want to train a model on. 
* **model_name** : str - the name of the model you want to train 
* **run_note** : str - specify a number or special note about this training run


### **Predict**

```
python -m predict
--project=TPS-FEV2021 
--model_name=LGBM
--run_note=test
```

the predict command is used to use a trained model on new test data and create a prediction file.

### Notebooks
the notebooks/ folder contain many Jupyter notebooks with baseline code for different task that can help when building a model for a project, things like model selection, feature selection, basic EDA, hyperparameter optimization, model evaluation etc...