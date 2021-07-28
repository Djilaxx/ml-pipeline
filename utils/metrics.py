import warnings
warnings.filterwarnings("ignore")

from sklearn import metrics 

metrics_dict = {
    "ACCURACY" : metrics.accuracy_score,
    "AUC" : metrics.roc_auc_score,
    "MSE" : metrics.mean_squared_error,
    "MSLE" : metrics.mean_squared_log_error,
    "MAE" : metrics.mean_absolute_error
}