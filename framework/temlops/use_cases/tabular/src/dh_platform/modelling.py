import os
import json
import pickle
import logging
import pandas as pd
from pickle import dump

from temlops.src.artifact_types import Data, Configuration, Model, Report, Status, Documentation
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from digitalhub_runtime_python import handler
from digitalhub import get_model
import digitalhub as dh

logging.basicConfig(level=logging.INFO)

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
MODEL_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "model")
REPORT_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")
STATUS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "status")


#################################################### Model Training ####################################################

@handler(outputs=["model"])
def train_model(product_name, config: Configuration):  
    project = dh.get_or_create_project(product_name)   
    train_fn = project.new_function(
        name="train-classifier",
        kind="python",
        python_version="PYTHON3_10",
        code_src="dh_modelling.py",
        handler="train_model_real",
        requirements=["scikit-learn", "numpy<2"],
        labels=["model_training", "baseline"],
    )
    train_ds = project.get_dataitem('train_data')
    train_fn.run(action="job", 
                            inputs={"data_train": train_ds.key},
                            parameters=config, 
                            wait=True
                            )   


######################################################################### Model Evaluation

def model_evaluation_accuracy_overall(product_name, model: Model, data_valid: Data, config: Configuration):
    project = dh.get_or_create_project(product_name) 
    model = model.load_model()
    valid_ds = data_valid.load_dataset()
    model_evaluation_accuracy_overall_fn = project.new_function(
        name="model-evaluation-accuracy-overall",
        kind="python",
        python_version="PYTHON3_10",
        code_src="dh_modelling.py",
        handler="model_evaluation_accuracy_overall_real",
        requirements=["scikit-learn", "numpy<2"],
        labels=["model_evaluation", "baseline"],
    )
    model_evaluation_accuracy_overall_fn.run(
        action="job",
        inputs={"model": model.key, "data_valid": valid_ds.key}, 
        parameters=config, 
        wait=True
    )



def model_evaluation_accuracy_demographic_groups(product_name, model: Model, data_valid: Data, config: Configuration):
    project = dh.get_or_create_project(product_name) 
    model = model.load_model()
    valid_ds = data_valid.load_dataset()
    model_evaluation_accuracy_demographic_groups_fn = project.new_function(
        name="model-evaluation-accuracy-demographic-groups",
        kind="python",
        python_version="PYTHON3_10",
        code_src="dh_modelling.py",
        handler="model_evaluation_accuracy_demographic_groups_real",
        requirements=["scikit-learn", "numpy<2"],
        labels=["model_evaluation", "baseline"],
    )
    model_evaluation_accuracy_demographic_groups_fn.run(
        action="job",
        inputs={"model": model.key, "data_valid": valid_ds.key}, 
        parameters=config, 
        wait=True
    )

    
    


######################################################## utils

def split_demographic_data_from_df(data, sensitive_features, id_feature, target_feature):
    """
    Splits a DataFrame into features (X), labels (y), and demographic data (dem).
    """
    filter_col = sensitive_features
    features = data.drop(columns=[id_feature, target_feature] + filter_col).columns
    y = data[target_feature].values  # Extract labels
    X = data[features].values  # Extract features
    dem = data[filter_col].copy()  # Extract demographics
    return X, y, dem  # Return features, labels, demographics