import os
import json
import pickle
import logging
import pandas as pd
from pickle import dump

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

logging.basicConfig(level=logging.INFO)

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
MODEL_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "model")
REPORT_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")
STATUS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "status")


#################################################### Model Training ####################################################

@handler(outputs=["model"])
def train_model(project, data_train, sensitive_features, random_state, id_feature, target_feature):        
    data_train = data_train.as_df()        
    # Get the feature matrix (X), target labels (y), and demographic data for both sets
    X_train, y_train, dem_train = split_demographic_data_from_df(data_train, sensitive_features, id_feature, target_feature)    

    # Define the model (RidgeClassifier) and train it on the training data
    model = RidgeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    
    model_output_path = "model"
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    
    with open(model_output_path + "/model_baseline.pkl", "wb") as model_file:
        dump(model, model_file, pickle.HIGHEST_PROTOCOL)       
    model = project.log_model(name="hiring_classifier", kind="sklearn", source="./model/")    
    return model

######################################################################### Model Evaluation

def model_evaluation_accuracy_overall(project, model, data_valid, sensitive_features, id_feature, target_feature):
    model_artifact = model
    model_path = model.download()
    model = pickle.load(open(model_path, "rb"))
    X_test, y_test, dem_test = split_demographic_data_from_df(data_valid.as_df(), sensitive_features, id_feature, target_feature)
    # Make predictions on the test set
    y_predict = model.predict(X_test)
    # Calculate and print the accuracy of the model on the test set
    metrics = {
        "f1_score": f1_score(y_test, y_predict),
        "accuracy": accuracy_score(y_test, y_predict),
        "precision": precision_score(y_test, y_predict),
        "recall": recall_score(y_test, y_predict),
    }
    model_artifact.log_metrics(metrics)
    artifact_output_path = "./artifacts"
    if not os.path.exists(artifact_output_path):
        os.makedirs(artifact_output_path)
    with open(os.path.join(artifact_output_path, "accuracy_overall.json"), "w") as file:
        json.dump(metrics, file, indent=4)  
        
    project.log_artifact(name="accuracy_overall", 
                        kind="artifact", 
                        labels=["baseline","model_evaluation", "report", "model_evaluation_accuracy_overall"],
                        source=f"./{artifact_output_path}/accuracy_overall.json")


def model_evaluation_accuracy_demographic_groups(project, model, data_valid, sensitive_features, id_feature, target_feature):
    accuracy_demographics = []
    model_path = model.download()
    model = pickle.load(open(model_path, "rb"))

    data_valid = data_valid.as_df()  
    # Get the feature matrix (X), target labels (y), and demographic data
    X_test, y_test, dem_test = split_demographic_data_from_df(data_valid, sensitive_features, id_feature, target_feature)
    y_pred_test = model.predict(X_test)
    
    for sensitive_feat in sensitive_features:
        logging.info(f"---- ACCURACY BY {sensitive_feat} ----")    
        # Calculate accuracy for each gender group
        dem_test = dem_test.reset_index(drop=True)
        for group in dem_test[sensitive_feat].unique():
            # Get the indices of the samples belonging to the current group
            idx_group = dem_test[dem_test[sensitive_feat] == group].index
            if group is None:
                continue
            # Calculate the accuracy for the current group
            acc = accuracy_score(y_test[idx_group], y_pred_test[idx_group])
            accuracy_demographics += [[f"Accuracy by {sensitive_feat}", group, "%.3f" % acc]]

    artifact_output_path = "./artifacts"
    if not os.path.exists(artifact_output_path):
        os.makedirs(artifact_output_path)
    acc_demographics_df = pd.DataFrame(accuracy_demographics, columns=["Accuracy type", "Accuracy Type Group", "Accuracy Value"])
    acc_demographics_df.to_json(os.path.join(artifact_output_path, "accuracy_demographic_groups.json"))
    
    project.log_artifact(name="accuracy_demographic_groups_artifact", 
                        kind="artifact", 
                        labels=["baseline", "model_evaluation", "report", "model_evaluation_accuracy_demographic_groups"],
                        source=f"./{artifact_output_path}/accuracy_demographic_groups.json")

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