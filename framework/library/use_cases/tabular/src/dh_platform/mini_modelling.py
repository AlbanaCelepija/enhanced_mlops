import os
import pickle
import logging
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
from utils import split_data_from_df

logging.basicConfig(level=logging.INFO)

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
MODEL_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "model")
REPORT_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")
STATUS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "status")


#################################################### Model Training ####################################################

@handler(outputs=["model"])
def train_model(project, data):
    
    def split_data_from_df(data, sensitive_features):
        """
        Splits a DataFrame into features (X), labels (y), and demographic data (dem).
        """
        filter_col = sensitive_features
        features = data.drop(columns=["Id", "decision"] + filter_col).columns
        y = data["decision"].values  # Extract labels
        X = data[features].values  # Extract features
        dem = data[filter_col].copy()  # Extract demographics
        return X, y, dem  # Return features, labels, demographics


    data = data.as_df()
    # Split the data into training and testing sets (70% training, 30% testing)
    data_train, data_test = train_test_split(
        data, test_size=0.3, random_state=4
    )
    
    # Get the feature matrix (X), target labels (y), and demographic data for both sets
    sensitive_features = ["nationality", "gender"]
    X_train, y_train, dem_train = split_data_from_df(data_train, sensitive_features)
    X_test, y_test, dem_test = split_data_from_df(data_test, sensitive_features)

    # Define the model (RidgeClassifier) and train it on the training data
    model = RidgeClassifier(random_state=4)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_predict = model.predict(X_test)

    model_output_path = "model"
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    
    with open(model_output_path + "/model_baseline.pkl", "wb") as model_file:
        dump(model, model_file, pickle.HIGHEST_PROTOCOL)
        
    # Calculate and print the accuracy of the model on the test set
    metrics = {
        "f1_score": f1_score(y_test, y_predict),
        "accuracy": accuracy_score(y_test, y_predict),
        "precision": precision_score(y_test, y_predict),
        "recall": recall_score(y_test, y_predict),
    }
    model = project.log_model(name="hiring_classifier", kind="sklearn", source="./model/")
    model.log_metrics(metrics)
    return model


def model_evaluation_accuracy_demographic_groups(data, config, model):
    accuracy_demographics = []
    model = get_model(model.key)

    # prepare a validation dataset for prediction and predict
    data_test = data.as_df()  
    # Get the feature matrix (X), target labels (y), and demographic data
    X_test, y_test, dem_test = split_data_from_df(data_test, config.sensitive_features)
    y_pred_test = model.predict(X_test)
    
    logging.info("---- OVERALL ACCURACY  ----")
    # Calculate the accuracy of the model on the test set
    acc = accuracy_score(y_test, y_pred_test)
    accuracy_demographics = [["Overall Accuracy", "All", "%.3f" % acc]]

    logging.info("---- ACCURACY BY GENDER ----")    
    # Calculate accuracy for each gender group
    dem_test = dem_test.reset_index(drop=True)
    for group in dem_test["gender"].unique():
        # Get the indices of the samples belonging to the current group
        idx_group = dem_test[dem_test["gender"] == group].index
        if group is None:
            continue
        # Calculate the accuracy for the current group
        acc = accuracy_score(y_test[idx_group], y_pred_test[idx_group])
        accuracy_demographics += [["Accuracy by gender", group, "%.3f" % acc]]

    logging.info("---- ACCURACY BY NATIONALITY ----")
    # Calculate accuracy for each ethnicity group
    for group in dem_test["nationality"].unique():
        # Get the indices of the samples belonging to the current group
        idx_group = dem_test[dem_test["nationality"] == group].index
        if group is None:
            continue
        # Calculate the accuracy for the current group
        acc = accuracy_score(y_test[idx_group], y_pred_test[idx_group])
        accuracy_demographics += [["Accuracy by ethnicity", group, "%.3f" % acc]]
    acc_demographics_df = pd.DataFrame(accuracy_demographics, columns=["Accuracy type", "Accuracy Type Group", "Accuracy Value"])
    
    #TODO produce report results
    #report.save_report(acc_demographics_df)
    #return report

