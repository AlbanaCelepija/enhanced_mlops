import os
import pickle
import numpy as np
import pandas as pd

# from holisticai.bias.mitigation import Reweighing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import digitalhub as dh
from digitalhub_runtime_python import handler

""" 
Data preparation stage containing 4 operations
Data Profiling
Data Validation
Data Preprocessing
Data Documentation
"""

####################################################### Evaluate if AI product can be reused for new use case

def data_drift_detection():
    """
    """
    pass
################################################################################################## Data Preprocessing

def split_demographic_data_from_df(data):
    """
    Splits a DataFrame into features (X), labels (y), and demographic data (dem).
    """
    filter_col = ["nationality", "gender"]
    features = data.drop(columns=["Id", "decision"] + filter_col).columns
    y = data["decision"].values  # Extract labels
    X = data[features].values  # Extract features
    dem = data[filter_col].copy()  # Extract demographics
    return X, y, dem  # Return features, labels, demographics

@handler(outputs=["dataset"])
def load_data(project):
    boolean_features = [
        "ind-debateclub",
        "ind-programming_exp",
        "ind-international_exp",
        "ind-entrepeneur_exp",
        "ind-exact_study",
        "decision",
    ]
    categorical_features = ["sport", "ind-degree", "company"]
    training_di = project.get_dataitem('recruitmentdataset.csv')
    data = training_di.as_df()
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(data[categorical_features])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(categorical_features)
    )
    data = pd.concat([encoded_df, data.drop(columns=categorical_features)], axis=1)
    data[boolean_features] = data[boolean_features].astype(int)
    
    # Split the data into training and testing sets (70% training, 30% testing)
    #data_train, data_test = train_test_split(data, test_size=0.3, random_state=4)
    #project.new_dataitem(name="training_set",
    #                      kind="table",
    #                      path=URL)
    #project.new_dataitem(name="test_set",
    #                      kind="table",
    #                      path=URL)
    
    return data

def split_train_valid_test_data(project, data):
    
    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Second split: train vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42  # 0.25 x 0.8 = 0.2
    )
    project.new_dataitem(name="training_set",
                          kind="table",
                          path=URL)
    project.new_dataitem(name="test_set",
                          kind="table",
                          path=URL)
