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
    return data

def split_train_valid_test_data(project, data, test_size, valid_size, random_state):
    # First split: train+val vs test
    X_train_val, X_test = train_test_split(
        data.as_df(), test_size=test_size, random_state=random_state
    )
    # Second split: train vs validation
    X_train, X_val = train_test_split(
        X_train_val, test_size=valid_size, random_state=random_state  # 0.25 x 0.8 = 0.2
    )
    project.log_dataitem(name="training_set_X",
                          kind="table",
                          data=X_train)
    project.log_dataitem(name="test_set_X",
                          kind="table",
                          data=X_test)
    project.log_dataitem(name="validation_set_X",
                          kind="table",
                          data=X_val)

def split_demographic_data_from_df(project, data, config):
    """
    Splits a DataFrame into features (X), labels (y), and demographic data (dem).
    """
    filter_col = config.sensitive_features # ["nationality", "gender"]
    features = data.drop(columns=["Id", "decision"] + filter_col).columns
    y = data["decision"].values  # Extract labels
    X = data[features].values  # Extract features
    dem = data[filter_col].copy()  # Extract demographics
    return X, y, dem  # Return features, labels, demographics


