import os
import pickle
import numpy as np
import pandas as pd

# from holisticai.bias.mitigation import Reweighing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

def load_data(project, data_name, url):
    project.new_dataitem(name=data_name,
                          kind="table",
                          path=url)
    

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


def preprocess_train_data(project, training_di, di_name, boolean_features, categorical_features):   
    data = training_di.as_df()
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(data[categorical_features])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(categorical_features)
    )
    data = pd.concat([encoded_df, data.drop(columns=categorical_features)], axis=1)
    data[boolean_features] = data[boolean_features].astype(int)
    project.log_dataitem(name=di_name,
                          kind="table",
                          data=data)
