import os
import pickle
import numpy as np
import pandas as pd

# from holisticai.bias.mitigation import Reweighing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from library.src.artifact_types import Data, Configuration, Report, Status
import digitalhub as dh
from digitalhub_runtime_python import handler

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
USE_CASE_FOLDER = os.path.join(parent_folder, "dh_platform")


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

def load_data(product_name, config: Configuration):
    project = dh.get_or_create_project(product_name)
    data_preparation_fn = project.new_function(name="load_data",
                                   kind="python",
                                   python_version="PYTHON3_10",
                                   code_src="framework/library/use_cases/tabular/src/dh_platform/dh_data_preparation.py",
                                   handler="load_data_real",
                                   requirements=["scikit-learn"],
                                   labels=["data_preprocessing", "baseline"],)
    data_preparation_fn = data_preparation_fn.run("job", parameters=config, wait=True)

def split_train_valid_test_data(product_name, data: Data, config: Configuration):
    project = dh.get_or_create_project(product_name)
    raw_training_di = data.load_dataset()
    split_fn = project.new_function(
        name="split-train-valid-test",
        kind="python",
        python_version="PYTHON3_10",
        code_src="framework/library/use_cases/tabular/src/dh_platform/dh_data_preparation.py",
        handler="split_train_valid_test_data_real",
        requirements=["scikit-learn", "numpy<2"],
        labels=["data_preprocessing", "baseline"]
    )
    split_fn.run(action="job", inputs={"data": raw_training_di.key}, parameters=config, wait=True)


def preprocess_train_data(product_name, data: Data, config: Configuration):
    project = dh.get_or_create_project(product_name)  
    training_di = data.load_dataset()
    preprocess_fn = project.new_function(
        name="preprocess_train_data",
        kind="python",
        python_version="PYTHON3_10",
        code_src="framework/library/use_cases/tabular/src/dh_platform/dh_data_preparation.py",
        handler="preprocess_train_data_real",
        requirements=["scikit-learn", "numpy<2"],
        labels=["data_preprocessing", "baseline"]
    )
    preprocess_fn.run(action="job", inputs={"training_di": training_di.key}, parameters=config.__dict__, wait=True)    
    