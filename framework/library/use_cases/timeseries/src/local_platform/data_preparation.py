import os
import gdown
import pickle
import numpy as np
import pandas as pd
from library.src.artifact_types import Data, Configuration, Report, Status

# data profiling
from ydata_profiling import ProfileReport
from ydata_profiling.utils.cache import cache_file

""" 
Data preparation stage containing 4 operations
1.Data Profiling
2.Data Validation
3.Data Preprocessing
4.Data Documentation
"""
FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")


def load_data(config: Configuration):
    file_name = cache_file(
        "pollution_us_2000_2016.csv",
        "https://query.data.world/s/mz5ot3l4zrgvldncfgxu34nda45kvb",
    )
    df = pd.read_csv(file_name, index_col=[0])
    # We will only consider the data from Arizone state for this example
    df = df[df["State"] == "Arizona"]
    df["Date Local"] = pd.to_datetime(df["Date Local"])
    return df


###################################################################### Data Profiling


def data_profiling(data: Data, report: Report):
    report = ProfileReport(data.get_dataset(), tsmode=True)
    report.to_file(report.name)
    return Report(report.name)

###################################################################### Data Preprocessing

# TODO synthetic data generation ydata_synthetic



##################################################################### Platform code


def run_on_platform():
    data_gen_fn = project.new_function(
        name="data-prep",
        kind="python",
        python_version="PYTHON3_10",
        code_src="src/data-prep.py",
        handler="data_generator",
    )
