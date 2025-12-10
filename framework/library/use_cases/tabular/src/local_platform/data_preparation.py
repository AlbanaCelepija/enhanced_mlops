import os
import gdown
import pickle
import numpy as np
import pandas as pd
from library.src.artifact_types import Data, Configuration, Report

# from holisticai.bias.mitigation import Reweighing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# import digitalhub as dh

""" 
Data preparation stage containing 4 operations
Data Profiling
Data Validation
Data Preprocessing
Data Documentation
"""

################################################################################################## Data Preprocessing

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")


def load_data_0(config: Configuration):
    gdown.download(config.url, config.original_filepath, quiet=False)
    # load data and remove all NaN values
    with open(config.original_filepath, "rb") as handle:
        raw_data = pickle.load(handle)
    data = raw_data.dropna()
    data = data.rename(columns={i: str(i) for i in range(500)})
    data.to_parquet(config.resulting_filepath)
    return Data(config.resulting_filepath, data)


def split_data_from_df(data):
    """
    Splits a DataFrame into features (X), labels (y), and demographic data (dem).
    """
    filter_col = ["nationality", "gender"]
    features = data.drop(columns=["Id", "decision"] + filter_col).columns
    y = data["decision"].values  # Extract labels
    X = data[features].values  # Extract features
    dem = data[filter_col].copy()  # Extract demographics
    return X, y, dem  # Return features, labels, demographics


def load_data(data: Data, config: Configuration):
    boolean_features = [
        "ind-debateclub",
        "ind-programming_exp",
        "ind-international_exp",
        "ind-entrepeneur_exp",
        "ind-exact_study",
        "decision",
    ]
    categorical_features = ["sport", "ind-degree", "company"]
    dataset = data.get_dataset()
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(dataset[categorical_features])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(categorical_features)
    )
    dataset = pd.concat(
        [encoded_df, dataset.drop(columns=categorical_features)], axis=1
    )
    dataset[boolean_features] = dataset[boolean_features].astype(int)
    output_path = os.path.join(DATA_ARTIFACTS_PATH, config.resulting_filepath)
    dataset.to_parquet(output_path)
    return Data(output_path)


def resample_equal(df, cat):
    """Resamples the DataFrame to balance categories by oversampling based on a combined category-label identifier."""
    df["uid"] = df[cat] + df["Label"].astype(
        str
    )  # Create unique identifier combining category and label
    enc = LabelEncoder()  # Initialize label encoder
    df["uid"] = enc.fit_transform(df["uid"])  # Encode the combined identifier
    res = imblearn.over_sampling.RandomOverSampler(
        random_state=6
    )  # Initialize oversampler
    df_res, euid = res.fit_resample(df, df["uid"].values)  # Apply oversampling
    df_res = pd.DataFrame(df_res, columns=df.columns)  # Convert to DataFrame
    df_res = df_res.sample(frac=1).reset_index(drop=True)  # Shuffle rows
    df_res["Label"] = df_res["Label"].astype(float)  # Convert label to float
    return df_res  # Return resampled DataFrame


def bias_mitigation_pre_reweighing(data: Data, config: Configuration) -> Data:
    """Reweighing is a pre-processing bias mitigation technique that amends the dataset to achieve statistical parity.
    This method adjusts the weights of the samples in the dataset to compensate for imbalances between
    different groups. By applying appropriate weights to each instance,
    it ensures that the model is not biased towards any particular group, thereby promoting fairness.
    The goal is to adjust the influence of each group so that the final model satisfies fairness criteria
    such as statistical parity or disparate impact."""
    # Initialise and fit the Reweighing model to mitigate bias
    rew = Reweighing()

    data = data.get_dataset()
    # Split the data into training and testing sets (70% training, 30% testing)
    data_train, data_test = train_test_split(
        data, test_size=config.test_size, random_state=config.random_state
    )
    # Get the feature matrix (X), target labels (y), and demographic data for both sets
    X_train, y_train, dem_train = split_data_from_df(data_train)
    X_test, y_test, dem_test = split_data_from_df(data_test)

    # Define the groups (Black and White) in the training data based on the 'Ethnicity' column
    group_a_train = dem_train["Ethnicity"] == "Black"  # Group A: Black ethnicity
    group_b_train = dem_train["Ethnicity"] == "White"  # Group B: White ethnicity

    # Fit the reweighing technique to adjust sample weights
    rew.fit(y_train, group_a_train, group_b_train)

    # Extract the calculated sample weights from the reweighing model
    sample_weights = rew.estimator_params["sample_weight"]
    data_train["sample_weights"] = sample_weights

    data_train.to_parquet(config.resulting_filepath)
    return Data(config.resulting_filepath, data_train)


def drift_detection():
    pass


##################################################################### Platform code


def run_on_platform():
    data_gen_fn = project.new_function(
        name="data-prep",
        kind="python",
        python_version="PYTHON3_10",
        code_src="src/data-prep.py",
        handler="data_generator",
    )
