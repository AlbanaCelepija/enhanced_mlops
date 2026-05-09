import os
import json
import gdown
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from library.src.artifact_types import Data, Model, Configuration, Report, Status, Documentation
from library.use_cases.tabular.src.local_platform.utils import *

# from holisticai.bias.mitigation import Reweighing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# data profiling
from ydata_profiling import ProfileReport


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
REPORTS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")

def load_data(config: Configuration):
    gdown.download(config.url, config.original_filepath, quiet=False)
    # load data and remove all NaN values
    with open(config.original_filepath, "rb") as handle:
        raw_data = pickle.load(handle)
    data = raw_data.dropna()
    data = data.rename(columns={i: str(i) for i in range(500)})
    

def data_profiling(data: Data, report: Report) -> Report:
    pass


def data_validation_check_quantity(data: Data, config: Configuration, output_status: Status) -> Status:
    pass


def split_train_valid_test_data(data, test_size, valid_size, random_state):
    # First split: train+val vs test
    X_train_val, X_test = train_test_split(
        data.load_dataset(), test_size=test_size, random_state=random_state
    )
    # Second split: train vs validation
    X_train, X_val = train_test_split(
        X_train_val, test_size=valid_size, random_state=random_state  # 0.25 x 0.8 = 0.2
    )
    output_path = os.path.join(DATA_ARTIFACTS_PATH, "training_set_X.csv")
    X_train.to_csv(output_path)
    output_path = os.path.join(DATA_ARTIFACTS_PATH, "test_set_X.csv")
    X_test.to_csv(output_path)
    output_path = os.path.join(DATA_ARTIFACTS_PATH, "validation_set_X.csv")
    X_val.to_csv(output_path)


def preprocess_train_data(data_input: Data, data_output: Data):
    data = pd.read_csv(data_input.config["training_filepath"])
    categorical_features = data_input.config["categorical_features"]
    boolean_features = data_input.config["boolean_features"]
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(data[categorical_features])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(categorical_features)
    )
    data_to_output = pd.concat([encoded_df, data.drop(columns=categorical_features)], axis=1)
    data_to_output[boolean_features] = data[boolean_features].astype(int)
    data_to_output.to_csv(data_output.config["preprocessed_training_filepath"], index=False)
    return data_output

def data_validation_demographics_qty(data: Data, config: Configuration, output_status: Status) -> Status:
    pass


def preprocess_reweighing(data_input: Data, data_output: Data):
    """Reweighing is a pre-processing bias mitigation technique that amends the dataset to achieve statistical parity.
    This method adjusts the weights of the samples in the dataset to compensate for imbalances between
    different groups. By applying appropriate weights to each instance,
    it ensures that the model is not biased towards any particular group, thereby promoting fairness.
    The goal is to adjust the influence of each group so that the final model satisfies fairness criteria
    such as statistical parity or disparate impact."""
    # Initialise and fit the Reweighing model to mitigate bias
    rew = Reweighing()

    data = data_input.load_dataset()
    # Split the data into training and testing sets (70% training, 30% testing)
    data_train, data_test = train_test_split(
        data, test_size=data_input.config["test_size"], random_state=data_input.config["random_state"]
    )
    # Get the feature matrix (X), target labels (y), and demographic data for both sets
    X_train, y_train, dem_train = split_data_from_df(
        data_train, data_input.config["sensitive_features"]
    )
    X_test, y_test, dem_test = split_data_from_df(data_test, data_input.config["sensitive_features"])

    # Define the groups (Black and White) in the training data based on the 'Ethnicity' column
    group_a_train = dem_train["Ethnicity"] == "Black"  # Group A: Black ethnicity
    group_b_train = dem_train["Ethnicity"] == "White"  # Group B: White ethnicity

    # Fit the reweighing technique to adjust sample weights
    rew.fit(y_train, group_a_train, group_b_train)

    # Extract the calculated sample weights from the reweighing model
    sample_weights = rew.estimator_params["sample_weight"]
    data_train["sample_weights"] = sample_weights

    data_train.to_csv(data_output.config["resulting_filepath"], index=False)
    return Data(data_output.config["resulting_filepath"], data_train)


def data_card_generation(data: Data, documentation: Documentation):
    pass

def data_profiling_custom(
    data: Data, 
    config: Configuration
    ):
    df = data.load_dataset()
    results = []
    for act in config.actions:
        action = act["name"]
        result_meta = {"action": action}
        if action == "summary":
            result_meta["n_rows"] = int(df.shape[0])
            result_meta["n_cols"] = int(df.shape[1])
            result_meta["memory_mb"] = float(
                df.memory_usage(deep=True).sum() / (1024 * 1024)
            )
        elif action == "dtypes":
            result_meta["dtypes"] = df.dtypes.apply(lambda x: str(x)).to_dict()
        elif action == "missing_values":
            mv = df.isna().sum()
            result_meta["missing_count"] = mv.to_dict()
            result_meta["missing_pct"] = (mv / len(df)).round(4).to_dict()
        elif action == "unique_values":
            cols = df.columns.tolist()
            uniques = {}
            for c in cols:
                if c in df.columns:
                    uniques[c] = {
                        "unique_count": int(df[c].nunique()),
                        # "top": df[c].mode().iloc[0] if not df[c].mode().empty else None,
                    }
            result_meta["unique"] = uniques
        elif action == "correlations":
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] >= 2:
                corr = numeric.corr()
                result_meta["correlation_head"] = corr.round(3).iloc[:5, :5].to_dict()
                # Save full correlation to file
                corr.to_csv(
                    os.path.join(
                        REPORTS_ARTIFACTS_PATH, f"{config.dataset_name}_correlation.csv"
                    )
                )
                result_meta["correlation_csv"] = (
                    f"{config.dataset_name}_correlation.csv"
                )
            else:
                result_meta["note"] = "Not enough numeric columns for correlation."
        elif action == "histograms":
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
            bins = 30
            img_paths = []
            for c in cols:
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                    fig, ax = plt.subplots()
                    df[c].dropna().hist(bins=bins, ax=ax)
                    ax.set_title(f"Histogram {c}")
                    img_path = os.path.join(
                        REPORTS_ARTIFACTS_PATH, f"{config.dataset_name}_hist_{c}.png"
                    )
                    fig.savefig(img_path, bbox_inches="tight")
                    plt.close(fig)
                    img_paths.append(img_path)
            result_meta["histograms"] = img_paths
        elif action == "outliers":
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
            outliers = {}
            for c in cols:
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                    q1 = df[c].quantile(0.25)
                    q3 = df[c].quantile(0.75)
                    iqr = q3 - q1
                    low = q1 - 1.5 * iqr
                    high = q3 + 1.5 * iqr
                    mask = (df[c] < low) | (df[c] > high)
                    outliers[c] = {
                        "n_outliers": int(mask.sum()),
                        "low": float(low),
                        "high": float(high),
                    }
            result_meta["outliers"] = outliers
        elif action == "profile_report":
            title = f"Profile report {config.dataset_name}"
            profile = ProfileReport(df, title=title, explorative=True)
            out_html = os.path.join(
                REPORTS_ARTIFACTS_PATH, f"{config.dataset_name}_profile.html"
            )
            profile.to_file(out_html)
            result_meta["profile_html"] = out_html
        else:
            result_meta["error"] = "action not implemented"
        results.append(result_meta)
        # Save results summary
        report_final_path = os.path.join(
            REPORTS_ARTIFACTS_PATH, f"{config.dataset_name}_final_report.json"
        )
        with open(report_final_path, "w", encoding="utf-8") as file:
            json.dump(
                {"actions": config.actions, "results": results},
                file,
                indent=2,
                ensure_ascii=False,
            )

    return results


   


def data_drift_detection(data: Data, config: Report):
    data = data.load_dataset()
    reference_data = data[:7000]
    current_data = data[-900:]
    data_drift_report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )
    data_drift_report.run(reference_data=reference_data, current_data=current_data)
    data_drift_report.save(f"{output.name}.html")
    return output


def data_drift_status(data: Data, output_status: Status):
    reference_data = data[:7000]
    current_data = data[-900:]
    data_drift_report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )
    data_drift_report.run(reference_data=reference_data, current_data=current_data)
    data_drift_result = data_drift_report.as_dict()
    final_status = (
        data_drift_result["metrics"][0]["result"]["dataset_drift"]
        & data_drift_result["metrics"][1]["result"]["dataset_drift"]
    )
    output_status.change_status(final_status)
    return output_status



if __name__ == "__main__":
    import yaml

    with open(
        "/home/albana/Desktop/Albana/DataScience/AI4DT/Projects/enhanced_mlops/framework/library/use_cases/tabular/metadata/aipc_local.yaml",
        "r",
    ) as f:
        aipc_config = yaml.safe_load(f)
    profiling_actions = list(
        filter(
            lambda x: x["name"] == "profiling_actions",
            aipc_config["artifacts"]["configuration"],
        )
    )
    data = Data(
        filepath=os.path.join(DATA_ARTIFACTS_PATH, "recruitmentdataset-2022.csv")
    )
    config = Configuration(config={"resulting_filepath": "data2.parquet"})
    profiling_config = Configuration(config=profiling_actions[0]["config"])

    # load_data(data, config)
    data_profiling_custom(data, profiling_config)
