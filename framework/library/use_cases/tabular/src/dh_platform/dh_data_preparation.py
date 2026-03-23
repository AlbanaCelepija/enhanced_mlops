import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
REPORTS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")

####################################################### Evaluate if AI product can be reused for new use case

def data_drift_detection():
    """
    """
    pass
################################################################################################## Data Preprocessing

def load_data_real(project, data_name, url):
    project.new_dataitem(name=data_name,
                          kind="table",
                          path=url)
    

def split_train_valid_test_data_real(project, data, test_size, valid_size, random_state):
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


def preprocess_train_data_real(project, training_di, di_name, boolean_features, categorical_features):   
    data = training_di.as_df()
    boolean_features = boolean_features.split(",")
    categorical_features = categorical_features.split(",")
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

def data_profiling_real(project, training_di):
    """
    Provide a set of predefined actions for data profiling to assess readiness for ML. 
    Identify key data quality issues, and produce a profile report
    Generate a report of the dataset's summary, missing values, unique values, correlations, and histograms.
    """
    df = training_di.as_df()
    config = {"actions": ["summary", "dtypes", "missing_values", "unique_values", "correlations", "histograms", "outliers"]}
    os.makedirs(REPORTS_ARTIFACTS_PATH, exist_ok=True)
    results = []
    for act in config.actions:
        action = act
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
                        #"top": df[c].mode().iloc[0] if not df[c].mode().empty else None,
                    }
            result_meta["unique"] = uniques
        elif action == "correlations":
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] >= 2:
                corr = numeric.corr()
                result_meta["correlation_head"] = corr.round(3).iloc[:5, :5].to_dict()
                # Save full correlation to file
                corr.to_csv(os.path.join(REPORTS_ARTIFACTS_PATH, f"{config.dataset_name}_correlation.csv"))
                result_meta["correlation_csv"] = f"{config.dataset_name}_correlation.csv"
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
                    img_path = os.path.join(REPORTS_ARTIFACTS_PATH, f"{config.dataset_name}_hist_{c}.png")
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
        #elif action == "profile_report":
        #    title = f"Profile report {config.dataset_name}"
        #    profile = ProfileReport(df, title=title, explorative=True)
        #    out_html = os.path.join(REPORTS_ARTIFACTS_PATH, f"{config.dataset_name}_profile.html")
        #    profile.to_file(out_html)
        #    result_meta["profile_html"] = out_html
        else:
            result_meta["error"] = "action not implemented"
        results.append(result_meta)
        # Save results summary
        report_final_path = os.path.join(REPORTS_ARTIFACTS_PATH, f"{config.dataset_name}_final_report.json")
        with open(report_final_path, "w", encoding="utf-8") as file:
            json.dump(
                {"actions": config.actions, "results": results},
                file,
                indent=2,
                ensure_ascii=False,
            )
        


def data_validation_check_quantity_real():
    pass

def data_validation_demographics_qty_real():
    pass

def preprocess_reweighing_real():
    pass

def data_card_generation_real():
    pass