import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import imblearn

from holisticai.bias.metrics import disparate_impact, statistical_parity, average_odds_diff
from holisticai.bias.mitigation import Reweighing, GridSearchReduction, EqualizedOdds

from lightgbm import LGBMClassifier

import evidently
np.float_ = np.float64
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


# =============================================================================
# DATA PREPARATION
# =============================================================================

# --- data_preprocessing ---

def load_data(parquet_path: str = "data/dataset_44270.pq") -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def split_data_from_df(data: pd.DataFrame):
    """Splits a DataFrame into features (X), labels (y), and demographic data (dem)."""
    y = data["Label"].values
    X = data[[str(i) for i in np.arange(50) if str(i) in data.columns]].copy()
    filter_col = (
        ["Ethnicity", "Gender"]
        + [col for col in data if str(col).startswith("Ethnicity_")]
        + [col for col in data if str(col).startswith("Gender_")]
    )
    dem = data[[c for c in filter_col if c in data.columns]].copy()
    return X, y, dem


def resample_equal(df: pd.DataFrame, cat: str) -> pd.DataFrame:
    """Resamples the DataFrame to balance categories by oversampling."""
    df["uid"] = df[cat] + df["Label"].astype(str)
    enc = LabelEncoder()
    df["uid"] = enc.fit_transform(df["uid"])
    res = imblearn.over_sampling.RandomOverSampler(random_state=6)
    df_res, _ = res.fit_resample(df, df["uid"].values)
    df_res = pd.DataFrame(df_res, columns=df.columns)
    df_res = df_res.sample(frac=1).reset_index(drop=True)
    df_res["Label"] = df_res["Label"].astype(float)
    return df_res


def filter_ethnicity_groups(
    data: pd.DataFrame, groups: list = None
) -> pd.DataFrame:
    """Filters dataset to only include specified ethnicity groups."""
    if groups is None:
        groups = ["Black", "White"]
    return data[data["Ethnicity"].isin(groups)].copy()


def apply_reweighing(
    y_train: np.ndarray,
    dem_train: pd.DataFrame,
    group_a_col: str = "Black",
    group_b_col: str = "White",
) -> np.ndarray:
    """Fits a Reweighing pre-processor and returns sample weights."""
    group_a = dem_train["Ethnicity"] == group_a_col
    group_b = dem_train["Ethnicity"] == group_b_col
    rew = Reweighing()
    rew.fit(y_train, group_a, group_b)
    return rew.estimator_params["sample_weight"]


# --- data_validation ---

def run_data_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_html_path: str = "../artifacts/report/data_drift.html",
    status_json_path: str = "../artifacts/status/status.json",
) -> dict:
    """Generates an Evidently data drift report and saves artefacts."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(report_html_path)
    result = report.as_dict()
    with open(status_json_path, "w") as f:
        json.dump(result, f)
    drift_status = (
        result["metrics"][0]["result"]["dataset_drift"]
        & result["metrics"][1]["result"]["dataset_drift"]
    )
    return {"report": result, "dataset_drift": drift_status}


# =============================================================================
# MODELLING
# =============================================================================

# --- feature_engineering ---

def permute_X(X: pd.DataFrame, j) -> pd.DataFrame:
    """Returns a copy of X with column j randomly permuted."""
    Xj = X.copy()
    Xj[j] = Xj[j].sample(frac=1).values
    return Xj


def compute_permutation_feature_importance(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    group_a_test: pd.Series,
    group_b_test: pd.Series,
    n_features: int = 50,
    n_iter: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes permutation feature importance for accuracy and disparate impact.

    Returns two DataFrames: one ranked by accuracy impact, one by bias impact.
    """
    np.random.seed(10)
    acc_base = accuracy_score(y_test, model.predict(X_test))
    di_base = disparate_impact(group_a_test, group_b_test, model.predict(X_test))

    accs = np.zeros((n_iter, n_features))
    dis = np.zeros((n_iter, n_features))

    for j in tqdm(range(n_features)):
        for i in range(n_iter):
            X_perm = permute_X(X_test, str(j))
            y_perm = model.predict(X_perm)
            accs[i, j] = accuracy_score(y_test, y_perm)
            dis[i, j] = disparate_impact(group_a_test, group_b_test, y_perm)

    acc_diff = accs - acc_base
    di_diff = dis - di_base

    def _build_df(diff):
        df = pd.DataFrame(np.arange(n_features), columns=["feature"])
        df["feature"] = df["feature"].astype(str)
        df["imp_mean"] = diff.mean(axis=0).round(3)
        df["imp_std"] = diff.std(axis=0).round(3)
        return df.sort_values("imp_mean", ascending=False).reset_index(drop=True)

    return _build_df(acc_diff), _build_df(di_diff)


def plot_feature_importance(df_imp: pd.DataFrame, title: str = "Feature Importance"):
    """Renders a bar chart of permutation feature importance."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=df_imp, y="imp_mean", x="feature", ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def retrain_without_feature(
    data: pd.DataFrame,
    drop_col: str,
    model_fn=None,
    test_size: float = 0.3,
    group_a_val: str = "Black",
    group_b_val: str = "White",
) -> dict:
    """Retrains a model after dropping a feature column and reports accuracy and DI."""
    if model_fn is None:
        model_fn = lambda: LGBMClassifier(random_state=42)
    data_train, data_test = train_test_split(data, test_size=test_size, random_state=3)
    data_train = data_train.drop(columns=[drop_col]).copy()
    data_test = data_test.drop(columns=[drop_col]).copy()
    X_train, y_train, _ = split_data_from_df(data_train)
    X_test, y_test, dem_test = split_data_from_df(data_test)
    model = model_fn()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    group_a = dem_test["Ethnicity"] == group_a_val
    group_b = dem_test["Ethnicity"] == group_b_val
    return {
        "model": model,
        "accuracy": accuracy_score(y_pred, y_test),
        "disparate_impact": disparate_impact(group_a, group_b, y_pred),
    }


# --- model_training ---

def train_ridge_classifier(
    X_train,
    y_train: np.ndarray,
    sample_weight: np.ndarray = None,
) -> RidgeClassifier:
    """Trains a RidgeClassifier, optionally with sample weights."""
    model = RidgeClassifier(random_state=42)
    kwargs = {} if sample_weight is None else {"sample_weight": sample_weight.ravel()}
    model.fit(X_train, y_train, **kwargs)
    return model


def train_lgbm_classifier(X_train, y_train: np.ndarray) -> LGBMClassifier:
    """Trains a LightGBM classifier."""
    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_with_grid_search_reduction(
    X_train,
    y_train: np.ndarray,
    dem_train: pd.DataFrame,
    group_a_col: str = "Black",
    group_b_col: str = "White",
) -> GridSearchReduction:
    """In-processing bias mitigation using Grid Search Reduction."""
    base_model = RidgeClassifier(random_state=42)
    gsr = GridSearchReduction()
    gsr.transform_estimator(base_model)
    group_a = dem_train["Ethnicity"] == group_a_col
    group_b = dem_train["Ethnicity"] == group_b_col
    gsr.fit(X_train, y_train, group_a, group_b)
    return gsr


# --- model_evaluation ---

def plot_cm(
    y_true,
    y_pred,
    labels: list = None,
    display_labels: list = None,
    ax=None,
):
    """Plots a single confusion matrix with annotations."""
    if labels is None:
        labels = [1, 0]
    if display_labels is None:
        display_labels = [1, 0]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="viridis",
        cbar=False,
        xticklabels=display_labels,
        yticklabels=display_labels,
        square=True,
        linewidths=2,
        linecolor="black",
        ax=ax,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_yticklabels(display_labels, fontsize=11)
    return cm


def plot_confusion_matrices(
    groups,
    data_test: pd.DataFrame,
    category: str,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
) -> dict:
    """Plots confusion matrices for overall data and each demographic group."""
    num_groups = len(groups) + 1
    fig, axes = plt.subplots(1, num_groups, figsize=(5 * num_groups, 4))
    cm = plot_cm(y_test, y_pred_test, ax=axes[0])
    axes[0].set_title("All", fontsize=14, fontweight="bold")
    cm_dict = {"All": cm}
    for i, group in enumerate(groups):
        ax = axes[i + 1]
        subset = data_test[data_test[category] == group]
        cm = plot_cm(subset["Label"], subset["Pred"], ax=ax)
        cm_dict[group] = cm
        ax.set_title(group, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return cm_dict


def calculate_tpr(cms: dict) -> dict:
    """Calculates True Positive Rate for each group from their confusion matrices."""
    return {g: cm[0, 0] / cm[0, :].sum() for g, cm in cms.items()}


def get_metrics(
    group_a: pd.Series,
    group_b: pd.Series,
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Returns a DataFrame of accuracy and fairness metrics for two groups."""
    metrics = [
        ["Model Accuracy", round(accuracy_score(y_true, y_pred), 2), 1],
        ["Black vs. White Disparate Impact", round(disparate_impact(group_a, group_b, y_pred), 2), 1],
        ["Black vs. White Statistical Parity", round(statistical_parity(group_a, group_b, y_pred), 2), 0],
        ["Black vs. White Average Odds Difference", round(average_odds_diff(group_a, group_b, y_pred, y_true), 2), 0],
    ]
    return pd.DataFrame(metrics, columns=["Metric", "Value", "Reference"])


def evaluate_accuracy_by_demographics(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    dem_test: pd.DataFrame,
) -> pd.DataFrame:
    """Computes per-group accuracy for Gender and Ethnicity."""
    dem_test = dem_test.reset_index(drop=True)
    records = []
    for group in dem_test["Gender"].dropna().unique():
        idx = dem_test[dem_test["Gender"] == group].index
        acc = accuracy_score(y_test[idx], y_pred_test[idx])
        records.append(["Accuracy by gender", group, f"{acc:.3f}"])
    for group in dem_test["Ethnicity"].dropna().unique():
        idx = dem_test[dem_test["Ethnicity"] == group].index
        acc = accuracy_score(y_test[idx], y_pred_test[idx])
        records.append(["Accuracy by ethnicity", group, f"{acc:.3f}"])
    return pd.DataFrame(records, columns=["Accuracy type", "Accuracy Type Group", "Accuracy Value"])


def evaluate_success_rate(
    data_test: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Computes mean predicted success rate grouped by Gender and Ethnicity."""
    pred_g_mean = data_test.groupby("Gender")["Pred"].mean()
    pred_e_mean = data_test.groupby("Ethnicity")["Pred"].mean()
    return pred_g_mean, pred_e_mean


def evaluate_statistical_parity(
    data_test: pd.DataFrame,
    y_pred_test: np.ndarray,
) -> pd.DataFrame:
    """Computes Statistical Parity for gender and ethnicity groups and plots results."""
    sr_male = y_pred_test[data_test["Gender"] == "Male"].mean()
    sr_female = y_pred_test[data_test["Gender"] == "Female"].mean()
    sr_white = y_pred_test[data_test["Ethnicity"] == "White"].mean()
    sr_black = y_pred_test[data_test["Ethnicity"] == "Black"].mean()
    sr_asian = y_pred_test[data_test["Ethnicity"] == "Asian"].mean()
    sr_hispanic = y_pred_test[data_test["Ethnicity"] == "Hispanic"].mean()

    groups = ["Female vs. Male", "Black vs. White", "Asian vs. White", "Hispanic vs. White"]
    values = [
        sr_female - sr_male,
        sr_black - sr_white,
        sr_asian - sr_white,
        sr_hispanic - sr_white,
    ]
    df = pd.DataFrame(zip(groups, values), columns=["Group", "Statistical Parity (SP)"])

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Group", y="Statistical Parity (SP)", data=df, palette="viridis", hue="Group", legend=False)
    plt.axhline(y=-0.1, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=0.1, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=0, linewidth=2, color="g", linestyle="-")
    plt.title("Statistical Parity by Group", fontsize=14)
    plt.xlabel("Group", fontsize=12)
    plt.ylabel("Statistical Parity", fontsize=12)
    plt.show()
    return df


def evaluate_disparate_impact(
    data_test: pd.DataFrame,
    y_pred_test: np.ndarray,
) -> pd.DataFrame:
    """Computes Disparate Impact for gender and ethnicity groups and plots results."""
    sr_male = y_pred_test[data_test["Gender"] == "Male"].mean()
    sr_female = y_pred_test[data_test["Gender"] == "Female"].mean()
    sr_white = y_pred_test[data_test["Ethnicity"] == "White"].mean()
    sr_black = y_pred_test[data_test["Ethnicity"] == "Black"].mean()
    sr_asian = y_pred_test[data_test["Ethnicity"] == "Asian"].mean()
    sr_hispanic = y_pred_test[data_test["Ethnicity"] == "Hispanic"].mean()

    groups = ["Female", "Black", "Asian", "Hispanic"]
    values = [sr_female / sr_male, sr_black / sr_white, sr_asian / sr_white, sr_hispanic / sr_white]
    df = pd.DataFrame(zip(groups, values), columns=["Group", "Disparate Impact (DI)"])

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Group", y="Disparate Impact (DI)", data=df, palette="viridis", hue="Group", legend=False)
    plt.axhline(y=0.8, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=1.2, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=1, linewidth=2, color="g", linestyle="-")
    plt.title("Disparate Impact by Group", fontsize=14)
    plt.xlabel("Group", fontsize=12)
    plt.ylabel("Disparate Impact", fontsize=12)
    plt.show()
    return df


def evaluate_equal_opportunity_difference(
    cms_gender: dict,
    cms_ethnicity: dict,
    pred_g_mean: pd.Series,
    pred_e_mean: pd.Series,
) -> pd.DataFrame:
    """Computes Equal Opportunity Difference (EOD) and plots results."""
    TPRs_gender = calculate_tpr(cms_gender)
    TPRs_ethnicity = calculate_tpr(cms_ethnicity)

    eod_fm = TPRs_gender["Female"] - TPRs_gender["Male"]
    eod_bw = TPRs_ethnicity["Black"] - TPRs_ethnicity["White"]
    eod_aw = TPRs_ethnicity["Asian"] - TPRs_ethnicity["White"]
    eod_hw = TPRs_ethnicity["Hispanic"] - TPRs_ethnicity["White"]

    eod_data = pd.DataFrame(
        {"Group": ["Female", "Black", "Asian", "Hispanic"], "EOD": [eod_fm, eod_bw, eod_aw, eod_hw]}
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(x="Group", y="EOD", data=eod_data, palette="viridis", hue="Group", legend=False)
    plt.axhline(y=-0.1, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=0.1, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=0, linewidth=2, color="g", linestyle="-")
    plt.title("Equal Opportunity Difference by Group", fontsize=14)
    plt.xlabel("Group", fontsize=12)
    plt.ylabel("EOD", fontsize=12)
    plt.show()
    return eod_data


def plot_mitigation_comparison(
    metrics_orig: pd.DataFrame,
    metrics_mitigated: pd.DataFrame,
    mitigation_label: str,
):
    """Plots a side-by-side bar chart comparing original vs mitigated metrics."""
    metrics_orig = metrics_orig.copy()
    metrics_mitigated = metrics_mitigated.copy()
    metrics_orig["mitigation"] = "None"
    metrics_mitigated["mitigation"] = mitigation_label
    metrics = pd.concat([metrics_orig, metrics_mitigated], axis=0, ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics, x="Metric", y="Value", hue="mitigation")
    plt.axhline(y=0.8, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=-0.05, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=1, linewidth=2, color="g")
    plt.axhline(y=0, linewidth=2, color="g")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.tight_layout()
    plt.show()
    return metrics


# --- model_validation ---

def validate_bias_thresholds(
    metrics_df: pd.DataFrame,
    sp_bounds: tuple = (-0.1, 0.1),
    di_bounds: tuple = (0.8, 1.2),
    eod_bounds: tuple = (-0.1, 0.1),
) -> dict:
    """
    Checks whether bias metrics meet fairness thresholds.

    Returns a dict with metric names as keys and pass/fail status as values.
    """
    status = {}
    for _, row in metrics_df.iterrows():
        metric, value = row["Metric"], row["Value"]
        if "Statistical Parity" in metric:
            status[metric] = sp_bounds[0] <= value <= sp_bounds[1]
        elif "Disparate Impact" in metric:
            status[metric] = di_bounds[0] <= value <= di_bounds[1]
        elif "Average Odds" in metric or "Equal Opportunity" in metric:
            status[metric] = eod_bounds[0] <= value <= eod_bounds[1]
        else:
            status[metric] = None
    return status


# =============================================================================
# OPERATIONALIZATION
# =============================================================================

# --- post_inference_transformations ---

def apply_equalized_odds(
    model,
    data_test: pd.DataFrame,
    group_col: str = "Ethnicity",
    group_a_val: str = "Black",
    group_b_val: str = "White",
    pp_test_size: float = 0.4,
    solver: str = "highs",
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Post-processing bias mitigation using Equalized Odds.

    Splits data_test into a post-processor training/test split, fits EqualizedOdds,
    and returns adjusted predictions alongside fairness metrics.
    """
    data_pp_train, data_pp_test = train_test_split(data_test, test_size=pp_test_size, random_state=seed)
    X_pp_train, y_pp_train, dem_pp_train = split_data_from_df(data_pp_train)
    X_pp_test, y_pp_test, dem_pp_test = split_data_from_df(data_pp_test)

    group_a_pp_train = dem_pp_train[group_col] == group_a_val
    group_b_pp_train = dem_pp_train[group_col] == group_b_val
    group_a_pp_test = dem_pp_test[group_col] == group_a_val
    group_b_pp_test = dem_pp_test[group_col] == group_b_val

    eq = EqualizedOdds(solver=solver, seed=seed)
    y_pred_pp_train = model.predict(X_pp_train)
    eq.fit(y_pp_train, y_pred_pp_train, group_a=group_a_pp_train, group_b=group_b_pp_train)

    y_pred_pp_test = model.predict(X_pp_test)
    d = eq.transform(y_pred_pp_test, group_a=group_a_pp_test, group_b=group_b_pp_test)
    y_pred_adjusted = d["y_pred"]

    metrics = get_metrics(group_a_pp_test, group_b_pp_test, y_pred_adjusted, y_pp_test)
    return y_pred_adjusted, metrics
