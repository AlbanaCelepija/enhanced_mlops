import os
import pickle
import logging
from pickle import dump

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from holisticai.bias.mitigation import EqualizedOdds, Reweighing, GridSearchReduction
from holisticai.bias.metrics import (
    disparate_impact,
    statistical_parity,
    average_odds_diff,
)

# from interpret.blackbox import LimeTabular
# from interpret import show

from library.src.artifact_types import Data, Configuration, Report, Model, Status
from library.use_cases.tabular.src.local_platform.utils import *

from scipy.stats import uniform
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
from lightgbm import LGBMClassifier

# import mlflow
# import mlflow.sklearn

logging.basicConfig(level=logging.INFO)

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
MODEL_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "model")
REPORT_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")
STATUS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "status")


#################################################### Feature Engineering


def permutation_feature_importance(
    data_test: Data, model: Model, config: Configuration, report: Report
) -> Report:
    """
    Computes permutation feature importance for both accuracy and disparate impact.

    Config attributes:
        sensitive_features, id_feature, target_feature,
        group_col, group_a_val, group_b_val, n_iter (optional, default 10)
    """
    model_obj = model.load_model()
    data = data_test.load_dataset()
    X_test, y_test, dem_test = split_demographic_data_from_df(
        data, config.sensitive_features, config.id_feature, config.target_feature
    )
    feature_cols = data.drop(
        columns=[config.id_feature, config.target_feature] + config.sensitive_features
    ).columns.tolist()
    X_test_df = data[feature_cols].copy()

    group_a_test = dem_test[config.group_col] == config.group_a_val
    group_b_test = dem_test[config.group_col] == config.group_b_val

    acc_base = accuracy_score(y_test, model_obj.predict(X_test))
    di_base = disparate_impact(group_a_test, group_b_test, model_obj.predict(X_test))

    n_features = X_test_df.shape[1]
    n_iter = getattr(config, "n_iter", 10)

    np.random.seed(10)
    accs = np.zeros((n_iter, n_features))
    dis = np.zeros((n_iter, n_features))

    for j in tqdm(range(n_features)):
        for i in range(n_iter):
            X_perm = X_test_df.copy()
            col = X_test_df.columns[j]
            X_perm[col] = X_perm[col].sample(frac=1).values
            y_perm = model_obj.predict(X_perm.values)
            accs[i, j] = accuracy_score(y_test, y_perm)
            dis[i, j] = disparate_impact(group_a_test, group_b_test, y_perm)

    def _build_imp_df(diff, cols):
        df = pd.DataFrame({"feature": cols})
        df["imp_mean"] = diff.mean(axis=0).round(3)
        df["imp_std"] = diff.std(axis=0).round(3)
        return df.sort_values("imp_mean", ascending=False).reset_index(drop=True)

    df_acc_imp = _build_imp_df(accs - acc_base, feature_cols)
    df_di_imp = _build_imp_df(dis - di_base, feature_cols)

    df_acc_imp["importance_type"] = "accuracy"
    df_di_imp["importance_type"] = "disparate_impact"
    combined = pd.concat([df_acc_imp, df_di_imp], axis=0, ignore_index=True)
    report.save_report(combined)
    return report


#################################################### Model Training ####################################################

def train_model(data: Data, config: Configuration):
    """Train a RidgeClassifier and persist model and split datasets."""
    data = data.load_dataset()
    data_train, data_test = train_test_split(
        data, test_size=config.test_size, random_state=config.random_state
    )
    artifact_data_train = Data(os.path.join(DATA_ARTIFACTS_PATH, "data_training.csv"))
    artifact_data_train.log_dataset(data_train)
    artifact_data_test = Data(os.path.join(DATA_ARTIFACTS_PATH, "data_testing.csv"))
    artifact_data_test.log_dataset(data_test)

    X_train, y_train, dem_train = split_demographic_data_from_df(
        data_train, config.sensitive_features, config.id_feature, config.target_feature
    )
    X_test, y_test, dem_test = split_demographic_data_from_df(
        data_test, config.sensitive_features, config.id_feature, config.target_feature
    )

    model = RidgeClassifier(random_state=config.random_state)
    model.fit(X_train, y_train)

    model_output_path = os.path.join(MODEL_ARTIFACTS_PATH, config.model_filepath)
    with open(model_output_path, "wb") as model_file:
        dump(model, model_file, pickle.HIGHEST_PROTOCOL)


def train_model_reweighing(data: Data, config: Configuration) -> Model:
    """Pre-processing bias mitigation using Reweighing (Kamiran and Calders, 2012).

    Adjusts sample weights before training so the model satisfies statistical parity.

    Config attributes:
        test_size, random_state, sensitive_features, id_feature, target_feature,
        group_col, group_a_val, group_b_val, model_filepath
    """
    data_df = data.load_dataset()
    data_train, _ = train_test_split(
        data_df, test_size=config.test_size, random_state=config.random_state
    )
    X_train, y_train, dem_train = split_demographic_data_from_df(
        data_train, config.sensitive_features, config.id_feature, config.target_feature
    )
    group_a_train = dem_train[config.group_col] == config.group_a_val
    group_b_train = dem_train[config.group_col] == config.group_b_val

    rew = Reweighing()
    rew.fit(y_train, group_a_train, group_b_train)
    sample_weights = rew.estimator_params["sample_weight"]

    model = RidgeClassifier(random_state=config.random_state)
    model.fit(X_train, y_train, sample_weight=sample_weights.ravel())

    model_output_path = os.path.join(MODEL_ARTIFACTS_PATH, config.model_filepath)
    with open(model_output_path, "wb") as model_file:
        dump(model, model_file, pickle.HIGHEST_PROTOCOL)
    return Model(model_path=model_output_path)


def hyperparameters_optimization(data: Data, config: Configuration):
    data = data.get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        data, test_size=config.test_size, random_state=config.random_state
    )
    lr = ElasticNet()
    distributions = dict(
        alpha=uniform(loc=0, scale=10),
        l1_ratio=uniform(),
    )
    clf = RandomizedSearchCV(
        estimator=lr,
        param_distributions=distributions,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_iter=100,
    )
    with mlflow.start_run(run_name="hyperparameter-tuning"):
        search = clf.fit(X_train, y_train)
        y_pred = clf.best_estimator_.predict(X_test)
        rmse, mae, r2 = eval_metrics(clf.best_estimator_, y_pred, y_test)
        mlflow.log_metrics(
            {
                "mean_squared_error_X_test": rmse,
                "mean_absolute_error_X_test": mae,
                "r2_score_X_test": r2,
            }
        )


############################################################## Evaluations - Performance metrics - Accuracy


def model_evaluation_accuracy(
    data: Data, config: Configuration, model: Model, report: Report
) -> Report:
    model = model.load_model()
    data_test = data.load_dataset()
    X_test, y_test, dem_test = split_demographic_data_from_df(
        data_test, config.sensitive_features, config.id_feature, config.target_feature
    )
    y_pred_test = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred_test)
    evaluation_metric = pd.DataFrame(
        columns=["Metric", "Value", "Reference"], data=[["Accuracy", acc, "1"]]
    )
    eval_acc_metric_report = Report(report.filepath).save_report(evaluation_metric)
    return eval_acc_metric_report


############## Evaluations - Performance and Fairness evaluation metrics


def model_evaluation_accuracy_simple(
    data: Data, config: Configuration, model: Model, metrics_baseline_report: Report
) -> Report:
    model = model.load_model()
    data_test = data.load_dataset()
    X_test, y_test, dem_test = split_demographic_data_from_df(
        data_test, config.sensitive_features, config.id_feature, config.target_feature
    )
    y_pred_test = model.predict(X_test)

    group_a_test = dem_test["nationality"] == "Dutch"
    group_b_test = dem_test["nationality"] == "Belgian"

    metrics_rw = get_metrics_classifier(
        group_a_test, group_b_test, y_pred_test, y_test, "Dutch vs Belgians"
    )
    metrics_baseline_report.save_report(metrics_rw)
    return metrics_baseline_report


def model_evaluation_accuracy_demographic_groups(
    model: Model, data_valid: Data, config: Configuration, report: Report
) -> Report:
    """Computes per-group accuracy for each sensitive feature and saves to report.

    Config attributes:
        sensitive_features, id_feature, target_feature
    """
    accuracy_demographics = []
    model = model.load_model()

    data_valid = data_valid.load_dataset()
    X_test, y_test, dem_test = split_demographic_data_from_df(
        data_valid, config.sensitive_features, config.id_feature, config.target_feature
    )
    y_pred_test = model.predict(X_test)

    for sensitive_feat in config.sensitive_features:
        logging.info(f"---- ACCURACY BY {sensitive_feat} ----")
        dem_test = dem_test.reset_index(drop=True)
        for group in dem_test[sensitive_feat].unique():
            idx_group = dem_test[dem_test[sensitive_feat] == group].index
            if group is None:
                continue
            acc = accuracy_score(y_test[idx_group], y_pred_test[idx_group])
            accuracy_demographics += [
                [f"Accuracy by {sensitive_feat}", group, "%.3f" % acc]
            ]

    acc_demographics_df = pd.DataFrame(
        accuracy_demographics,
        columns=["Accuracy type", "Accuracy Type Group", "Accuracy Value"],
    )
    report.save_report(acc_demographics_df)
    return report


def calculate_success_rate(
    data: Data, config: Configuration, model: Model, report: Report
) -> Report:
    """Computes the proportion of positive predictions for each group within group_col.

    Config attributes:
        sensitive_features, id_feature, target_feature, group_col
    """
    model_obj = model.load_model()
    data_df = data.load_dataset()
    X_test, y_test, dem_test = split_demographic_data_from_df(
        data_df, config.sensitive_features, config.id_feature, config.target_feature
    )
    y_pred = model_obj.predict(X_test)
    dem_test = dem_test.reset_index(drop=True)

    data_df = data_df.copy()
    data_df["Pred"] = y_pred

    pred_by_group = data_df.groupby(config.group_col)["Pred"].mean().reset_index()
    pred_by_group.columns = ["Group", "Success Rate"]

    report.save_report(pred_by_group)
    return report


def calculate_statistical_parity(
    data: Data, config: Configuration, model: Model, report: Report
) -> Report:
    """Computes Statistical Parity and Disparate Impact for each group vs group_b_val.

    Config attributes:
        sensitive_features, id_feature, target_feature, group_col, group_b_val
    """
    model_obj = model.load_model()
    data_df = data.load_dataset()
    X_test, y_test, dem_test = split_demographic_data_from_df(
        data_df, config.sensitive_features, config.id_feature, config.target_feature
    )
    y_pred = model_obj.predict(X_test)
    dem_test = dem_test.reset_index(drop=True)

    records = []
    for group in dem_test[config.group_col].dropna().unique():
        if group == config.group_b_val:
            continue
        group_a = dem_test[config.group_col] == group
        group_b = dem_test[config.group_col] == config.group_b_val
        sp_val = float(statistical_parity(group_a, group_b, y_pred))
        di_val = float(disparate_impact(group_a, group_b, y_pred))
        records.append({
            "Group": f"{group} vs {config.group_b_val}",
            "Statistical Parity": round(sp_val, 2),
            "Disparate Impact": round(di_val, 2),
            "SP Fair (-0.1, 0.1)": -0.1 <= sp_val <= 0.1,
            "DI Fair (0.8, 1.2)": 0.8 <= di_val <= 1.2,
        })

    sp_df = pd.DataFrame(records)
    report.save_report(sp_df)
    return report


def confusion_matrices(
    data: Data, config: Configuration, model: Model, report: Report
) -> Report:
    """Plots per-group confusion matrices and saves True Positive Rates to report.

    Config attributes:
        sensitive_features, id_feature, target_feature, group_col
    """
    model_obj = model.load_model()
    data_df = data.load_dataset()
    X_test, y_test, dem_test = split_demographic_data_from_df(
        data_df, config.sensitive_features, config.id_feature, config.target_feature
    )
    y_pred = model_obj.predict(X_test)
    dem_test = dem_test.reset_index(drop=True)

    data_test = data_df.copy()
    data_test["Label"] = y_test
    data_test["Pred"] = y_pred

    groups = dem_test[config.group_col].dropna().unique()
    cms = plot_confusion_matrices(groups, data_test, config.group_col, y_test, y_pred)
    tprs = calculate_tpr(cms)

    tpr_df = pd.DataFrame(list(tprs.items()), columns=["Group", "TPR"])
    report.save_report(tpr_df)
    return report


def calculate_equal_opportunity_difference(
    data: Data, config: Configuration, model: Model, report: Report
) -> Report:
    """Computes Equal Opportunity Difference (TPR gap) for each group vs group_b_val.

    Config attributes:
        sensitive_features, id_feature, target_feature, group_col, group_b_val
    """
    model_obj = model.load_model()
    data_df = data.load_dataset()
    X_test, y_test, dem_test = split_demographic_data_from_df(
        data_df, config.sensitive_features, config.id_feature, config.target_feature
    )
    y_pred = model_obj.predict(X_test)
    dem_test = dem_test.reset_index(drop=True)

    data_test = data_df.copy()
    data_test["Label"] = y_test
    data_test["Pred"] = y_pred

    groups = dem_test[config.group_col].dropna().unique()
    cms = plot_confusion_matrices(groups, data_test, config.group_col, y_test, y_pred)
    tprs = calculate_tpr(cms)

    tpr_ref = tprs.get(config.group_b_val, np.nan)
    records = []
    for group in groups:
        if group == config.group_b_val:
            continue
        eod_val = float(tprs.get(group, np.nan) - tpr_ref)
        records.append({
            "Group": f"{group} vs {config.group_b_val}",
            "TPR": round(float(tprs.get(group, np.nan)), 3),
            "EOD": round(eod_val, 2),
            "Fair (-0.1, 0.1)": -0.1 <= eod_val <= 0.1,
        })

    eod_df = pd.DataFrame(records)
    report.save_report(eod_df)
    return report


############# Model Validation


def model_validation_baseline(
    report: Report, config: Configuration, status: Status
) -> Status:
    """Checks all fairness metrics in the report against configurable thresholds.

    Config attributes:
        sp_bounds (optional, default (-0.1, 0.1)),
        di_bounds (optional, default (0.8, 1.2)),
        eod_bounds (optional, default (-0.1, 0.1))
    """
    metrics_df = report.load_report()
    if metrics_df is None:
        status.change_status(False)
        return status

    sp_bounds = getattr(config, "sp_bounds", (-0.1, 0.1))
    di_bounds = getattr(config, "di_bounds", (0.8, 1.2))
    eod_bounds = getattr(config, "eod_bounds", (-0.1, 0.1))

    all_pass = True
    for _, row in metrics_df.iterrows():
        metric, value = row["Metric"], row["Value"]
        if "Statistical Parity" in metric:
            if not (sp_bounds[0] <= value <= sp_bounds[1]):
                all_pass = False
        elif "Disparate Impact" in metric:
            if not (di_bounds[0] <= value <= di_bounds[1]):
                all_pass = False
        elif "Average Odds" in metric or "Equal Opportunity" in metric:
            if not (eod_bounds[0] <= value <= eod_bounds[1]):
                all_pass = False

    status.change_status(all_pass)
    return status


def check_all_fairness_metrics(report: Report, config: Configuration) -> dict:
    """Returns a dict of metric name → pass/fail for all fairness metrics in the report.

    Config attributes:
        sp_bounds (optional), di_bounds (optional), eod_bounds (optional)
    """
    metrics_df = report.load_report()
    if metrics_df is None:
        return {}

    sp_bounds = getattr(config, "sp_bounds", (-0.1, 0.1))
    di_bounds = getattr(config, "di_bounds", (0.8, 1.2))
    eod_bounds = getattr(config, "eod_bounds", (-0.1, 0.1))

    results = {}
    for _, row in metrics_df.iterrows():
        metric, value = row["Metric"], row["Value"]
        if "Statistical Parity" in metric:
            results[metric] = bool(sp_bounds[0] <= value <= sp_bounds[1])
        elif "Disparate Impact" in metric:
            results[metric] = bool(di_bounds[0] <= value <= di_bounds[1])
        elif "Average Odds" in metric or "Equal Opportunity" in metric:
            results[metric] = bool(eod_bounds[0] <= value <= eod_bounds[1])
    return results


############################################################## Bias Mitigation techniques


def bias_mitigation_in_process_train(
    data: Data, config: Configuration, report: Report
) -> Report:
    """In-processing bias mitigation using Grid Search Reduction (Agarwal et al., 2018).

    Trains a RidgeClassifier wrapped in GridSearchReduction to enforce demographic parity
    during the training procedure itself.

    Config attributes:
        test_size, random_state, sensitive_features, id_feature, target_feature,
        group_col, group_a_val, group_b_val
    """
    data_df = data.load_dataset()
    data_train, data_test = train_test_split(
        data_df, test_size=config.test_size, random_state=config.random_state
    )
    X_train, y_train, dem_train = split_demographic_data_from_df(
        data_train, config.sensitive_features, config.id_feature, config.target_feature
    )
    X_test, y_test, dem_test = split_demographic_data_from_df(
        data_test, config.sensitive_features, config.id_feature, config.target_feature
    )

    group_a_train = dem_train[config.group_col] == config.group_a_val
    group_b_train = dem_train[config.group_col] == config.group_b_val
    group_a_test = dem_test[config.group_col] == config.group_a_val
    group_b_test = dem_test[config.group_col] == config.group_b_val

    base_model = RidgeClassifier(random_state=config.random_state)
    gsr = GridSearchReduction()
    gsr.transform_estimator(base_model)
    gsr.fit(X_train, y_train, group_a_train, group_b_train)

    y_pred_test = gsr.predict(X_test)
    group_label = f"{config.group_a_val} vs {config.group_b_val}"
    metrics_gsr = get_metrics(group_a_test, group_b_test, y_pred_test, y_test, group_label)

    report.save_report(metrics_gsr)
    return report


############################################################## Explainability


def explain_model_predictions(
    blackbox_model: Model, X_train: Data, y_test: Data, config: Configuration
):
    seed = config.seed
    lime = LimeTabular(blackbox_model, X_train, random_state=seed)
    show(lime.explain_local(X_test[:5], y_test[:5]), 0)


############################################################## Post Processing Bias Mitigation


def post_process_bias_mitigation_eq_odds(
    data_test: Data, model: Model, config: Configuration, report: Report
) -> Report:
    """Post-processing bias mitigation using Equalized Odds (Hardt et al., 2016).

    Splits the test set into a post-processor calibration split and an evaluation split,
    fits EqualizedOdds on calibration predictions, then adjusts evaluation predictions.

    Config attributes:
        sensitive_features, id_feature, target_feature, group_col, group_a_val, group_b_val,
        pp_test_size (optional, default 0.4), random_state (optional, default 42)
    """
    model_obj = model.load_model()
    data_df = data_test.load_dataset()
    pp_test_size = getattr(config, "pp_test_size", 0.4)
    random_state = getattr(config, "random_state", 42)

    data_pp_train, data_pp_test = train_test_split(
        data_df, test_size=pp_test_size, random_state=random_state
    )
    X_pp_train, y_pp_train, dem_pp_train = split_demographic_data_from_df(
        data_pp_train, config.sensitive_features, config.id_feature, config.target_feature
    )
    X_pp_test, y_pp_test, dem_pp_test = split_demographic_data_from_df(
        data_pp_test, config.sensitive_features, config.id_feature, config.target_feature
    )

    group_a_pp_train = dem_pp_train[config.group_col] == config.group_a_val
    group_b_pp_train = dem_pp_train[config.group_col] == config.group_b_val
    group_a_pp_test = dem_pp_test[config.group_col] == config.group_a_val
    group_b_pp_test = dem_pp_test[config.group_col] == config.group_b_val

    eq = EqualizedOdds(solver="highs", seed=random_state)
    y_pred_pp_train = model_obj.predict(X_pp_train)
    eq.fit(y_pp_train, y_pred_pp_train, group_a=group_a_pp_train, group_b=group_b_pp_train)

    y_pred_pp_test = model_obj.predict(X_pp_test)
    d = eq.transform(y_pred_pp_test, group_a=group_a_pp_test, group_b=group_b_pp_test)
    y_pred_adjusted = d["y_pred"]

    group_label = f"{config.group_a_val} vs {config.group_b_val}"
    metrics_eq = get_metrics(group_a_pp_test, group_b_pp_test, y_pred_adjusted, y_pp_test, group_label)
    report.save_report(metrics_eq)
    return report


############################################################## Robustness


def robustness_evaluation():
    import lightgbm as lgb
    import numpy as np
    from art.attacks.evasion import ZooAttack
    from art.estimators.classification import LightGBMClassifier
    from art.utils import load_mnist

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = (
        load_mnist()
    )

    x_test = x_test[0:5]
    y_test = y_test[0:5]

    nb_samples_train = x_train.shape[0]
    nb_samples_test = x_test.shape[0]
    x_train = x_train.reshape((nb_samples_train, 28 * 28))
    x_test = x_test.reshape((nb_samples_test, 28 * 28))

    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": 10,
        "force_col_wise": True,
    }
    train_set = lgb.Dataset(x_train, label=np.argmax(y_train, axis=1))
    test_set = lgb.Dataset(x_test, label=np.argmax(y_test, axis=1))
    model = lgb.train(
        params=params, train_set=train_set, num_boost_round=100, valid_sets=[test_set]
    )

    classifier = LightGBMClassifier(
        model=model, clip_values=(min_pixel_value, max_pixel_value)
    )

    predictions = classifier.predict(x_test)
    accuracy = np.sum(
        np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)
    ) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    attack = ZooAttack(
        classifier=classifier,
        confidence=0.5,
        targeted=False,
        learning_rate=1e-1,
        max_iter=200,
        binary_search_steps=100,
        initial_const=1e-1,
        abort_early=True,
        use_resize=False,
        use_importance=False,
        nb_parallel=250,
        batch_size=1,
        variable_h=0.01,
    )
    x_test_adv = attack.generate(x=x_test)

    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(
        np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)
    ) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))





if __name__ == "__main__":
    data = Data(filepath="artifacts/data/data_training.parquet")
    data_testing = Data(filepath="artifacts/data/data_testing.csv")
    config_model = Configuration(
        config={
            "test_size": 0.3,
            "random_state": 4,
            "model_filepath": "model_baseline.pickle",
            "sensitive_features": ["nationality", "gender"],
            "id_feature": "Id",
            "target_feature": "decision",
            "group_col": "nationality",
            "group_a_val": "Dutch",
            "group_b_val": "Belgian",
        }
    )
    model = Model(model_path="artifacts/model/model_baseline.pickle")
    report_filepath = os.path.join(REPORT_ARTIFACTS_PATH, "report_accuracy_demographics.csv")
    report = Report(filepath=report_filepath)
    status = Status(
        "model_validation", os.path.join(STATUS_ARTIFACTS_PATH, "model_validation.json")
    )

    train_model(data, config_model)
    # model_evaluation_accuracy(data_testing, config_model, model, report)
    # model_evaluation_accuracy_demographic_groups(model, data_testing, config_model, report)
    # calculate_success_rate(data_testing, config_model, model, report)
    # calculate_statistical_parity(data_testing, config_model, model, report)
    # confusion_matrices(data_testing, config_model, model, report)
    # calculate_equal_opportunity_difference(data_testing, config_model, model, report)
    # check_all_fairness_metrics(report, config_model)
    # model_validation_baseline(report, config_model, status)
    # train_model_reweighing(data, config_model)
    # bias_mitigation_in_process_train(data, config_model, report)
    # post_process_bias_mitigation_eq_odds(data_testing, model, config_model, report)
    # permutation_feature_importance(data_testing, model, config_model, report)
