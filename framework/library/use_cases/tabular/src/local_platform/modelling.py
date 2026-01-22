import os
import pickle
from pickle import dump

# import holisticai
# from holisticai.bias.metrics import (
#    disparate_impact,
#    statistical_parity,
#    average_odds_diff,
# )
# from interpret.blackbox import LimeTabular
# from interpret import show

from library.src.artifact_types import Data, Configuration, Report, Model
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

# import mlflow
# import mlflow.sklearn


FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
MODEL_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "model")
REPORT_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")

#################################################### Feature Engineering


def permutation_feature_importance(data_test: Data, model: Model):
    # Function to permute a specific feature column in the dataset
    def permute_X(X, j):
        Xj = X.copy()
        Xj[j] = Xj[j].sample(frac=1).values  # Shuffle feature values
        return Xj

    X_test = data_test.get_dataset()
    # Define parameters
    np.random.seed(10)
    n_features = len(X_test.columns)
    n_iter = 10

    # Initialize arrays to store accuracy and disparate impact results
    accs = np.zeros((n_iter, n_features))
    dis = np.zeros((n_iter, n_features))

    # Iterate over features and perform permutation testing
    for j in tqdm(range(n_features)):
        for i in range(n_iter):
            # Shuffle feature j and make predictions
            X_test_permuted = permute_X(X_test, str(j))
            y_pred_test_permuted = model.predict(X_test_permuted)

            # Compute accuracy and disparate impact and store results
            accs[i, j] = accuracy_score(y_test, y_pred_test_permuted)
            dis[i, j] = disparate_impact(
                group_a_test, group_b_test, y_pred_test_permuted
            )


def get_metrics_classifier(group_a, group_b, y_pred, y_true):
    """
    Returns a DataFrame of model accuracy and fairness metrics for two groups.
    """
    metrics = [
        ["Model Accuracy", round(accuracy_score(y_true, y_pred), 2), 1]
    ]  # Calculate accuracy
    metrics += [
        [
            "Black vs. White Disparate Impact",
            round(disparate_impact(group_a, group_b, y_pred), 2),
            1,
        ]
    ]  # Calculate disparate impact
    metrics += [
        [
            "Black vs. White Statistical Parity",
            round(statistical_parity(group_a, group_b, y_pred), 2),
            0,
        ]
    ]  # Calculate statistical parity
    metrics += [
        [
            "Black vs. White Average Odds Difference",
            round(average_odds_diff(group_a, group_b, y_pred, y_true), 2),
            0,
        ]
    ]  # Calculate average odds difference
    return pd.DataFrame(
        metrics, columns=["Metric", "Value", "Reference"]
    )  # Return metrics as DataFrame


#################################################### Model Training ####################################################


def train_model_mlflow(data: Data, config: Configuration):
    """ """
    data = data.get_dataset()
    # Split the data into training and testing sets (70% training, 30% testing)
    data_train, data_test = train_test_split(
        data, test_size=config.test_size, random_state=config.random_state
    )
    train_dataset = mlflow.data.from_pandas(data_train, name="train")
    test_dataset = mlflow.data.from_pandas(data_test, name="test")

    # Get the feature matrix (X), target labels (y), and demographic data for both sets
    X_train, y_train, dem_train = split_data_from_df(data_train)
    X_test, y_test, dem_test = split_data_from_df(data_test)

    # Set th experiment name
    # mlflow.set_experiment("hiring_classification")
    with mlflow.start_run(run_name="train_no_fairness"):

        # Define the model (RidgeClassifier) and train it on the training data
        model = RidgeClassifier(random_state=config.random_state)
        model.fit(X_train, y_train)

        model_output_path = os.path.join(MODEL_ARTIFACTS_PATH, config.model_filepath)
        with open(model_output_path, "wb") as model_file:
            dump(model, model_file, pickle.HIGHEST_PROTOCOL)

        # Make predictions on the test set
        y_pred_test = model.predict(X_test)

        # Define the groupings for fairness analysis (Black and White) in the test set
        group_a_test = dem_test["nationality"] == "Dutch"
        group_b_test = dem_test["nationality"] == "Belgian"

        metrics_rw = get_metrics_classifier(
            group_a_test, group_b_test, y_pred_test, y_test
        )
        metrics_rw.to_csv("metrics_accuracy.csv")

        mlflow.log_param("model_type", "RidgeClassifier")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="RidgeClassifier",
            registered_model_name="RidgeClassifier",
            input_example=X_train,
        )
        mlflow.log_metrics(
            metrics={metric.Metric: metric.Value for metric in metrics_rw.itertuples()},
            dataset=train_dataset,
            model_id=model_info.model_id,
        )
        print(model_info.model_id)

        # Add the model's predictions to the data_test DataFrame for easier analysis
        data_test = data_test.copy()
        data_test["Pred"] = y_pred_test
        # TODO generate report to return from this method

def train_model(data: Data, config: Configuration, report: Report):
    """ """
    data = data.get_dataset()
    # Split the data into training and testing sets (70% training, 30% testing)
    data_train, data_test = train_test_split(
        data, test_size=config.test_size, random_state=config.random_state
    )
    print(data_train)
    artifact_data_train = Data(os.path.join(DATA_ARTIFACTS_PATH, "data_training.csv"))
    artifact_data_train.log_dataset(data_train)
    artifact_data_test = Data(os.path.join(DATA_ARTIFACTS_PATH, "data_testing.csv"))
    artifact_data_test.log_dataset(data_train)
    # Get the feature matrix (X), target labels (y), and demographic data for both sets
    X_train, y_train, dem_train = split_data_from_df(data_train, config.sensitive_features)
    X_test, y_test, dem_test = split_data_from_df(data_test, config.sensitive_features)

    # Define the model (RidgeClassifier) and train it on the training data
    model = RidgeClassifier(random_state=config.random_state)
    model.fit(X_train, y_train)

    model_output_path = os.path.join(MODEL_ARTIFACTS_PATH, config.model_filepath)
    with open(model_output_path, "wb") as model_file:
        dump(model, model_file, pickle.HIGHEST_PROTOCOL)

    # Make predictions on the test set
    y_pred_test = model.predict(X_test)

    # Define the groupings for fairness analysis (Black and White) in the test set
    group_a_test = dem_test["nationality"] == "Dutch"
    group_b_test = dem_test["nationality"] == "Belgian"

    metrics_rw = get_metrics_classifier(
        group_a_test, group_b_test, y_pred_test, y_test
    )
    path = os.path.join(REPORT_ARTIFACTS_PATH, report.report_filepath)
    metrics_baseline_report = Report(path)
    metrics_baseline_report.save_report(metrics_rw)       

def hyperparameters_optimization(data: Data, config: Configuration):
    data = data.get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        data, test_size=config.test_size, random_state=config.random_state
    )
    lr = ElasticNet()
    # Define distribution to pick parameter values from
    distributions = dict(
        alpha=uniform(loc=0, scale=10),  # sample alpha uniformly from [-5.0, 5.0]
        l1_ratio=uniform(),  # sample l1_ratio uniformlyfrom [0, 1.0]
    )
    # Initialize random search instance
    clf = RandomizedSearchCV(
        estimator=lr,
        param_distributions=distributions,
        # Optimize for mean absolute error
        scoring="neg_mean_absolute_error",
        # Use 5-fold cross validation
        cv=5,
        # Try 100 samples. Note that MLflow only logs the top 5 runs.
        n_iter=100,
    )
    # Start a parent run
    with mlflow.start_run(run_name="hyperparameter-tuning"):
        search = clf.fit(X_train, y_train)

        # Evaluate the best model on test dataset
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
):
    with open(model.filepath, "rb") as model_file:
        model = load(model_file)
    # prepare a validation dataset for prediction and predict
    data_test = data.load_dataset()  
    X_test, y_test, dem_test = split_data_from_df(data_test)
    y_pred_test = model.predict(X_test)

    # Calculate the accuracy of the model on the test set
    acc = accuracy_score(y_test, y_pred_test)
    evaluation_metric = pd.DataFrame(
        columns=["Metric", "Value"], data=[["Accuracy", acc]]
    )
    eval_acc_metric_report = Report(report.filepath).save_report(evaluation_metric)
    return eval_acc_metric_report


############################################################## Evaluations - Performance and Fairness evaluation metrics


def model_evaluation_accuracy_overall(
    data: Data, config: Configuration, model: Model
) -> Report:
    model_name = model.name
    model_version = model.vesion

    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")

    # prepare a validation dataset for prediction and predict
    data = data.get_dataset()  # TODO load the test dataset
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=4)
    # Get the feature matrix (X), target labels (y), and demographic data for both sets
    X_test, y_test, dem_test = split_data_from_df(data_test)
    y_pred_test = model.predict(X_test)

    # Calculate the accuracy of the model on the test set
    acc = accuracy_score(y_test, y_pred_test)
    evaluation_metric = pd.DataFrame(
        columns=["Metric", "Value"], data=[["Accuracy", acc]]
    )
    report = Report(config.file_path).save_report(evaluation_metric)
    return report


def model_evaluation_accuracy_demographic_groups(
    data: Data, config: Configuration, model: Model
) -> Report:
    model_name = model.name
    model_version = model.vesion

    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")

    # prepare a validation dataset for prediction and predict
    data = data.get_dataset()  # TODO load the test dataset
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=4)
    # Get the feature matrix (X), target labels (y), and demographic data for both sets
    X_test, y_test, dem_test = split_data_from_df(data_test)
    y_pred_test = model.predict(X_test)

    evaluation_metrics = []
    print("---- ACCURACY BY GENDER ----")
    # Calculate accuracy for each gender group
    dem_test = dem_test.reset_index(drop=True)
    for group in dem_test["Gender"].unique():
        # Get the indices of the samples belonging to the current group
        idx_group = dem_test[dem_test["Gender"] == group].index

        # Calculate the accuracy for the current group
        acc = accuracy_score(y_test[idx_group], y_pred_test[idx_group])
        print(group, "Accuracy = %.3f" % acc)
        metrics += [["Accuracy by gender", group, "%.3f" % acc]]
    print("---- ACCURACY BY ETHNICITY ----")
    # Calculate accuracy for each ethnicity group
    for group in dem_test["Ethnicity"].unique():
        # Get the indices of the samples belonging to the current group
        idx_group = dem_test[dem_test["Ethnicity"] == group].index

        # Calculate the accuracy for the current group
        acc = accuracy_score(y_test[idx_group], y_pred_test[idx_group])
        print(group, "Accuracy = %.3f" % acc)
    evaluation_results = pd.DataFrame(
        metrics, columns=["Metric Type", "Group", "Value"]
    )
    report = Report(evaluation_results)
    report.save_report(config.filepath)
    return report


############################################################## Bias Mitigation techniques


def bias_mitigation_in_process_train(data: Data, config: Configuration, report: Report):
    data = data.get_dataset()
    # Split the data into training and testing sets (70% training, 30% testing)
    data_train, data_test = train_test_split(
        data, test_size=config.test_size, random_state=config.random_state
    )
    train_dataset = mlflow.data.from_pandas(data_train, name="train")
    test_dataset = mlflow.data.from_pandas(data_test, name="test")

    # Get the feature matrix (X), target labels (y), and demographic data for both sets
    X_train, y_train, dem_train = split_data_from_df(data_train)
    X_test, y_test, dem_test = split_data_from_df(data_test)

    sample_weights = data_train["sample_weights"]
    with mlflow.start_run():
        # Train the model using the sample weights calculated through reweighing
        model = RidgeClassifier(random_state=config.random_state)
        model.fit(
            X_train, y_train, sample_weight=sample_weights.ravel()
        )  # Fit model with sample weights

        y_pred_test = model.predict(X_test)

        # Define the groupings for fairness analysis (Black and White) in the test set
        group_a_test = dem_test["Ethnicity"] == "Black"
        group_b_test = dem_test["Ethnicity"] == "White"

        # Get the fairness and accuracy metrics after applying reweighing
        metrics_rw = get_metrics_classifier(
            group_a_test, group_b_test, y_pred_test, y_test
        )
        print(metrics_rw)

        mlflow.log_param("model_type", "RidgeClassifier_SampleWeights")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="RidgeClassifier_SampleWeights",
            registered_model_name="RidgeClassifier_SampleWeights",
            input_example=X_train,
        )
        mlflow.log_metrics(
            metrics={metric.Metric: metric.Value for metric in metrics_rw.itertuples()},
            dataset=train_dataset,
            model_id=model_info.model_id,
        )

        # Add a 'mitigation' column to both metrics dataframes to label them accordingly
        metrics_orig = pd.read_csv(
            "metrics_accuracy.csv"
        )  # TODO: read as input Report artifact
        metrics_orig["mitigation"] = "None"
        metrics_rw["mitigation"] = "Reweighing"

        metrics = pd.concat([metrics_orig, metrics_rw], axis=0, ignore_index=True)
        print(metrics)
        compare_metrics(metrics)


############################################################## Explainability


def explain_model_predictions(blackbox_model, X_train, y_test):
    seed = 4
    lime = LimeTabular(blackbox_model, X_train, random_state=seed)
    show(lime.explain_local(X_test[:5], y_test[:5]), 0)


############################################################## Robustness


def robustness_evaluation():
    import lightgbm as lgb
    import numpy as np
    from art.attacks.evasion import ZooAttack
    from art.estimators.classification import LightGBMClassifier
    from art.utils import load_mnist

    # Step 1: Load the MNIST dataset

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = (
        load_mnist()
    )

    # Step 1a: Flatten dataset

    x_test = x_test[0:5]
    y_test = y_test[0:5]

    nb_samples_train = x_train.shape[0]
    nb_samples_test = x_test.shape[0]
    x_train = x_train.reshape((nb_samples_train, 28 * 28))
    x_test = x_test.reshape((nb_samples_test, 28 * 28))

    # Step 2: Create the model

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

    # Step 3: Create the ART classifier

    classifier = LightGBMClassifier(
        model=model, clip_values=(min_pixel_value, max_pixel_value)
    )

    # Step 4: Train the ART classifier

    # The model has already been trained in step 2

    # Step 5: Evaluate the ART classifier on benign test examples

    predictions = classifier.predict(x_test)
    accuracy = np.sum(
        np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)
    ) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    # Step 6: Generate adversarial test examples
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

    # Step 7: Evaluate the ART classifier on adversarial test examples

    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(
        np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)
    ) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))


if __name__ == "__main__":
    data = Data(filepath="artifacts/data/data_training.parquet")
    config = Configuration(config={"test_size": 0.3, "random_state": 4, 
                                   "model_filepath": "model_baseline.pickle",
                                   "sensitive_features": ["nationality", "gender"]})
    train_model(data, config)
    