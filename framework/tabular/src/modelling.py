from pickle import dump

import holisticai
from holisticai.bias.metrics import disparate_impact,statistical_parity, average_odds_diff
from interpret.blackbox import LimeTabular
from interpret import show

from artifact_types import Data, Configuration, Report, Model
from fairness.holisticAI.src.utils import *
from fairness.holisticAI.src.data_preparation import *

from scipy.stats import uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import mlflow
import mlflow.sklearn

def train_model(data: Data, config: Configuration):
    """
    
    """
    data = data.get_dataset()
    # Split the data into training and testing sets (70% training, 30% testing)
    data_train, data_test = train_test_split(data, test_size=config.test_size, random_state=config.random_state)
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
        
        with open(config.model_filepath, 'wb') as model_file:
            dump(model, model_file, pickle.HIGHEST_PROTOCOL)

        # Make predictions on the test set
        y_pred_test = model.predict(X_test)
        
        # Define the groupings for fairness analysis (Black and White) in the test set
        group_a_test = (dem_test['Ethnicity']=='Black')
        group_b_test = (dem_test['Ethnicity']=='White')

        metrics_rw = get_metrics_classifier(group_a_test, group_b_test, y_pred_test, y_test)
        metrics_rw.to_csv("metrics_accuracy.csv")   
        
        mlflow.log_param("model_type", "RidgeClassifier")
        model_info = mlflow.sklearn.log_model(
            sk_model=model, 
            name="RidgeClassifier",
            registered_model_name="RidgeClassifier",  
            input_example=X_train)
        mlflow.log_metrics(
            metrics={
                metric.Metric: metric.Value
                for metric in metrics_rw.itertuples()
            },
            dataset=train_dataset,
            model_id=model_info.model_id,
        )
        print(model_info.model_id)

        # Add the model's predictions to the data_test DataFrame for easier analysis
        data_test = data_test.copy()
        data_test['Pred'] = y_pred_test
        #TODO generate report to return from this method 
        
def hyperparameters_optimization(data: Data, config: Configuration):
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

def plot_cm(y_true, y_pred, labels=[1, 0], display_labels=[1, 0], ax=None):
    """
    Plots a single confusion matrix with annotations
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)  # Compute confusion matrix

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))  # Create new figure if no axis is provided

    # Create heatmap for confusion matrix
    sns.heatmap(
        cm, annot=True, fmt="g", cmap="viridis", cbar=False,
        xticklabels=display_labels, yticklabels=display_labels,
        square=True, linewidths=2, linecolor="black", ax=ax, annot_kws={"size": 14}
    )

    # Label and format axes
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_yticklabels(display_labels, fontsize=11)

    return cm  # Return confusion matrix


def plot_confusion_matrices(groups, data_test, category, y_test, y_pred_test):
    """
    Plots confusion matrices for each group in a given category.
    """
    num_groups = len(groups) + 1  # Number of groups to display
    fig, axes = plt.subplots(1, num_groups, figsize=(5 * num_groups, 4))  # Create subplot grid

    # Plot confusion matrix for overall data
    cm = plot_cm(y_test, y_pred_test, ax=axes[0])
    axes[0].set_title("All", fontsize=14, fontweight="bold")

    # Plot confusion matrices for each group in the dataset
    cm_dict = {"All": cm}  # Store overall confusion matrix
    for i, group in enumerate(groups):
        ax = axes[i + 1]  # Get axis for group
        subset = data_test[data_test[category] == group]  # Filter data for group
        cm = plot_cm(subset["Label"], subset["Pred"], ax=ax)  # Plot confusion matrix for group
        cm_dict[group] = cm  # Store confusion matrix for group
        ax.set_title(group, fontsize=14, fontweight="bold")

    plt.tight_layout()  # Adjust layout
    plt.show()  # Display plot
    return cm_dict  # Return dictionary of confusion matrices for each group


def calculate_tpr(cms):
    """
    Calculates True Positive Rates (TPR) for each group,
    given a set of confusion matrices.
    """
    tprs = {g: cm[0, 0] / cm[0, :].sum() for g, cm in cms.items()}  # Calculate TPR
    return tprs  # Return dictionary of TPRs

############################################################## Accuracy and fairness evaluation metrics

def model_evaluation(data: Data, config: Configuration, model: Model):
    model_name = model.name
    model_version = model.vesion

    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")    
    
    # prepare a validation dataset for prediction and predict
    data = data.get_dataset()
    y_pred_new = model.predict(data)
    # TODO compare metrics and generate report (csv/json)

############################################################## Bias Mitigation techniques

def bias_mitigation_in_process_train(data: Data, config: Configuration, report: Report):
    data = data.get_dataset()   
    # Split the data into training and testing sets (70% training, 30% testing)
    data_train, data_test = train_test_split(data, test_size=config.test_size, random_state=config.random_state)
    train_dataset = mlflow.data.from_pandas(data_train, name="train")
    test_dataset = mlflow.data.from_pandas(data_test, name="test")

    # Get the feature matrix (X), target labels (y), and demographic data for both sets
    X_train, y_train, dem_train = split_data_from_df(data_train)
    X_test, y_test, dem_test = split_data_from_df(data_test)
    
    sample_weights = data_train["sample_weights"]
    with mlflow.start_run():
        # Train the model using the sample weights calculated through reweighing
        model = RidgeClassifier(random_state=config.random_state)    
        model.fit(X_train, y_train, sample_weight=sample_weights.ravel())  # Fit model with sample weights

        y_pred_test = model.predict(X_test)

        # Define the groupings for fairness analysis (Black and White) in the test set
        group_a_test = (dem_test['Ethnicity']=='Black')
        group_b_test = (dem_test['Ethnicity']=='White')

        # Get the fairness and accuracy metrics after applying reweighing
        metrics_rw = get_metrics_classifier(group_a_test, group_b_test, y_pred_test, y_test)
        print(metrics_rw)
        
        mlflow.log_param("model_type", "RidgeClassifier_SampleWeights")
        model_info = mlflow.sklearn.log_model(
            sk_model=model, 
            name="RidgeClassifier_SampleWeights",
            registered_model_name="RidgeClassifier_SampleWeights",  
            input_example=X_train)
        mlflow.log_metrics(
            metrics={
                metric.Metric: metric.Value
                for metric in metrics_rw.itertuples()
            },
            dataset=train_dataset,
            model_id=model_info.model_id,
        )

        # Add a 'mitigation' column to both metrics dataframes to label them accordingly
        metrics_orig = pd.read_csv("metrics_accuracy.csv") # TODO: read as input Report artifact
        metrics_orig['mitigation'] = 'None'
        metrics_rw['mitigation'] = 'Reweighing'

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

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

    # Step 1a: Flatten dataset

    x_test = x_test[0:5]
    y_test = y_test[0:5]

    nb_samples_train = x_train.shape[0]
    nb_samples_test = x_test.shape[0]
    x_train = x_train.reshape((nb_samples_train, 28 * 28))
    x_test = x_test.reshape((nb_samples_test, 28 * 28))

    # Step 2: Create the model

    params = {"objective": "multiclass", "metric": "multi_logloss", "num_class": 10, "force_col_wise": True}
    train_set = lgb.Dataset(x_train, label=np.argmax(y_train, axis=1))
    test_set = lgb.Dataset(x_test, label=np.argmax(y_test, axis=1))
    model = lgb.train(params=params, train_set=train_set, num_boost_round=100, valid_sets=[test_set])

    # Step 3: Create the ART classifier

    classifier = LightGBMClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value))

    # Step 4: Train the ART classifier

    # The model has already been trained in step 2

    # Step 5: Evaluate the ART classifier on benign test examples

    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
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
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))