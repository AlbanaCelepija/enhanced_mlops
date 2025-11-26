import pandas as pd
from io import BytesIO
from datetime import datetime

from holisticai.bias.metrics import (
    disparate_impact,
    statistical_parity,
    average_odds_diff,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from model_card_toolkit import ModelCardToolkit

# visualization
import matplotlib.pyplot as plt
import seaborn as sns


def get_metrics_classifier(group_a, group_b, y_pred, y_true):
    """
    Function to calculate and return model accuracy and fairness metrics for two groups
    Returns a DataFrame of model accuracy and fairness metrics for two groups.
    """
    metrics = [
        ["Model Accuracy", round(accuracy_score(y_true, y_pred), 2), 1]
    ]  # Calculate accuracy
    metrics += [
        ["Precision", round(precision_score(y_true, y_pred), 2), 1]
    ]  # Calculate precision: Of the predited positives (TP + FP), how many are correctly predicted
    metrics += [
        ["Recall", round(recall_score(y_true, y_pred), 2), 1]
    ]  # Calculate recall: Of the actual positives (TP + FN), how many were correctly predicted
    metrics += [
        ["F1 Score", round(f1_score(y_true, y_pred), 2), 1]
    ]  # Calculate f1-score
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


def compare_metrics(metrics):
    now = datetime.now()
    # Plot the comparison of metrics between the original model and the model with reweighing
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics, x="Metric", y="Value", hue="mitigation")
    plt.axhline(y=0.8, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=-0.05, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=1, linewidth=2, color="g")
    plt.axhline(y=0, linewidth=2, color="g")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    # plt.show()
    # Save the figure
    plt.savefig(f"metrics_comparison_{now}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_to_str():
    img = BytesIO()
    plt.savefig(img, format="png")
    return base64.encodebytes(img.getvalue()).decode("utf-8")


def generate_model_card():
    # TODO
    mct = ModelCardToolkit()
    model_card = mct.scaffold_assets()
