import ollama
import requests
from utils import *
from artifact_types import Data, Configuration, Report
from sklearn.metrics import accuracy_score, confusion_matrix

from evidently.report import Report
from evidently.metrics import *
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

################################################################################ Model deployment


def model_deployment(config: Configuration):
    model_name = config.name
    model_path = config.path
    ollama.create(model=model_name, from_=model_path)


################################################################################ Model monitoring
# experiments tracking: efficient


def model_monitoring():
    """mlflow"""
    pass

################################################################################ Production data monitoring

def production_data_monitoring_data_drift(prod_data: Data, reference_data: Data) -> Report:
    report = Report(
        metrics=[
            # Drift in numerical embedding space
            DataDriftPreset(),            
            # Drift in predictions
            TargetDriftPreset(),             
            # Extra: classification quality (if true labels in production exist)
            ClassificationQualityMetric() if "label" in current_data else None
        ]
    )
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("text_and_prediction_drift_report.html")

################################################################################ proxy for inference service


def pre_inference_transformation():
    pass


def post_inference_transformation():
    pass
