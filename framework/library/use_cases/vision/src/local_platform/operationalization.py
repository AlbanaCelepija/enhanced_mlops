import ollama
import requests
from utils import *
from artifact_types import Data, Configuration, Report
from sklearn.metrics import accuracy_score, confusion_matrix

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


################################################################################ proxy for inference service


def pre_inference_transformation():
    pass


def post_inference_transformation():
    pass
