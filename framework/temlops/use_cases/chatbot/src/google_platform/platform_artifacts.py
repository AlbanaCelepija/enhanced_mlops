import os
import pickle
import pandas as pd
from pickle import load, dump
from temlops.src.artifact_types import Data, Model, Configuration, Report, Status, Documentation

from google.cloud import aiplatform
from google.cloud.aiplatform.prediction import LocalModel


FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
REPORTS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")
MODEL_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "model")


class ModelText(Model):
    def __init__(self, model_path):
        super().__init__(model_path)
        
    def load_model(self):
        with open(self.model_path, "rb") as model_file:
            model = load(model_file)
            return model
        
    def save_model(self, model):
        model_output_path = os.path.join(MODEL_ARTIFACTS_PATH, self.model_path)
        with open(model_output_path, "wb") as model_file:
            dump(model, model_file, pickle.HIGHEST_PROTOCOL)
            

def push_to_artifact_registry():
    # Define your container details in Google Artifact Registry
    CUSTOM_HANDLER_IMAGE_URI = "us-central1-docker.pkg.dev/your-project-id/llm-repo/cpr-llm:v1"

    # Build the container automatically using Vertex AI SDK
    local_model = LocalModel.build_cpr_model(
        src_dir=".",                          # Directory containing predictor.py and requirements.txt
        output_image_uri=CUSTOM_HANDLER_IMAGE_URI,
        predictor=LLMPredictor,               # Your custom class reference
        requirements_path="requirements.txt"
    )

    # Push the container to your Artifact Registry
    local_model.push()