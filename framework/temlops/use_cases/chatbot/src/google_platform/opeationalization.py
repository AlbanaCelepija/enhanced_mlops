import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud import aiplatform

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class LLMPredictor(Predictor):
    def __init__(self):
        self._pipeline = None

    def load(self, artifacts_uri: str):
        """Loads the tokenizer and model weights from GCS or local directory."""
        # Vertex AI automatically downloads your model artifacts to the local directory
        model_dir = os.environ.get("AIP_STORAGE_URI", artifacts_uri)
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self._pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer
        )

    def preprocess(self, prediction_input: dict) -> dict:
        """Processes the incoming JSON payload before inference."""
        instances = prediction_input.get("instances", [])
        if not instances:
            raise ValueError("No instances provided for prediction.")
        
        # Example preprocessing: extract prompt and inject a fallback system instruction
        payload = instances[0]
        prompt = payload.get("prompt", "")
        system_instruction = "System: Respond only in concise JSON format.\n"
            # Set up the engine, loads the NLP module (spaCy model by default) 
        # and other PII recognizers
        analyzer = AnalyzerEngine()
        # Call analyzer to get results
        results = analyzer.analyze(text=prompt,
                                entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "IBAN_CODE"],
                                language='en')
        # Analyzer results are passed to the AnonymizerEngine for anonymization
        anonymizer = AnonymizerEngine()
        anonymized_prompt = anonymizer.anonymize(text=prompt,analyzer_results=results)
        
        return {"formatted_prompt": f"{system_instruction}User: {anonymized_prompt}\nAI:"}

    def predict(self, preprocess_output: dict) -> dict:
        """Executes the LLM text generation loop."""
        prompt = preprocess_output["formatted_prompt"]
        
        # Run generation
        outputs = self._pipeline(
            prompt,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True
        )
        return {"generated_text": outputs[0]["generated_text"]}

    def postprocess(self, prediction_output: dict) -> dict:
        """Formats the generated text, stripping out metadata or bad words."""
        raw_text = prediction_output["generated_text"]
        
        # Clean up output formatting
        clean_text = raw_text.split("AI:")[-1].strip()
        
        # Vertex AI expects a dictionary with a top-level "predictions" key
        return {"predictions": [{"response": clean_text}]}


def model_deploy():
    # Initialize Vertex AI SDK
    aiplatform.init(project="your-project-id", location="us-central1")
    # Define your container details in Google Artifact Registry
    CUSTOM_HANDLER_IMAGE_URI = "us-central1-docker.pkg.dev/your-project-id/llm-repo/cpr-llm:v1"

    # 1. Register the model with your custom CPR container
    model = aiplatform.Model.upload(
        display_name="custom-llm-cpr",
        artifact_uri="gs://your-bucket-name/model-weights-folder/", # GCS path containing config.json/safetensors
        serving_container_image_uri=CUSTOM_HANDLER_IMAGE_URI
    )

    # 2. Deploy the model to an online inference endpoint with an L4 GPU
    endpoint = model.deploy(
        machine_type="g2-standard-4", # Select machine type based on model footprint
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        min_replica_count=1,
        max_replica_count=1
    )
    # Print out the exact identifiers to use later
    print(f"Numerical Endpoint ID: {endpoint.name}")
    print(f"Full Resource Name: {endpoint.resource_name}")


def pre_processing_transform(text):
    # NeMo Guardrails
    # presidio-analyzer presidio-anonymizer
    # LLM Guard

    # Set up the engine, loads the NLP module (spaCy model by default) 
    # and other PII recognizers
    analyzer = AnalyzerEngine()
    # Call analyzer to get results
    results = analyzer.analyze(text=text,
                            entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "IBAN_CODE"],
                            language='en')
    # Analyzer results are passed to the AnonymizerEngine for anonymization
    anonymizer = AnonymizerEngine()
    anonymized_text = anonymizer.anonymize(text=text,analyzer_results=results)

    return anonymized_text


def example_inference():
    # Initialize Vertex AI SDK
    aiplatform.init(project="your-project-id", location="us-central1")

    # 1. Get the deployed endpoint
    endpoint = aiplatform.Endpoint("projects/your-project-id/locations/us-central1/endpoints/your-endpoint-id")
    # Reference the existing endpoint using its saved ID
    #my_deployed_endpoint = aiplatform.Endpoint(endpoint_name="4815162342109876543")

    # 2. Prepare the input payload
    input_payload = {
        "instances": [
            {"prompt": "Explain the benefits of using MLOps in production."}
        ]
    }

    # 3. Make the prediction request
    prediction_response = endpoint.predict(instances=input_payload["instances"])

    # 4. Print the response
    print(prediction_response)