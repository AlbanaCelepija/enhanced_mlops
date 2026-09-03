"""Build the Unsloth image and submit the fine-tuning job to Vertex AI."""
import argparse
import subprocess

from google.cloud import aiplatform
from src.config import GCP, DATA


def build_and_push(tag: str = "unsloth-train:v1") -> str:
    uri = f"{GCP.region}-docker.pkg.dev/{GCP.project_id}/{GCP.ar_repo}/{tag}"
    # --platform matters on an ARM Mac; Vertex only runs linux/amd64.
    subprocess.run(
        ["docker", "build", "--platform", "linux/amd64",
         "-f", "Dockerfile.unsloth", "-t", uri, "."],
        check=True,
    )
    subprocess.run(["docker", "push", uri], check=True)
    return uri


def submit(image_uri: str, train_uri: str, name: str = "unsloth-sft", **overrides):
    aiplatform.init(project=GCP.project_id, location=GCP.region,
                    staging_bucket=GCP.staging_uri)

    args = ["--train-uri", train_uri]
    for key, value in overrides.items():
        args += [f"--{key.replace('_', '-')}", str(value)]

    job = aiplatform.CustomContainerTrainingJob(
        display_name=name,
        container_uri=image_uri,
    )
    model_dir = f"{GCP.bucket_uri}/models/{name}"
    job.run(
        args=args,
        base_output_dir=model_dir,          # becomes AIP_MODEL_DIR + "/model"
        # Phi-3-mini in 4-bit fits comfortably on one L4 — far cheaper than an
        # A100 and the quota is usually granted by default.
        machine_type="g2-standard-12",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        replica_count=1,
        service_account=GCP.service_account or None,
        environment_variables={"HF_TOKEN": ""},   # set for gated base models
    )
    return f"{model_dir}/model"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-uri",
                    default=f"{GCP.bucket_uri}/{DATA.processed_prefix}/{DATA.train_file}")
    ap.add_argument("--name", default="unsloth-sft")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--skip-build", action="store_true")
    a = ap.parse_args()

    image = (f"{GCP.region}-docker.pkg.dev/{GCP.project_id}/{GCP.ar_repo}/unsloth-train:v1"
             if a.skip_build else build_and_push())
    artifact_uri = submit(image, a.train_uri, a.name, epochs=a.epochs)
    print("\nDeploy with:")
    print(f"  python scripts/03_build_and_deploy.py --artifact-uri {artifact_uri}")
