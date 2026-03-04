# promote model

import os
import mlflow


def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "abhirupray14"
    repo_name = "mlops-mini-project"

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    client = mlflow.MlflowClient()

    model_name = "my_model"

    # Get latest staging versions
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

    if not staging_versions:
        print("No model currently in Staging. Skipping promotion.")
        return

    latest_version_staging = staging_versions[0].version

    # Archive current production models
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])

    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote staging model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )

    print(f"Model version {latest_version_staging} promoted to Production")


if __name__ == "__main__":
    promote_model()