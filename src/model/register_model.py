# register model

import json
import mlflow
import logging
import os
import dagshub
from dotenv import load_dotenv


# Set up DagsHub credentials for MLflow tracking
load_dotenv()  # loads .env file

dagshub_token = os.getenv("DAGSHUB_PAT")
dagshub_username = os.getenv("DAGSHUB_USERNAME")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "abhirupray14"
repo_name = "mlops-mini-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry and transition to Staging."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model (creates a version)
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        print(f"Model registered. Version: {model_version.version}")

        client = mlflow.tracking.MlflowClient()

        # Wait until model version is READY (important for remote servers like DagsHub)
        import time
        time.sleep(5)

        # Transition to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=True  # IMPORTANT
        )

        print(f"Model version {model_version.version} transitioned to Staging.")

    except Exception as e:
        print("REGISTRATION ERROR:", e)
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()