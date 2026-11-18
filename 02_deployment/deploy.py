"""Deployment: package the trained model, upload it, create, check and delete the endpoint."""

import argparse
import tarfile

import boto3
import joblib
import lightgbm
import sagemaker
from botocore.exceptions import ClientError
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.sklearn.model import SKLearnModel

import config

logger = config.get_logger("deploy")

def _client():
    return boto3.client("sagemaker", region_name=config.AWS_REGION)

def endpoint_exists():
    try:
        _client().describe_endpoint(EndpointName=config.ENDPOINT_NAME)
        return True
    except ClientError as error:
        if "Could not find endpoint" in error.response["Error"]["Message"]:
            return False
        raise

def package_model():
    """Convert the saved joblib model into model.tar.gz (no retraining).

    SageMaker needs a tarball, and a LightGBM text file avoids pickle version clashes
    between your laptop and the AWS container.
    """
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}")
    model = joblib.load(config.MODEL_PATH)

    trained = set(model.feature_name_)
    expected = set(config.RAW_COLUMNS + config.ENGINEERED_COLUMNS)
    if trained != expected or len(trained) != config.EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Model features differ from the 69 expected: {trained ^ expected}")

    config.BUILD_DIR.mkdir(exist_ok=True)
    booster_path = config.BUILD_DIR / "model.txt"
    model.booster_.save_model(str(booster_path))

    tar_path = config.BUILD_DIR / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(booster_path, arcname="model.txt")

    # The container installs this file at start-up; the version matches your local one.
    (config.ENDPOINT_CODE_DIR / "requirements.txt").write_text(
        "lightgbm==4.6.0\n")
    logger.info("Packaged model at %s", tar_path)
    return tar_path

def deploy_endpoint():
    if endpoint_exists():
        logger.info("Endpoint %s already exists. Nothing to do.", config.ENDPOINT_NAME)
        return

    tar_path = package_model()
    session = sagemaker.Session(boto_session=boto3.Session(region_name=config.AWS_REGION))
    model_uri = session.upload_data(str(tar_path), bucket=config.S3_BUCKET,
                                    key_prefix=config.S3_MODEL_PREFIX)
    logger.info("Uploaded model to %s", model_uri)

    sm_model = SKLearnModel(
        model_data=model_uri,
        role=config.SAGEMAKER_ROLE_ARN,
        entry_point="inference.py",
        source_dir=str(config.ENDPOINT_CODE_DIR),
        framework_version=config.SKLEARN_FRAMEWORK_VERSION,
        py_version="py3",
        sagemaker_session=session,
    )
    try:
        if config.DEPLOY_MODE == "serverless":
            sm_model.deploy(
                endpoint_name=config.ENDPOINT_NAME,
                serverless_inference_config=ServerlessInferenceConfig(
                    memory_size_in_mb=config.SERVERLESS_MEMORY_MB,
                    max_concurrency=config.SERVERLESS_MAX_CONCURRENCY))
        else:
            sm_model.deploy(endpoint_name=config.ENDPOINT_NAME, initial_instance_count=1,
                            instance_type=config.REALTIME_INSTANCE_TYPE)
    except ClientError as error:
        logger.error("Deployment failed: %s. Check the role ARN and CloudWatch logs.", error)
        raise
    logger.info("Endpoint %s is InService", config.ENDPOINT_NAME)

def endpoint_status():
    if not endpoint_exists():
        logger.info("Endpoint %s does not exist", config.ENDPOINT_NAME)
        return
    status = _client().describe_endpoint(EndpointName=config.ENDPOINT_NAME)["EndpointStatus"]
    logger.info("Endpoint %s status: %s", config.ENDPOINT_NAME, status)

def delete_endpoint():
    """Remove the endpoint, its config and model so nothing keeps billing."""
    if not endpoint_exists():
        logger.info("Nothing to delete")
        return
    client = _client()
    config_name = client.describe_endpoint(
        EndpointName=config.ENDPOINT_NAME)["EndpointConfigName"]
    variants = client.describe_endpoint_config(
        EndpointConfigName=config_name)["ProductionVariants"]
    model_names = [variant["ModelName"] for variant in variants]

    client.delete_endpoint(EndpointName=config.ENDPOINT_NAME)
    client.delete_endpoint_config(EndpointConfigName=config_name)
    for name in model_names:
        client.delete_model(ModelName=name)
    logger.info("Deleted endpoint, endpoint config and model(s)")

def main():
    parser = argparse.ArgumentParser(description="Manage the bankruptcy-risk endpoint")
    parser.add_argument("action", choices=["deploy", "status", "delete"])
    action = parser.parse_args().action
    {"deploy": deploy_endpoint, "status": endpoint_status, "delete": delete_endpoint}[action]()

if __name__ == "__main__":
    main()