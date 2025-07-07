"""MLflow model registry utilities."""
import mlflow
from typing import Optional, Dict, Any
from .config import MODEL_NAME, MODEL_STAGE_PRODUCTION
import re

from mlflow.tracking import MlflowClient

_ALLOWED = re.compile("[^0-9A-Za-z_-]")

def sanitize_model_name(name: str) -> str:
    """
    Sanitize model name to comply with MLflow's naming restrictions.
    Only allows alphanumeric characters, dashes, and underscores.
    """
    return re.sub(r'[^A-Za-z0-9_-]', '_', name)


def make_model_name(model_type: str, metric_name: str, metric_value: float) -> str:
    """
    Create a standardized model name from type and metric.
    Example: 'rf_accuracy_99_93' for a Random Forest with 99.93% accuracy
    """
    # Convert float to string with 2 decimal places and remove the dot
    metric_str = f"{metric_value:.2f}".replace('.', '_')
    name = f"{model_type}_{metric_name}_{metric_str}"
    return sanitize_model_name(name)


def register_model(
    model_uri: str,
    model_name: str,
    description: Optional[str] = None,
    overwrite: bool = True,
) -> str:
    """
    Register *model_uri* under *model_name*.
    If a registered model with that exact name already exists and
    ``overwrite is True`` it is deleted (all versions) before re-creation.
    
    Args:
        model_uri: URI of the model to register (e.g. 'runs:/run_id/model')
        model_name: Name to register the model under
        description: Optional description for the model version
        overwrite: If True, delete existing model with same name before registering
    
    Returns:
        Version number of the newly registered model as a string
    """
    client = MlflowClient()
    model_name = sanitize_model_name(model_name)

    if overwrite:
        try:                          # -- delete whole entry if present
            client.delete_registered_model(model_name)
            print(f"🗑️  Removed previous '{model_name}'")
        except Exception:
            pass                      # not present → nothing to delete

    client.create_registered_model(model_name)
    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        description=description,
    )
    print(f"✅ Registered {model_name} v{mv.version}")
    return mv.version


def promote_model_to_stage(model_name: Optional[str] = None,
                           version: Optional[str] = None,
                           stage: str = MODEL_STAGE_PRODUCTION) -> None:
    """
    Promote a model version to a specific stage using the fluent client.
    
    Args:
        model_name: Name of the registered model
        version: Version to promote (if None, promotes latest)
        stage: Target stage
    """
    name = model_name or MODEL_NAME
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Get latest version if not specified
        if version is None:
            latest = client.get_latest_versions(name, stages=["None"])
            if not latest:
                raise ValueError(f"No versions found for model {name}")
            version = latest[0].version
        
        # Transition to stage
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage
        )
        print(f"Promoted model {name} version {version} to {stage}")
        
    except Exception as e:
        print(f"Failed to promote model: {e}")
        raise


def load_model_from_registry(model_name: Optional[str] = None,
                             stage: str = MODEL_STAGE_PRODUCTION):
    """
    Load a model from the registry by name and stage.
    
    Args:
        model_name: Name of the registered model
        stage: Stage to load from
        
    Returns:
        Loaded model
    """
    name = model_name or MODEL_NAME
    model_uri = f"models:/{name}/{stage}"
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model {name} from {stage} stage")
        return model
    except Exception as e:
        print(f"Failed to load model from registry: {e}")
        raise


def load_model_from_run(run_id: str, artifact_path: str = "model"):
    """
    Load a model from a specific run.
    
    Args:
        run_id: MLflow run ID
        artifact_path: Path to the model artifact
        
    Returns:
        Loaded model
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model from run {run_id}")
        return model
    except Exception as e:
        print(f"Failed to load model from run: {e}")
        raise


def get_model_info(model_name: Optional[str] = None,
                   stage: str = MODEL_STAGE_PRODUCTION) -> Dict[str, Any]:
    """
    Get information about a registered model using the fluent client.
    
    Args:
        model_name: Name of the registered model
        stage: Stage to get info for
        
    Returns:
        Model information dictionary
    """
    name = model_name or MODEL_NAME
    client = mlflow.tracking.MlflowClient()
    
    try:
        model_version = client.get_latest_versions(name, stages=[stage])[0]
        
        return {
            "name": model_version.name,
            "version": model_version.version,
            "stage": model_version.current_stage,
            "description": model_version.description,
            "creation_timestamp": model_version.creation_timestamp,
            "last_updated_timestamp": model_version.last_updated_timestamp,
            "run_id": model_version.run_id
        }
    except Exception as e:
        print(f"Failed to get model info: {e}")
        raise 
