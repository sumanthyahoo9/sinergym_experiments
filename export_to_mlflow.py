"""
MLflow Model Versioning Script for HVAC RL Models
"""
import mlflow
import mlflow.pyfunc
import os
from pathlib import Path
from mlflow.tracking import MlflowClient

# Simple wrapper class for logging model artifacts
class RLModelWrapper(mlflow.pyfunc.PythonModel):
    """Simple wrapper to package RL model artifacts for MLflow"""
    
    def load_context(self, context):
        """Load model artifacts"""
        pass
    
    def predict(self, context, model_input):
        """Placeholder predict method"""
        return model_input

# Configuration
DATABRICKS_HOST = "https://dbc-9e9c29a5-e035.cloud.databricks.com"
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")  # Set this as environment variable

# Model paths
PRODUCTION_MODEL_DIR = "/home/sumanthmurthy/sinergym_experiments/SAC-Eplus-5zone-hot-continuous-stochastic-v1-episodes-100_2025-10-05_04-34-res1/evaluation/"
# Unity Catalog requires: catalog.schema.model format
REGISTERED_MODEL_NAME = "optivai_hvac_rl_model.default.hvac_rl_model"

# Set tracking URI - keep as "databricks" for tracking
mlflow.set_tracking_uri("databricks")
# Set registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
print(f"The Databricks host is {DATABRICKS_HOST}")
print(f"The Databricks API token is {DATABRICKS_TOKEN}")

def log_production_model():
    """Log the current production model to MLflow"""
    
    # Create or get experiment
    experiment_name = "/Users/smurthy@optivai.co.uk/OptivAI_HVAC_RL"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created experiment: {experiment_name}")
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    
    model_dir = Path(PRODUCTION_MODEL_DIR)
    
    with mlflow.start_run(run_name="optivai_hvac_rl_model_testing_v2") as run:
        
        # Log all model artifacts (.pth files) - these are just artifacts, not the "model"
        for file in model_dir.glob("*.pth"):
            mlflow.log_artifact(str(file), artifact_path="pytorch_files")
            print(f"Logged artifact: {file.name}")
        
        # Log data file if exists
        if (model_dir / "data").exists():
            mlflow.log_artifact(str(model_dir / "data"), artifact_path="pytorch_files")
        
        # Log metadata
        mlflow.log_params({
            "model_status": "production",
            "model_type": "SAC_RL",
            "framework": "pytorch"
        })
        
        # Log the model directory as an MLflow model using pyfunc
        # Unity Catalog requires a signature, so create one
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec
        
        # Define input/output signature for RL model (state -> action)
        input_schema = Schema([ColSpec("double", "state")])
        output_schema = Schema([ColSpec("double", "action")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RLModelWrapper(),
            artifacts={"model_dir": str(model_dir)},
            signature=signature
        )
        
        print("\n✓ Model artifacts and MLflow model logged successfully!")
        
        # Now register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        
        try:
            result = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
            print(f"✓ Model registered in Unity Catalog as: {REGISTERED_MODEL_NAME}")
            print(f"Version: {result.version}")
        except Exception as e:
            print(f"\n❌ Model registration failed: {e}")
        
        print(f"\nRun ID: {run.info.run_id}")
        print(f"🏃 View run at: https://dbc-9e9c29a5-e035.cloud.databricks.com/ml/experiments/{experiment_id}/runs/{run.info.run_id}")

"""
def transition_to_production(model_name, version):
    #Transition a model version to production stage
    
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    print(f"✓ Model version {version} transitioned to Production")
"""

def list_available_catalogs():
    """List all available catalogs in Unity Catalog"""
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)
        catalogs = list(w.catalogs.list())
        print("\n📁 Available catalogs:")
        for cat in catalogs:
            print(f"  - {cat.name}")
        return [cat.name for cat in catalogs]
    except Exception as e:
        print(f"⚠ Could not list catalogs: {e}")
        print("\nPlease create a catalog in Databricks UI:")
        print("1. Go to Databricks → Catalog")
        print("2. Click 'Create Catalog'")
        print("3. Name it (e.g., 'optivai' or 'hvac')")
        return []

if __name__ == "__main__":
    print("Starting MLflow model versioning...\n")
    
    # Check if token is set
    if not DATABRICKS_TOKEN:
        print("ERROR: Please set DATABRICKS_TOKEN environment variable")
        print("Run: export DATABRICKS_TOKEN='your_api_key'")
        exit(1)
    
    # List available catalogs first
    print("Checking available Unity Catalog catalogs...")
    catalogs = list_available_catalogs()
    
    if catalogs:
        print(f"\n✓ Found {len(catalogs)} catalog(s)")
        print(f"Current model name: {REGISTERED_MODEL_NAME}\n")
    else:
        print("\n❌ No catalogs found. You must create a catalog first!")
        print("\nSteps:")
        print("1. Go to https://dbc-9e9c29a5-e035.cloud.databricks.com")
        print("2. Click 'Catalog' in sidebar")
        print("3. Click 'Create Catalog', name it (e.g., 'optivai')")
        print(f"4. Update REGISTERED_MODEL_NAME in script to: '<catalog_name>.default.hvac_rl_model'")
        exit(1)
    
    # Log production model
    log_production_model()
    
    print("\nTo transition to Production stage:")
    print(f"  python mlflow_version_model.py --transition <version_number>")