import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def register_model(
    model,
    model_name:str,
    experiment_name:str = "Default",
    run_name:str = "run_1",
    stage: str = "production"):
    
    """
    Register the model to mlflow register 
    """
    
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        ## log the model
        mlflow.sklearn.log_model(
            sk_model = model,
            artifact_path = "model",
            register_model_name = model_name
        )
        
        
    if stage:
        client = MlflowClient()
        latest_version = client.get_latest_versions(name=model_name, stages=["None"])[0].version
        client.transition_model_version_stage(
            name = model_name,
            version = latest_version,
            stage = stage
           
        )
        
        print(f"model registered: {model_name}, version:{latest_version}, stage:{stage}")
        
        return latest_version
    
    
    