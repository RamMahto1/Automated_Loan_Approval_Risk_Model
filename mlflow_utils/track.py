import mlflow
import mlflow.sklearn



def track_experiment(
    model,
    params:dict,
    metrics:dict,
    experiment_name:str = "Default",
    run_name:str = "run_1"):
    
    
    """mlflow tracked the mlflow experiment 
    """
    
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        
        ## loag the params and metrics
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
            
            
            ## log the model
            mlflow.sklearn.log_model(sk_model = model, artifact_path="model")
            
            
            
            run = mlflow.active_run()
            print(f"experiment tracked: {experiment_name}, run name:{run_name}")
            
            
            return run.info.run_id