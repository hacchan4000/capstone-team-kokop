import mlflow
import mlflow.keras
from keras.models import load_model
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Iris_Tracking")

model = load_model("/Users/mac/Desktop/CAPSTONE/capstone-team-kokop/EyeTrackerModel.h5")

with mlflow.start_run(run_name="iris_tracker_manual"):
    mlflow.keras.log_model(
        model,
        artifact_path="model",
        input_example=np.zeros((1,250,250,3)),
        registered_model_name=None
    )