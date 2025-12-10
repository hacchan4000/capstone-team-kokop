import mlflow
import mlflow.keras
from keras.models import load_model

model_path = "/Users/mac/Desktop/CAPSTONE/capstone-team-kokop/EyeTrackerModel.h5"

# Load Keras model
model = load_model(model_path)

# Log to MLflow as a Keras model
with mlflow.start_run():
    mlflow.keras.log_model(model, "model")
    print("Model successfully logged!")