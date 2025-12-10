from keras.models import load_model

model = load_model("/Users/mac/Desktop/CAPSTONE/capstone-team-kokop/EyeTrackerModel.h5")
model.save("served_model/1")
