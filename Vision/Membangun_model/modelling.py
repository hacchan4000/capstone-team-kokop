import matplotlib.pyplot as plt
import tensorflow as tf
import mlflow

from keras.models import Sequential, load_model
from keras.layers import Input, Conv2D, Reshape, Dropout
from keras.applications import ResNet152V2

from PreprocessData import train, test, val

# ---- MLflow configuration ----
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Iris_Tracking")
mlflow.autolog()

# ---- Feature extractor (ResNet) ----
layer1 = Sequential([
    Input(shape=(250,250,3)),
    ResNet152V2(include_top=False, weights=None, input_shape=(250,250,3))
])
layer1.trainable = False  # freeze backbone

# ---- Additional layers ----
layer2 = Sequential([
    Conv2D(256, 3, padding='same', activation='relu'),
    Conv2D(256, 3, padding='same', activation='relu'),
    Conv2D(128, 3, strides=2, padding='same', activation='relu'),
    Conv2D(128, 3, strides=2, padding='same', activation='relu'),
])

# ---- Output layers ----
from keras.layers import GlobalAveragePooling2D

layer3 = Sequential([
    Dropout(0.05),
    Conv2D(4, 1, activation=None),   # shape: (batch, H, W, 4)
    GlobalAveragePooling2D(),        # → shape: (batch, 4)
])

# ---- Build full model ----
model = Sequential([
    layer1,
    layer2,
    layer3
])

model.summary()

# ---- Compile ----
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=0.0007)
loss = tf.keras.losses.MeanSquaredError()

model.compile(optimizer=optimizer, loss=loss)

# ---- Train ----
hist = model.fit(train, epochs=15, validation_data=val)

# ---- Save & reload ----
model.save("EyeTrackerModel.h5")
model = load_model("EyeTrackerModel.h5")
