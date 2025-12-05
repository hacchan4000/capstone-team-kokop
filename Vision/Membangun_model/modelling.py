import matplotlib.pyplot as plt
import mlflow

from keras.models import Sequential
from keras.layers import Input, Conv2D, Reshape, Dropout
from keras.applications import ResNet152V2

mlflow.set_experiment("Iris_Detection")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

layer1 = Sequential([ #layer input gambar
    Input(shape=(250,250,3)),
    ResNet152V2(include_top=False, input_shape=(250,250,3))
])
layer2 = Sequential([ #layer input gambar
    Conv2D(),
    Conv2D(),
    Conv2D(),
    Conv2D(),
])
layer3 = Sequential([ #layer hidden
    Dropout(0.05),
    Conv2D(4,2,2),
    Reshape((4,))
    
])

mlflow.autolog()

model = Sequential([ #layer output
    layer1,
    layer2,
    layer3,
])

model.summary()

hist = model.fit()