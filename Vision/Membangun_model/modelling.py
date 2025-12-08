import matplotlib.pyplot as plt
import tensorflow as tf

import mlflow

from keras.models import load_model
from PreprocessData import train,test,val


from keras.models import Sequential
from keras.layers import Input, Conv2D, Reshape, Dropout
from keras.applications import ResNet152V2


mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Iris_Tracking")
mlflow.autolog()

layer1 = Sequential([ #layer input gambar
    Input(shape=(250,250,3)),
    ResNet152V2(include_top=False, weights=None, input_shape=(250,250,3))
])
layer1.trainable = False  # freeze backbone
layer2 = Sequential([ #layer hidden model
    Conv2D(256, 3, padding='same', activation='relu'),
    Conv2D(256, 3, padding='same', activation='relu'),
    Conv2D(128, 3, 2, padding='same', activation='relu'),
    Conv2D(128, 2, 2, activation='relu'),
])
layer3 = Sequential([ #layer output
    Dropout(0.05),
    Conv2D(4, 1, activation=None),
    Reshape((4,))
])

mlflow.autolog()

model = Sequential([ #gabungin model
    layer1,
    layer2,
    layer3,
])

model.summary()

#setup loss function dan optimizer

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0007)
loss = tf.keras.losses.MeanSquaredError()

model.compile(optimizer=optimizer,loss=loss)

#TRAIN
hist = model.fit(train, epochs=15, validation_data=val)

""" Review performa model"""

""" Save Model"""
model.save('EyeTrackerModel.h5')

model = load_model('EyeTrackerModel.h5')
