import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

def loadGambarku(img_path):
    img_byte = tf.io.read_file(img_path)
    gambar = tf.io.decode_jpeg(img_byte)
    return gambar

#data train
train_images = tf.data.Dataset.list_files("Data/train/images/*", shuffle=False) # ini bakal ngembaliin path gambar dr images training
train_images = train_images.map(loadGambarku) # ubah gambar jd list matrix intensitas piksel (angka)
train_images = train_images.map(lambda x: tf.image.resize(x, (250,250))) # ubah ukuran gambar jd 250 x 250
train_images = train_images.map(lambda x: x/255) #normalisasi gambar jd rentang 0 ~ 1
#data test
test_images = tf.data.Dataset.list_files("Data/test/images/*", shuffle=False) # ini bakal ngembaliin path gambar dr images testing
test_images = test_images.map(loadGambarku) # ubah gambar jd list matrix intensitas piksel (angka)
test_images = test_images.map(lambda x: tf.image.resize(x, (250,250))) # ubah ukuran gambar jd 250 x 250
test_images = test_images.map(lambda x: x/255) #normalisasi gambar jd rentang 0 ~ 1
#data val
val_images = tf.data.Dataset.list_files("Data/val/images/*", shuffle=False) # ini bakal ngembaliin path gambar dr images testing
val_images = test_images.map(loadGambarku) # ubah gambar jd list matrix intensitas piksel (angka)
val_images = test_images.map(lambda x: tf.image.resize(x, (250,250))) # ubah ukuran gambar jd 250 x 250
val_images = test_images.map(lambda x: x/255) #normalisasi gambar jd rentang 0 ~ 1

