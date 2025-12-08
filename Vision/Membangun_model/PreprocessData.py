import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import json

def loadGambarku(img): # untuk baca file gambar
    img_byte = tf.io.read_file(img) # representasi gambar dalam byte
    gambar = tf.io.decode_jpeg(img_byte) # kembaliin ke img ori
    return gambar

""" Load & Preprocess Gambar-gambar yang udah di augmentasi"""

#load FILE PATH dr masing-masing subset
train_images = tf.data.Dataset.list_files("aug_Data/train/images/* .jpg", shuffle=False) #list_files tugasnya liat ke folder trs tunjukin semwa files format .jpg
test_images = tf.data.Dataset.list_files("aug_Data/test/images/* .jpg", shuffle=False)
val_images = tf.data.Dataset.list_files("aug_Data/val/images/* .jpg", shuffle=False)

#aplikasikan fungsi loadGambarku ke setiap instance dalam subset pake .map
train_images = train_images.map(loadGambarku) # ini ngembaliin gambarnya itu sendiri dalam format array of ints
test_images = test_images.map(loadGambarku)
val_images = val_images.map(loadGambarku)

#resize ukuran gambar pake lambda function spy consise
train_images = train_images(lambda img: tf.image.resize(img,(250,250))) # ngubah dimensi supaya lebih gampang dimasukin model
test_images = test_images(lambda img: tf.image.resize(img,(250,250)))
val_images = val_images(lambda img: tf.image.resize(img,(250,250)))

#normalisasi gambar
train_images = train_images(lambda img: img/255) #ubahh tiap nilai intensitas piksel di gambar jd rentang 0~1
test_images = test_images(lambda img: img/255)
val_images = val_images(lambda img: img/255)

""" Prepare Labels """