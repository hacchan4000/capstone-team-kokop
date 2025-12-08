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

#load FILE PATH GAMBAR dr masing-masing subset
train_images = tf.data.Dataset.list_files("aug_Data/train/images/*.jpg", shuffle=False) #list_files tugasnya liat ke folder trs tunjukin semwa files format .jpg
test_images = tf.data.Dataset.list_files("aug_Data/test/images/*.jpg", shuffle=False)
val_images = tf.data.Dataset.list_files("aug_Data/val/images/*.jpg", shuffle=False)

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

def LoadLabels(label_path):
    with open(label_path.numpy(), 'r', encoding='utf-8') as f: #ngebuka trs baca file label yang berformat json
        label = json.load(f)
    return [label['keypoints']]

#load FILE PATH LABEL dr masing-masing subset
train_labels = tf.data.Dataset.list_files("aug_Data/train/labels/*.json", shuffle=False)
test_labels = tf.data.Dataset.list_files("aug_Data/test/labels/*.json", shuffle=False)
val_labels = tf.data.Dataset.list_files("aug_Data/val/labels/*.json", shuffle=False)

#load masing-masing labels
train_labels = train_labels.map(lambda img: tf.py_function(LoadLabels, [img], [tf.float16])) #py_function itu untuk klo pingin pake semua base functionality dlm python utk func spesifik
test_labels = test_labels.map(lambda img: tf.py_function(LoadLabels, [img], [tf.float16])) #trs bakal ngembaliin koordinat dr mata  array([X_left_eye,Y_left_eye,X_right_eye,Y_right_eye], dtype=float)
val_labels = val_labels.map(lambda img: tf.py_function(LoadLabels, [img], [tf.float16])) 


""" Gabungin Label dan Gambar """
#kaya mslh supervised learning pd umumnya ada fitur(gambar) dan target(label)

#concatenate gambar dan label
train = tf.data.Dataset.zip((train_images, train_labels))
test = tf.data.Dataset.zip((test_images, test_labels))
val = tf.data.Dataset.zip((val_images, val_labels))

train = train.shuffle(3000)
test = test.shuffle(2000)
val = val.shuffle(2000)

train = train.batch(16)
test = test.batch(16)
val = val.batch(16)

train = train.prefetch(4)
test = test.prefetch(4)
val = val.prefetch(4)

#liat sample

data_sample = train.as_numpy_iterator()
hasil = data_sample.next()

cols = 4
fig, ax = plt.subplots(ncols=cols, figsize=(20,20))
for i in range(cols):
    img_sample = hasil[0][i]
    koordinat_sample = hasil[1][0][i]
    
    #ngegambar lingkaran untuk masing-masing mata
    cv2.circle(img_sample, tuple(np.multiply(koordinat_sample[:2], [250,250]).astype(int)), 2, (255,0,0), -1)#mata kiri
    cv2.circle(img_sample, tuple(np.multiply(koordinat_sample[2:], [250,250]).astype(int)), 2, (0,255,0), -1)#mata kanan
    
    ax[i].imshow(img_sample)

