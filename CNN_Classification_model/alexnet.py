import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def build_alexnet():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(96,(3,3) ,activation='relu',input_shape = (32,32,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(256,(3,3) , activation='relu',padding = 'same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(384,(3,3) , activation='relu',padding = 'same'),
        tf.keras.layers.Conv2D(384,(3,3) , activation='relu',padding = 'same'),
        tf.keras.layers.Conv2D(256,(3,3) , activation='relu',padding = 'same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096,activation = 'relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(4096,activation = 'relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(10,activation = 'relu'),
        ])
    return model

