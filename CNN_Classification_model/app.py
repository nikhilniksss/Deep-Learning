import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from LENET import build_lenet
from alexnet import build_alexnet
from vgg16 import build_vgg

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_text = x_train/255.0,x_test/255.0

if not os.path.exists("save_models"):
    os.mkdir('save_models')

def train_and_evaluate(model,model_name):

    model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

    checkpointing = tf.keras.callbacks.ModelCheckpoint(
        filepath = f"save_models/{model_name}.h5",
        monitor = 'val_accuracy',
        save_best_only = True,
        verbose = 1
    )

    model.fit(x_train,y_train,epochs=50,batch_size=64,validation_split=.2,verbose=1, callbacks = [checkpointing])
    
    model.evaluate(x_test,y_test,verbose=1)

# calling 3 timeas as we have created 3 model architecture

# calling lenet model
train_and_evaluate(build_lenet(),'lenet')

# calling alexnet model
train_and_evaluate(build_alexnet(),'alexnet')

# calling vgg model
train_and_evaluate(build_vgg(),'vgg')
