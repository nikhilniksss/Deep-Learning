# importing important libraries

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDatagenerator


# defining image size

IMG_SIZE = (128,128)
BATCH_SIZE = 16

# create data augumentation
# rescaling the image

train_datagen = ImageDatagenerator(rescale = 1./255)
val_datagen = ImageDatagenerator(resale = 1./255)

# load a data by going into data directory

# train data
train_data = train_datagen.flow_from_directory(
    '/Users/nick_mac/Desktop/FSDS/cnn_image_classification/data/train',
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical'
)

# validation data
val_data = train_datagen.flow_from_directory(
    '/Users/nick_mac/Desktop/FSDS/cnn_image_classification/data/val',
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical'
)

# defining the number of class

num_class = len(train_data.class_indices)

############# Let's design the CNN architecture ############

models.Sequential([

    layers.Conv2d(32,(3,3),activation = 'relu',input_shape = (128,128,3)), #------layer 1
    layers.MaxPooling2d(2,2),

    layers.Conv2d(64,(3,3),activation = 'relu'), #------layer 2
    layers.MaxPooling2d(2,2),

    layers.Conv2d(128,(3,3),activation = 'relu'), #------layer 3
    layers.MaxPooling2d(2,2),

    layers.Conv2d(32,(3,3),activation = 'relu'), #------layer 4
    layers.MaxPooling2d(2,2),

    layers.Conv2d(254,(3,3),activation = 'relu'), #------layer 5
    layers.MaxPooling2d(2,2),

    layers.Conv2d(128,(3,3),activation = 'relu'), #------layer 6
    layers.MaxPooling2d(2,2),

    layers.Flatten(),
    layers.Dense(128,activation = 'relu'),
    layers.Dropout(.3),
    layers.Dense(num_class,activation = 'softmax')   # output layer
]
)

models.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

# model fit code

models.fit(
    train_data,
    epochs = 10,
    validation_data = val_data
)

# save the model

models.save("CNN_classifier.h5")

# print model summary

print(models.summary())