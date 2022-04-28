import os
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import string

#set the directory from which we'll be getting our data
BASE_DIR = '/content/drive/My Drive/Datasets/LEGO DATASET/'
super_classes = ['harry-potter/','jurassic-world/','marvel/','star-wars/']

if not os.path.isdir(BASE_DIR + 'train/'):
  os.makedirs(BASE_DIR + 'train/')
  os.makedirs(BASE_DIR + 'val/')

#move files 

folder_source = BASE_DIR + 'train_dataset/'
destination_folder = BASE_DIR + 'train/'
for superclass in super_classes:
  for folder_name in (os.listdir(folder_source + superclass)):
    folder = folder_source + superclass + folder_name
    destination_source = destination_folder + str(np.random.randint(5000))
    shutil.move(folder,destination_source)

#instantiate the image data generator for the trainig and the test datasets
train_gen = keras.preprocessing.image.ImageDataGenerator(rescale =1./255,rotation_range=20,
    horizontal_flip = True,
    width_shift_range=0.2
,validation_split =0.25) 
test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.0)

# here we get our train_gen object and apply it to the data flowing from our directory.

train_batches = train_gen.flow_from_directory(
    BASE_DIR + 'train/',
    subset = 'training',
    target_size=(256,256),
    batch_size = 32,
    shuffle = True,
    class_mode = 'sparse',
    color_mode ='rgb',
)

val_batches = train_gen.flow_from_directory(
    BASE_DIR + 'train/',
    subset = 'validation',
    batch_size = 32,
    target_size=(256,256),
    class_mode = 'sparse',
    shuffle= True,
    color_mode ='rgb',

)

#creating the model itself. The model is a multilayer CNN with a last layer using the softmax activation.

model = keras.models.Sequential([
                                 keras.layers.Conv2D(128,3,1,padding='valid', activation='relu'),
                                 keras.layers.MaxPool2D((2,2)),
                                 keras.layers.Conv2D(128,3,padding='valid'),
                                 keras.layers.MaxPool2D((2,2)),
                                keras.layers.Conv2D(128,3,padding='valid'),
                                 keras.layers.MaxPool2D((2,2)),
                                keras.layers.Conv2D(128,3,padding='valid'),
                                 keras.layers.MaxPool2D((2,2)),
                                 keras.layers.Flatten(),
                                 keras.layers.Dense(128,activation='relu'),
                                 keras.layers.Dense(37, activation='softmax')
])

#setting the optmizer, loss, metrics. Compiling the model with the parameters.
# defining a callback based on the validation accuracy.
optmizer = keras.optimizers.Adam(lr=0.001)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics =['accuracy']

model.compile(optimizer=optmizer, loss = loss, metrics=metrics)

my_callbacks = [ keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5)]


#training the model
model.fit(train_batches,validation_data=val_batches ,epochs=250,callbacks=my_callbacks)

#saving the model
model.save('Classifying lego all data')