#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:09:09 2020

@author: kumet
"""
#data preproccessing was done manually via the file structure

#PART 1  BUILDING THE CNN

# Importing the Keras libraries and packages 
# each package corresponds to one step in building the CNN
from keras.models import Sequential #initialize NN (graph or sequential)
from keras.layers import Convolution2D #images are 2d, whilst videos are 3d
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense #add fully connected layers in a classic ANN

#Initializing the CNN
classifier = Sequential()

#Step 1 - Convolution (making the convolutional layer of feature maps)
#tensor flow backend
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation = 'relu'))

#Step2 - Max Pooling (taking the max of a 2x2 table to make a new, reduced feature map. For all feature maps)
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step2.5 add a new layer (we do not need to specify the input_shape since it's already decided.)
classifier.add(Convolution2D(32,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step3 - Flattening 
classifier.add(Flatten())

#Step4 - Full Connection (making the fully connected layers, with a single binary output layer)
#hidden layer (fully connected layer)
classifier.add(Dense(output_dim = 128, activation = 'relu'))
#output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) #binary outcome so use sigmoid

#Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#PART 2  FITTING THE CNN TO THE IMAGES

#image preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64, 64),#the size of the images
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,#number of images we have in the training set
        epochs=25, #don't make this too big if you don't want to wait too long
        validation_data=test_set,
        validation_steps=2000) #number of images in test set