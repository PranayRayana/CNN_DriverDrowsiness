#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Driver Drowsiness is one of the main reasons for the road accidents. The dataset is downloaded from kaggle ("Driver Drowsiness Dataset (DDD)"). This dataset has been used for training and testing the CNN architecture. Transfer Learning (ResNet50) concept is used in this project.

# ### 1. Importing Dataset

# In[ ]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy = True')


# In[ ]:


import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content'


# In[ ]:


#API Token has to be imported
get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')


# In[ ]:


# Importing a dataset from kaggle
get_ipython().system('kaggle datasets download -d ismailnasri20/driver-drowsiness-dataset-ddd')


# In[ ]:


# The dataset downloaded is zipped. We need to unzip to proceed further
import zipfile
zip = zipfile.ZipFile('/content/driver-drowsiness-dataset-ddd.zip')
zip.extractall('/content')
zip.close()


# In[ ]:


# Installing split-folders library
pip install split-folders


# In[ ]:


import splitfolders
data_dir = '/content/Driver Drowsiness Dataset (DDD)'
output_dir = '/content/splitted_Data'
#In splitfolders.ratio(), ratio parameter is given in format (a,b,c). a = training set ratio, b = validation set ratio, c = test set ratio
splitfolders.ratio(data_dir, output=output_dir, seed=101, ratio=(.8, 0.15, 0.05))


# ### 2. Relevant Libraries

# In[ ]:


# Importing Relevant Libraries
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf


# ### 3. Data import

# In[ ]:


train_path = '/content/splitted_Data/train'
test_path = '/content/splitted_Data/test'
val_path = '/content/splitted_Data/val'


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

val_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)


# In[ ]:


train_data = train_datagen.flow_from_directory(train_path,
                                                 target_size = IMAGE_SIZE,
                                                 batch_size = 128,
                                                 class_mode = 'binary')


# In[ ]:


val_data = val_datagen.flow_from_directory(val_path,
                                                 target_size = IMAGE_SIZE,
                                                 batch_size = 128,
                                                 class_mode = 'binary')


# In[ ]:


test_data = test_datagen.flow_from_directory(test_path,
                                                 target_size = IMAGE_SIZE,
                                                 batch_size = 128,
                                                 class_mode = 'binary')


# ### 4. Model Creation

# In[ ]:


# Since we are using ResNet50, it's default input layer size is (224, 224). We are keeping its same.
IMAGE_SIZE = [224,224]


# In[ ]:


#Creating ResNet50 model. Parameters are explained below.
# IMAGE_SIZE = input image size, [3] indicates it's RGB channel
#weights = 'imagenet'. It means we are taking weights as per standard ResNet50.
#include_top = False. Input and output layers are given by us.
resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[ ]:


resnet.summary()


# In[ ]:


# don't train existing weights. we are just using the trained weights
for layer in resnet.layers:
    layer.trainable = False


# In[ ]:


x = Flatten()(resnet.output)


# In[ ]:


# The number of nodes in the output layer is 1. (Binary image classification)
prediction = Dense(1, activation='sigmoid')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)


# In[ ]:


model.summary()


# In[ ]:


# Chaecking the number of classes
folders = glob('/content/splitted_Data/train/*')
folders


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


#Checking random images
fig, axes = plt.subplots(8, 4, figsize=(15, 30))
class_indices = train_data.class_indices

for i in range(8):
    images, labels = next(train_data)
    for j in range(4):

        ax = axes[i, j]
        ax.imshow(images[j])
        ax.axis('off')
        label = int(labels[j])
        label_name = list(class_indices.keys())[list(class_indices.values()).index(label)]
        ax.set_title(f'{label_name} ({label})')

plt.tight_layout()
plt.show()


# In[ ]:


r = model.fit(train_data, validation_data= val_data, epochs=5)


# In[ ]:


test_image = cv.imread('/content/splitted_Data/test/NonDrowsy/a0075.png')
plt.imshow(test_image)


# In[ ]:


test_image.shape


# In[ ]:


test_image = cv.resize(test_image, (224,224))
test_input = test_image.reshape(1,224,224,3)


# In[ ]:


pred = model.predict(test_input)
if (pred < 0.5):
   print("Drowsy")
else:
   print('NonDrowsy')


# In[ ]:


model.evaluate(test_data)


# In[ ]:




