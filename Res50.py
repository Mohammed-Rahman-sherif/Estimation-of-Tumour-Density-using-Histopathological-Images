import os
import cv2
import json
import glob
import scipy
import random
import pickle
import os.path
import itertools
import numpy as np
from os import listdir
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.models import Sequential
from skimage.transform import resize
from os.path import isdir, join, isfile
from keras.layers import Dense, Flatten, Conv2D
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array, apply_affine_transform

class DirectoryDataGenerator(keras.utils.Sequence):
    def __init__(self, base_directories, labels, augmentor=True, preprocessors=None, batch_size=8, target_sizes=(224,224), channel_last=True, nb_channels=3, shuffle=False, verbose=True, isRegression=True):
        #this may be adapted so we don't care about these kind of errors or handle them differently, but currently we expect for each target size to have a corresponding pre-processor. If there is no preprocessing just pass a function that returns the values the same for now
#        if verbose:
#            if not preprocessors:
#                print('Warning: no preprocessor was supplied.')
#            elif not len(preprocessors) is len(target_sizes):
#                #raise Exception('Number of target sizes does not match number of preprocessors.')
#                print('Warning: number of target sizes does not match number of preprocessors. Make sure this is as intended.') #we can't print the length here because len((1,2,3)) == 3 and len([(1,2,3)]) == 1, which is what we want, but we must only accept arrays then! @TODO: Fix this
#            if not augmentor:
#                print('Warning: no augmentor was supplied. Images will be unchanged.')
        self.base_directories = base_directories
        self.augmentor = augmentor
        self.preprocessors = preprocessors #should be a function that can be directly called with an image 
        self.batch_size = batch_size
        self.target_sizes = target_sizes
        self.channel_last = channel_last
        self.nb_channels = nb_channels
        self.shuffle = shuffle
        #self.reg_label=reg_label 
        self.is_regression=isRegression
        
        self.class_names = []
        files = []
        print('base', base_directories)

        files = sorted([
        os.path.join(base_directories, fname)
        for fname in os.listdir(base_directories)
        if fname.endswith(".png")])

        self.nb_classes = 1
        self.nb_files = len(files)
        self.files = files
        self.labels = labels
        #print(labels)
        self.on_epoch_end() #initialise indexes
        print("total number of data", len(files), len(labels))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.nb_files / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, indexes)
        return [X,], y
        
        #return [X,self.rois], {"region_output": y, "whole_image_output": y, "combine_output": y}

    def get_indexes(self):
        return self.indexes

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nb_files)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def scipy_image_augmentation(self, img, theta=15, tx=0., ty=0., zoom=0.15):
        
        if zoom != 1:
            #zx, zy = np.random.uniform(1 - zoom, 1 + zoom, 2)
            zx = zy = np.random.uniform(1 - zoom, 1 + zoom)
        else:
            zx, zy = 1, 1
            
        if theta != 0:      
            theta = np.random.uniform(-theta, theta)    
        
        #m_inv = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), theta, scale)
        
        
        ''' ADD translation to Rotation Matrix '''
        if tx != 0. or ty != 0.:
            h, w = img.shape[0], img.shape[1]
            ty = np.random.uniform(-ty, ty) * h
            tx = np.random.uniform(-tx, tx) * w

           
        return apply_affine_transform(img, theta=theta, tx=tx, ty=ty, zx=zx, zy=zy)

    def __data_generation(self, list_IDs_temp, indexes):
  
        # Initialization
        X = np.empty((self.batch_size, self.target_sizes[0],self.target_sizes[1], self.nb_channels), dtype = np.uint8)#K.floatx())

        y = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            
            img = load_img(ID) #this won't work if we have channel first
            img = img_to_array(img)
            
            if self.augmentor:
                img = self.scipy_image_augmentation(img, theta=15, tx=0., ty=0., zoom=0.15)
                img = resize(img, (224,224)) # resize image on-the-fly
            else:
                img = resize(img, self.target_sizes) # resize image on-the-fly              

            
            if self.preprocessors:                   
                img = self.preprocessors(img) #used for the pre-processing step of the model, e.g. VGG16 or Resnets  
            
            X[i,] = img
            y[i] = self.labels[indexes[i]]
            
        return X, y

def build_model():
    resnet_model = Sequential()
    pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(224,224,3),
                   pooling='avg',classes=1,
                   weights='imagenet')

    pretrained_model.summary()

    resnet_model.add(pretrained_model)
    resnet_model.add(keras.layers.Dropout(0.2))
    resnet_model.add(Dense(1, activation='sigmoid'))
    return resnet_model

resnet_model = build_model()

#######################
dictlist = []
def read_json(file):
  data = json.load(file)
  for key, value in data.items():
    val = list(value.values())
    tumour = val[0]
    total = val[1]
    result = tumour / total
    dictlist.append(result)
    np.save('Density', dictlist)

with open("test.json") as f:
    read_json(f)

labels = np.load('Density.npy')
#labels = labels.tolist() #y as array dtype = np.float32

print('labels length: ', len(labels))
tr_label = labels[:1000]
vl_label = labels[1000:]    
print('train labels length: ', len(tr_label))
print('val labels length: ', len(vl_label))    

#######################

base_directories = "C:\\Users\\smart\\Documents\\Part Time - EHU\\Workspace\\NuCLS\\NuCLS data\\PsAreTruth_E-20211020T171415Z-001-F3\\PsAreTruth_E\\rgbs\\new\\train"
val_directories = "C:\\Users\\smart\\Documents\\Part Time - EHU\\Workspace\\NuCLS\\NuCLS data\\PsAreTruth_E-20211020T171415Z-001-F3\\PsAreTruth_E\\rgbs\\new\\validation"

train_gen = DirectoryDataGenerator(base_directories, labels = tr_label)
val_gen = DirectoryDataGenerator(val_directories, labels = vl_label, augmentor=False)
resnet_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='mean_squared_error', metrics=["MeanAbsoluteError", "MeanAbsolutePercentageError"])#standard regression metrics 

#print('#################', train_gen[1])# {issue here.. have a look!!!}

callbacks = [
    keras.callbacks.CSVLogger('C:\\Users\\smart\\Documents\\Part Time - EHU\\Workspace\\NuCLS\\LR_0_00001\\Log_Files\\Res50\\log0.csv', separator = ',', append = True),
    keras.callbacks.ModelCheckpoint("C:\\Users\\smart\\Documents\\Part Time - EHU\\Workspace\\NuCLS\\LR_0_00001\\Models\\Res50\\model0.{epoch:02d}-{val_loss:.2f}.h5", save_best_only=True)
]

history = resnet_model.fit(train_gen, validation_data=val_gen, epochs=25)

#DirectoryDataGenerator(base_directories).__getitem__(0)

fig1 = plt.gcf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Metrics Visualization')
plt.ylabel('Loss')
plt.xlabel('Val_Loss')
plt.legend(['Loss', 'Val_Loss'])
plt.show()

fig2 = plt.gcf()
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
#plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Metrics Visualization')
plt.ylabel('mean_absolute_error')
plt.xlabel('val_mean_absolute_error')
plt.legend(['MAE', 'Val_MAE'])
plt.show()