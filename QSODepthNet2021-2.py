"""
###################################################################################################
########################################### QSOCNN2021 ############################################
###################################################################################################
__author__ = "Astrid E. San-Martin-Jimenez"
__email__ = "aesanmar@uc.cl"
__date__ = "january 30th, 2021"
__paper__= "Depthwise Architecture for Multi-Band Automatic Quasars Classification in ATLAS,
	    San-Martin-Jimenez, Pichara, Barrientos, Rojas, Moya-Sierralta & ATLAS, 2021"                                                                                             #
###################################################################################################
###################################################################################################
###################################################################################################
"""


#!/usr/bin/env python
# coding: utf-8



import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import astropy
from astropy.io import fits

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle

import time
from tqdm import tqdm
from datetime import datetime, timedelta, tzinfo
import pickle
import itertools

import tensorflow as tf


from tensorflow.keras.models import Sequential,Model,model_from_json
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputLayer, Dense, Activation, Flatten, Reshape, Conv2D, MaxPool2D, Input
from tensorflow.keras.layers import Convolution2D, SeparableConv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Dropout, Concatenate, Cropping2D
from tensorflow.keras.layers import BatchNormalization, Lambda, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.backend import reshape, spatial_2d_padding, resize_images
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow import keras
import DataAugmentation2 as DA
import tensorboard


plt.rcParams['figure.figsize'] = [15, 15]
plt.rcParams.update({'font.size': 22})



"""
Activate GPU
"""

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)



"""
To stop potential randomness
"""

seed = 41
rng = np.random.RandomState(seed)




"""
class IMAGES -> to load files, images and separate between training and testing set.
	xsize,ysize : image size x,y
	data_dir    : working directory
	_loadfile_ -> load files for training, qso objects, nonqso objects and objects for qso selection, returns each file in pandas dataframe.
		train_file     : all objects file
		qso_file       : qso objects file
		nonqso_file    : nonqso objects file
		selection_file : objects for qso selection
	_images_ -> read and load images from loaded files, returns images array
		settp : setting type, 'trainfile' loads train_file images, 'bal-unbal' loads a specific number of qso and nonqso from qso_file and nonqso_file
		nq    : number of qso to select from qso_file
		nnq   : number of nonqso to select from nonqso_file
	_splittrain_ -> splits loaded images from _images_ into training and testing set.
		trainsplit : splitting proportion
"""

class IMAGES(object):
    def __init__(self,xsize,ysize, data_dir):
        self.xsize = xsize
        self.ysize = ysize
        self.xarcsec = 0.21*self.xsize
        self.yarcsec = 0.21*self.ysize
        self.data_dir = data_dir
        self.im_dir = 'ImagesCutout'
        
    def _loadfile_(self,train_file,qso_file,nonqso_file,selection_file):
    	self.train_file = train_file
    	self.qso_file = qso_file
    	self.nonqso_file = nonqso_file
    	self.selection_file = selection_file
        train = pd.read_csv(os.path.join(self.data_dir,train_file))
        trainq = pd.read_csv(os.path.join(self.data_dir,qso_file))
        trainnq = pd.read_csv(os.path.join(self.data_dir,nonqso_file))
        selection = pd.read_csv(os.path.join(self.data_dir,selection_file))
        self.train = shuffle(train)
        self.trainq = shuffle(trainq)
        self.trainnq = shuffle(trainnq)
        self.selection = shuffle(selection)
        return self.train,self.trainq,self.trainnq,self.selection
            
    def _images_(self,settp,nq,nnq): 
        self.settp = settp
        self.nq = nq
        self.nnq = nnq
        
        if self.settp=='trainfile':
            file_ = shuffle(self.train)
        elif self.settp=='bal-unbal':
            f_ = self.trainnq.sample(n=self.nnq)
            f__ = self.selection
            file_ = shuffle(self.trainq.sample(n=self.nq).append(f_))
            print(len(f_),len(f__))
            
        temp = []
        for img_name in tqdm(f__.Filename):
            image_path = os.path.join(self.data_dir, self.im_dir, img_name)
            hdu = fits.open(image_path)
            
            if hdu[0].header["DATAOK"] == 1:
                imgu = hdu[0].data
            elif hdu[0].header["DATAOK"] == 0:
                imgu = hdu[1].data
                
            if hdu[0].header["FILT"] == 'u1':
                imgg = hdu[2].data
                imgr = hdu[3].data
                imgi = hdu[4].data
                imgz = hdu[5].data
            elif hdu[0].header["FILT"] == 'u':
                imgg = hdu[1].data
                imgr = hdu[2].data
                imgi = hdu[3].data
                imgz = hdu[4].data
                
            img = np.dstack((imgu,imgg,imgr,imgi,imgz))
            img = img.astype('float32')
            temp.append(img)
            self.select_x = np.stack(temp)

        temp = []
        for img_name in tqdm(file_.Filename):
            image_path = os.path.join(self.data_dir, self.im_dir, img_name)
            hdu = fits.open(image_path)
            
            if hdu[0].header["DATAOK"] == 1:
                imgu = hdu[0].data
            elif hdu[0].header["DATAOK"] == 0:
                imgu = hdu[1].data
                
            if hdu[0].header["FILT"] == 'u1':
                imgg = hdu[2].data
                imgr = hdu[3].data
                imgi = hdu[4].data
                imgz = hdu[5].data
            elif hdu[0].header["FILT"] == 'u':
                imgg = hdu[1].data
                imgr = hdu[2].data
                imgi = hdu[3].data
                imgz = hdu[4].data
                
            img = np.dstack((imgu,imgg,imgr,imgi,imgz))
            img = img.astype('float32')
            temp.append(img)
            self.train_x = np.stack(temp)
            
        self.train_y = tf.keras.utils.to_categorical(file_.Label.values)
        return self.train_x,self.train_y,self.select_x
    
    def _splittrain_(self,trainsplit):
        self.trainsplit = trainsplit
        split_size = int(self.train_x.shape[0]*self.trainsplit)
        self.train_x_, self.test_x_ = self.train_x[:split_size], self.train_x[split_size:]
        self.train_y_, self.test_y_ = self.train_y[:split_size], self.train_y[split_size:]
        return self.train_x_, self.test_x_, self.train_y_, self.test_y_
    


"""
class QSOCNN -> Convolutional Neural Network for Automatic QSO Classification.
	train_x   : training images array
	test_x    : testing images array
	train_y   : training label array
	test_y    : testing label array
	data_dir  : working directory
	settype   : set type 'Balanced' or 'Unbalanced' configuration
	augmented : with data augmentation True or False
	_split_ -> load files for training, qso objects, nonqso objects and objects for qso selection, returns each file in pandas dataframe.
		train_file     : all objects file
		qso_file       : qso objects file
		nonqso_file    : nonqso objects file
		selection_file : objects for qso selection
	_augment_ -> generate data augmentation images, returns augmentes images saved to directory.
		nim : number of augmented images to generate
	_model_ -> convolutional neral network model.
		input_shape      : input shape of the model
		hidden_num_units : number of hidden units in fully connected network
		output_num_units : number of output units in fully connected network
		pool_size        : pooling size
		nepochs          : number of training epochs
		batch_size       : batch size
	_modelcompile_ -> compile model.
	_modelfit_ -> fit model.
	_savemodel_ -> saves trained model, weights and training history.
	_modelpredict_ -> make prediction with trained model,returns predicted labels.
	_savemetrics_ -> saves training metrics (loss, validation loss, accuracy and validation accuracy) as numpy array format.
"""

class QSOCNN(object):
    def __init__(self,train_x,test_x,train_y,test_y,data_dir,settype,augmented):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.data_dir = data_dir
        self.settype = settype
        self.augmented = augmented
    
    def _split_(self,tx,ty,trainsplit):
        self.tx = tx
        self.ty = ty
        self.trainsplit = trainsplit
        split_size = int(self.tx.shape[0]*self.trainsplit)
        self.tx_, self.vx_ = self.tx[:split_size], self.tx[split_size:]
        self.ty_, self.vy_ = self.ty[:split_size], self.ty[split_size:]
        return self.tx_, self.vx_, self.ty_, self.vy_
    
    def _augment_(self,nim):
        self.nim = int(nim)
        
        if self.augmented == True:
            augdata_path = os.path.join(self.data_dir, 'AugmentedData')
            imgen_train = DA.ImageDataGenerator(rotation_range=90,
                                                horizontal_flip=True,
                                                vertical_flip=True)
            print('DA')
            imtrain = imgen_train.fit(self.train_x)
            print('fit')
            train_generator = imgen_train.flow(self.train_x,self.train_y,
                                batch_size=self.batch_size,
                                save_to_dir=augdata_path,
                                save_prefix="train_aug",
                                save_format="fits")                                
            print('flow')
            for j in tqdm(range(int(self.nim/5))):
                train_generator.next()
            print('next')
    

    def _model_(self,input_shape, hidden_num_units, output_num_units, pool_size, nepochs, batch_size):
        self.input_shape = input_shape
        self.hidden_num_units = hidden_num_units
        self.output_num_units = output_num_units
        self.pool_size = pool_size
        self.nepochs = nepochs
        self.batch_size = batch_size
        
        
        self.inputs = Input(shape=self.input_shape)

        self.x = SeparableConv2D(25, (5, 5), strides=(1, 1), activation='relu', padding='same')(self.inputs)
        self.x = SeparableConv2D(25, (5, 5), strides=(1, 1), activation='relu', padding='same')(self.x)

        self.x = MaxPooling2D(pool_size=self.pool_size, strides=(2, 2), padding='same')(self.x)
        self.x = Dropout(0.25)(self.x)
        self.x = Convolution2D(25, (5, 5), strides=(1, 1), activation='relu', padding='same')(self.x)
        self.x = Convolution2D(25, (5, 5), strides=(1, 1), activation='relu', padding='same')(self.x)
        self.x = MaxPooling2D(pool_size=self.pool_size, strides=(2, 2), padding='same')(self.x)
        self.x = Convolution2D(25, (5, 5), activation='relu',strides=(1, 1), padding='same')(self.x)
        self.x = Convolution2D(25, (3, 3), activation='relu',strides=(1, 1), padding='same')(self.x)
        self.x = MaxPooling2D(pool_size=self.pool_size, strides=(2, 2), padding='same')(self.x)
        self.x = Dropout(0.25)(self.x)
        self.x = Convolution2D(25, (5, 5), strides=(1, 1), activation='relu', padding='same')(self.x)
        self.x = Convolution2D(25, (5, 5), strides=(1, 1), activation='relu', padding='same')(self.x)
        self.x = MaxPooling2D(pool_size=self.pool_size, strides=(2, 2), padding='same')(self.x)
        self.x = Convolution2D(25, (5, 5), activation='relu',strides=(1, 1), padding='same')(self.x)
        self.x = Convolution2D(25, (3, 3), activation='relu',strides=(1, 1), padding='same')(self.x)
        self.x = MaxPooling2D(pool_size=self.pool_size, strides=(2, 2), padding='same')(self.x)
        self.x = Dropout(0.25)(self.x)
        self.x = Convolution2D(35, (3, 3), activation='relu',strides=(1, 1), padding='same')(self.x)
        self.x = Convolution2D(35, (3, 3), activation='relu',strides=(1, 1), padding='same')(self.x)
        self.x = MaxPooling2D(pool_size=self.pool_size, strides=(2, 2), padding='same')(self.x)
        self.x = Convolution2D(64, (3, 3), activation='relu',strides=(1, 1), padding='same')(self.x)
        self.x = MaxPooling2D(pool_size=self.pool_size, strides=(2, 2), padding='same')(self.x)
        self.x = Dropout(0.25)(self.x)

        self.x = Flatten()(self.x)
        self.x = Dense(units=64, activation='relu')(self.x)

        self.x = Dropout(0.25)(self.x)

        self.x = Dense(units=self.hidden_num_units, activation='relu')(self.x)

        self.x = Dropout(0.25)(self.x)

        self.outputs = Dense(units=self.output_num_units, input_dim=self.hidden_num_units, activation='softmax')(self.x)
        self.model = Model(inputs=self.inputs, outputs=self.outputs, name="QSOCNN2021-2")
        
        

    def _modelcompile_(self):
        self.model.compile(loss='binary_crossentropy', 
                           optimizer='adam', 
                           metrics=['accuracy'])
        # checkpoint
        self.filepath = os.path.join(self.data_dir,"QSOCNN2021-weights-improvement-{epoch:02d}-{val_loss:.2f}-%s-%s.hdf5" % (self.settype,self.augmented))
        self.checkpoint = ModelCheckpoint(self.filepath, 
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min')
        # early stopping
        self.early_stopping = EarlyStopping(monitor='val_loss', 
                                       patience=15, 
                                       verbose=2, 
                                       mode='min',
                                       min_delta=0.001,
                                       restore_best_weights=True)
        # tensorboard
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.data_dir,"QSOCNN2021-logs-%s-%s" % (self.settype,self.augmented)), 
                                                                   write_graph=True,
                                                                   write_images=True, 
                                                                   update_freq='epoch', 
                                                                   profile_batch=2)

        # callbacks
        self.callbacks_list = [self.checkpoint, self.early_stopping, self.tensorboard_callback]

    def _modelfit_(self):
        if self.augmented == False:
            self.history = self.model.fit(self.train_x, self.train_y, 
                                    epochs=self.nepochs, 
                                    batch_size=self.batch_size, 
                                    shuffle=True,
                                    validation_split=0.2,
                                    callbacks=self.callbacks_list)
        elif self.augmented == True:
            trainx,valx,trainy,valy = self._split_(self.train_x, self.train_y,0.8)
            print('Data Augmentation: ',self.augmented,'--- Shapes: ',trainx.shape,valx.shape,trainy.shape,valy.shape)
            imgen_train = ImageDataGenerator(rotation_range=90, horizontal_flip=True, vertical_flip=True)
            imgen_train.fit(trainx)
            gentrain = imgen_train.flow(trainx,trainy, batch_size=self.batch_size) 
        
            self.history = self.model.fit(gentrain,
                                    steps_per_epoch=trainx.shape[0]//self.batch_size,
                                    epochs=self.nepochs, 
                                    batch_size=self.batch_size, 
                                    shuffle=True,
                                    validation_data=(valx, valy),
                                    callbacks=self.callbacks_list)


    def _savemodel_(self):
        self.model_json = self.model.to_json()
        with open(os.path.join(self.data_dir,"QSOCNN2021-%s-%s.json" % (self.settype,self.augmented)), "w") as json_file:
            json_file.write(self.model_json)

        self.model.save_weights(os.path.join(self.data_dir,"weights_QSOCNN2021-%s-%s.h5" % (self.settype,self.augmented)))

        with open(os.path.join(data_dir,"trainHistoryDict_QSOCNN2021-%s-%s.h5" % (self.settype,self.augmented)), 'wb') as file_pi:
                pickle.dump(self.history.history, file_pi)

    def _modelpredict_(self):
        self.pred = self.model.predict(self.test_x,verbose=1)
        return self.pred
    
    def _savemetrics_(self):
        self.loss = np.array(self.history.history['loss'])#.reshape(-1, 1)
        self.val_loss = np.array(self.history.history['val_loss'])#.reshape(-1, 1)
        self.acc = np.array(self.history.history['accuracy'])#.reshape(-1, 1)
        self.val_acc = np.array(self.history.history['val_accuracy'])#.reshape(-1, 1)
        return self.loss,self.val_loss,self.acc,self.val_acc



"""
Function to returns time in hours, minutes and seconds format.
"""

def _time_(time_):
    seconds = time_.total_seconds()
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    print(hours,':',minutes,':',seconds)
    



if __name__ == "__main__":

	"""
	Define variables
	"""

	xsize = 57 # pixel
	ysize = 57 # pixel
	input_shape = (xsize, ysize, 5)

	hidden_num_units = 250
	output_num_units = 2

	pool_size = (2,2)
	nepochs = 100
	batch_size = 10

	settype='Balanced-Arch1' # 'Balanced-Arch1' - 'Unbalanced-Arch1'
	augmented = False

	train_file = 'Train.csv'
	qso_file = 'QSO.csv'
	nonqso_file = 'NonQSO.csv'
	selection_file = 'ToSelect.csv'



	"""
	Load images
	"""

	images = IMAGES(xsize,ysize, data_dir)
	trainfile,trainqfile,trainnqfile,selectionfile = images._loadfile_(train_file,qso_file,nonqso_file,selection_file)

	train_x,train_y,select_x = images._images_('bal-unbal',15000,15000) # trainfile/bal-unbal,num qso,num nonqso
	train_x_, test_x_, train_y_, test_y_ = images._splittrain_(0.8)
	print(select_x.shape)
	print(train_x_.shape,test_x_.shape)

	unique1, counts1 = np.unique(np.argmax(train_y_,axis=-1), return_counts=True)
	unique2, counts2 = np.unique(np.argmax(test_y_,axis=-1), return_counts=True)
	print(dict(zip(unique1, counts1)))
	print(dict(zip(unique2, counts2)))



	"""
	Train model
	"""

	startTime = datetime.now()

	qsocnn = QSOCNN(train_x_,test_x_,train_y_,test_y_,data_dir,settype,augmented)
	model = qsocnn._model_(input_shape,hidden_num_units,output_num_units,pool_size,nepochs,batch_size)
	qsocnn._modelcompile_()
	qsocnn._modelfit_()
	qsocnn._savemodel_()

	print("\n" * 5, "Time taken:", datetime.now() - startTime, "\n" * 5)

	loss,val_loss,acc,val_acc = qsocnn._savemetrics_()
	metrics_ = [loss,val_loss,acc,val_acc]
	pred_ = qsocnn._modelpredict_()
	matrix_ = confusion_matrix(np.argmax(test_y_,axis=-1), np.argmax(pred_,axis=-1))
	print(classification_report(np.argmax(test_y_,axis=-1), np.argmax(pred_,axis=-1)))

	#!tensorboard --logdir=/home/QSOCNN2021-logs
