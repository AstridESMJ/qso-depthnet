"""
__author__ = "Astrid E. San-Martin-Jimenez"
__copyright__ = " "
__credits__ = ["Astrid E. San-Martin-Jimenez"]
__license__ = " "
__version__ = " "
__maintainer__ = "Astrid E. San-Martin-Jimenez"
__email__ = "aesanmar@uc.cl"
__status__ = " "
__date__ = "february 21th, 2020"
"""




import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import astropy
from astropy.io import fits

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle

import time
from tqdm import tqdm
from datetime import datetime
import multiprocessing

import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras_tqdm import TQDMNotebookCallback
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten, 
                                 MaxPooling2D, Reshape, InputLayer, SeparableConv2D,  
                                 LocallyConnected2D
from keras.utils import np_utils, plot_model
from keras import backend as K
from keras import applications

import DataAugmentation2 as DA


# GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

device_name = '/device:XLA_GPU:0'
if device_name == "XLA_GPU":
    device_name = "/device:XLA_GPU:0"
else:
    device_name = "/device:CPU:0"
    
tf.debugging.set_log_device_placement(True)

# To stop potential randomness
seed = 41
rng = np.random.RandomState(seed)

############################################ 


""""""""""""""""""""""""""""""""""""
# Directories
""""""""""""""""""""""""""""""""""""
root_dir = os.path.abspath('/home/')
data_dir = os.path.join(root_dir, 'cnn/')

# check for existence
print(os.path.exists(root_dir))
print(os.path.exists(data_dir))

""""""""""""""""""""""""""""""""""""
# Load files
""""""""""""""""""""""""""""""""""""
@loadfile
def _loadfile_(data_dir,i,datasettype):
    train = pd.read_csv(os.path.join(data_dir, 'Train', 'Train-ITER-%i_PROG-%s.csv' 
                                                        % (i,datasettype)))
    test = pd.read_csv(os.path.join(data_dir, 'Test-ITER-%i_PROG-%s.csv' 
                                                        % (i,datasettype)))
    train = shuffle(train)
    test = shuffle(test)
    return train, test
    
    
""""""""""""""""""""""""""""""""""""
# Split train/validation
""""""""""""""""""""""""""""""""""""
@splitvaltrain
def _splitvaltrain_(train, train_x, train_y):
    split_size = int(train_x.shape[0]*0.7)

    train_x, val_x = train_x[:split_size], train_x[split_size:]
    train_y, val_y = train_y[:split_size], train_y[split_size:]

    train.Label.ix[split_size:]
    return train_x, val_x, train_y, val_y

    
""""""""""""""""""""""""""""""""""""
# Data Augmentation
""""""""""""""""""""""""""""""""""""
@dataaug
def _dataaug_(train_x):
    imgen_train = DA.ImageDataGenerator(rotation_range=90, horizontal_flip=True,  
                                         vertical_flip=True)
    imgen_train.fit(train_x)
    return imgen_train    
 
    
""""""""""""""""""""""""""""""""""""
# Model 1
""""""""""""""""""""""""""""""""""""
@model
def _model_(input_shape, hidden_num_units, output_num_units, pool_size, nepochs, batch_size):

    model = Sequential()

    model.add(InputLayer(input_shape=input_shape))
    model.add(SeparableConv2D(25, (5, 5), strides=(1, 1), activation='relu', padding='same'))
    model.add(SeparableConv2D(25, (5, 5), strides=(1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2)))
    model.add(Convolution2D(25, (5, 5), activation='relu',strides=(1, 1), padding='same'))
    model.add(Convolution2D(25, (3, 3), activation='relu',strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2)))
    model.add(Convolution2D(35, (3, 3), activation='relu',strides=(1, 1), padding='same'))
    model.add(Convolution2D(35, (3, 3), activation='relu',strides=(1, 1)))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2)))
    model.add(Convolution2D(64, (3, 3), activation='relu',strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=model.output_shape[1], activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=hidden_num_units, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=output_num_units, input_dim=hidden_num_units, activation='softmax'))

    print(model.summary())

    return model

""""""""""""""""""""""""""""""""""""
# Model 2
""""""""""""""""""""""""""""""""""""
@model2
def _model2_(input_shape, hidden_num_units, output_num_units, pool_size, nepochs, batch_size):

    model2 = Sequential()

    model2.add(InputLayer(input_shape=input_shape))
    model2.add(SeparableConv2D(25, (5, 5), strides=(1, 1), activation='relu', padding='same'))
    model2.add(SeparableConv2D(25, (5, 5), strides=(1, 1), activation='relu', padding='same'))
    model2.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2), padding='same'))
    model2.add(Dropout(0.25))
    model2.add(Convolution2D(25, (5, 5), strides=(1, 1), activation='relu', padding='same'))
    model2.add(Convolution2D(25, (5, 5), strides=(1, 1), activation='relu', padding='same'))
    model2.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2), padding='same'))
    model2.add(Convolution2D(25, (5, 5), activation='relu',strides=(1, 1), padding='same'))
    model2.add(Convolution2D(25, (3, 3), activation='relu',strides=(1, 1), padding='same'))
    model2.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2), padding='same'))
    model2.add(Dropout(0.25))
    model2.add(Convolution2D(25, (5, 5), strides=(1, 1), activation='relu', padding='same'))
    model2.add(Convolution2D(25, (5, 5), strides=(1, 1), activation='relu', padding='same'))
    model2.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2), padding='same'))
    model2.add(Convolution2D(25, (5, 5), activation='relu',strides=(1, 1), padding='same'))
    model2.add(Convolution2D(25, (3, 3), activation='relu',strides=(1, 1), padding='same'))
    model2.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2), padding='same'))
    model2.add(Dropout(0.25))
    model2.add(Convolution2D(35, (3, 3), activation='relu',strides=(1, 1), padding='same'))
    model2.add(Convolution2D(35, (3, 3), activation='relu',strides=(1, 1), padding='same'))
    model2.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2), padding='same'))
    model2.add(Convolution2D(64, (3, 3), activation='relu',strides=(1, 1), padding='same'))
    model2.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2), padding='same'))
    model2.add(Dropout(0.25))    
    model2.add(Flatten())
    model2.add(Dense(units=model.output_shape[1], activation='relu'))
    model2.add(Dropout(0.25))
    model2.add(Dense(units=hidden_num_units, activation='relu'))
    model2.add(Dropout(0.25))
    model2.add(Dense(units=output_num_units, input_dim=hidden_num_units, activation='softmax'))

    print(model2.summary())

    return model2


""""""""""""""""""""""""""""""""""""
# Save model 
"""""""""""""""""""""""""""""""""""" 
@savemodel
def _savemodel_(model, root_dir, i, datasettype, augmented):
    model.summary()
    model.save(root_dir + '/model_CNN_QSO-ITER-%i_PROG-%s-%s.h5' % (i,datasettype,augmented))
    model.save_weights(root_dir + '/modelweights_CNN_QSO-ITER-%i_PROG-%s-%s.h5' 
                                  % (i,datasettype,augmented))
    return


""""""""""""""""""""""""""""""""""""
# Model compile
""""""""""""""""""""""""""""""""""""
@modelcompile
def _modelcompile_(model, i, datasettype,augmented):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10),
             ModelCheckpoint(filepath='best_model_CNN_QSO-ITER-%i_PROG-%s-%s.h5' 
                                   % (i,datasettype,augmented), monitor='val_loss',  
                                      ∫save_best_only=True, mode='min',verbose=1)]

    return callbacks


""""""""""""""""""""""""""""""""""""
# Training dataset 
""""""""""""""""""""""""""""""""""""
@dataset
def _dataset_(data_dir, i, datasettype, isizex, isizey):
    train, test, train_x, test_x, train_y, test_y = _array_(data_dir, i, datasettype, 
                                                           isizex, isizey)
    train_x, val_x, train_y, val_y = _splitvaltrain_(train, train_x, train_y)
    return train_x, test_x, train_y, test_y, val_x, val_y


""""""""""""""""""""""""""""""""""""
# Fit model
""""""""""""""""""""""""""""""""""""
@fitmodel
def _fitmodel_(augmented, i, iteration, datasettype, data_dir, input_shape, hidden_num_units, 
               output_num_units, pool_size, nepochs, batch_size):
    # Open file to save verbose
    FileW = open(os.path.join(root_dir, 'Output_ITER-%i_PROG-%s-%s.txt')  
                                        % (i,datasettype,augmented),'w') 
    sys.stdout = FileW   
    
    if augmented==True:
        print('Iteracion %i de %i con datos %s y' % (i,iteration,datasettype), 
                                  ' Data Augmentation\n')
    else:
        print('Iteracion %i de %i con datos %s y' % (i,iteration,datasettype), 
                                  ' NO-Data Augmentation\n')

    
    train_x, test_x, train_y, test_y, val_x, val_y = _dataset_(data_dir, i, 
                                                               datasettype,isizex, isizey)
    imgen_train = _dataaug_(train_x)
    
    #  Choose model: _model_ or _model2_
    model = _model_(input_shape, hidden_num_units, output_num_units, pool_size,  
                                  nepochs, batch_size)
	#model = _model2_(input_shape, hidden_num_units, output_num_units, pool_size,  
    #                              nepochs, batch_size)

    historial = _modelcompile_(model, i, datasettype,augmented)
    if augmented == True:
        augdata_path = os.path.join(data_dir, 'AugmentedImages')
        gentrain = imgen_train.flow(train_x, train_y, batch_size=batch_size)
        
        startTime = datetime.now()
        trained_model_conv = model.fit_generator(gentrain, steps_per_epoch=train_x.shape[0]//batch_size, 
                                             epochs=nepochs, validation_data=(val_x, val_y),  
                                  callbacks=historial)
        print("Time taken:", datetime.now() - startTime)
    else:
        startTime = datetime.now()
        trained_model_conv = model.fit(train_x, train_y, batch_size=batch_size, 
                                             epochs=nepochs, validation_data=(val_x, val_y), 
                                             shuffle=True, callbacks=historial)
        print("Time taken:", datetime.now() - startTime)

    print(trained_model_conv.history.keys())
    print('val_acc : ',trained_model_conv.history['val_acc'])
    print('acc : ',trained_model_conv.history['acc'])
    print('val_loss : ',trained_model_conv.history['val_loss'])
    print('loss : ',trained_model_conv.history['loss'])
    _savemodel_(model, root_dir, i, datasettype, augmented)
    
    _plotacc_(trained_model_conv,i,datasettype,augmented)
    _predict_(model, test_x, test_y,i,datasettype,augmented)
    
    FileW.close()
    
    return 



############################################
#Main
############################################
def main(): 
    
	""""""""""""""""""""""""""""""""""""
	# define variables
	""""""""""""""""""""""""""""""""""""

	isizex = 57 # pixel
	isizey = 57 # pixel
	input_shape = (isizex, isizey, 5) # x, y size in pixel (x,y, nº band)

	hidden_num_units = 250
	output_num_units = 2

	pool_size = (2,2)
	nepochs = 100
	batch_size = 10

	datasettype='Unbalanced-Arch1' # 'Balanced-Arch1' - 'Unbalanced-Arch1'
	                               # 'Balanced-Arch2' - 'Unbalanced-Arch2'
	iteration=10 # Number iterations
	balanced = False #True - False
	augmented = False #True - False  
	
	""""""""""""""""""""""""""""""""""""
	# train
	""""""""""""""""""""""""""""""""""""
	for i in range(0,int(iteration)):
		_fitmodel_(augmented, i, iteration, datasettype, data_dir, input_shape, 
		           hidden_num_units, output_num_units,pool_size, nepochs, batch_size)
		
		if i==iteration-1:
			print('last iteration:',i)
			break	 
    
    
    
    
    
############################################    
 main()   
    
    
 