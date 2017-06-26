#Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Block SSE instruction messages
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, SpatialDropout2D
from keras.callbacks import Callback
from keras import backend as K

#Create flat model
def create_flat_model(rows, cols, channels):
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, \
					 input_shape=(rows, cols, channels)))
	model.add(Flatten())
	model.add(Dense(1))
	return model

#Create LeNet
def create_lenet_model(rows, cols, channels):
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, \
					 input_shape=(rows, cols, channels)))
	model.add(Cropping2D(cropping=((70, 25), (0, 0))))
	model.add(Conv2D(6, (5, 5), strides=(1, 1), activation="relu"))
	model.add(MaxPooling2D())
	model.add(Conv2D(6, (5, 5), strides=(1, 1), activation="relu"))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))
	return model 
  
#Create Nvidia Model
#https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
def create_nvidia_model(rows, cols, channels):
	model = Sequential()
	model.add(Lambda(lambda x: x / 127.5 - 1., \
					 input_shape=(rows, cols, channels))) 			#Normalize
	model.add(Cropping2D(cropping=((69, 15), (0, 0))))				#76x320x3
	model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))#36x158x24
	model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))#16x77x36
	model.add(Dropout(0.5))											#Dropout
	model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))#6x37x48
	model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))#4x35x64
	model.add(Dropout(0.5))											#Dropout
	model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))#2x33x64
	model.add(Flatten())											#4224x1
	model.add(Dense(100))											#100x1
	model.add(Dropout(0.5))											#Dropout
	model.add(Dense(50))											#50x1
	model.add(Dense(10))											#10x1
	model.add(Dense(1))												#Output
	return model

#Create network with resizing
def create_resize_model(rows, cols, channels):
	model = Sequential()
	model.add(Lambda(lambda x: K.tf.image.resize_images(x, (80, 160)), \
					 input_shape=(rows, cols, channels)))			#Resize 80x160x3
	model.add(Lambda(lambda x: x / 127.5 - 1.))						#Normalize
	model.add(Cropping2D(cropping=((25, 5), (0, 0))))				#Crop->50x160x3
	model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))#Conv2D->23x78x24
	model.add(SpatialDropout2D(0.2))								#2D-Dropout
	model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))#Conv2D->10x37x36
	model.add(SpatialDropout2D(0.2))								#2D-Dropout
	model.add(Conv2D(48, (5, 5), strides=(1, 1), activation="relu"))#Conv2D->6x33x48
	model.add(SpatialDropout2D(0.2))								#2D-Dropout
	model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))#Conv2D->4x31x64
	model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))#Conv2D->2x29x64
	model.add(Flatten())											#Flatten->3712x1
	model.add(Dense(100))											#Fully connected->100x1
	model.add(Dropout(0.5))											#Dropout
	model.add(Dense(50))											#Fully connected->50x1
	model.add(Dropout(0.5))											#Dropout
	model.add(Dense(10))											#Fully connected->10x1
	model.add(Dense(1))												#Output
	return model