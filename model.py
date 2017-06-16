#Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Block SSE instruction messages
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, SpatialDropout2D

#Define some constants
DATA_PATH = './my-data/'
STEERING_CUTOFF = 0.5
BATCH_SIZE = 32
EPOCHS = 5

#Augment function
def augment(image):
	#Todo

	return

#Get training data
lines = []
with open(DATA_PATH + 'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

#Load training data
def generator(lines, batch_size=32):
	num_samples = len(lines)
	print(num_samples)
	while True:
		sklearn.utils.shuffle(lines)
		for offset in range(0, num_samples, batch_size):
			batch_lines = lines[offset:offset+batch_size]

			images = []
			measurements = []
			for line in batch_lines:
				c_filename = line[0].split('/')[-1]
				l_filename = line[1].split('/')[-1]
				r_filename = line[2].split('/')[-1]
				current_path = DATA_PATH + 'IMG/'
				c_image = cv2.imread(current_path + c_filename)
				#l_image = cv2.imread(current_path + l_filename)
				#r_image = cv2.imread(current_path + r_filename)
				if abs(float()) > STEERING_CUTOFF:
					augment(c_image)
					#augment(l_image)
					#augment(r_image)
				images.append(c_image)
				measurement = float(line[3])
				measurements.append(measurement)
				#Add flipped data
				c_image = cv2.flip(c_image, 1)
				#l_image = cv2.flip(r_image, 1)
				#r_image = cv2.flip(l_image, 1)
				if abs(float()) > STEERING_CUTOFF:
					augment(c_image)
					#augment(l_image)
					#augment(r_image)
				images.append(c_image)
				measurement *= -1.
				measurements.append(measurement)

			#Create numpy arrays
			X_train = np.array(images)
			y_train = np.array(measurements)
			yield sklearn.utils.shuffle(X_train, y_train) 

#Compile and train the model using the generator function
train_generator = generator(train_samples, BATCH_SIZE)
validation_generator = generator(validation_samples, BATCH_SIZE)

#Create Model
#https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60, 20), (0, 0)))) #(80, 320, 3)
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(SpatialDropout2D(0.1))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(SpatialDropout2D(0.1))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(SpatialDropout2D(0.1))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, \
					steps_per_epoch=np.ceil(len(train_samples) / BATCH_SIZE), \
					epochs=EPOCHS, \
					verbose=1, \
					callbacks=None, \
					validation_data=validation_generator, \
					validation_steps=np.ceil(len(validation_samples) / BATCH_SIZE), \
					class_weight=None, \
					max_q_size=10, \
					workers=1, \
					pickle_safe=False, \
					initial_epoch=0)

model.save('model.h5')
exit()
