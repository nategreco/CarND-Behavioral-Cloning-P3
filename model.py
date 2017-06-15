#Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Block SSE instruction messages
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

#Get training data
lines = []
with open('./example-data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

#Load training data
images = []
measurements = []
for line in lines:
	c_filename = line[0].split('/')[-1]
	l_filename = line[1].split('/')[-1]
	r_filename = line[2].split('/')[-1]
	current_path = './example-data/IMG/'
	c_image = cv2.imread(current_path + c_filename)
	#l_image = cv2.imread(current_path + l_filename)
	#r_image = cv2.imread(current_path + r_filename)
	images.append(c_image)
	measurement = float(line[3])
	measurements.append(measurement)
	#Add flipped data
	c_image = cv2.flip(c_image, 1)
	#l_image = cv2.flip(r_image, 1)
	#r_image = cv2.flip(l_image, 1)
	images.append(c_image)
	measurement *= -1.
	measurements.append(measurement)

#Create numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

#Create Model
#https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)))) #(65, 320, 3)
model.add(Conv2D(24, (5, 5), subsample=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), subsample=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), subsample=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')
exit()
