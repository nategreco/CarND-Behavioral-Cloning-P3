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

#Double set with flipped images
def add_flipped_set(X, y):
	#Check for a valid data set
	assert(len(X) == len(y))

	#Add flipped values
	for i in range(len(X)):	
		np.append(X, cv2.flip(X[i], 1))
		np.append(y, (y[i] * -1.0))
	return

#Get training data
lines = []
with open('./example-data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	c_filename = line[0].split('/')[-1]
	l_filename = line[1].split('/')[-1]
	r_filename = line[2].split('/')[-1]
	current_path = './example-data/IMG/'
	c_image = cv2.imread(current_path + c_filename)
	l_image = cv2.imread(current_path + l_filename)
	r_image = cv2.imread(current_path + r_filename)
	images.append(c_image)
	measurement = float(line[3])
	measurements.append(measurement)

#Add flipped set
X_train = np.array(images)
y_train = np.array(measurements)
add_flipped_set(X_train, y_train)
print(y_train[300], ',', y_train[len(lines) + 300])

#Create Model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

model.save('model.h5')
exit()
