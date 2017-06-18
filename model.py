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
from keras.callbacks import Callback

#Define some constants
DATA_PATH = './my-data/'
LOG_PATH = './training.txt'
INPUT_COLS = 320
INPUT_ROWS = 160
INPUT_CHANNELS = 3
SIDE_IMAGE_OFFSET = 0.88
STEERING_CUTOFF = 0.3
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DECAY_RATE = 1.0
EPOCHS = 4

def print_training(history):
	file = open(LOG_PATH, 'w')
	file.write(str(history.losses))
	file.close
	return

#Helper functions
def prepare_image(image): # Get image to correct shape
	#Work with new copy
	working_image = image.copy()

	#Resize if too tall, but maintain aspect ratio
	if working_image.shape[0] > INPUT_ROWS:
		new_height = INPUT_ROWS
		new_width = int((working_image.shape[1] / working_image.shape[0]) * float(INPUT_ROWS))
		working_image = cv2.resize(working_image, (new_width, new_height))
	#Resize if too wide, but maintain aspect ratio
	if working_image.shape[1] > INPUT_COLS:
		new_height = int((working_image.shape[0] / working_image.shape[1]) * float(INPUT_COLS))
		new_width = INPUT_COLS
		working_image = cv2.resize(working_image, (new_width, new_height))
		
	#Pad to input shape
	working_image = cv2.copyMakeBorder(working_image, \
									   int((INPUT_ROWS - working_image.shape[0]) / 2), \
									   int((INPUT_ROWS - working_image.shape[0]) / 2), \
									   int((INPUT_COLS - working_image.shape[1]) / 2), \
									   int((INPUT_COLS - working_image.shape[1]) / 2), \
									   cv2.BORDER_CONSTANT, \
									   value=0)
	
	return working_image

def augment_image(image): #Augment images
	#References - http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html

	#Scale
	sf_x = .06 * np.random.rand() - 3. #Limit +/- 3%
	sf_y = .06 * np.random.rand() - 3. #Limit +/- 3%
	working_image = image.copy()
	working_image = cv2.resize(image.copy(), \
							   None, \
							   fx=(sf_x / 100.) + 1., \
							   fy=(sf_y / 100.) + 1., \
							   interpolation = cv2.INTER_CUBIC)

	#Rotate and Skew
	center_x = int(working_image.shape[1] / 2.)
	center_y = int(working_image.shape[0] / 2.)
	angle = 10. * np.random.rand() - 5. #Limit +/- 5 degrees
	matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
	working_image = cv2.warpAffine(working_image, \
								   matrix, \
								   (working_image.shape[1], working_image.shape[0]))

	#Shift
	shift_x = int(.06 * working_image.shape[1] * np.random.rand() - working_image.shape[1] * .03) #Limit +/- 3%
	shift_y = int(.06 * working_image.shape[1] * np.random.rand() - working_image.shape[0] * .03) #Limit +/- 3%
	matrix = np.float32([[1, 0, shift_x],[0, 1, shift_y]])
	working_image = cv2.warpAffine(working_image, \
								   matrix, \
								   (working_image.shape[1], working_image.shape[0]))
	
	#Get back to correct shape
	working_image = prepare_image(working_image.copy())

	return working_image

#Get training data
lines = []
with open(DATA_PATH + 'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

current_path = DATA_PATH + 'IMG/'
for line in lines:
	c_filename = line[0].split('/')[-1]
	c_image = cv2.imread(current_path + c_filename)
	cv2.imshow('Original', c_image)
	c_image = augment_image(c_image)
	cv2.imshow('Augmented', c_image)
	cv2.waitKey(0)
	

#Load training data
def generator(lines, batch_size=32):
	num_samples = len(lines)
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
				c_image = cv2.imread(current_path + c_filename)
				l_image = cv2.imread(current_path + l_filename)
				r_image = cv2.imread(current_path + r_filename)
				if abs(float()) > STEERING_CUTOFF:
					c_image = augment_image(c_image)
					l_image = augment_image(l_image)
					r_image = augment_image(r_image)
				images.append(c_image)
				measurement = float(line[3])
				measurements.append(measurement)
				images.append(l_image)
				measurements.append(measurement + SIDE_IMAGE_OFFSET)
				images.append(r_image)
				measurements.append(measurement - SIDE_IMAGE_OFFSET)
				#Add flipped data
				c_image = cv2.flip(c_image, 1)
				l_image = cv2.flip(r_image, 1)
				r_image = cv2.flip(l_image, 1)
				if abs(float()) > STEERING_CUTOFF:
					c_image = augment_image(c_image)
					l_image = augment_image(l_image)
					r_image = augment_image(r_image)
				images.append(c_image)
				measurement *= -1.
				measurements.append(measurement)
				images.append(l_image)
				measurements.append(measurement + SIDE_IMAGE_OFFSET)
				images.append(r_image)
				measurements.append(measurement - SIDE_IMAGE_OFFSET)

			#Create numpy arrays
			X_train = np.array(images)
			y_train = np.array(measurements)
			yield sklearn.utils.shuffle(X_train, y_train)

#Create a callback for logging loss
#https://keras.io/callbacks/#create-a-callback
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

#Compile and train the model using the generator function
train_generator = generator(train_samples, BATCH_SIZE)
validation_generator = generator(validation_samples, BATCH_SIZE)

#Create Model
#https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(INPUT_ROWS, INPUT_COLS, INPUT_CHANNELS))) #Normalize
model.add(Cropping2D(cropping=((50, 10), (0, 0)))) #(100, 320, 3)
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(SpatialDropout2D(0.5))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(SpatialDropout2D(0.5))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(SpatialDropout2D(0.5))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
model.add(SpatialDropout2D(0.5))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
model.add(SpatialDropout2D(0.5))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))

history = LossHistory()
model.compile(loss='mse', optimizer='adam')
model.optimizer.lr.assign(LEARNING_RATE)
model.optimizer.decay.assign(DECAY_RATE)
model.fit_generator(train_generator, \
					steps_per_epoch=np.ceil(6 * len(train_samples) / BATCH_SIZE), \
					epochs=EPOCHS, \
					verbose=1, \
					callbacks=[history], \
					validation_data=validation_generator, \
					validation_steps=np.ceil(6 * len(validation_samples) / BATCH_SIZE), \
					class_weight=None, \
					max_q_size=10, \
					workers=1, \
					pickle_safe=False, \
					initial_epoch=0)
print_training(history)
model.save('model.h5')
exit()
