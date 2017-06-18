# **Behavioral Cloning** 

## Behavior cloning with End-to-End learning

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Original.JPG "Original"
[image2]: ./Augmented.JPG "Augmented"
[image3]: ./Original2.JPG "Original2"
[image4]: ./Augmented2.JPG "Augmented2"
[image5]: ./Bridge.jpg "Bridge"
[image6]: ./BridgePOV.jpg "BridgePOV"
[image7]: ./TurnOff.jpg "TurnOff"
[image8]: ./TurnOffPOV.jpg "TurnOffPOV"
[image9]: ./Model.jpg "Model"

---

### Writeup

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/nategreco/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/nategreco/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/nategreco/CarND-Behavioral-Cloning-P3/blob/master/model.h5) containing a trained convolution neural network 
* [README.md](https://github.com/nategreco/CarND-Behavioral-Cloning-P3/blob/master/writeup/README.md) the report you are reading now


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The initial model was based off of the Nvidia end-to-end model shown [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), consisting of 5 convolutional layers and 4 fully connected layers with a normalized input and flattening layer inbetween.  Input was normalized with a Keras lambda layer (model.py:187-188), and then cropped with a Keras Cropping2D layer(model.py:189) to remove noise from the image (sky and hood of the car).  All activation functions used were ReLU and the output was a normalized steering position.  The overall model structure can be seen here: [model.py](https://github.com/nategreco/CarND-Behavioral-Cloning-P3/blob/master/model.py):189-207.


#### 2. Attempts to reduce overfitting in the model

Overfitting in the model was prevented in a number of ways. First, a Keras SpatialDropout2D layer was added between each Convolution layer (model.py:191,193,195,197,199), and similarily a Keras Dropout layer was added after each fully connected layer (model.py:202,204,206).  Dropout probabilities of 50% were used.

Next, image augmentation was implemented on the training and validation sets for images with non-zero steering position.  Augmentation (model.py:65-98) consisted of the following:

* Random scaling in both X and Y, +/- 3% - (model.py:68-76)
* Random rotation about image center, +/- 5 degrees - (model.py:78-85)
* Random translation in both X and Y, +/- 3% - (model.py:87-93)
* Preparing of image to bring back to Keras model input shape - (model.py:39-63)

Before and after augmentation:

![Before][image1] ![After][image2]

Also, all images used in the training sets were flipped to prevent the model from favoring left turns due to a counter-clockwise track.


#### 3. Model parameter tuning

Critical parameters to the model were listed at the top (model.py:16-29) to allow quick tweaking and re-training of the model.  An adam optimizer was used and default training rates worked well with approximately 5 epochs.  After this overfitting could be observed by reaching a limit of the validation loss.  A batch size of 64 was used due to the limit of GPU memory in the AWS instance that was being used.  A batch size of 128 would through an exception for not enough resources.

The parameter that was found to be most critical in the training was actually the 'SIDE_IMAGE_OFFSET' (model.py:22).  This is because the left and right images were added to the training set specfically to teach the model recovery, therefore the larger the value, the more agressive the attempt to steer towards the middle, and the lower the value, the more sluggish the steering response to an out of center condtion.


#### 4. Appropriate training data

I used my own training data for training, which included 3 laps counter-clockwise on track 1, 3 laps clockwise on track 1, and 1 lap clockwise on track 2.  Both the clockwise and counter-clockwise laps helped train prevent a bias in the model towards left versus right steering, similarly to the flipping of the images.  The data from track 2 helped generalize the model to using other road markings and image ques for position indication.  Overall, 17,925 car positions were captured, 3 images of each, and all images were flipped.  This made the total sample of data points 107,550.  Training and validation data was handled as follows:

* Training samples were shuffled - (model.py:124)
* Center image and left/right images with steering offset were added - (model.py:141-148)
* Images of non-zero steering positions were augmented as described above - (model.py:137-140,153-156)
* All 3 images were then flpped and added again, making 6 samples total per position - (model.py:149-164)
* Images were normalized in first layer - (model.py:187-188)
* Images were cropped in second layer - (model.py:189)


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As stated above, initial model began with the Nvidia End-to-End model, the only additional layer being the 2nd for cropping to remove irrelevant data from the model such as the sky or hood of the car.  Next, both 2D and 1D dropout were added to prevent overfitting.  With this model simulation showed the vehicle maintaining the center, but as soon as it began to get out of center it would go off course.  This was because no data had been provided for recovery situations, when a greater steering angle is required due to off center.

My first attempt to correct this condition was to use training data with the car heading towards the road edge and steering away, so I recorded data with a lot of swerving.  This did not help, and in fact made my vehicle want to drive off the road on a straight line.  This is because the capture data also included the erroneous data of me intentionally steering towards the road edge to create the recovery condition.  A better alternative I found was utilization of the left and right images.  So these images were added to the training set as well but with a steering position offset, which turned out to be an essential parameter for tuning the behavior of the car.  A larger offset would result in more aggressive steering, and a smaller one would be sluggish to react.  But with this approach, the car would track straight on an otherwise straight road.

The next issue was the brdige.

Problematic turn-off:

![Bridge Above][image5]

![Bridge POV][image6]


The final issue was the tendency of the car to drive off course after the bridge on track 1.  This was due to a 'dirt turn-off' area where the road line was defined by just a transition of pavement to dirt and no clear road markings similar to the rest of the track.  The solution of the problem was to generalize the model.  This was done by adding both data from the second track and augmenting the image.  Creating much more of a variety of input data allowed the model to reduce its reliance on a yellow or hatched white/red road marking in the corners and then the model would steer away from the dirt turnoff.

Problematic turn-off:

![Turn-off Above][image7]

![Turn-off POV][image8]


The final model and training values provided desireable behavior on track 1, and drove relatively well on track 2 although occasional lange changes would happen due to the model following the road width instead of lane markings.  The final touch was implementing smoother steering control by using a moving average of the steering command (drive.py:35-38).  This reduced 'jitter' in the steering and made it more natural and time weighted.

Track 1 video:

[![Track 1 video](https://youtu.be/fzEnhFJG6dU/0.jpg)](https://youtu.be/fzEnhFJG6dU)

Track 2 video:

[![Track 2 video](https://youtu.be/fzEnhFJG6dU/0.jpg)](https://youtu.be/fzEnhFJG6dU)


#### 2. Final Model Architecture

My final model:

| Layer         		| Description		        						|
|:---------------------:|:-------------------------------------------------:|
| Input         		| 160x320x3 BGR image 								|
| Resize 		     	| Resized to 80x160x3 								|
| Lambda 		     	| Image normalized with 0 mean 						|
| Cropping2D	     	| Outputs 50x160x3 									|
| Convolution	     	| 2x2 stride, 5x5 filter, valid, outputs 23x78x24	|
| RELU					| 													|
| SpatialDropout2D 		| 80% keep probability 								|
| Convolution	     	| 2x2 stride, 5x5 filter, valid, outputs 10x37x36 	|
| RELU					| 													|
| SpatialDropout2D 		| 80% keep probability 								|
| Convolution	     	| 1x1 stride, 3x3 filter, valid, outputs 8x35x48 	|
| RELU					| 													|
| SpatialDropout2D 		| 80% keep probability 								|
| Convolution	     	| 1x1 stride, 3x3 filter, valid, outputs 6x33x64 	|
| RELU					| 													|
| Convolution	     	| 1x1 stride, 3x3 filter, valid, outputs 4x31x64 	|
| RELU					| 													|
| Flatten				| Outputs 7936x1									|
| Fully connected 		| Outputs 100x1 									|
| Dropout				| 50% keep probability 								|
| Fully connected 		| Outputs 50x1 										|
| Dropout				| 50% keep probability 								|
| Fully connected 		| Outputs 10x1 										|
| Fully connected 		| Outputs 1 (steering position normalized) 			|

![Model][image9]

#### 3. Creation of the Training Set & Training Process

Training data was shuffled, augmented, and split into training and validation sets.

Before and after example 2:

![Before][image3] ![After][image4]