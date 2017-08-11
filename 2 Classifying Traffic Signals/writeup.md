# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/signs.png "Visualization"
[image2]: ./examples/barchart.png "Class Size"
[image4]: ./examples/i1.jpg "Traffic Sign 1"
[image5]: ./examples/i2.jpeg "Traffic Sign 2"
[image6]: ./examples/i3.jpeg "Traffic Sign 3"
[image7]: ./examples/i4.jpeg "Traffic Sign 4"
[image8]: ./examples/i5.jpeg "Traffic Sign 5"
[image9]: ./examples/f1.png "Layer1"
[image10]: ./examples/f2.png "Layer2"

---

### Data Set Summary & Exploration

#### 1. Following is a basic summary of the data set. 

Pandas library is used to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Following is an exploratory visualization of the data set. Samples from each sign label are randomly selected and displayed together with corresponding number of images available for training.

![alt text][image1]

The following figure shows the count of images in each class available for training.
![alt text][image2]

* The maximum count of images under a sign label is 2010
* The minimum count of images under a sign label is 180

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to use only "Y" channel so that the model is not dependent on any one color.

As a last step, I normalized the image data by dividing the Y channel values by 255. This will ensure that the pixel values are in the range [0,1].

I decided to generate additional data because the number of images available in each sign label varies. Thus, there might be a bias to certain sign labels due to this inequality in the training size. 

To add more data to the the data set, I randomly sampled the images to increase the image count to the maximum count of 2010 for each sign label.




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 image      							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x108 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x108 				|
| Dropout   	      	| 50% during training, outputs 16x16x108		|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x200 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x200  				|
| Dropout   	      	| 50% during training, outputs 8x8x200  		|
| Flatten               | Flatten and concatenate Layers 1 & 2 outputs  |
| Fully connected Layer1| Inputs 1x40448, Outputs 1x120        			|	
| Fully connected Layer2| Inputs 1x120, Outputs 1x84        			|
| Fully connected Layer3| Inputs 1x120, Outputs 1x43        			|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer to minimize the softmax cross entropy.

* Batch size is 128
* Number of epochs is 10
* Learning rate is 0.001
* The variables were initialized using truncated normal distribution with mean 0 and standard deviation 1/10.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of 0.961 
* test set accuracy of 0.947

An iterative approach was chosen:

* The first architecture was LeNet's architecture.
* Unable to reach the validation accuracy greater than 0.93.
* The architecture was adjusted using the architecture proposed in "Traffic Sign Recognition with Multi-Scale Convolutional Networks" by P. Sermanet and Y. LeCun.
* The ouputs from both Layer 1 and Layer 2 were used to form the inputs to the fully connected layers.
* Additionally a dropout of 50% was used during training in the Layer 1 and Layer 2 convolutional layers. This droput ensured that the model is robust and able to generalize well for any new images.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult as it is has vegetation in the background. The second image has watermarking and might also be challenging.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout mandatory  | Roundabout mandatory   						| 
| Speed limit (60km/h)  | Speed limit (60km/h) 							|
| Keep right			| Keep right									|
| Turn right ahead	   	| Turn right ahead				 				|
| Stop       			| Stop              							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.947

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 24th cell of the Ipython notebook.

For the first image, the model is very sure that this is a roundabout mandatory (probability of 0.96), and the image does contain a roundabout mandatory sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .96         			| Roundabout mandatory  						| 
| .34     				| Speed limit (100km/h) 						|
| .00					| Speed limit (120km/h)							|
| .00	      			| Priority road					 				|
| .00				    | Vehicles over 3.5 metric tons prohibitted     |

For the second image, the model is very sure that this is a speed limit (60km/h) (probability of 0.96), and the image does contain a speed limit (60km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (60km/h)  						| 
| .00    				| Speed limit (80km/h) 	     					|
| .00					| Speed limit (50km/h)							|
| .00	      			| Go straight or right			 				|
| .00				    | End of speed limit (80km/h)                   |

For the third image, the model predicted a keep right (probability of 0.42), and the image does contain a keep right sign. Though the prediction is correct, the probability was not very high as observed in other images. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .42        			| Keep right             						| 
| .35    				| Speed limit (60km/h) 	     					|
| .21					| Children crossing 							|
| .01	      			| Speed limit (80km/h)   		 				|
| .01				    | Slippery road                                 |

For the fourth image, the model is very sure that this is a turn right ahead (probability of 0.99), and the image does contain a turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999         		| Turn right ahead       						| 
| .0001    				| No vehicles 	     					        |
| .00					| Go straight or left							|
| .00	      			| General caution			 				    |
| .00				    | Turn left ahead                               |

For the fifth image, the model is almost sure that this is a stop (probability of 0.86), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .860        			| Stop  				                  		| 
| .100    				| Speed limit (60km/h) 	     					|
| .032					| Speed limit (30km/h)							|
| .004	      			| Speed limit (20km/h)			 				|
| .004				    | Keep right                                    |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The covolutional layer 1 output is shown in the figure below. This layer tries to detect the boundaries of the symbols present in the image.

![alt text][image9]

The covolutional layer 1 output is shown in the figure below. This layer tries to detect the features within the detected boundries in the image.

![alt text][image10]
