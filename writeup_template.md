# Traffic Sign Recognition 

Implementation Report 

This report describes all the steps and techniques used to load and  pre-processing an dataset of the German Traffic Signs, to train and evalutate a ConvNet to predict and reconize the signs labels. Below is describes the ahrdware as well the enrironment used in the project.

Processor:Intel® Core™ i5-5200U CPU @ 2.20GHz × 4 

Graphics: GeForce 930M/PCIe/SSE2

OS: Linux Ubuntu 17.04 64 bits

Environment: Anaconda 3 / TensorFlow-GPU / CUDA Toolkit 8.0 


# Build a Traffic Sign Recognition Project 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./datasethist.png "DataSet Distribution"
[image2]: ./index.png "Original DataSet"
[image3]: ./preprocessed.jpg "Grayscaled and Normalized DataSet"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

# Rubric Points 

## 1.Required files: 

  For this project will be submmited The Traffic_Sign_Classifier.ipynb notebook file with all questions answered and all code          cells executed and displaying output, a HTML file with th jupyter notebook code, 5 images downloaded from internet in 32X32 format used as external samples, this Report  in MD format , as well a link to GitHub where will be included all files related to the project.
  
## 2.DataSet summarizing: 

 To sumarize the data was used Numpy library, with the folowing results for each aspect of the data set.
 * The size of training set is ?
   n_train = len(X_train)
 * The size of the validation set is ?
   n_validation = len(X_valid)
 * The size of test set is ?
   n_test = len(X_test)
 * The shape of a traffic sign image is ?
   image_shape = X_train.shape[1:]
 * The number of unique classes/labels in the data set is ?
   n_classes = len(np.unique(y_valid))
   
 The notebook output returned as follow:
 
 >>  Number of training examples = 34799
 
 >>  Number of validation examples = 4410
 
 >>  Number of testing examples = 12630
 
 >>  Image data shape = (32, 32, 3)
 
 >>  Number of classes = 43
 
  

## 3.DataSet Exploration 


The graph shows the dataset distribution, where is possible to see how variant is. This variation affect directly the model accuracy, and show how import is the pre-processing step.
Red - Training DataSet
Blue - Test DataSet
Green - Validation DataSet

![alt text][image1]

Below the Bar Graphics, was plotted the first sample of each label without any pre-processing method, just the original DataSet.

![alt text][image2]







## 4.Design and Test a Model Architecture

To pre-process the DataSet, the first step was to grayscale using cv2.cvtColor funtion followed by a normalization. For the model I got better result just divinding the image array by 255 than using (x-128)/128 technique. I tried others tools as cv2.normalize , but as before mentioned, no improvement was verifyed. 


![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


## 5. Model Architeture 

Structure of Weights, Biases and HyperParameters

mu = 0
    sigma = 0.1
    
    strides={'str1': [1,1,1,1],
             'str2': [1,2,2,1]}
    
    
    maxpool_filter=[1,2,2,1]
        
                   

    weights={
            'wl1':tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), mean = mu, stddev = sigma)),
            'wl2':tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 128), mean = mu, stddev = sigma)),
            'wl2_a':tf.Variable(tf.truncated_normal(shape=(5, 5, 128, 800), mean = mu, stddev = sigma)),
            'wl3':tf.Variable(tf.truncated_normal(shape=(800, 420), mean = mu, stddev = sigma)),
            'wl4':tf.Variable(tf.truncated_normal(shape=(420, 200), mean = mu, stddev = sigma)),
            'wl5':tf.Variable(tf.truncated_normal(shape=(200, 43), mean = mu, stddev = sigma))}
    
    biases={
            'bl1':tf.Variable(tf.zeros(32)),
            'bl2':tf.Variable(tf.zeros(128)),
            'bl2_a':tf.Variable(tf.zeros(800)),
            'bl3':tf.Variable(tf.zeros(420)),
            'bl4':tf.Variable(tf.zeros(200)),
            'bl5':tf.Variable(tf.zeros(43))}


6 Layers Model 

| Layer 1         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScaled image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|


| Layer 2        		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScaled image   							| 
| Convolution 5x5     	| 2x2 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|


| Layer 2_a         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScaled image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|

| Layer 3         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScaled image   							| 
| Fully Connected     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|

| Layer 4         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScaled image   							| 
| Fully Connected     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|

| Layer 5         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScaled image   							| 
| Fully Connected     	| 1x1 stride, same padding, outputs 32x32x64 	|




 


## 6. Model Training 

. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


