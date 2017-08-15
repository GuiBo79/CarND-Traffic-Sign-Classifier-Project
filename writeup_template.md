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
[image3]: ./preprocessed.png "Grayscaled and Normalized DataSet"
[image4]: ./img1.png "Traffic Sign 1"
[image5]: ./img2.png "Traffic Sign 2"
[image6]: ./img3.png "Traffic Sign 3"
[image7]: ./img4.png "Traffic Sign 4"
[image8]: ./img5.png "Traffic Sign 5"

# Rubric Points 

## 1.Required files: 

  For this project will be submmited The Traffic_Sign_Classifier.ipynb notebook file with all questions answered and all code          cells executed and displaying output, a HTML file with th jupyter notebook code, 5 images downloaded from internet in 32X32 format used as external samples, thiimg2s Report  in MD format , as well a link to GitHub where will be included all files related to the project.
  
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

The model was trained using these hyperparamter:
  Learning Rate=0.00097
  EPOCHS=30
  Batch=128
In the beggining I was just doing trial and error to understand how the hyperparameters, as well the architeture and depth of the NetWork were influnecing in the performance. The whole time I was looking for a simple code, focused in the NetWork problem. 
Since the beggining was clear for me how import is the pre-processing . After lots of changes in the NetWork, a simple change in the normalization technique made my code jumps from 0.70 accuracy to 0.90. 



The first architeture was a Vanilla LetNet. Rest very clear for me how good LeNet can handle with image, but to fit it exatcly to the problem is the challenge. 

Changes and Experiences with LeNet
  Use of Dropout to prevent overfitting: Doesn't make any difference out of a slower code to run
  Inclusion of an intermediate Conv layer called Layer 2_a to increase the depth of the NetWork: This change improve the accuracy in about 5%
  
Each test was followed by an evalutation of the NetWotk accuracy related to the three DataSet, always looking for a convergency of the result. Due the difference between the datasets was already expected differents accuracies, but is very interesting how the behavior of the network can change, with a same setup one dataset can converge while another diverge the accuracy.


Final Model Accuracy:

* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

## 7.Test on New Images

All the images were downloaded from internet and ajusted to 32 X 32 whith ShotWell



![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

y_final_labels=np.array([4,32,40,36,35])

4.Speed limit (70km/h)
32.End of all speed and passing limits
40.Roundabout mandatory
36.Go straight or right
35.Ahead only


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


