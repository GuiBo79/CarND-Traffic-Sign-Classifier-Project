# Traffic Sign Recognition 

Implementation Report 

This report describes all the steps and techniques used to load and  pre-processing an dataset of the German Traffic Signs, to train and evalutate a ConvNet to predict and reconize the signs labels. Below is described the hardware as well the environment used in the project.

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
[image2]: ./original.png "Original DataSet"
[image3]: ./preprocessed.png "Grayscaled and Normalized DataSet"
[image4]: ./img1.png "Traffic Sign 1"
[image5]: ./img2.png "Traffic Sign 2"
[image6]: ./img3.png "Traffic Sign 3"
[image7]: ./img4.png "Traffic Sign 4"
[image8]: ./img5.png "Traffic Sign 5"
[image9]: ./performancegraph.png "Performance Graph"
[image10]: ./featuresmap.png "Features Map"

# Rubric Points 

## 1.Required files: 

  For this project will be submitted The Traffic_Sign_Classifier.ipynb notebook file with all questions answered and all code          cells executed and displaying output, a HTML file with th jupyter notebook code, 5 images downloaded from internet in 32X32 format used as external samples, this Report  in MD format , as well a link to GitHub where will be included all files related to the project.
  
## 2.DataSet summarizing: 

 To summarize the data was used Numpy library, with the following results for each aspect of the data set.
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


The graph shows the dataset distribution, where is possible to see how variant is. This variation affect directly the model accuracy, and show how important is the pre-processing step.
Red - Training DataSet
Blue - Test DataSet
Green - Validation DataSet

![alt text][image1]

Below the Bar Graphics, was plotted the first sample of each label without any pre-processing method, just the original DataSet.

![alt text][image2]







## 4.Design and Test a Model Architecture

To pre-process the DataSet, the first step was to grayscale using cv2.cvtColor function followed by a normalization. For the model I got better result just dividing the image array by 255 than using (x-128)/128 technique. I tried others tools as cv2.normalize , but as before mentioned, no improvement was verifyed. 

The Validation dataset was augmented just concatenating the dataset itself, followed by a code (enhance_fig function) who randomly rotate and modify the sharpness of all dataset.


![alt text][image3]


## 5. Model Architecture 

Structure of Weights, Biases and HyperParameters

EPOCHS = 30
BATCH_SIZE = 156
dropout=0.75
mu = 0
sigma = 0.1
    
    strides={'str1': [1,1,1,1],
             'str2': [1,2,2,1]}
    
    
    maxpool_filter=[1,2,2,1]
        
                   

    weights={
            'wl1':tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), mean = mu, stddev = sigma)),
            'wl2':tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean = mu, stddev = sigma)),
            'wl2_a':tf.Variable(tf.truncated_normal(shape=(5, 5, 64, 400), mean = mu, stddev = sigma)),
            'wl3':tf.Variable(tf.truncated_normal(shape=(400, 200), mean = mu, stddev = sigma)),
            'wl4':tf.Variable(tf.truncated_normal(shape=(200, 100), mean = mu, stddev = sigma)),
            'wl5':tf.Variable(tf.truncated_normal(shape=(100, 43), mean = mu, stddev = sigma))}
    
    biases={
            'bl1':tf.Variable(tf.zeros(32)),
            'bl2':tf.Variable(tf.zeros(64)),
            'bl2_a':tf.Variable(tf.zeros(400)),
            'bl3':tf.Variable(tf.zeros(200)),
            'bl4':tf.Variable(tf.zeros(100)),
            'bl5':tf.Variable(tf.zeros(43))}




6 Layers Model 

| Layer 1         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScaled image   	| 
| Convolution 3x3     	| 1x1 stride, VALID padding	| output 28X28X32
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|


| Layer 2        		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 14x14x32 GrayScaled image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding	| output 10X10X64
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|


| Layer 2_a         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 5x5x64  							| 
| Convolution 5x5     	| 1x1 stride, VALID padding 	| output 400
| RELU					|		output 400										|

| Layer 3         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 400  							| 
| Fully Connected     	| 	|
| RELU					|				output 200								|

| Layer 4         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 200  							| 
| Fully Connected     	|  	|
| RELU					|				output 100								|

| Layer 5         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 100   							| 
| Fully Connected     	|  	|
| RELU					|				output 43

SoftMax Funtion: softmax_cross_entropy_with_logits

Optimizer: AdamOptimizer


 


## 6. Model Training 

The model was trained using these hyperparamter:
  Learning Rate=0.00097
  EPOCHS=30
  Batch=156
In the begining I was just doing trial and error to understand how the hyperparameters, as well the architeture and depth of the NetWork were influencing  the performance. The whole time I was looking for a simple code, focused in the NetWork problem. 
Since the begining was clear for me how import is the pre-processing . After lots of changes in the NetWork, a simple change in the normalization technique made my code jumps from 0.70 accuracy to 0.90. 

The first architecture was a Vanilla LetNet. Rest very clear for me how good LeNet can handle with image, but to fit it exatcly to the problem is the challenge. 

Changes and Experiences with LeNet
  Use of Dropout to prevent over-fitting: Doesn't make any difference out of a slower code. 
  Inclusion of an intermediate Conv layer called Layer 2_a to increase the depth of the NetWork: This change improve the accuracy in about 5%
  
  
Each test was followed by an evaluation of the NetWotk accuracy related to the three DataSet, always looking for a convergence of the result. Due the difference between the datasets was already expected different accuracies, but is very interesting how the behavior of the network can change, with a same setup one dataset can converge while another diverge the accuracy.


Final Model Accuracy:

* training set accuracy of 100%
* validation set accuracy of 95.9%
* test set accuracy of 94.2% 

![alt text][image9]



## 7.Test on New Images

All the images were downloaded from internet and ajusted to 32 X 32 whith ShotWell just using the CRÔP tool, no further treatment was done. Was already expected that differences in constrast and brightness could make more difficult to predict the correct sign. Fortunately for the downloaded images the predictions were 100% correct, probably due the good quality of the images. 



![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

y_final_labels=np.array([14,13,40,36,35])

14.Stop

13.Yield

40.Roundabout mandatory

36.Go straight or right

35.Ahead only

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop    		| Stop  									| 
| Yield     			| Yield								|
| Roundabout mandatory					| Roundabout mandatory											|
| Go straight or right	      		| Go straight or right				 				|
| Ahead only			| Ahead only     							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the valid set 96.4%

Image 1 - STOP

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop   									| 
| 0.00     				| Keep left										|
| 0.00					| Turn right ahead										|
| 0.00      			| Road work					 				|
| 0.00				    | Turn left ahead      							|

Image 2 - Yield

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield 									| 
| 0.00     				| 	Ahead only									|
| 0.00					| 		Speed limit (60km/h)							|
| 0.00	      			| 	Speed limit (20km/h)				 				|
| 0.00				    |   Speed limit (30km/h)  							|

Image 3 - Roundabout mandatory

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Roundabout mandatory									| 
| 0.01     				| 	Priority road									|
| 0.00					| 		Speed limit (100km/h)							|
| 0.00	      			| 	Speed limit (80km/h)				 				|
| 0.00				    |   Speed limit (50km/h)  							|

Image 4 - Go straight or right

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Go straight or right									| 
| 0.01     				| 		Turn right ahead						|
| 0.00					| 			Traffic signals						|
| 0.00	      			| 	Ahead only				 				|
| 0.00				    |   Right-of-way at the next intersection 							|

Image 5 - Ahead only

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Ahead only									| 
| 0.00     				| 		Priority road						|
| 0.00					| 			Go straight or right					|
| 0.00	      			| 	No passing				 				|
| 0.00				    |   Turn left ahead							|


## 8. Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

Third Layer Visualization:
The features MAP show us how the network classify the signs. Each feature information is compared with the image , and after calculate de SoftMax function the prediction is done in base of the probabilities.

![alt text][image10]


