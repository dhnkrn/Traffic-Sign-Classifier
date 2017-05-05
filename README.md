#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/german_1.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


---

###Data Set Summary & Exploration

####1. Summary of the dataset.

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
pixel value max= 255 min= 0
Number of classes = 43

####2. Vizualization of the dataset.

Here is an exploratory vizualization of the data set. It is a bar chart showing how the data ...

![alt text](https://raw.githubusercontent.com/dhnkrn/Traffic-Sign-Classifier/master/images/label_distribution.png)
ClassID SignName                                                % Training  % Validation
0       Speed limit (20km/h)                                     0.52       0.68
1       Speed limit (30km/h)                                     5.69       5.44
2       Speed limit (50km/h)                                     5.78       5.44
3       Speed limit (60km/h)                                     3.62       3.40
4       Speed limit (70km/h)                                     5.09       4.76
5       Speed limit (80km/h)                                     4.74       4.76
6       End of speed limit (80km/h)                              1.03       1.36
7       Speed limit (100km/h)                                    3.71       3.40
8       Speed limit (120km/h)                                    3.62       3.40
9       No passing                                               3.79       3.40
10      No passing for vehicles over 3.5 metric tons             5.17       4.76
11      Right-of-way at the next intersection                    3.36       3.40
12      Priority road                                            5.43       4.76
13      Yield                                                    5.52       5.44
14      Stop                                                     1.98       2.04
15      No vehicles                                              1.55       2.04
16      Vehicles over 3.5 metric tons prohibited                 1.03       1.36
17      No entry                                                 2.84       2.72
18      General caution                                          3.10       2.72
19      Dangerous curve to the left                              0.52       0.68
20      Dangerous curve to the right                             0.86       1.36
21      Double curve                                             0.78       1.36
22      Bumpy road                                               0.95       1.36
23      Slippery road                                            1.29       1.36
24      Road narrows on the right                                0.69       0.68
25      Road work                                                3.88       3.40
26      Traffic signals                                          1.55       1.36
27      Pedestrians                                              0.60       0.68
28      Children crossing                                        1.38       1.36
29      Bicycles crossing                                        0.69       0.68
30      Beware of ice/snow                                       1.12       1.36
31      Wild animals crossing                                    1.98       2.04
32      End of all speed and passing limits                      0.60       0.68
33      Turn right ahead                                         1.72       2.04
34      Turn left ahead                                          1.03       1.36
35      Ahead only                                               3.10       2.72
36      Go straight or right                                     0.95       1.36
37      Go straight or left                                      0.52       0.68
38      Keep right                                               5.34       4.76
39      Keep left                                                0.78       0.68
40      Roundabout mandatory                                     0.86       1.36
41      End of no passing                                        0.60       0.68
42      End of no passing by vehicles over 3.5 metric tons       0.60       0.68

###Design and Test a Model Architecture
The pre-processing steps I used are-
1. Conversion to gray scale - color is not a distinguishing feature for traffic signs. IOW there are no two traffic signs with different color but same symbols. 
2. Center the image values -  (x_train-128.0)/128.0 as these values work with CNNs that have RELU activations.

Images before and after pre-processing.
![alt text](https://raw.githubusercontent.com/dhnkrn/Traffic-Sign-Classifier/master/images/before.png)
![alt text](https://raw.githubusercontent.com/dhnkrn/Traffic-Sign-Classifier/master/images/after.png)

My model consisted of the following layers based on LeNet architecture.


| Input         	        	| grayscale image   						    |        32x32x1          	 
| Convolution1     5x5x1x6| 1x1x1 stride, VALID padding |outputs 28x28x6 	 
| RELU1				               |			
| Max pooling	      	2x2x1| 2x2x1 stride, VALID padding |outputs 14x14x6 			
| Convolution2 	  5x5x6x16| 1x1x1 stride, VALID padding |outputs 10x10x16			
| RELU2			               	|
| Max pooling	      	2x2x1| 2x2x1 stride, VALID padding |outputs 5x5x16
| Fully connected 	400x120|                             |outputs 120
| RELU3				               |			
| Dropout1                | 0.7
| Fully connected 	 120x84|                             |outputs 84
| RELU4			               	|			
| Dropout2                | 0.7
| Fully connected				84x43|                             |outputs 43
| Softmax               		|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?  0.944898
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
        Prediction                              Actual
Sign 1: Road work                  Road work
Sign 2: Roundabout mandatory       No Stopping - Not in the dataset
Sign 3: Right-of-way at the next intersection       Right-of-way at the next intersection
Sign 4: Speed limit (60km/h)       Speed limit (60km/h)
Sign 5: Children crossing          No Parking - Not in the dataset


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 60.0%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

['1.00000', '0.00000', '0.00000', '0.00000', '0.00000']
['0.72244', '0.27636', '0.00061', '0.00033', '0.00010']
['1.00000', '0.00000', '0.00000', '0.00000', '0.00000']
['0.99994', '0.00004', '0.00002', '0.00000', '0.00000']
['0.63904', '0.36041', '0.00031', '0.00019', '0.00003']

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


