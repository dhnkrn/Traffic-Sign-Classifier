#**Traffic Sign Recognition** 

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
[image4]: ./images/german_1.jpg "Traffic Sign 1"
[image5]: ./images/german_2.jpg "Traffic Sign 2"
[image6]: ./images/german_3.jpg "Traffic Sign 3"
[image7]: ./images/german_4.jpg "Traffic Sign 4"
[image8]: ./images/german_5.jpg "Traffic Sign 5"


---

###Data Set Summary & Exploration
Some key  statistics:
  Number of training examples = 34799
  Number of validation examples = 4410
  Number of testing examples = 12630
  Image data shape = (32, 32, 3) i.e., image has a resolution of 32x32 pixels with each made of RGB components
  pixel value ranges between max=255 and  min=0
  Number of classes = 43 unique traffic signs

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
The image input goes through standard pre-processing steps meant for images. The steps employed are-
1. Grayscale conversion - image color is not a distinguishing feature for traffic signs. IOW there are no two traffic signs with different colors and same symbol.
2. Centering the image values -  (x_train-128.0)/128.0 as these values work well with CNNs that have RELU activations.

Images before and after pre-processing.
![alt text](https://raw.githubusercontent.com/dhnkrn/Traffic-Sign-Classifier/master/images/before.png)

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

To train the model, I used Stochastic Gradient Descent optimized by AdamOptimizer at a learning rate of 0.001. Each batch was a randomized sample of 128 training samples. The loss converged for the validation set at around 15 epochs training on CPU.

The approach to classify the traffic symbols was to implement a standard Lenet-5 CNN and iteratively tune it to improve performance for this specific dataset. The Lenet-5 model comprises of a stack of two convolution layers and three fully connected layers with RELU activations interleaved betweeen them. The convolutions layers outputs are also fed through MaxPooling layers after RELU. One of the changes that improved performance for this dataset is the inclusion of dropout layers connected to fully-connected layers. This was added when I noticed the model was overfitting to the training data set. Learning rate, batch size and the probablity for the dropout layers were the most important hyperparameters that I had to tune. My initial learning rate of 0.1 with the GradientDescent optimizer was failing to train, possibly getting stuck at a local optima. Reducing learning rate by an order was sufficient to get the model to train. I also switched the optimizer to AdamOptimizer as it converged significantly faster than GradientDescent. 

![alt text](https://raw.githubusercontent.com/dhnkrn/Traffic-Sign-Classifier/master/images/accuracy_loss.png)


The final model results, with just 14 epochs of training on CPU were - 
* Training set accuracy of 0.993534
* Validation set accuracy of  0.944898
* Test set accuracy 0.923674 

###Test a Model on New Images

Apart for testing the model on test data, I tested the model with five images of German traffic signs downloaded from the internet.
<img src="https://raw.githubusercontent.com/dhnkrn/Traffic-Sign-Classifier/master/images/german_1.jpg" width="200" height="200" />
<img src="https://raw.githubusercontent.com/dhnkrn/Traffic-Sign-Classifier/master/images/german_2.jpg" width="200" height="200" />
<img src="https://raw.githubusercontent.com/dhnkrn/Traffic-Sign-Classifier/master/images/german_3.jpg" width="200" height="200" />
<img src="https://raw.githubusercontent.com/dhnkrn/Traffic-Sign-Classifier/master/images/german_4.jpg" width="200" height="200" />
<img src="https://raw.githubusercontent.com/dhnkrn/Traffic-Sign-Classifier/master/images/german_5.jpg" width="200" height="200" />


One of the interesting things I noticed was the model fails to classify a "known" traffic sign if the sign is not centered  or does not cover a significant part of the image. Cropping the image to mostly include just the sign gives 100% accuracy. This shows that the dataset is insufficient and makes a good case for augmenting the data set with transformed images.

Another observation is that model appears to have low precision in some cases. Testing with an unseen input - "No stopping"
results in the model classifying it with 70% accuracy as a "Roundabout mandatory". I believe, this probablity would have been lesser if the color components were included in images used for training.
The second and last image are not in the training dataset. 

Here are the results of the prediction:
        Prediction                              Actual
Sign 1: Road work                               Road work
Sign 2: Roundabout mandatory                    No Stopping - Not in the dataset
Sign 3: Right-of-way at the next intersection   Right-of-way at the next intersection
Sign 4: Speed limit (60km/h)                    Speed limit (60km/h)
Sign 5: Children crossing                       No Parking - Not in the dataset

The model classifies 3 of the 5 traffic signs correctly but all 3 signs known to the model are classified with 100% accuracy.

The top five soft max probabilities for the 5 test data are below. The model classifies the first, third and fourth signs with almost 100% certainty. The rest two, second and fifth, are negative test cases where the model is expected to be not certain as these traffic signs are not in the training data.

         Prediction                              Actual
Sign 1: Road work                               Road work                             '1.00', '0.00', '0.00', '0.00', '0.00'

Sign 2: Roundabout mandatory                    No Stopping - Not in the dataset      '0.72', '0.28', '0.00', '0.00', '0.00'

Sign 3: Right-of-way at the next intersection   Right-of-way at the next intersection '1.00', '0.00', '0.00', '0.00', '0.00'

Sign 4: Speed limit (60km/h)                    Speed limit (60km/h)                  '0.99', '0.00', '0.00', '0.00', '0.00'

Sign 5: Children crossing                       No Parking - Not in the dataset       '0.64', '0.36', '0.00', '0.00', '0.00'


### Visualizing the Neural Network
Vizualizing the parameters of the first convolution layer for a 60 km/hr traffic sign looks like this 

![alt text](https://raw.githubusercontent.com/dhnkrn/Traffic-Sign-Classifier/master/images/60_1.png)

