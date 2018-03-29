# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run.mp4 a video of the car driving one round along the test route in autonomous mode
* run.tar.gz the images that were output by the simulator and used to compile run.mp4

Here is a link to my [project code repository](https://github.com/amilendra/CarND-Behavioral-Cloning-P3)

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I got most of the tips from the behaviour cloning cheatsheet put together by Paul Heraty which was suggested by Udacity in the project introductions.

I went with the [Network Architecture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) published by Nvidia because it was suggested in the course, cheatsheet, and also because I wanted to use something that had a proven track record.

This model consists of a convolution neural network with three 5x5 convolution layers(line 45-47), 
two 3x3 convolution layers(line 49-50), and four fully-connected layers with respectively 100, 50, 10 and 1 neurons(line 53-56). 
All the convolution layers use RELU activations to introduce nonlinearity.

Before passing the images through the network, they go through two preprocessing layers.
* image normalization using a Keras Lambda layer(line 41)
* image cropping using the Keras Cropping2D layer(line 43)

I initially tried using only the center image to train. However I found that the car does not recover once it drifted to the sides. Even adding my own data where the car recovers from the side of the road, driving to the opposite side did not improve results. So I decided to use images from both cameras, mainly because that seemed the obvious solution and it made sense to use already available data. My driving also was a bit jittery so I was not sure if using my own training data was the best solution.

Because I could get a working model using only the training data, 
I did not find the need to resort to flipping the images to generate more data,
and also did not need to use Generators to handle large amount of data.

The correction parameter for the side images were determined by trial and error.
I used the initial correction of 0.2 which was shown in the project introduction videos. 
However that gave worse performance so I increased the correction parameter to 0.5.
This resulted in worse performance but to the other side, so I gradually reduced it and settled at 0.3.

My next step was to try adding dropout layers if things did not improve after the above, 
but after tuning the correction parameter, the vehicle was able to drive autonomously around the track without leaving the road,
so I stopped work on it because I am already a bit behind schedule as it is.

#### 2. Creation of the Training Set & Training Process

The model was trained and validated on different data sets by shuffling the data to ensure that the model was not overfitting (code line 10-16). 20% of the data was used for validation. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. I used an adam optimizer so that manually training the learning rate wasn't necessary.
I settled down for 6 epochs because anything further did not improve the validation loss. 
