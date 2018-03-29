import csv
import cv2
import numpy as np

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)# Skip header
    for line in reader:
        lines.append(line)
                                                                                                                                                
images =[]
measurements  = []
CORRECTION = 0.3
# these are the corrections applied to the images from different cameras
#               center,       left,       right
corrections = [      0, CORRECTION, -CORRECTION]
for line in lines:
    for idx in [0,1,2]: #(0: center, 1: left, 2: right)
        source_path = line[idx]
        filename = source_path.split('/')[-1]
        current_path = '../data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurement = measurement + corrections[idx]
        measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Based on nvidia's self driving car network architecture.
# https://devblogs.nvidia.com/deep-learning-self-driving-cars/
model = Sequential()
# Image normalizing layer
model.add(Lambda(lambda x:x /255.0 - 0.5, input_shape=(160,320,3)))
# image cropping layer
model.add(Cropping2D(cropping=((70,25),(0,0))))
# 5x5 convolution layers with relu activation
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
# 3x3 convolution layers with relu activation
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
# Fully connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

model.save('model.h5')
