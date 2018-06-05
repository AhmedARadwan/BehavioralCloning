import csv
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Lines in the csv file
lines = []

# Reading the csv file and save each line in lines array
with open('TrainingV3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Each line contain: Center image, left image, right image, steering value, throttle, brake, speed
# and the tag of each one is found in lines[0], so the data starts from lines[1]
print("First row: ", lines[0])
print("Second row (data): ", lines[1])
print()
print("Lines Number: ", len(lines))




# correction for the left and the right images of the sample data
steering_correction = 0.2
# Images array for the input data
images = []
# Measurements will be the labels for the Images
measurements = []
counter = 0

for line in lines[1:]:
    center_path = line[0]
    left_path = line[1]
    right_path = line[2]
    cent_measurement = float(line[3])
    throttle = float(line[4])
    # print(cent_measurement)

    center_filename = (center_path.split('/')[-1]).split('\\')[-1]
    left_filename = (left_path.split('/')[-1]).split('\\')[-1]
    right_filename = (right_path.split('/')[-1]).split('\\')[-1]

    center_current_path = 'TrainingV3' + '/IMG/' + center_filename
    left_current_path = 'TrainingV3' + '/IMG/' + left_filename
    right_current_path = 'TrainingV3' + '/IMG/' + right_filename

    cent_image = cv2.imread(center_current_path)[..., ::-1]
    left_image = cv2.imread(left_current_path)[..., ::-1]
    right_image = cv2.imread(right_current_path)[..., ::-1]

    left_measurement = cent_measurement + steering_correction
    right_measurement = cent_measurement - steering_correction

    cent_image = cv2.resize(cent_image[40:140, :], (64, 64))
    images.append(cent_image)
    measurements.append(tuple((cent_measurement, throttle)))
    # images.append(cent_image_flipped)

    left_image = cv2.resize(left_image[40:140, :], (64, 64))
    images.append(left_image)
    measurements.append(tuple((left_measurement, throttle)))
    # images.append(left_image_flipped)


    right_image = cv2.resize(right_image[40:140, :], (64, 64))
    images.append(right_image)
    measurements.append(tuple((right_measurement, throttle)))
    counter += 1
    print(counter)


# Convert images and measurements into numpy arrays for keras

# Covert to numpy arrays
images = np.array(images)
print(images.shape)
measurements = np.array(measurements)
print(measurements.shape)
print(measurements[0])
print('data in arrays')


import tensorflow as tf


# Normalization, Mean Centering, and Cropping


# In[37]:

# from keras.models import Sequential
# from keras.layers import Flatten, Dense, Activation, Lambda, Cropping2D
# from keras.layers.convolutional import Convolution2D
# from keras.layers.pooling import MaxPooling2D
# from keras.layers.core import Dropout

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout, Cropping2D, Activation
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU



# We will depend on Nvidia Model

def Nvidia_model(input_shape):
    # Normalizing, Mean Cenetring, Cropping
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, name="image_normalization", input_shape=input_shape))


    model.add(Convolution2D(24, 5, 5, name="convolution_1", subsample=(2, 2), activation='relu', border_mode="valid",
                            init='he_normal'))

    # model.add(Convolution2D(nb_filter=36,nb_row=5,nb_co                                                                                                                                                                                                           l=5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36, 5, 5, name="convolution_2", subsample=(2, 2), border_mode="valid", init='he_normal',
                            activation='relu'))

    # model.add(Convolution2D(nb_filter=48,nb_row=5,nb_col=5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48, 5, 5, name="convolution_3", subsample=(2, 2), border_mode="valid", init='he_normal',
                            activation='relu'))

    # model.add(Convolution2D(nb_filter=64,nb_row=3,nb_col=3,activation='relu'))
    model.add(Convolution2D(64, 3, 3, name="convolution_4", border_mode="valid", init='he_normal', activation='relu'))

    # model.add(Convolution2D(nb_filter=64,nb_row=3,nb_col=3,activation='relu'))
    model.add(Convolution2D(64, 3, 3, name="convolution_5", border_mode="valid", init='he_normal', activation='relu'))

    model.add(Flatten())

    # Adding Dropout layer
    model.add(Dropout(p=0.2))
    model.add(Activation('relu'))
    model.add(Dense(1000))

    # Adding Dropout layer
    model.add(Dropout(p=0.5))
    model.add(Activation('relu'))
    model.add(Dense(500))

    # Adding Dropout layer
    model.add(Dropout(p=0.5))
    model.add(Activation('relu'))
    model.add(Dense(100))

    # Adding Dropout layer
    model.add(Dropout(p=0.2))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Dense(2))  # Steering angle action and speed

    return model


# In[38]:


# Training data

X_train = images
print(X_train.shape)
y_train = measurements
print(y_train.shape)


# Run the model
epochs_arr = [20, 30, 40, 50]

for x in range(0, len(epochs_arr)):
    Model = Nvidia_model(input_shape=(64, 64, 3))
    learning_rate = 0.001
    print('created Model')
    # Mean square error and adam optimizer
    adam = Adam(lr=learning_rate)
    Model.compile(optimizer=adam, loss='mse')
    print('compiled Model')
    Model.summary()
    epochs = epochs_arr[x]
    batch_size = 512
    # 15% validation and apply shuffling
    Model.fit(x=X_train, y=y_train, nb_epoch=epochs, batch_size=batch_size,  validation_split=0.15, shuffle=True)
    # Save the model for testing it and give it as a paramter when running drive.py
    Model.save('model_6_str_throttle_' + str(learning_rate) + 'lr_' + str(epochs) + 'epoch.h5')



