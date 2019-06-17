import os
import numpy as np
import keras
from keras.models import Sequential
from datetime import datetime
import matplotlib.image as mpimg
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random


datadir = './Track'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
pd.set_option('display.max_colwidth', -1)
data.head()
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail
data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data.head()
num_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1]+ bins[1:]) * 0.5


print('total data:', len(data))
remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)
 
print('removed:', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('remaining:', len(data))
 
hist, _ = np.histogram(data['steering'], (num_bins))


print(data.iloc[1])
def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
        # left image append
        image_path.append(os.path.join(datadir,left.strip()))
        steering.append(float(indexed_data[3])+0.15)
        # right image append
        image_path.append(os.path.join(datadir,right.strip()))
        steering.append(float(indexed_data[3])-0.15)
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings
 
image_paths, steerings = load_img_steering(datadir + '/IMG', data)
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))



def zoom(image):
    """
    return zoomed image |1-1.3|
    """

    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image


def pan(image):
    """
    return image displace
    """

    pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image


def bright(image):
    """
    return image more or less bright
    """

    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image


def flip(image, steering_angle):
    """
    return flipped image with steering angle
    """

    image = cv2.flip(image,1)
    steering_angle = -steering_angle
    return image, steering_angle



def random_augment(image, steering_angle):
    """
    return randmonly augment image & steering angle
    """

    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
      image = pan(image)
    if np.random.rand() < 0.5:
      image = zoom(image)
    if np.random.rand() < 0.5:
      image = bright(image)
    if np.random.rand() < 0.5:
      image, steering_angle = flip(image, steering_angle)
    
    return image, steering_angle



def img_preprocess(img):
    """
    input image
    1 - image cropped
    2 - changing the color map (YUV)
    3 - split the filter
    4 - reduce noise
    5 - resize the image
    6 - divide by 255 to normalize
    return the image preprocessed and 3 YUV filters

    """

    img = img[60:135,:,:] #bellow 60 and over 135
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) # change cmap
    y, u, v = cv2.split(img) # split filter
    img = cv2.GaussianBlur(img,  (3, 3), 0) # reduce noise
    img = cv2.resize(img, (200, 66)) # resize
    img = img/255 # normalization
    return img, y, u, v


def batch_generator(image_paths, steering_ang, batch_size, istraining):
    """
    input image, steering batchsize (100), istraining (boolean)
    """

    while True:
        batch_img = []
        batch_steering = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1) # random int between 0 and total image

            if istraining:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])

            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]

            im, _, _, _ = img_preprocess(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))
        

x = np.array(list(map(img_preprocess, X_train)))
print(x.shape)
x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))

def nvidia_model():
    """
    Using NVIDIA model, the normalization (img_preprocess func) is out of my model
    """

    model = Sequential()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='elu')) # conv2d_1
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu')) # conv2d_2
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu')) # conv2d_3
    model.add(Convolution2D(64, 3, 3, activation='elu')) # conv2d_4
    model.add(Convolution2D(64, 3, 3, activation='elu')) # conv2d_5
    model.add(Dropout(.9)) # dropout at 0.9
  
  
    model.add(Flatten()) # flatten
  
    model.add(Dense(100, activation = 'elu')) # dense_1
  
    model.add(Dense(50, activation = 'elu')) # dense_2
  
    model.add(Dense(10, activation = 'elu')) # dense_3
 
    model.add(Dense(1)) # dense_4
  
    optimizer = Adam(lr=1e-3) # learning rate 0.001
    model.compile(loss='mse', optimizer=optimizer)
    return model


from keras.callbacks import ModelCheckpoint


checkpointer = ModelCheckpoint(filepath='./checkpoint/weights.h5', verbose=1, save_best_only=True)
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir) # cmd = tensorboard --logdir logs



model = nvidia_model()

print(model.summary())

model.fit_generator(batch_generator(X_train, y_train, 100, 1),
                                  steps_per_epoch=300, 
                                  epochs=10,
                                  validation_data=batch_generator(X_valid, y_valid, 100, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle=1,
                                  callbacks=[checkpointer, tensorboard_callback])


model.save('model.h5') # saving model
