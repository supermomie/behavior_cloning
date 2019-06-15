[//]: # (Image References)

[image1]: ./demo/plthist.png "Collecte"
[image2]: ./demo/plthistremoved.png "Balance"
[image3]: ./demo/trainandvalid.png "Train&Valid"
[image4]: ./demo/zoomed.png "zoom"
[image5]: ./demo/panned.png "panned"
[image6]: ./demo/bright.png "brightness"
[image7]: ./demo/flipped.png "flipped"
[image8]: ./demo/YUV.png "YUV"
[image9]: ./demo/resizeandpreproc.png "resizeandpreproc"
[image10]: ./demo/generator.png "generator"
[image11]: ./demo/summary.png "summary"


# README

# Drive

`python drive.py`
### Simulator
- Simulator for Linux OS [https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
- Simulator for Mac OS [https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
- Simulator for windows OS [https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)
# Collecting Data
For my collecting data I did 3 laps (track 1) in both directions.

![alt text][image1]
# Balancing Data

If I were to train our convolutional neural network based on this data then the model could become biased towards driving straight all the time. So I must flatten the data distribution and cut off extraneous samples for specific bins whose frequency exceed 400.

![alt text][image2]

# Training & Validation Split

![alt text][image3]

# Augmentation Data

- ### Zoom
    ![alt text][image4]
- ### pan
    ![alt text][image5]
- ### brigth
    ![alt text][image6]
- ### flip
    ![alt text][image7]

# Preprocessing
- ### YUV
    ![alt text][image8]
- ### Resize
    ![alt text][image9]

# Batch Generator

```python
def batch_generator(image_paths, steering_ang, batch_size, istraining):
  
  while True:
    batch_img = []
    batch_steering = []
    
    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths) - 1)
      
      if istraining:
        im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
     
      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_ang[random_index]
      
      im, _, _, _ = img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering))  

x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))

```

![alt text][image10]

# NVIDIA model


```python
def nvidia_model():
    
    model = Sequential()
    
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))

    model.add(Flatten())
  
    model.add(Dense(100, activation = 'elu'))
    model.add(Dense(50, activation = 'elu'))
    model.add(Dense(10, activation = 'elu'))
    
    model.add(Dense(1))
    
    optimizer = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model
```


![alt text][image11]


# Result


[YouTube Link](https://www.youtube.com/watch?v=T9FdAELz1KA)