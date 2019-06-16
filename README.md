[//]: # (Image References)

[video1]: ./output_vid/output_vid.mp4 "videomp4"
[video2]: https://www.youtube.com/watch?v=T9FdAELz1KA "videoYT"
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
[image12]: ./demo/tensorboardMainGraph_1.png "graph"
[link1]: "https://docs.opencv.org/ref/2.4.13.3/d0/de9/structcv_1_1gpu_1_1device_1_1color__detail_1_1RGB2YUV.html" "RGB2YUV"

# README

# Drive

`python drive.py`
### Simulator
- Simulator for Linux OS [https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
- Simulator for Mac OS [https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
- Simulator for windows OS [https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)
# Collecting Data

For my first collecting, I recorded 3 laps, but after training and test in autonomus mode I realized the car make only left turn, so to overcome this problem I need collect 3 other laps in the reverse.

In summary I did 3 laps (track 1) in both directions.


# Balancing Data
This plot indicate the number of times the steering angle are used and the angle for each one.
As we can see we have unbalance data, so we need to balance it.


![alt text][image1]

If I were to train our convolutional neural network based on this data then the model could become biased towards driving straight all the time. So I must flatten the data distribution and cut off extraneous samples for specific bins whose frequency exceed 400.

![alt text][image2]





# Augmentation Data
I use four method for augment my poor data

- ### Zoom

```python
    def zoom(image):
        zoom = iaa.Affine(scale=(1, 1.3))
        image = zoom.augment_image(image)
        return image
```
![alt text][image4]


- ### pan
```python
    def pan(image):
      pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
      image = pan.augment_image(image)
      return image
```
![alt text][image5]


- ### brigth

```python
    def img_random_brightness(image):
        brightness = iaa.Multiply((0.2, 1.2))
        image = brightness.augment_image(image)
        return image
```
![alt text][image6]



- ### flip

```python
    def flip(image, steering_angle):
        image = cv2.flip(image,1)
        steering_angle = -steering_angle
        return image, steering_angle
```
![alt text][image7]



# Preprocessing

For the preprocessing I used YUV filter and I resize it

- ### YUV  

    [Opencv Doc rgb to yuv][link1]

    ![alt text][image8] I change the color space to Y U V, it is much lighter and really easy to used

    ```
    cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    ```

- ### Resize

    ![alt text][image9]

    ```
    cv2.resize(img, (200, 66))
    ```
    I decide de resize because this will allow for faster computations as a smaller image is easier to work with.

# Batch Generator

My generator will take input data create a defined number of augmented sample images along with labels and then returns these augmented images with their respective labels.
The main benefit of the generator is that it can create augmented images on the fly rather than augmenting all my images at one time and storing them using valuable memory space.

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

![alt text][image3]
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

![alt text][image12]


# Result


[Video mp4][video1]
[YouTube Link][video2]{:target="_blank"}