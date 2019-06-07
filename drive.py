import os
import socketio
import eventlet
import numpy as np
from datetime import datetime
from flask import Flask
from tensorflow.python.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
 
sio = socketio.Server()
 
app = Flask(__name__) #'__main__'
speed_limit = 50
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
 
 
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    imageDecode = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(imageDecode)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('ANGLE:{} THROTTLE:{:.3f} SPEED:{:.1f}'.format(steering_angle, throttle*100, speed))
    send_control(steering_angle, throttle)
    # save frame
    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    image_filename = os.path.join('./output_vid/', timestamp)
    #imageDecode.save('{}.jpg'.format(image_filename))
 
 
 
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)
 
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
 
 
if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
