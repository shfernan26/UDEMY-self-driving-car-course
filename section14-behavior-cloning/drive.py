import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()

app = Flask(__name__) #'__main__'
speed_limit = 10

def img_preprocess(img):
  img = img[ 60:135, : , : ] # cropping out top (sky) and bottom (car hood) of image
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) # color scheme recommended for NVIDIA neural model used later on
  img = cv2.GaussianBlur(img, (3,3), 0) # reduce noise, focus on important features
  img = cv2.resize(img, (200, 66)) # matches input size of images used by NVIDIA model
  img = img/255 # normalize pixel intensities to reduce deviation
  return img

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image']))) # simulator sends data that is base 64 encoded by default
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

@sio.on('connect') # connect, message, disconnect are reserved sio handler names
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle':steering_angle.__str__(),
        'throttle': throttle.__str__()
    })



if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)