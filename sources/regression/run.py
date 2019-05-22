# set the matplotlib backend so figures can be saved in the background
import socket

from cv2.cv2 import cvtColor, COLOR_RGB2GRAY

from sources.regression.config import *
from utils.filter_frame import birdeye

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 8059)
sock.connect(server_address)
import matplotlib.pyplot as plt

USE_JIT = True
NUMBER_OF_TRIALS = 30
sum_of_trials = 0
IMAGE_DIMENSIONS = (CONFIG['input_width'], CONFIG['input_height'])
import argparse

from tensorflow.python.keras.preprocessing.image import img_to_array
import numpy as np
import cv2 as cv
import os
if os.uname()[1] == 'sqldeveloper':
    from picamera.array import PiRGBArray
    from picamera import PiCamera
from tensorflow.python.keras.models import load_model
import time

def predict_image(model, image, image_path=None, true_values=None):
    h, w = image.shape[:2]

    warped, M, Minv = birdeye(image)
    ah, aw = warped.shape[:2]
    cropped_bird = warped[int(ah * CONFIG['crop_start_height_ratio']):int(ah - (ah * CONFIG['crop_end_height_ratio'])), ]
    image = cv.resize(cropped_bird, IMAGE_DIMENSIONS)
    image = cvtColor(image, COLOR_RGB2GRAY)
    save = image.copy()
    image = img_to_array(image)
    image = np.array(image, dtype="float") / 255.0
    image = np.expand_dims(image, axis=0)
    now = time.time()
    predictions = model.predict(image)
    then = time.time()
    if args['debug']:
        plt.figure(1)
        # if image_path is not None:
        #     plt.xlabel(image_path)
        if true_values is not None:
            plt.xlabel('{} {}\n{} {}'.format(true_values[0], true_values[1], int(predictions[0][0]*100), int(predictions[0][1]*100)), )
        plt.imshow(save, cmap='gray')
        plt.show()
    global sum_of_trials
    difference = then - now
    sum_of_trials = sum_of_trials + difference
    print('spent time: {:0.2f}'.format(difference))
    print('predicted_throttle: {}, predicted_steering: {}'.format(int(predictions[0][0] * 100), int(predictions[0][1] * 100)))
    return predictions


def run_through_camera(model):
    camera = PiCamera()
    camera.resolution = (640, 480)
    rawCapture = PiRGBArray(camera, size=(640, 480))
    time.sleep(2)
    try:
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            predictions = predict_image(model, frame.array)
            message = ""
            message = message + chr(1)
            message = message + chr(int(predictions[0][0]* 100))
            left = int(predictions[0][1] *100)
            right = int(predictions[0][2] *100)
            direction = 0 if left > right else 1
            message = message + chr(direction)
            steering = left if direction == 0 else right
            message = message + chr(steering)
            print('sending {}'.format(message))
            if CONNECT_TO_SOCKET:
                sent = sock.sendto(bytes(message, encoding='ascii'), server_address)
            key = cv.waitKey(1) & 0xFF
            rawCapture.truncate(0)
            if key == ord('q'):
                break
    finally:
        if CONNECT_TO_SOCKET:
            print('closing socket')
            sock.close()


def run_through_images(model):
    dataset_file = open(args['dataset'], 'r')
    dataset = dataset_file.readlines()
    dataset_file.close()

    dataset = np.array(dataset)
    test_data = np.random.choice(dataset, NUMBER_OF_TRIALS)
    try:
        for test_datum in test_data:
            image_path, throttle, steering = test_datum.split(',')
            print('image_path: {}'.format(image_path))
            throttle = int(throttle.rstrip())
            steering = int(steering.rstrip())
            image = cv.imread(image_path)
            predictions = predict_image(model, image, image_path, (throttle, steering))
            print("throttle: {}, steering: {}".format(throttle, steering))
            message = ""
            message = message + chr(1)
            message = message + chr(int(predictions[0][0] * 100) )
            left = int(predictions[0][1] *100)

            #TODO do not forget to change axis of right prediction
            right = int(predictions[0][1] *100)
            direction = 0 if left > right else 1
            message = message + chr(direction)
            steering = left if direction == 0 else right
            message = message + chr(steering)
            print('sending {}'.format(message))
            print('\n\n')
            if CONNECT_TO_SOCKET:
                sent = sock.sendto(bytes(message, encoding='ascii'), server_address)
            key = cv.waitKey(500) & 0xFF
            if key == ord('q'):
                break
    finally:
        if CONNECT_TO_SOCKET:
            print('closing socket')
            sock.close()


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to input model")
    ap.add_argument("-d", "--dataset", type=str, required=True,
                    help="Dataset to compare with predictions")
    ap.add_argument("--debug", dest='debug', action='store_true',
                    help="Debug options")
    ap.add_argument("--socket", dest='socket', action='store_true',
                    help="Debug options")
    args = vars(ap.parse_args())

    CONNECT_TO_SOCKET = args['socket']

    # Create a UDP socket

    model = load_model('models/{}.h5'.format(args['model']))

    if args['dataset'] != '0':
        run_through_images(model)
        print("average prediction time over {} number of trials: {:0.2f}".format(NUMBER_OF_TRIALS, float(sum_of_trials) / float(NUMBER_OF_TRIALS)))
    else:
        run_through_camera(model)
