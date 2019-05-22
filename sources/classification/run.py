# set the matplotlib backend so figures can be saved in the background
import random
import socket
from binascii import unhexlify
from io import BytesIO

from cv2.cv2 import cvtColor, COLOR_RGB2GRAY
from imutils import paths
from numpy import uint8

from sources.classification.config import *
from sources.utils.filter_frame import birdeye
from PIL import Image
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 8059)
sock.bind(server_address)
import io
import matplotlib.pyplot as plt

classifiers = ['green_traffic', 'left', 'parking', 'pedestrian', 'red_traffic', 'right', 'stop', 'woman']

LEFT = 0
CENTER = 1
RIGHT = 2

FRAME_PROPERTIES = (100, 55, 3)
SOCKET_BUFFER_SIZE = 49500
FRAME_SIZE = 16500

USE_JIT = True
NUMBER_OF_TRIALS = 30
sum_of_trials = 0
IMAGE_DIMENSIONS = (CONFIG['input_width'], CONFIG['input_height'])
import argparse

from tensorflow.python.keras.preprocessing.image import img_to_array
import numpy as np
import cv2 as cv

import os
from tensorflow.python.keras.models import load_model
import time

def predict_image(model=None, images=None, path=None, many=False):
    if not many:
        images = [images]
    inp = []
    for image in images:
        image = cv.resize(image, (CONFIG['input_width'], CONFIG['input_height']))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        save = image
        image = img_to_array(image)
        image = np.array(image, dtype="float") / 255.0
        inp.append(image)
    inp = np.array(inp)
    now = time.time()
    predictions = model.predict(inp)
    then = time.time()
    if args['debug']:
        plt.figure(1)
        # if image_path is not None:
        #     plt.xlabel(image_path)
        plt.xlabel('{} {}'.format(classifiers[predictions[0].argmax()], predictions[0][predictions[0].argmax()]))
        plt.imshow(save, cmap='gray')
        plt.show()
    global sum_of_trials
    difference = then - now
    sum_of_trials = sum_of_trials + difference
    print('spent time: {:0.2f}'.format(difference))
    print('{}  {} {}'.format(path, classifiers[predictions[0].argmax()], predictions[0][predictions[0].argmax()]))
    return predictions


def run_through_socket(model):
    try:
        print('receiving from Sher')
        while True:
            data, address = sock.recvfrom(SOCKET_BUFFER_SIZE)
            left = data[0:FRAME_SIZE]
            left = np.array(list(left))
            left = left.reshape((100, 55, 3))
            left = left.astype(uint8)
            center = data[FRAME_SIZE:FRAME_SIZE+FRAME_SIZE]
            center = np.array(list(center))
            center = center.reshape((100, 55, 3))
            center = center.astype(uint8)
            right = data[FRAME_SIZE+FRAME_SIZE:]
            right = np.array(list(right))
            right = right.reshape((100, 55, 3))
            right = right.astype(uint8)
            inp = np.array([left, center, right])
            assert model is not None
            now = time.time()
            predictions = predict_image(model, inp, many=True)
            l_max = predictions[0].argmax()
            c_max = predictions[1].argmax()
            r_max = predictions[2].argmax()

            data = bytes('{}{}{}'.format(l_max, c_max, r_max).encode('ascii'))
            # predictions = [list(map(lambda v: '{0:.2f}'.format(v),p)) for p in predictions]
            then = time.time()
            print('spent time: {}'.format(then - now))
            cv.imshow('left', left)
            cv.displayStatusBar('left', '{} {}'.format(classifiers[predictions[0].argmax()], predictions[0][predictions[0].argmax()]))
            cv.imshow('right', right)
            cv.displayStatusBar('right', '{} {}'.format(classifiers[predictions[2].argmax()], predictions[2][predictions[2].argmax()]))
            cv.imshow('center', center)
            cv.displayStatusBar('center', '{} {}'.format(classifiers[predictions[1].argmax()], predictions[1][predictions[1].argmax()]))
            cv.waitKey(15)

            sock.sendto(data, address)
    finally:
        if CONNECT_TO_SOCKET:
            print('closing socket')
            sock.close()


def run_through_images(model):
    imagePaths = sorted(list(paths.list_images(args['dataset'])))



    test_data = np.random.choice(imagePaths, NUMBER_OF_TRIALS)
    try:
        for test_datum in test_data:
            image = cv.imread(test_datum)
            predictions = predict_image(model, image, test_datum)
            print(test_datum)
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
    ap.add_argument("-m", "--model", required=False,
                    help="path to input model")
    ap.add_argument("-d", "--dataset", type=str, required=True,
                    help="Dataset to compare with predictions")
    ap.add_argument("--debug", dest='debug', action='store_true',
                    help="Debug options")
    ap.add_argument("--socket", dest='socket', action='store_true',
                    help="Debug options")
    args = vars(ap.parse_args())

    CONNECT_TO_SOCKET = args['socket']

    model = None
    # Create a UDP socket
    if args['model'] != None:
        model = load_model('models/{}.h5'.format(args['model']))

    if args['dataset'] != '0':
        run_through_images(model)
        print("average prediction time over {} number of trials: {:0.2f}".format(NUMBER_OF_TRIALS, float(sum_of_trials) / float(NUMBER_OF_TRIALS)))
    else:
        run_through_socket(model)
