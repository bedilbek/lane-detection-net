# set the matplotlib backend so figures can be saved in the background
import argparse
from time import time

import tensorflow as tf
import matplotlib
from tensorflow.python.keras.preprocessing.image import img_to_array

matplotlib.use("Agg")
import numpy as np
import cv2 as cv
from tensorflow.python.keras.models import load_model

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to input model")
ap.add_argument("-d", "--dataset", type=str, required=True,
                help="Dataset to compare with predictions")
ap.add_argument("-p", "--image_path", type=str, required=True,
                help="Dataset to compare with predictions")
ap.add_argument("--debug", dest='debug', action='store_true',
                help="Debug options")
args = vars(ap.parse_args())

model = load_model(args['model'])

dataset_file = open(args['dataset'], 'r')
dataset = dataset_file.readlines()
dataset_file.close()


image_path = args['image_path']
truth_results = [value for value in dataset if value.split(',')[0] == image_path]
throttle, steering = list(map(lambda t: int(t.rstrip()), truth_results[0].split(',')[1:]))
image = cv.imread(image_path)
h, w = image.shape[:2]
image = image[int(h * 0.5):, :]
image = cv.resize(image, (256, 256))
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('frame', image)
image = img_to_array(image)
image = np.array(image, dtype="float") / 255.0
image = np.expand_dims(image, axis=0)

now = time()
print(now)
dd = model.predict(image)
then = time()
print(then)
print('spent time: {}'.format(then-now))
print("throttle: {}, steering: {}".format(throttle, steering))
print('predicted_throttle: {}, predicted_steering_left: {}, predicted_steering_rigt: {}'.format(int(dd[0][0]*100), int(dd[0][1]*100), int(dd[0][2]*-1*100)))
cv.waitKey()
