import cv2
import numpy as np
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import keras.backend as K
from config import *
from utils.filter_frame import birdeye


def split_train_val(csv_driving_data, test_size=0.2):
    """
    Splits the csv containing driving data into training and validation
    :param csv_driving_data: file path of Udacity csv driving data
    :return: train_split, validation_split
    """
    with open(csv_driving_data, 'r') as f:
        reader = csv.reader(f)
        driving_data = [row for row in reader][1:]

    train_data, val_data = train_test_split(driving_data, test_size=test_size, random_state=1)

    return train_data, val_data


def preprocess(frame_bgr, verbose=False, bird_eye=True):
    """
    Perform preprocessing steps on a single bgr frame.
    These inlcude: cropping, resizing, eventually converting to grayscale
    :param frame_bgr: input color frame in BGR format
    :param verbose: if true, open debugging visualization
    :return:
    """
    # set training images resized shape
    h, w = CONFIG['input_height'], CONFIG['input_width']

    if bird_eye:
        bird, M, Minv = birdeye(frame_bgr)
    else:
        bird = frame_bgr

    # crop image (remove useless information)
    frame_cropped = bird[CONFIG['crop_height'], :, :]

    # resize image
    frame_resized = cv2.resize(frame_cropped, dsize=(w, h))

    # eventually change color space
    if CONFIG['input_channels'] == 1:
        frame_resized = np.expand_dims(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)[:, :, 0], 2)

    if verbose:
        plt.figure(1), plt.imshow(cv2.cvtColor(frame_bgr, code=cv2.COLOR_BGR2RGB))
        plt.figure(2), plt.imshow(cv2.cvtColor(bird, code=cv2.COLOR_BGR2RGB))
        plt.figure(3), plt.imshow(cv2.cvtColor(frame_cropped, code=cv2.COLOR_BGR2RGB))
        plt.figure(4), plt.imshow(cv2.cvtColor(frame_resized, code=cv2.COLOR_BGR2RGB))
        plt.show()

    return frame_resized.astype('float32')


def load_data_batch(data, batchsize=CONFIG['batchsize'], data_dir='data', augment_data=True, bias=0.5):
    """
    Load a batch of driving data from the "data" list.
    A batch of data is constituted by a batch of frames of the training track as well as the corresponding
    steering directions.
    :param data: list of training data in the format provided by Udacity
    :param batchsize: number of elements in the batch
    :param data_dir: directory in which frames are stored
    :param augment_data: if True, perform data augmentation on training data
    :param bias: parameter for balancing ground truth distribution (which is biased towards steering=0)
    :return: X, Y which are the batch of input frames and steering angles respectively
    """
    # set training images resized shape
    h, w, c = CONFIG['input_height'], CONFIG['input_width'], CONFIG['input_channels']

    # prepare output structures
    X = np.zeros(shape=(batchsize, h, w, c), dtype=np.float32)
    y_steer = np.zeros(shape=(batchsize,), dtype=np.float32)
    y_throttle = np.zeros(shape=(batchsize,), dtype=np.float32)

    # shuffle data
    shuffled_data = shuffle(data)

    loaded_elements = 0
    while loaded_elements < batchsize:

        path, throttle, steer = shuffled_data.pop()

        # cast strings to float32
        steer = np.float32(steer) / 100.0
        throttle = np.float32(throttle) / 100.0

        frame = preprocess(cv2.imread(path.strip()))

        if augment_data:

            # mirror images with chance=0.5
            if random.choice([True, False]):
                frame = frame[:, ::-1, :]
                steer *= -1.

            # if color images, randomly change brightness
            if CONFIG['input_channels'] == 3:
                frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2HSV)
                frame[:, :, 2] *= random.uniform(CONFIG['augmentation_value_min'], CONFIG['augmentation_value_max'])
                frame[:, :, 2] = np.clip(frame[:, :, 2], a_min=0, a_max=255)
                frame = cv2.cvtColor(frame, code=cv2.COLOR_HSV2BGR)

        X[loaded_elements] = frame
        y_steer[loaded_elements] = steer
        y_throttle[loaded_elements] = throttle
        loaded_elements += 1

    if K.backend() == 'theano':
        X = X.transpose(0, 3, 1, 2)

    return X, y_steer, y_throttle


def generate_data_batch(data, batchsize=CONFIG['batchsize'], data_dir='data', augment_data=True, bias=0.5):
    """
    Generator that indefinitely yield batches of training data
    :param data: list of training data in the format provided by Udacity
    :param batchsize: number of elements in the batch
    :param data_dir: directory in which frames are stored
    :param augment_data: if True, perform data augmentation on training data
    :param bias: parameter for balancing ground truth distribution (which is biased towards steering=0)
    :return: X, Y which are the batch of input frames and steering angles respectively
    """
    while True:

        X, y_steer, y_throttle = load_data_batch(data, batchsize, data_dir, augment_data, bias)

        yield X, y_steer


if __name__ == '__main__':

    # debugging purpose
    train_data, val_data = split_train_val(csv_driving_data='data/driving_log.csv')
