import glob
import random
from uuid import uuid4

import cv2, os
import numpy as np
import matplotlib.image as mpimg
from imutils import paths
from skimage import transform
from skimage import util

from keras_preprocessing.image import img_to_array

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image):
    """
    Randomly flipt the image left <-> right
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    return image

def random_rotation(image):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    rotated = transform.rotate(image, random_degree)
    rotated *= 255
    return rotated.astype(np.uint8)

def random_noise(image):
    # add random noise to the image
    noise = util.random_noise(image)
    noise *= 255
    return noise.astype(np.uint8)



def random_translate(image, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    return cv2.warpAffine(image, trans_m, (width, height))


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    h, w = image.shape[:2]
    x1, y1 = w * np.random.rand(), 0
    x2, y2 = w * np.random.rand(), h
    xm, ym = np.mgrid[0:h, 0:w]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] = hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(image, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    # image = random_flip(image)
    translated = random_translate(image, range_x, range_y)
    # cv2.imshow('translated', translated)
    rotated = random_rotation(image)
    # cv2.imshow('rotated', rotated)
    noised = random_noise(image)
    # cv2.imshow('noised', noised)
    shadowed = random_shadow(image)
    # cv2.imshow('shadowed', shadowed)
    brighted = random_brightness(image)
    return [translated, rotated, noised, shadowed, brighted]


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers


if __name__ == '__main__':
    imagePaths = sorted(list(paths.list_images('test')))
    categoryLabels = []

    saved_pictures_path = 'augmented_pictures.txt'

    saved_picture_path_lines = list()
    if os.path.exists(saved_pictures_path):
        saved_videos_file = open(saved_pictures_path, 'r')
        saved_picture_path_lines = saved_videos_file.readlines()
        saved_videos_file.close()
    saved_picture_path_lines = list(map(lambda s: s.rstrip(), saved_picture_path_lines))

    saved_videos_file = open(saved_pictures_path, 'a')

    # loop over the input images
    for imagePath in imagePaths:
        if imagePath in saved_picture_path_lines:
            continue
        data = []
        image = cv2.imread(imagePath)
        data.append(image)
        folders = imagePath.split(os.path.sep)[:-1]
        folders[0] = 'augmented'
        folder_path = os.path.sep.join(folders)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for i in range(10):
            augmenteds = augument(image)
            data = data + augmenteds
        for augmented in data:
            img_path = '{}/{}.jpg'.format(folder_path, str(uuid4()))
            cv2.imwrite(img_path, augmented)
        print('{} finished'.format(imagePath))
        saved_videos_file.write(imagePath + '\n')
        category = folders[-1]
        # cv2.imshow('org', image)
        # cv2.imshow('augmented', augmented)
        # cv2.waitKey()
