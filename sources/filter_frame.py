import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from cv2.cv2 import imread

yellow_th_min = np.array([108, 108, 144])
yellow_th_max = np.array([255, 255, 255])


def thresh_frame_in_LAB(frame, min_values, max_values, verbose=False):
    """
    Threshold a color frame in HSV space
    """

    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

    min_th_ok = np.all(HSV > min_values, axis=2)
    max_th_ok = np.all(HSV < max_values, axis=2)

    out = np.logical_and(min_th_ok, max_th_ok)
    h,w = frame.shape[:2]

    binary = np.zeros(shape=(h, w), dtype=np.uint8)
    binary = np.logical_or(binary, out)

    if verbose:
        plt.imshow(binary, cmap='gray')
        plt.show()

    binary = binary.astype(np.uint8)
    binary *= 255

    return binary


def birdeye(img, verbose=False):
    """
    Apply perspective transform to input frame to get the bird's eye view.
    :param img: input color frame
    :param verbose: if True, show the transformation result
    :return: warped image, and both forward and backward transformation matrices
    """
    height, width = img.shape[:2]

    w_out = 200
    bhratio = 1
    thratio = 0.408
    wtratio = 0.3
    wbratio = 1 - wtratio
    ratio = 1.6


    width += w_out

    src = np.float32([[(width - w_out) * wtratio, height * thratio],  # br
                      [(width - w_out) * wbratio, height * thratio],  # bl
                      [w_out / (-2), height * bhratio],  # tl
                      [width - w_out / 2, height * bhratio]])  # tr

    dst = np.float32([[0, 0],  # br
                      [width, 0],  # bl
                      [0, width * ratio],  # tl
                      [width, width * ratio]])  # tr

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (width, round(width * ratio)), flags=cv2.INTER_LINEAR)

    if verbose:
        cv2.namedWindow('birdeye', cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('birdeye', warped)

    return warped, M, Minv

if __name__ == '__main__':
    img = imread('sources/test1.jpg')
    bird, w, i = birdeye(img)
    resized = bird[350:1150,]

    plt.figure(1), plt.imshow(cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB))
    plt.figure(2), plt.imshow(cv2.cvtColor(bird, code=cv2.COLOR_BGR2RGB))
    plt.figure(3), plt.imshow(cv2.cvtColor(resized, code=cv2.COLOR_BGR2RGB))
    plt.show()

    # dataset_file = open('dataset.csv', 'r')
    # dataset = dataset_file.readlines()
    # dataset_file.close()
    #
    # for data in dataset:
    #     img_path, throttle, steering = data.split(',')
    #     img = cv2.imread(img_path)
    #     warped, M, Minv = birdeye(img)
    #     thresh = thresh_frame_in_LAB(warped, yellow_th_min, yellow_th_max, False)
    #     cv2.imwrite('binary/' + img_path.split('/')[1], thresh)
