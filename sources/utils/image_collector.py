import csv
from uuid import uuid4
import os
from typing import List, AnyStr

import cv2
import glob

from imutils import paths


regions = []

crop_area = (250, 150)
classifiers = ['green_traffic', 'red_traffic', 'left', 'right', 'stop', 'pedestrian', 'parking']

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

def on_classifier_button_change(counter):
    cv2.displayOverlay('image', classifiers[counter], 500)
    global classifier
    classifier = classifiers[counter]

def on_mouse(event, x, y, flags, params):
    global regions
    if event == cv2.EVENT_LBUTTONUP:
        if len(regions) == 2:
            return
        regions.append([x, y])
        print('point - {}'.format(regions))


imagePaths = sorted(list(paths.list_images('signs_ready')))
saved_images_path = 'saved_images.txt'


cv2.setMouseCallback('image', on_mouse)
# cv2.createTrackbar('PLAYBACK', 'image', 1, 100, len(imagePaths))
cv2.createTrackbar('Classifier', 'image', 0, len(classifiers)-1, on_classifier_button_change)
saved_file = open(saved_images_path, 'a')

dataset_file = open('saved_images.txt', 'r')
dataset = dataset_file.readlines()
dataset_file.close()

for image_path in imagePaths:
    if image_path in list(map(lambda dat: dat.split(',')[0], dataset)):
        continue
    saved = False
    playing = True
    starting_frame = 0
    ending_frame = 0
    option = 0
    frame = cv2.imread(image_path)
    save = frame.copy()
    while True:
        if len(regions) == 2:
            cv2.rectangle(frame, tuple(regions[0]), tuple(regions[1]), (0,255,0), thickness=3)
            cv2.imshow('image', frame)
        else:
            cv2.imshow('image', save)
        option = cv2.waitKey(15) & 0xFF
        if option == ord('n'):
            break
        if option == ord('u'):
            print('regions clear')
            regions = list()
        if option == ord('o'):
            if len(regions) == 2:
                cropped = frame[regions[0][1]:regions[1][1], regions[0][0]:regions[1][0]]
                cv2.imshow('cropped', cropped)
                save = cv2.waitKey() & 0xFF
                if save == ord('o'):
                    saved_file = open(saved_images_path, 'a')
                    message = '{},{},{},{},{},{}\n'.format(image_path, regions[0][0], regions[0][1], regions[1][0], regions[1][1], image_path.split(os.path.sep)[-2])
                    print(message)
                    saved_file.write(message)
                    saved_file.close()
                    print('{} saved'.format(image_path))
                    saved = True
        if saved or option == ord('n'):
            saved = False
            break
        # first time

print('finished')
