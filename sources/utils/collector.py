import csv
from uuid import uuid4
import os
from typing import List, AnyStr

import cv2
import glob



video = None
frame_counter = -1

IMAGE_CLASSIFICATION = True

regions = [[250, 150]]

crop_area = (250, 150)
classifiers = ['green_traffic', 'red_traffic', 'left', 'right', 'up', 'stop', 'pedestrian', 'round', 'parking', 'woman']
classifier = classifiers[0]
image = None
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

def on_classifier_button_change(counter):
    cv2.displayOverlay('image', classifiers[counter], 500)
    global classifier
    classifier = classifiers[counter]

def on_mouse(event, x, y, flags, params):
    global region, image_frame

    if event == cv2.EVENT_LBUTTONUP:
        regions[0] = [x, y]
        print('point - {}'.format(regions[0]))

def on_playback_slidebar_change(frame):
    global frame_counter
    frame_counter = frame
    video.set(cv2.CAP_PROP_POS_FRAMES, frame)

def read_lines(lines: List[AnyStr], starting_line_number: int, ending_line_number: int):
    for index, line in enumerate(lines):
        if index >= starting_line_number:
            yield line
        if index > ending_line_number:
            break

video_files_path = glob.glob('signs/*.avi')
ready_folder = 'signs_ready'
data_save_path = 'signs.csv'
saved_videos_path = 'saved_videos.txt'


saved_video_lines = list()
if os.path.exists(saved_videos_path):
    saved_videos_file = open(saved_videos_path, 'r')
    saved_video_lines = saved_videos_file.readlines()
    saved_videos_file.close()
saved_video_lines = list(map(lambda s: s.rstrip(), saved_video_lines))

cv2.setMouseCallback('image', on_mouse)
cv2.createTrackbar('PLAYBACK', 'image', 1, 100, on_playback_slidebar_change)
cv2.createTrackbar('Classifier', 'image', 0, len(classifiers)-1, on_classifier_button_change)

for video_file_path in video_files_path:
    if video_file_path in saved_video_lines:
        continue
    data_list = list()
    saved = False
    if not IMAGE_CLASSIFICATION:
        vehicle_control_file_path = video_file_path.split('.')[0] + '.txt'
        vehicle_control_file = open(vehicle_control_file_path, 'r')
        lines = vehicle_control_file.readlines()
        vehicle_control_file.close()
    frame_counter = -1
    video = cv2.VideoCapture(video_file_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, 1)
    video.set(cv2.CAP_PROP_FPS, 10)
    number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.setTrackbarMax('PLAYBACK', 'image', number_of_frames)
    cv2.setTrackbarPos('PLAYBACK', 'image', 1)
    playing = True
    starting_frame = 0
    ending_frame = 0
    option = 0
    while True:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
        video.grab()
        flag, frame = video.retrieve()
        cv2.setTrackbarPos('PLAYBACK', 'image', frame_counter)
        if not flag or frame is None:
            frame_counter -= 1
        if flag:
            if len(regions) == 1:
                second = tuple([regions[0][0] + crop_area[1], regions[0][1] + crop_area[0]])
                cv2.rectangle(frame, tuple(regions[0]), second, (0,255,0), thickness=3)
            cv2.imshow('image', frame)
        option = cv2.waitKey(15) & 0xFF
        if playing:
            frame_counter += 1
        if option == ord('p'):
            playing = not playing
            print('playing: {}'.format(playing))
        if option == ord('s'):
            starting_frame = frame_counter
            print('starting_frame: {}'.format(starting_frame))
            if not IMAGE_CLASSIFICATION:
                print(lines[starting_frame])
        if option == ord('n'):
            break
        if option == ord('6'):
            if frame_counter < number_of_frames:
                frame_counter += 1
        if option == ord('5'):
            if frame_counter > 0:
                frame_counter -= 1
        if option == ord('e'):
            ending_frame = frame_counter
            print('ending_frame: {}'.format(ending_frame))
            print(lines[ending_frame])
        if option == ord('r'):
            if starting_frame < ending_frame:
                video.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
                found_lines = read_lines(lines, starting_frame, ending_frame)
                for i in range(starting_frame, ending_frame+1):
                    video.grab()
                    flag, image = video.retrieve()
                    data_img_path = 'signs_ready/' + str(uuid4()) + '.jpg'
                    cv2.imwrite(data_img_path, image)
                    steering, throttle = lines[i].split(',')
                    data_list.append([data_img_path, int(steering), int(throttle)])
                with open(data_save_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(data_list)
                    saved = True
                    data_list = list()
                file.close()
                print('file: {} saved with starting_frame - {} and ending_frame - {}'.format(video_file_path, starting_frame, ending_frame))
        if option == ord('o'):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
            video.grab()
            flag, image = video.retrieve()
            if len(regions) == 1:
                cropped_img = image[regions[0][1]:regions[0][1]+crop_area[0], regions[0][0]:regions[0][0]+crop_area[1], ]
                print(cropped_img.shape)
                cv2.imshow('cropped', cropped_img)
                save = cv2.waitKey() & 0xFF
                if save == ord('o'):
                    if not os.path.exists('{}/{}'.format(ready_folder, classifier)):
                        os.makedirs('{}/{}'.format(ready_folder, classifier))
                    data_img_path = '{}/{}/{}.jpg'.format(ready_folder, classifier, str(uuid4()))
                    cv2.imwrite(data_img_path, cropped_img)
                    saved = True
                    print('frame {} saved: {}'.format(frame_counter, data_img_path))
    if saved or option == ord('n'):
        saved_videos_file = open(saved_videos_path, 'a')
        if len(saved_video_lines) >= 1 and all(saved_line != video_file_path for saved_line in saved_video_lines):
            saved_videos_file.write(video_file_path + '\n')
        # first time
        elif len(saved_video_lines) == 0:
            saved_videos_file.write(video_file_path + '\n')
        saved_videos_file.close()
    video.release()



print('finished')
