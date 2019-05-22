# set the matplotlib backend so figures can be saved in the background
import csv
import os
import pickle

from imutils import paths
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import LabelBinarizer

from utils.filter_frame import birdeye
from utils.loader import generate_data_batch, split_train_val
from sources.classification.model import TrafficSignNet
from sources.classification.config import *
import matplotlib.pyplot as plt
# import the necessary packages
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import random
import cv2


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset (i.e., directory of images)")
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-c", "--converted", required=True,
                    help="path to converted output weights model")
    ap.add_argument("-r", "--result", type=str, default="result",
                    help="base filename for generated plots and results")
    ap.add_argument("--generator", dest='generator', action='store_true',
                    help="Use generator")
    ap.add_argument("--debug", dest='debug', action='store_true',
                    help="Debug options")
    args = vars(ap.parse_args())

    # initialize our TrafficSignNet multi-output network
    model = TrafficSignNet.build((CONFIG['input_width'], CONFIG['input_height'], CONFIG['input_channels']), CONFIG['number_of_classes'])

    # initialize the optimizer and compile the model
    print("[INFO] compiling model...")
    # opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    opt = SGD(lr=CONFIG['learning_rate'], decay=CONFIG['learning_rate'] / CONFIG['epochs'], momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', loss_weights={'traffic_sign_output': 1.0}, metrics=["accuracy"])
    model.summary()

    # json dump of model architecture
    with open('logs/traffic_{}.json'.format(args['model']), 'w') as f:
        f.write(model.to_json())

    checkpoint_path = 'checkpoints/traffic_{}_weights'.format(args['model'])

    checkpointer = ModelCheckpoint(checkpoint_path + '.{epoch:02d}-{val_loss:.3f}.hdf5')
    logger = CSVLogger(filename='logs/traffic_{}_history.csv'.format(args['model']))

    print("[INFO] loading images...")

    imagePaths = sorted(list(paths.list_images(args['dataset'])))
    random.seed(42)
    random.shuffle(imagePaths)

    data = []
    class_labels = []

    # loop over the data of images and throttle, steering values
    # loop over the input images
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (CONFIG['input_width'], CONFIG['input_height']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img_to_array(image)
        data.append(image)
        folders = imagePath.split(os.path.sep)[:-1]
        category = folders[-1]
        class_labels.append(category)


    # scale the raw pixel intensities to the range [0, 1] and convert to
    # a NumPy array
    data = np.array(data, dtype='float') / 255.0


    print("[INFO] data matrix: {} images ({:.2f}MB)".format(
        len(data), data.nbytes / (1024 * 1000.0)))

    class_labels = np.array(class_labels)

    print('[INFO] binarizing labels...')
    class_label = LabelBinarizer()
    class_labels = class_label.fit_transform(class_labels)

    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    split = train_test_split(data, class_labels, test_size=0.2,
                             random_state=42)

    (train_x, test_x, train_y, test_y) = split


    print("[INFO] training generator model...")
    H = model.fit(train_x, train_y,
                  validation_data=(test_x, test_y),
                  epochs=CONFIG['epochs'],
                  batch_size=CONFIG['batchsize'],
                  verbose=1,
                  callbacks=[checkpointer, logger])

    # save the model to disk
    print("[INFO] serializing network...")
    model.save('models/classification_{}.h5'.format(args["model"]))

    # save the category binarizer to disk
    print("[INFO] serializing category label binarizer...")
    f = open('models/classification_{}.pickle'.format(args["model"]), "wb")
    f.write(pickle.dumps(class_label))
    f.close()

    # converting the model to weights file and saving to disk
    print("[INFO] converting network...")
    # export_model(model, args["converted"])

    # print("[INFO] visualizing model graph...")
    # from tensorflow.python.keras.utils import model_to_dot
    # pydot: Dot = model_to_dot(model, rankdir='LR', show_shapes=True)
    # pydot.write('{}_model_graph.png'.format(args['result']), format='png')

    # plot the total loss, throttle loss, steering_right loss and steering_left loss
    print("[INFO] visualizing losses and accuracies over epochs...")
    lossNames = ['loss']
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots()

    # loop over the loss names
    for (i, l) in enumerate(lossNames):
        # plot the loss for both the training and validation data
        title = "Loss for {}".format(l) if l != "loss" else "Total loss"
        ax.set_title(title)
        ax.set_xlabel("Epoch #")
        ax.set_ylabel("Loss")
        ax.plot(np.arange(0, CONFIG['epochs']), H.history[l], label=l)
        ax.plot(np.arange(0, CONFIG['epochs']), H.history["val_" + l],
                   label="val_" + l)
        ax.legend()

    # save the losses figure and create a new figure for the accuracies
    plt.tight_layout()
    plt.savefig("results/classification_{}_model_losses.png".format(args["result"]))
    plt.close()

    # create a new figure for the accuracies
    accuracyNames = ['acc']
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots()

    # loop over the accuracy names
    for (i, l) in enumerate(accuracyNames):
        # plot the loss for both the training and validation data
        ax.set_title("Accuracy for {}".format(l))
        ax.set_xlabel("Epoch #")
        ax.set_ylabel("Accuracy")
        ax.plot(np.arange(0, CONFIG['epochs']), H.history[l], label=l)
        ax.plot(np.arange(0, CONFIG['epochs']), H.history["val_" + l],
                   label="val_" + l)
        ax.legend()

    # save the accuracies figure
    plt.tight_layout()
    plt.savefig("results/classification_{}_model_accs.png".format(args["result"]))
    plt.close()
