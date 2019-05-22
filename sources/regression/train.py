# set the matplotlib backend so figures can be saved in the background
import csv

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam

from utils.filter_frame import birdeye
from utils.loader import generate_data_batch, split_train_val
from sources.regression.model import LaneControlNet
from sources.regression.config import *
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

    # initialize our FashionNet multi-output network
    # model = LaneControlNet.build((IMAGE_DIMS[0], IMAGE_DIMS[1]), final_act='linear')
    model = LaneControlNet.build_sequential((CONFIG['input_width'], CONFIG['input_height'], CONFIG['input_channels']), final_act='sigmoid')


    # initialize the optimizer and compile the model
    print("[INFO] compiling model...")
    # opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=["accuracy"])
    model.summary()

    # json dump of model architecture
    with open('logs/regression_{}.json'.format(args['model']), 'w') as f:
        f.write(model.to_json())

    checkpoint_path = 'checkpoints/regression_{}_weights'.format(args['model'])

    checkpointer = ModelCheckpoint(checkpoint_path + '.{epoch:02d}-{val_loss:.3f}.hdf5')
    logger = CSVLogger(filename='logs/{}_history.csv'.format(args['model']))


if args['generator']:
    # split csv data into training and validation
    train_data, val_data = split_train_val(csv_driving_data=args['dataset'])

    # train the model
    print("[INFO] training generator model...")

    H = model.fit_generator(generator=generate_data_batch(train_data, augment_data=True, bias=CONFIG['bias']),
                            steps_per_epoch=10*CONFIG['batchsize'],
                            epochs=CONFIG['epochs'],
                            validation_data=generate_data_batch(val_data, augment_data=False, bias=1.0),
                            validation_steps=3*CONFIG['batchsize'],
                            callbacks=[checkpointer, logger])
else:
    print("[INFO] loading images...")

    dataset = []
    with open(args['dataset'], 'r') as f:
        reader = csv.reader(f)
        dataset = [row for row in reader][1:]
    random.seed(42)
    random.shuffle(dataset)

    # initialize the data, clothing category labels (i.e., shirts, jeans,
    # dresses, etc.) along with the color labels (i.e., red, blue, etc.)
    data = []
    throttle_values = []
    steering_values = []

    # loop over the data of images and throttle, steering values
    for datum in dataset:
        # load the image, pre-process it, and store it in the data list
        image_path, throttle, steering = datum
        if args['debug']:
            print('image_path: {}'.format(image_path))
        throttle = int(throttle) / 100.0
        if args['debug']:
            print('throttle: {}'.format(throttle))
        steering = int(steering) / 100.0
        if args['debug']:
            print('steering: {}'.format(steering))
        image = cv2.imread(image_path)
        throttle_values.append(throttle)
        steering_values.append(steering)

        if image is None:
            print('no image is identified in path: {}'.format(image_path))
            continue

        h, w = CONFIG['input_height'], CONFIG['input_width']
        warped, M, Minv = birdeye(image)
        ah, aw = warped.shape[:2]
        cropped_bird = warped[int(ah*CONFIG['crop_start_height_ratio']):int(ah-(ah*CONFIG['crop_end_height_ratio'])),]
        img = cv2.resize(cropped_bird, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if args['debug']:
            plt.figure(1)
            plt.xlabel(image_path)
            plt.imshow(image)
            plt.figure(2)
            plt.xlabel(image_path)
            plt.imshow(warped)
            plt.figure(3)
            plt.xlabel(image_path)
            plt.imshow(img, cmap='gray')
            plt.show()

        img = img_to_array(img)
        data.append(img)

    # convert throttle and steering values into niumpy array
    throttle_values = np.array(throttle_values)
    steering_values = np.array(steering_values)

    # scale the raw pixel intensities to the range [0, 1] and convert to
    # a NumPy array
    data = np.array(data, dtype="float") / 255.0
    print("[INFO] data matrix: {} images ({:.2f}MB)".format(
        len(dataset), data.nbytes / (1024 * 1000.0)))

    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    split = train_test_split(data, throttle_values, steering_values, test_size=0.2,
                             random_state=42)
    (
    train_x, test_x, train_throttle_y, test_throttle_y, train_steering_y, test_steering_y) = split

    train_y = np.array(list(zip(train_throttle_y, train_steering_y)))
    test_y = np.array(list(zip(train_throttle_y, test_steering_y)))
    # train_y = np.array(train_throttle_y, train_steering_y)
    # test_y = np.array(train_throttle_y, test_steering_y)
    print(train_y[:10])
    print(test_y[:10])
    # train the model
    print("[INFO] training generator model...")

    H = model.fit(train_x, train_y,
                  validation_data=(test_x, test_y),
                  epochs=CONFIG['epochs'],
                  batch_size=CONFIG['batchsize'],
                  verbose=1,
                  callbacks=[checkpointer, logger])

    # save the model to disk
    print("[INFO] serializing network...")
    model.save('models/regression_{}.h5'.format(args["model"]))

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
    plt.savefig("results/regression_{}_model_losses.png".format(args["result"]))
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
    plt.savefig("results/regression_{}_model_accs.png".format(args["result"]))
    plt.close()
