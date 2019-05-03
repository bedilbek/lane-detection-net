# set the matplotlib backend so figures can be saved in the background
import locale

from tensorflow.python.keras.optimizer_v2.adam import Adam

from model import LaneControlNet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import the necessary packages
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import random
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="output",
                help="base filename for generated plots")
ap.add_argument("--debug", dest='debug', action='store_true',
                help="Debug options")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 20
INIT_LR = 1e-3
BS = 16
IMAGE_DIMS = (256, 256, 1)

print("[INFO] loading images...")


dataset_file = open(args['dataset'], 'r')
dataset = dataset_file.readlines()
dataset_file.close()
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
    image_path, throttle, steering = list(map(lambda x: x.rstrip(), datum.split(',')))
    if args['debug']:
        print('image_path: {}'.format(image_path))
    throttle = int(throttle)
    if args['debug']:
        print('throttle: {}'.format(throttle))
    steering = int(steering)
    if args['debug']:
        print('steering: {}'.format(steering))
    image = cv2.imread(image_path)
    throttle_values.append(throttle)
    steering_values.append(steering)

    if image is None:
        print('no image is identified in path: {}'.format(image_path))
        continue
    h, w = image.shape[:2]

    image = image[int(h * 0.5):, :]

    if args['debug']:
        cv2.imshow('image with region of interest', image)
        if cv2.waitKey() & 0xFF == ord('p'):
            exit(0)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if args['debug']:
        cv2.imshow('resized grayscal image with region of interest', image)
        if cv2.waitKey() & 0xFF == ord('p'):
            exit(0)
    image = img_to_array(image)
    data.append(image)

# convert throttle and steering values into niumpy array
throttle_values = np.array(throttle_values)
steering_values = np.array(steering_values)
steering_left_values = np.copy(steering_values)
steering_right_values = np.copy(steering_values)

# filter steering values into left and right steering
steering_left_values[steering_left_values < 0] = 0
steering_right_values[steering_right_values > 0] = 0

# invert values of the steering_right_values in order to bring into appropriate scale of positive numbers
steering_right_values *= -1

# convert into [0,1] scale to make better predictions
throttle_values = throttle_values / 100
steering_left_values = steering_left_values / 100
steering_right_values = steering_right_values / 100

# scale the raw pixel intensities to the range [0, 1] and convert to
# a NumPy array
data = np.array(data, dtype="float") / 255.0
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
    len(dataset), data.nbytes / (1024 * 1000.0)))

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, throttle_values, steering_left_values, steering_right_values, test_size=0.2, random_state=42)
(train_x, test_x, train_throttle_y, test_throttle_y, train_steering_left_y, test_steering_left_y, train_steering_right_y, test_steering_right_y) = split

# initialize our FashionNet multi-output network
model = LaneControlNet.build(256, 256, final_act='linear')

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
    "throttle_output": "mean_squared_error",
    "left_steering_output": "mean_squared_error",
    "right_steering_output": "mean_squared_error",
}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
opt = Adam(lr=0.001, decay=1e-5)
model.compile(optimizer=opt, loss='mse', metrics=["accuracy"])
model.summary()

# train the model
print("[INFO] training model...")
train_y = np.array(list(zip(train_throttle_y, train_steering_left_y, train_steering_right_y)))
test_y = np.array(list(zip(test_throttle_y, test_steering_left_y, test_steering_right_y)))
H = model.fit(train_x, train_y,
              validation_data=(test_x, test_y),
              epochs=EPOCHS,
              batch_size=BS,
              verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])


# plot the total loss, throttle loss, steering_right loss and steering_left loss
# lossNames = ['loss', 'throttle_output_loss', 'left_steering_output_loss', 'right_steering_output_loss']
lossNames = ['loss']
plt.style.use("ggplot")
(fig, ax) = plt.subplots(1, 1, figsize=(13, 13))

# loop over the loss names
for (i, l) in enumerate(lossNames):
    # plot the loss for both the training and validation data
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
               label="val_" + l)
    ax[i].legend()

# save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plt.savefig("{}_losses.png".format(args["plot"]))
plt.close()


# create a new figure for the accuracies
# accuracyNames = ['throttle_output_accuracy', 'left_steering_output_accuracy', 'right_steering_output_accuracy']
accuracyNames = ['accuracy']
plt.style.use("ggplot")
(fig, ax) = plt.subplots(1, 1, figsize=(8, 8))

# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
    # plot the loss for both the training and validation data
    ax[i].set_title("Accuracy for {}".format(l))
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Accuracy")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
               label="val_" + l)
    ax[i].legend()

# save the accuracies figure
plt.tight_layout()
plt.savefig("{}_accs.png".format(args["plot"]))
plt.close()
