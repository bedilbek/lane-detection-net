import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Input, Flatten, Dense,Lambda, Activation, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Lambda, Convolution2D


class TrafficSignNet:

    @staticmethod
    def build(width_height_channel, num_classes):
        # initialize the input shape and channel dimension (this code
        # assumes you are using TensorFlow which utilizes channels
        #  last ordering
        (width, height, channel) = width_height_channel
        input_shape = (height, width, channel)

        inputs = Input(shape=input_shape)

        init = 'glorot_uniform'
        padding = 'same'

        # x = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer=init, padding=padding)(inputs)
        x = Conv2D(32, kernel_size=(3, 3), activation='relu', strides=2, kernel_initializer=init, padding=padding)(inputs)
        x = Dropout(0.15)(x)
        # x = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer=init, padding=padding)(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=2, kernel_initializer=init, padding=padding)(x)
        x = Dropout(0.15)(x)
        # x = Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer=init, padding=padding)(x)
        x = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=2, kernel_initializer=init, padding=padding)(x)
        x = Dropout(0.15)(x)

        # Flatten the volume, FC => DENSE
        # define branch of output layers for the regression of steering
        x = Flatten()(x)

        x = Dense(100, activation='relu', kernel_initializer=init)(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax', kernel_initializer=init, name='traffic_sign_output')(x)

        # create model using input (batch of images) and
        # number of outputs for image classification
        model = Model(inputs=inputs, outputs=x, name='TrafficSignNet')

        return model
