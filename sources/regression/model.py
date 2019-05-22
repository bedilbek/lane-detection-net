import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Input, Flatten, Dense,Lambda, Activation, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Lambda, Convolution2D


class LaneControlNet:

    @staticmethod
    def __build_branch(inputs, final_act='linear'):
        # utilize a lambda layer to convert the 3 channel input to a
        # grayscale representation
        # x = Lambda(lambda c: tf.image.rgb_to_grayscale(c)(inputs))

        init = 'glorot_uniform'
        padding = 'valid'

        x = Lambda(lambda z: z / 127.5 -1.0)(inputs)
        x = Conv2D(24, kernel_size=(5, 5), activation='elu', strides=2, kernel_initializer=init, padding=padding)(x)
        x = Conv2D(36, kernel_size=(5, 5), activation='elu', strides=2, kernel_initializer=init, padding=padding)(x)
        x = Conv2D(48, kernel_size=(5, 5), activation='elu', strides=2, kernel_initializer=init, padding=padding)(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='elu', kernel_initializer=init, padding=padding)(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='elu', kernel_initializer=init, padding=padding)(x)

        # Flatten the volume, FC => DENSE
        # define branch of output layers for the regression of steering
        x = Flatten()(x)

        x = Dense(100, activation='elu', kernel_initializer=init)(x)
        x = Dropout(0.5)(x)
        x = Dense(50, activation='elu', kernel_initializer=init)(x)
        x = Dropout(0.5)(x)
        x = Dense(10, activation='elu', kernel_initializer=init)(x)
        x = Dense(1, kernel_initializer=init)(x)

        return x

    @staticmethod
    def build_sequential(inputs, final_act='linear'):
        # utilize a lambda layer to convert the 3 channel input to a
        # grayscale representation
        # x = Lambda(lambda c: tf.image.rgb_to_grayscale(c)(inputs))

        # CONV => RELU => BN => POOL
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', strides=2, padding='valid', input_shape=inputs))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', strides=2, padding='valid'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', strides=2, padding='valid'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='valid'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='valid'))
        model.add(Dropout(0.5))

        # Flatten the volume, FC => DENSE
        # define branch of output layers for the regression of steering
        model.add(Flatten())

        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(2, activation=final_act))

        return model

    @staticmethod
    def build(width_height_channel, final_act='linear'):
        # initialize the input shape and channel dimension (this code
        # assumes you are using TensorFlow which utilizes channels
        #  last ordering
        (width, height, channel) = width_height_channel
        input_shape = (height, width, channel)

        inputs = Input(shape=input_shape)
        dense = LaneControlNet.__build_branch(inputs, final_act=final_act)

        # create model using input (batch of images) and
        # two separate outputs -- one for throttle and one for steering, respectively
        model = Model(inputs=inputs, outputs=dense, name='LaneControlNet')

        return model
