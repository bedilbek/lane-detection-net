import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Flatten, Dense, Activation, Dropout, BatchNormalization, Conv2D, MaxPooling2D


class LaneControlNet:

    @staticmethod
    def __build_branch(inputs, final_act='sigmoid'):
        # utilize a lambda layer to convert the 3 channel input to a
        # grayscale representation
        # x = Lambda(lambda c: tf.image.rgb_to_grayscale(c)(inputs))

        # CONV => RELU => BN => POOL
        x = Conv2D(16, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(inputs)
        x = Conv2D(32, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(x)

        # Flatten the volume, FC => DENSE
        # define branch of output layers for the regression of steering
        x = Flatten()(x)

        x = Dense(32, activation='relu')(x)
        x = Dense(3, activation=final_act)(x)

        return x

    @staticmethod
    def build(width, height, final_act='sigmoid'):
        # initialize the input shape and channel dimension (this code
        # assumes you are using TensorFlow which utilizes channels
        #  last ordering
        input_shape = (height, width, 1)

        inputs = Input(shape=input_shape)
        dense = LaneControlNet.__build_branch(inputs, final_act=final_act)

        # create model using input (batch of images) and
        # two separate outputs -- one for throttle and one for steering, respectively
        model = Model(inputs=inputs, outputs=dense, name='LaneControlNet')

        return model