from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        """Builds AlexNet model

        # Args:
            `width`: width of input image.
            `height`: height of input image.
            `depth`: number of channels in input image.
            `classes`: number of unique classes of image.
            `reg`: penalty. Defaults to 0.0002.
        """
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        input_shape = (height, width, depth)
        channel_dimension = -1
        
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            channel_dimension = 1
        
        # Block #1: first CONV => RELU layer set
        model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="same", activation="relu", kernel_regularizer=l2(reg), input_shape=input_shape))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
        model.add(Dropout(0.25))
        
        # Block #2: second CONV => RELU layer set
        model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
        model.add(Dropout(0.25))
        
        # Block #3: CONV => RELU => CONV => RELU => CONV => RELU
        # stacking multiple convolutional[and RELU activation]
        # layers before applying pooling/dropout
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
        model.add(Dropout(0.25))
        
        # Block #4: first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(Dropout(0.5))
        
        # Block #5: second set of FC => RELU layers
        model.add(Dense(units=4096, activation="relu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(Dropout(0.5))
        
        # softmax classifier
        model.add(Dense(units=classes, activation="softmax", kernel_regularizer=l2(reg)))
        
        # return the constructed network architecture
        return model