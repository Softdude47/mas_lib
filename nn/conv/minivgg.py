from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import image_data_format as k
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.layers import BatchNormalization as BN


class MiniVGG:
    @staticmethod
    def build(width, height, depth, classes):
        """Build MiniVGG model

        # Arguments:
            `width`: width of the input image
            `height`: height of the input image
            `depth`: number of channels in the image
            `classes`: number of uniques classes.
        """
        input_shape = (height, width, depth)
        channel_dimension = -1

        if k == "channels_first":
            input_shape = (depth, height, width)
            channel_dimension = 1
        
        # initialize model
        model = Sequential(name="MiniVGG")
        
        # constructs model architecture
        model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=input_shape))
        model.add(BN(axis=channel_dimension))
        
        model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(BN(axis=channel_dimension))
        model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
        model.add(Dropout(rate=0.25))
        
        model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(BN(axis=channel_dimension))
        
        model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(BN(axis=channel_dimension))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        
        model.add(Flatten())
        model.add(Dense(units=500, activation="relu"))
        model.add(BN(axis=channel_dimension))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=classes, activation="softmax"))
        
        # return constructed architecture
        return model