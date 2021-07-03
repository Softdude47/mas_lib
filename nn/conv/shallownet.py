from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        """Builds ShallowNet model

        # Args:
            `width`: width of the input image
            `height`: height of the input image
            `depth`: number of channels in the image
            `classes`: number of uniques classes.
        """
        
        input_shape = (width, height, depth)
        if K.image_data_format() == "channel_first":
            input_shape = (depth, width, height)
            
        # builds model architecture
        model = Sequential(name="shallownet")
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=input_shape))
        model.add(Dense(classes, activation="softmax"))
        
        # return model
        return model