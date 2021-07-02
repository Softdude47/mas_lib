from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization


class MiniGoogLeNet:
    @staticmethod
    def conv_module(x, filters, kernel_size, strides, channel_dim, padding):
        """Defines `CONV => BN => RELU` pattern
        # Args:
            `x`: Input layer to the function.
            `filters`: The number of filters the `CONV` layer is going to learn.
            `kernel_size`: The size of each of the filter that will be learned.
            `strides`: The stride of the `CONV` layer.
            `channel_dim`: The channel dimension, which is either derived from 'channel first' or 'channel last' ordering.
            `padding`: Type of padding to be appliced to the `CONV` layer.
        """
        
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization(axis=channel_dim)(x)
        x = Activation(activation="relu")(x)
        
        # return the block
        return x
    
    @staticmethod
    def inception_module(x, filters_1x1, filters_3x3, channel_dim):
        """Creates two blocks of `CONV => BN => RELU`. With each `CONV` layer
        in both blocks having a kernel size of a kernel size of
        `(1 x 1)` and `(3 x 3)` respectively.A tensor, which is
        the concatenation of the inputs(the blocks) alongside `channel_dim` axis is returned.

        # Args:
            `x`: Input layer to the function.
            `filters_1x1`: The number of filters the `CONV` layer( of `(1 x 1)` kernel size) is going to learn.
            `filters_3x3`: The number of filters the `CONV` layer( of `(3 x 3)` kernel size) is going to learn.
            `channel_dim`: The channel dimension, which is either derived from `channel_first` or `channel last` ordering.
        """
        conv_1x1 = MiniGoogLeNet.conv_module(x=x, filters=filters_1x1, kernel_size=(1, 1), strides=(1, 1), channel_dim=channel_dim, padding="same")
        conv_3x3 = MiniGoogLeNet.conv_module(x=x, filters=filters_3x3, kernel_size=(3, 3), strides=(1, 1), channel_dim=channel_dim, padding="same")
        merge = concatenate([conv_1x1, conv_3x3], axis=channel_dim)
        return merge
    
    @staticmethod
    def downsample_module(x, filters, channel_dim):
        """Creates a block of `CONV => BN => RELU` and a `MaxPool` layer. With the `CONV` layer
        in block having a kernel size `(3 x 3)`. Both(the block and `MaxPool` layer) having a stride of `(3 x 3)`.
        A tensor, which is the concatenation of the inputs(the block and M) alongside `channel_dim` axis is returned.

        # Args:
            `x`: Input layer the the function.
            `filters`: The number of filters the `CONV` layer is going to learn.
            `channel_dim`: The channel_dimension, which is either derived from 'channel first' or 'channel last' ordering.
        """
        conv_3x3 = MiniGoogLeNet.conv_module(x=x, filters=filters, kernel_size=(3, 3), strides=(2, 2), channel_dim=channel_dim, padding="same")
        max_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
        merge = concatenate([conv_3x3, max_pool], axis=channel_dim)
        return merge
    
    @staticmethod
    def build(width, height, depth, classes):
        """Builds MiniGoogLeNet model

        Args:
            `width`: width of input image
            `height`: height of input image
            `depth`: number of channels in input image
            `classes`: number of classes
        """
        input_shape = (width, height, depth)
        channel_dim = -1
        
        if K.image_data_format() == "channel_first":
            input_shape = (depth, width, height)
            channel_dim = 1
        
        # constructs model architecture
        inputs = Input(shape=input_shape, name="input layer")
        x = MiniGoogLeNet.conv_module(inputs, 96, (3, 3), (1, 1), channel_dim, "same")
        x = MiniGoogLeNet.inception_module(x, 32, 32, channel_dim)
        x = MiniGoogLeNet.inception_module(x, 32, 48, channel_dim)
        x = MiniGoogLeNet.downsample_module(x, 80, channel_dim)
        x = MiniGoogLeNet.inception_module(x, 112, 48, channel_dim)
        x = MiniGoogLeNet.inception_module(x, 96, 64, channel_dim)
        x = MiniGoogLeNet.inception_module(x, 80, 80, channel_dim)
        x = MiniGoogLeNet.inception_module(x, 48, 96, channel_dim)
        x = MiniGoogLeNet.downsample_module(x, 96, channel_dim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, channel_dim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, channel_dim)
        x = AveragePooling2D((7, 7), padding="same")(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(classes, "softmax")(x)
        
        # creates model
        model = Model(inputs, x, name="googlenet")
        
        # returns constructed network architecture
        return model