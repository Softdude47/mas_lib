from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization


class DeeperGoogLeNet:
    @staticmethod
    def conv_module(x, filters, kernel_size, strides, channel_dim, padding, name, reg):
        """Defines `CONV => BN => RELU` pattern
        # Arguments:
            `x`: Input layer to the function.
            `filters`: The number of filters the `CONV` layer is going to learn.
            `kernel_size`: The size of each of the filter that will be learned.
            `strides`: The stride of the `CONV` layer.
            `channel_dim`: The channel dimension, which is either derived from 'channel first' or 'channel last' ordering.
            `padding`: Type of padding to be appliced to the `CONV` layer.
            `name`: Name of current block.
            `reg`: L2 regularization strength.
        """
        conv_name, bn_name, act_name = (None, None, None)
        if name is not None:
            bn_name = str(name) + "_bn"
            act_name = str(name) + "_act"
            conv_name = str(name) + "_conv"
        
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_regularizer=l2(reg),
            name=conv_name
        )(x)
        x = BatchNormalization(axis=channel_dim, name=bn_name)(x)
        x = Activation(activation="relu", name=act_name)(x)
        
        # return the block
        return x
    
    @staticmethod
    def inception_module(x, filters_1x1, filters_3x3_reducer, filters_3x3,
        filters_5x5_reducer, filters_5x5, pool_proj, channel_dim, stage, reg=521e-5):
        """creates four branches of `conv_module`.

        # Arguments:
            `x`: Input to the inception module
            `filters_1x1`: The number of filters in `(1 x 1)` convolution.
            `filters_3x3_reducer`: The number of filters in dimensionality reduction convolution prior to `(3 x 3)`.
            `filters_3x3`: The number of filters in `(3 x 3)` convolution.
            `filters_5x5_reducer`: The nmber of filters in dimensionality reduction convolution prior to `(5 x 5)`.
            `filters_5x5`: The number of filters in `(5 x 5)` convolution.
            `pool_proj`: The number of filters in `POOL` projection `CONV` layer.
            `channel_dim`: The channel dimension, which is either derived from 'channel first' or 'channel last' ordering.
            `stage`: name of current `inception_module`.
            `reg`: L2 regularization strength applied to `conv_module` within current `inception_module`.

        # Returns:
             A tensor, which is the concatenation of the inputs(the branches) along the `channel_dim` axis is returned.
        """
        # first branch of inception module which consists of (1 x 1) convolutions
        first = DeeperGoogLeNet.conv_module(
            x=x,
            filters=filters_1x1,
            kernel_size=(1, 1),
            strides=(1, 1),
            channel_dim=channel_dim,
            padding="same",
            name=stage+"_first",
            reg=reg
        )
        
        # second branch of inception module which consists of (1 x 1) and
        # (3 x 3) convolutions
        second = DeeperGoogLeNet.conv_module(
            x=x,
            filters=filters_3x3_reducer,
            kernel_size=(3, 3),
            strides=(1, 1),
            channel_dim=channel_dim,
            padding="same",
            name=stage+"_second1",
            reg=reg
        )
        second = DeeperGoogLeNet.conv_module(
            x=second,
            filters=filters_3x3,
            strides=(1, 1),
            channel_dim=channel_dim,
            padding="same",
            name=stage+"_second2",
            reg=reg
        )
        
        # third branch of the inception module which consists of (1 x 1)
        # and (5 x 5) convolutions
        third = DeeperGoogLeNet.conv_module(
            x=x,
            filters=filters_5x5_reducer,
            strides=(1, 1),
            channel_dim=channel_dim,
            padding="same",
            name=stage+"_third1",
            reg=reg
        )
        third = DeeperGoogLeNet.conv_module(
            x=third,
            filters=filters_5x5,
            strides=(1, 1),
            channel_dim=channel_dim,
            padding="same",
            name=stage+"_third2",
            reg=reg
        )
        
        # fourth(and final) branch of the inception module which is the
        # `POOL` projection
        fourth = MaxPooling2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding="same",
            name=stage+"_fourth1"
        )(x)
        fourth = DeeperGoogLeNet.conv_module(
            x=fourth,
            filters=pool_proj,
            strides=(1, 1),
            channel_dim=channel_dim,
            paddin="same",
            name=stage+"_fourth2",
            reg=reg
        )
        merge = concatenate([first, second, third, fourth], axis=channel_dim)
        return merge
    
    def build(width, height, depth, classes, reg=503e-5):
        """Builds DeeperGoogLeNet model

        # Arguments:
            `width`: width of input image.
            `height`: height of input image.
            `depth`: number of channel in input image.
            `classes`: number of classes.
            `reg`: L2 regularization strength.
        """
        
        # input shape and channel dimension
        input_shape = (height, width, depth)
        channel_dim = -1
        if K.img_data_format == "channel_first":
            channel_dim = 1
        
        # block #1: input layer follow by CONV => POOL => CONV * 2 => POOL
        input_layer = Input(shape=input_shape)
        x = DeeperGoogLeNet.conv_module(input_layer, 64, (5, 5), (1, 1), channel_dim, padding="same", name="block1", reg=reg)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool1")(x)
        x = DeeperGoogLeNet.conv_module(x, 64, (1, 1), (1, 1), channel_dim, padding="same", name="block2", reg=reg)
        x = DeeperGoogLeNet.conv_module(x, 192, (3, 3), (1, 1), channel_dim, padding="same", name="block3", reg=reg)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool2")(x)
        
        # block #2: consists of CONV => CONV => POOL
        x = DeeperGoogLeNet.inception_module(x, 64, 96, 128, 16, 32, 32, channel_dim, reg=reg, stage="3a")
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, channel_dim, reg=reg, stage="3b")
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool3")(x)
        
        # block #3: consists of (CONV * 4) => (CONV * 4) => (CONV * 4) => (CONV * 4) => (CONV * 4) => POOL
        x = DeeperGoogLeNet.inception_module(x, 192, 96, 208, 16, 48, 64, channel_dim, reg=reg, stage="4a")
        x = DeeperGoogLeNet.inception_module(x, 160, 112, 224, 24, 64, 64, channel_dim, reg=reg, stage="4b")
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 256, 24, 64, 64, channel_dim, reg=reg, stage="4c")
        x = DeeperGoogLeNet.inception_module(x, 112, 114, 288, 32, 64, 64, channel_dim, reg=reg, stage="4d")
        x = DeeperGoogLeNet.inception_module(x, 256, 160, 320, 32, 128, 128, channel_dim, reg=reg, stage="4e")
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool4")(x)
        
        # applies average pooling and dropout
        x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding="same", name="pool5")(x)
        x = Dropout(rate=0.4, name="dropout")(x)
        
        # SOFTMAX classifier
        x = Flatten(name="flatten")(x)
        x = Dense(classes, activation="linear", name="labels")(x)
        x = Activation(activation="softmax", name="softmax")(x)
        
        # creates model
        model = Model(input_layer, x, name="deepergooglenet")
        
        # return constructed network architechture
        return model