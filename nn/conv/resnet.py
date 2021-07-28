from tensorflow.keras.layers import add
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization


class ResNet:
    
    @staticmethod
    def pre_activation_module(data, filters, filter_size, strides, channel_dim, reg, bn_epsilon, bn_momentum):
        """A module consisting of BN => RELU => CONV layer.

        # Args:
            `data`: Input of module, usually output of another module.
            `filters`: The number of filters of size `filter_size` to learn.
            `filter_size`: The size of each kernel/filters in the `Conv2D` layer.
            `strides`: The number of pixels shifts over the input in the `Conv2D` layer.
            `channel_dim`: Image/input channel axis, either `1` or `-1`.
            `reg`: L2 regularization strength.
            `bn_epsilon`: Epsilon value on the batch normalization layer to prevent zero-division.
            `bn_momentum`: Momentum values on the batch normalization layer to reduce noise.
        """
        # created module to reduce lines of code under the residual_module function
        # BN => ACT => CONV
        bn = BatchNormalization(axis=channel_dim, momentum=bn_momentum, epsilon=bn_epsilon)(data)
        act = Activation("relu")(bn)
        conv = Conv2D(filters=filters, kernel_size=filter_size, strides=strides, padding="same", kernel_regularizer=l2(reg), use_bias=False)(act)
        return conv
    
    @staticmethod
    def residual_module(data, filters, strides, channel_dim, reduce=False, reg=0.0002, bn_epsilon=2e-5, bn_momentum=0.9):
        """A block consisting of a shortcut(i.e input) and three stacked-together `pre_activation` module.

        # Args:
            `data`: Input of module, usually output of another module.
            `filters`: The number of filters learn in the last module, the one-fourth of the `filter` is learnt in `pre_activation` 1 and 2.
            `strides`: The number of pixels shifts over the input of the `shortcut` when being used as a reducer. at least `(2, 2)`.
            `channel_dim`: Image/input channel axis, either `1` or `-1`.
            `reg`: L2 regularization strength.
            `bn_epsilon`: Epsilon value on the batch normalization layer to prevent zero-division
            `bn_momentum`: Momentum values on the batch normalization layer to reduce noise.
        """
        # shortcut branch of the ResNet module initialized as the input(identity) data
        shortcut = data
        
        # first block of ResNet module, consisting of 1x1 pre_activation_module
        pre_1 = ResNet.pre_activation_module(
            data,
            filters=filters//4,
            filter_size=(1, 1),
            strides=(1, 1),
            channel_dim=channel_dim,
            reg=reg,
            bn_epsilon=bn_epsilon,
            bn_momentum=bn_momentum,
        )
        
        # second block of ResNet module, consisting of 3x3 pre_activation_module
        pre_2 = ResNet.pre_activation_module(
            pre_1,
            filters=filters//4,
            filter_size=(3, 3),
            strides=strides,
            channel_dim=channel_dim,
            reg=reg,
            bn_epsilon=bn_epsilon,
            bn_momentum=bn_momentum,
        )
        
        # third block of ResNet module, consisting of 1x1 pre_activation_module
        pre_3 = ResNet.pre_activation_module(
            pre_2,
            filters=filters,
            filter_size=(1, 1),
            strides=(1, 1),
            channel_dim=channel_dim,
            reg=reg,
            bn_epsilon=bn_epsilon,
            bn_momentum=bn_momentum,
        )
        
        if reduce:
            # if to reduce the spatial dimension,
            # apply pre_activation_module to shortcut
            shortcut = ResNet.pre_activation_module(
                data,
                filters=filters,
                filter_size=(1, 1),
                strides=strides,
                channel_dim=channel_dim,
                reg=reg,
                bn_epsilon=bn_epsilon,
                bn_momentum=bn_momentum,
            )
        
        # adds the shortcut to the output of the third block
        x = add([shortcut, pre_3])
        
        # returns the added outputs
        return x
        
        
    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bn_epsilon=2e-5, bn_momentum=0.9, dataset="cifar10"):
        """Builds ResNet model. Implementation of the paper `Identity Mappings in Deep Residual Networks by He et al`.

        # Args:
            `width`: Width of the input image.
            `height`: Height of the input image.
            `depth`: The number of color channels in the input image.
            `classes`: Number of classes.
            `stages`: A tuple or list containing integers which represents groups of same stacked-together `residual_module` .
            `filters`: A tuple or list containing integers that specifies number of filters by `residual_module` created by `stages`.
            `reg`: L2 regularization strength.
            `bn_epsilon`: Epsilon value on the batch normalization layer to prevent zero-division.
            `bn_momentum`: Momentum values on the batch normalization layer to reduce noise.
            `dataset`: Specifies dataset, as certain dataset requires certain operations.
            
        # Returns:
            A constructed ResNet model architecture
        """
        # at first, initialize the input shape and channel dimension in the 
        # ordering of "channel_last"
        input_shape = (height, width, depth)
        channel_dim = -1
        
        # changes the channel dimension and input shape if image format is in "channel_first" ordering
        if K.image_data_format() == "channel_first":
            input_shape = (depth, height, width)
            channel_dim = 1
        
        # input layer followed by a batch normalization layer
        inputs = Input(shape=input_shape)
        x = BatchNormalization(axis=channel_dim, momentum=bn_momentum, epsilon=bn_epsilon)(inputs)
        
        # adds a CONV layer based on dataset
        if dataset == "cifar10":
            x = Conv2D(
                filters=filters[0],
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                kernel_regularizer=l2(reg),
                use_bias=False
            )(x)
            
        # loop over integers in the stages list/tuple
        for i in range(0, len(stages)):
            
            # the first entry uses 1x1 stride while the others uses 2x2 strides
            strides = (1, 1) if i == 0 else (2, 2)
            
            # applies residual module to learn filters and reduce spatial dimension
            x = ResNet.residual_module(
                data=x,
                filters=filters[i + 1],
                strides=strides,
                channel_dim=channel_dim,
                reduce=True,
                reg=reg,
                bn_epsilon=bn_epsilon,
                bn_momentum=bn_momentum
            )
            
            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                x = ResNet.residual_module(
                    data=x,
                    filters=filters[i + 1],
                    strides=(1, 1),
                    channel_dim=channel_dim,
                    reduce=False,
                    reg=reg,
                    bn_epsilon=bn_epsilon,
                    bn_momentum=bn_momentum
                )
        
        # applies BN => ACT => POOL 
        x = BatchNormalization(axis=channel_dim, momentum=bn_momentum, epsilon=bn_epsilon)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)
        
        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)
        
        # creat the model
        model = Model(inputs, x, name="resnet")
        
        # returns the constructed network architecture
        return model