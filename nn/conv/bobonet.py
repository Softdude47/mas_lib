import math
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


class BoboNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        """Builds BoboNet model
        
        # Arguments:
            `width`: width of the input image
            `height`: height of the input image
            `depth`: number of channels in the image
            `classes`: number of uniques classes.
            `reg`: regularization. Defaults to 0.0002.
        """
        # specify input shape
        input_shape = (height, width, depth)
        if K.image_data_format() == "channel_first":
            input_shape = (depth, height, width)

        # A good rule of thumb for assigning number of node in current layer
        # is to take the square root of the previous number of nodes in the
        # layer and then find the closest power of 2 (eg 100 -> 8 -> ... -> output)
        unit1 = get_nearest_pow_of_2(input_shape[-1])
        unit2 = get_nearest_pow_of_2(unit1)
        
        # build model with architecture:
        # input => square-root(prev_node) => square-root(prev_node) => output
        model = Sequential(name="BoboNet")
        model.add(Dense(units=unit1, activation="relu", kernel_regularizer=l2(reg), input_shape=input_shape))
        model.add(Dense(units=unit2, activation="relu", kernel_regularizer=l2(reg)))
        model.add(Dropout(0.1))
        model.add(Dense(units=classes, activation="softmax", kernel_regularizer=l2(reg)))
        
        return model

def get_nearest_pow_of_2(num:"int | float"):
    # calculates the nearest power of 2 to the square root of 'num'
    sqrt = math.sqrt(num)
    next_pow_of_2 = pow(2, math.ceil(math.log(sqrt)/math.log(2)))
    
    prev_pow_of_2 = next_pow_of_2 / 2
    
    nearest_pow = next_pow_of_2
    if abs(sqrt - prev_pow_of_2) < abs(sqrt - next_pow_of_2):
        nearest_pow = prev_pow_of_2
    
    return nearest_pow