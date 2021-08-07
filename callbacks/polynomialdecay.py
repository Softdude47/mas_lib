from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback


class PolynomialDecay(Callback):
    def __init__(self, INIT_LR, POWER):
        super(PolynomialDecay, self).__init__()
        self.INIT_LR = INIT_LR
        self.POWER = POWER
        
    def on_epoch_begin(self, epoch, logs={}):
        # get total number of epochs
        max_epochs = self.params.get("nb_epoch") or self.params.get("epochs")
        
        # calculate new learning rate
        # lr = initial_lr * (1- (current_epoch/total_epochs)) ** 2
        current_lr = self.INIT_LR * pow(1 - (epoch/max_epochs), self.POWER)
        
        # apply new learning rate to model
        K.set_value(self.model.optimizer.learning_rate, current_lr)