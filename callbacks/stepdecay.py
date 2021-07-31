import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback



class StepDecay(Callback):
    def __init__(self, INIT_LR, factor, drop_every):
        # initialize base class
        super(StepDecay, self).__init__()
        
        # set parameters as class properties
        self.factor = factor
        self.INIT_LR = INIT_LR
        self.drop_every = drop_every
        self.H = {}
        
    def on_epoch_end(self, epoch, logs):
        # update metric history
        for (key, val) in logs.items():
            existing = self.H.get(key, [])
            existing.append(float(val))
            self.H[key] = existing
        
    def on_train_begin(self, logs={}):
        # get current epoch by adding one to the length
        # of any key in metrics history(self.H)
        random_key = list(self.H.keys())[0]
        epoch = 1 + len(logs[random_key])
        
        # perform learning decay
        max_epochs = self.params.get("nb_epoch")
        current_lr = self.INIT_LR * pow(self.factor, np.floor(1 - ((epoch + 1)/ max_epochs)))
        K.set_value(self.model.optimizer.learning_rate, current_lr)