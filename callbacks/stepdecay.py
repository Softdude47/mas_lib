import numpy as np
from keras import backend as K
from keras.callbacks import LambdaCallback



class StepDecay(LambdaCallback):
    def __init__(self, init_lr, factor, drop_every):
        
        self.factor = factor
        self.init_lr = init_lr
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
        current_lr = self.init_lr * pow(self.factor, np.floor(1 - ((epoch + 1)/ max_epochs)))
        K.set_value(self.model.optimizer.learning_rate, current_lr)