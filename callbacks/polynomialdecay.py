from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback


class PolynomialDecay(Callback):
    def __init__(self, INIT_LR, POWER):
        super(PolynomialDecay, self).__init__()
        self.INIT_LR = INIT_LR
        self.POWER = POWER
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
        current_lr = self.INIT_LR * pow(1 - (epoch/max_epochs), self.POWER)
        K.set_value(self.model.optimizer.learning_rate, current_lr)