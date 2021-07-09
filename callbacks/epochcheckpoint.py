from tensorflow.keras.callbacks import BaseLogger

class EpochCheckpoint(BaseLogger):
    def __init__(self, interval, path):
        # initialize base class and store parameters
        super(EpochCheckpoint, self).__init__()
        self.interval = interval
        self.path = path
        
    def on_epoch_end(self, epoch, logs={}):
        # checkpoint model at given interval
        if epoch % self.interval == 0:
            self.model.save(self.path)