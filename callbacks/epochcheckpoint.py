
import os
from pathlib import Path
from tensorflow.keras.callbacks import BaseLogger

class EpochCheckpoint(BaseLogger):
    def __init__(self, path, interval):
        # initialize base class and store parameters
        super(EpochCheckpoint, self).__init__()
        self.interval = interval
        self.path = path
        
    def on_epoch_end(self, epoch, logs={}):
        
        # model metrics
        val_loss = logs.get("val_loss")
        val_acc = logs.get("val_acc")
        loss = logs.get("loss")
        acc = logs.get("acc")
        
        
        # checkpoint model at given interval
        if epoch % self.interval == 0:
            
            # frees up memory
            self.clear_recent_checkpoints(os.path.dirname(self.path))
            
            self.model.save(str(self.path).format(
                val_loss=val_loss,
                val_acc = val_acc,
                loss=loss,
                acc=acc,
                epoch=epoch,
            ))
            
    def clear_recent_checkpoints(self, parent_dir):
        # get recent checkpoints
        recent_checkpoint = os.listdir(parent_dir)
        
        # delete all recent checkpoints
        for file in recent_checkpoint:
            if file.endswith("h5") or file.endswith("hdf5"):
                os.remove(file)