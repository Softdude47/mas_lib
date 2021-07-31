from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self, fig_path=None, json_path=None, start_at=0, **kwargs):
        # store the output path for the plot figures,
        # the json serialized file and the starting epoch
        super(TrainingMonitor, self).__init__()
        
        plot_path = kwargs.get("plot_path", fig_path)
        self.fig_path = kwargs.get("fig_path", plot_path)
        self.json_path = json_path
        self.start_at = start_at
        
        # creates the directory to each file
        os.makedirs(os.path.dirname(self.fig_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        
    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}
        
        # if a valid path was provided for the json path
        if self.json_path is not None and os.path.exists(str(self.json_path)):
            
            # load the json values into the history dictionary
            self.H = json.loads(open(self.json_path).read())
            
            # checks to see if a starting epoch was supplied
            if self.start_at > 0:
                # loop over the entries in the history log and
                # trim any entries that are past the starting epoch
                for k in self.H.keys():
                    self.H[k] = self.H[k][: self.start_at]
                    self.H[k] = [float(val) for val in self.H[k]]
                    
    def on_epoch_end(self, epoch, logs={}):
        # loops over the logs and update the metrics[loss, accuracy, etc]
        # for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l
        
        if self.json_path is not None and self.fig_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()
        
        if len(self.H["loss"]) > 0:
            
            # plot the metrics
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="loss")
            plt.plot(N, self.H["accuracy"], label="accuracy")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["val_accuracy"], label="val_accuracy")
            plt.title(f"Training Loss and Accuracy [Epoch {len(N)}]")
            plt.xlabel("Epoch")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            
            # save plot figure
            plt.savefig(self.fig_path)
            plt.close()