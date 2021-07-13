import os
import cv2
import progressbar


class ImageDatasetWriter:
    def __init__(self, db_path, target_size=(224, 224), preprocessors=None, aug=None, show_progress=True, n_images=None):
        """Builds datasets into the `db_path/label/image.png` format

        # Args:
            `db_path`: directory to build dataset on.
            `target_size`: size of each image in the dataset.
            `preprocessors`: list of preprocessors to apply on each images. must accept an image as parameter and returns the image.
            `aug`: data augmentation function(or class that accept parameters with `.flow()` method) like `keras.preprocessing.image.ImageDataGenerator`.
                    must accept image and `batch_size` parameter and returns image in batches.
            `show_progress`: wether or not to show progressbar.
            `n_images`: total number of images.
        """
        # stores values
        self.aug = aug
        self.db_path = db_path
        self.target_size = target_size
        self.preprocessors = preprocessors or []
        
        # creates and start progressbar base on show_progress value
        if show_progress:
            widgets = ["[INFO] Building Dataset ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
            self.progress = progressbar.ProgressBar(max_value=n_images, widgets=widgets).start()
            self.close = self.__close
            
            
        self.__label = {}
        
    def add(self, images, labels):
        """adds list of images and laels to dataset"""
        
        # loop over image and labels
        for (image, label) in zip(images, labels):
            # construct image path
            idx = self.__label.get(label, 0) + 1
            filename = f"{label}_{idx}.png"
            path = os.path.sep.join([self.db_path, label, filename])
            
            # apply self.preprocessor
            for preprocessor in self.preprocessors:
                try:
                    image = preprocessor.preprocess(image)
                except:
                    image = preprocessor(image)
            
            # apply self.aug
            if self.aug:
                try:
                    image = self.aug.flow([image,], batch_size=32)[0]
                except:
                    image = self.aug([image,], batch_size=32)[0]
                    
            # save image
            cv2.imwrite(path, image)
            
    def __close(self):
        """ stops progressbar """
        if hasattr(self, "progress"):
            self.progress.finish()