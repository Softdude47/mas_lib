import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical

class HDF5DatasetGenerator:
    """ HDF5 based dataset generator"""
    def __init__(self, db_path, feature_ref_name="data", batch_size=32, preprocessors=None, aug=None, binarize=True, classes=2):
        # stores the batch size, preprocessors, augmentor,
        # wether or not to binarize labels, number of unique classes
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        self.feature_ref_name = feature_ref_name
        
        # opens the HDF5 database for reading and determining
        # the total number of entries in the database
        self.db = h5py.File(db_path, mode="r")
        self.num_images = self.db[self.feature_ref_name].shape[0]
        
    def generate(self, passes=np.inf):
        # initialize the epoch count
        epoch = 0
        
        # loops infinitely till we reach a desired number
        # of epochs
        while epoch < passes:
            # loops over the HDF5 dataset
            for i in np.arange(0, self.num_images, self.batch_size):
                
                # extracts the images and labels from the HDF5 datasets
                images = self.db[self.feature_ref_name][i : i + self.batch_size]
                labels = self.db["labels"][i : i + self.batch_size]
                
                # checks to see if the labels should be binarize
                if self.binarize:
                    labels = to_categorical(labels, self.classes)
                
                # checks if any preprocessor was provided
                if self.preprocessors is not None:
                    
                    # initialize an empty list to store all preprocessed images
                    processed_images = []
                    
                    # loops and preprocess all images found and adds to the 
                    # list of preprocessed images
                    for image in images:
                        for preprocessor in self.preprocessors:
                            image = preprocessor.preprocess(image)
                            
                        processed_images.append(image)
                    # updates the image array to be that
                    # of the processed images
                    images = np.array(processed_images)
                    
                # applies augmentor if any was provided
                if self.aug:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size=self.batch_size))
                
                yield (images, labels)
            epoch += 1
    
    def close(self,):
        self.db.close()