import os
import cv2
import random
import numpy as np
from imutils.paths import list_images
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

class ImageDatasetGenerator(Sequence):
    def __init__(self, paths, batch_size=32, preprocessors=None, aug=None, binarize=True, target_size=(224, 224), validation_split=0.2, **kwargs):
        """Reads image from directory

        # Args:
            `paths`: Parent directory of dataset.
            `batch_size`: Dataset batch size.
            `preprocessors`: Preprocessing functions to apply.
            `aug`: class that generates augmented images. keras.preprocessing.image.ImagaDataGenerators.
            `binarize`: wether to apply one hot encoding for labels.
            `target_size`: image size.
        """
        for (key, value) in kwargs.items():
            setattr(self, key, value)
        
        # stores values
        self.aug = kwargs.get("aug", aug)
        self.paths = kwargs.get("paths", paths)
        self.binarize = kwargs.get("binarize", binarize)
        self.batch_size = kwargs.get("batch_size", batch_size)
        self.target_size = kwargs.get("target_size", target_size)
        self.preprocessors = kwargs.get("preprocessors", preprocessors)
        self.validation_split = kwargs.get("validation_split", validation_split)
        
        # list and shuffle image paths
        # and extract labels
        paths = list(list_images(paths))
        random.shuffle(paths)
        self.paths = paths
        self.labels = [self.get_label(path) for path in self.paths]
        self.num_classes = len(np.unique(self.labels))
        print(f"[INFO] Found {len(self.paths)} images... belonging to {self.num_classes} classes")
        
        steps = len(paths) // batch_size
        self.steps = (steps + 1) if (steps * batch_size) < len(paths) else steps
        
    def __len__(self):
        # returns number of batches in datasets
        return self.steps
       
    def __getitem__(self, index):
        # get the batch at the index-th posiition
        paths = self.paths[index * self.batch_size : (index + 1) * self.batch_size]
        labels = self.labels[index * self.batch_size : (index + 1) * self.batch_size]
        labels = to_categorical(labels, num_classes=np.unique(self.labels))
        images = [self.transform(path) for path in paths]
    
        return np.array([images, labels])
    
    def __get_subset_paths(self, name):
        # gets train/test data path
        if "train" in name:
            split = 1 - self.validation_split
            split = int(split * len(self.paths))
            return self.paths[: split]
        split = int(self.validation_split * len(self.paths))
        return self.paths[split: ]
    
    def get_label(self, path):
        # extract label from path
        p = path.replace("/", os.path.sep)
        label = p.split(os.path.sep)[-2]
        return label
    
    
    def generate(self, subset="train", passes=np.inf):
        i = 0
        # generates data
        paths = self.__get_subset_paths(subset)
        labels = [self.get_label(path) for path in paths]
        
        # split into batches
        labels = [labels[i : i + self.batch_size] for i in np.arange(0, len(labels), self.batch_size)]
        paths = [paths[i : i + self.batch_size] for i in np.arange(0, len(paths), self.batch_size)]
        while i < passes:
            for (batch_path, batch_labels) in zip(paths, labels):
                
                # preprocess each batchs images and labels
                batch_labels = to_categorical(batch_labels, self.num_classes) if self.binarize else batch_labels
                batch_images = [self.transform(path, getattr(self, "val_preprocessors", [])) for path in batch_path]
                if "train" in subset and self.aug is not None:
                    batch_images = [self.transform(path, self.preprocessors) for path in batch_path]
                    batch_images = self.aug.flow(batch_images, batch_size=self.batch_size)
                    
                # returns batch
                yield np.array(batch_images, batch_labels)
            i += 1
        
        
    def transform(self, path, preprocessors):
        # load and preprocess an image from path
        image = cv2.imread(path)
        image = cv2.resize(image, self.target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for preprocessor in preprocessors:
            image = preprocessor.preprocess(image)
        
        return image