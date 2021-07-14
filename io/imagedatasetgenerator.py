import os
import cv2
import random
import numpy as np
from imutils.paths import list_images
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelBinarizer

class ImageDatasetGenerator(Sequence):
    def __init__(self, paths, batch_size=32, preprocessors=None, aug=None, binarize=True, target_size=(224, 224), validation_split=0.2, **kwargs):
        """Reads image from directory

        # Args:
            `paths`: Parent directory of dataset.
            `batch_size`: Dataset batch size.
            `preprocessors`: Preprocessing functions to apply.
            `aug`: data augmentation function(or class with `flow` method), accept and returns images in batch
            `binarize`: wether to apply one hot encoding for labels.
            `target_size`: image `width x height` dimension.
            `validation_split`: 
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
        self.num_images = len(paths)
        
        # encode and store labels
        self.__lb = LabelBinarizer()
        self.labels = [self.get_label(path) for path in self.paths]
        self.labels = self.__lb.fit_transform(self.labels) if self.binarize else self.labels
        
        self.classes = [self.get_label(path) for path in self.paths]
        self.classes = np.unique(self.classes)
        num_classes = len(self.classes)
        print(f"[INFO] Found {len(self.paths)} images... belonging to {num_classes} classes")
        
        # store generator steps
        steps = len(paths) // batch_size
        self.steps = (steps + 1) if (steps * batch_size) < len(paths) else steps
        
    def __len__(self):
        """
        Number of batch in the Sequence.
        # Returns:
            The number of batches in the Sequence.
        """
        return self.steps
       
    def __getitem__(self, index):
        """
        Gets batch at position `index`.
        # Args:
            `index`: position of the batch in the Sequence.
        # Returns:
            A batch of images and corresponding labels each with a size of `self.batch_size`.
        """
        paths = self.paths[index * self.batch_size : (index + 1) * self.batch_size]
        labels = self.labels[index * self.batch_size : (index + 1) * self.batch_size]
        
        # load and preprocess image
        images = [self.transform(path, preprocessors=self.preprocessors) for path in paths]
        if self.aug is not None:
            images = next(self.aug.flow(images, batch_size=len(images)))
        
        return (np.array(images), labels)
    
    def __get_subset_paths(self, name):
        """
        Finds list of path under datasets's split name.

        # Args:
            name: dataset's split name, either `train` or `val`.

        # Returns:
            List of path under dataset split.
        """
        if "train" in name:
            # compute number of train image paths
            split = 1 - self.validation_split
            split = int(split * len(self.paths))
            return self.paths[: split]
        
        # compute number of validation image paths
        split = int(self.validation_split * len(self.paths))
        return self.paths[split: ]
    
    def get_label(self, path):
        """
        Gets image label from image path.

        # Args:
            path: full image path.

        # Returns:
            Encoded label from image path.
        """
        # extract label from path
        label_idx = getattr(self, "label_index", -2)
        p = path.replace("/", os.path.sep)
        label = p.split(os.path.sep)[label_idx]
        return label
    
    
    def generate(self, subset="train", passes=np.inf):
        """
        Generates image and labels.

        # Args:
            subset: name of dataset split either `train` or `val`.
            passes: number of times to iterate.

        # Yields:
            A tuple with the structure of (`image_batch`, `label_batch`).
        """
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
                batch_labels = self.__lb.transform(batch_labels, self.num_images) if self.binarize else batch_labels
                
                if "val" in subset:
                    # checks for any preprocessors made for validation image split
                    val_preprocessors = getattr(self, "val_preprocessors", None)
                    val_preprocessors = val_preprocessors or getattr(self, "validation_preprocessors", [])
                    
                    # loads image and applies preprocesors(if any)
                    batch_images = [self.transform(path, val_preprocessors) for path in batch_path]
                
                if "train" in subset:
                    # loads image and applies preprocesors(if any)
                    batch_images = [self.transform(path, self.preprocessors) for path in batch_path]
                    
                    # applies data additional augmentation to loaded images
                    if self.aug is not None:
                        try:
                            batch_images = next(self.aug.flow(batch_images, batch_size=self.batch_size))
                        except:
                            batch_images = next(self.aug(batch_images, batch_size=self.batch_size))
                    
                # returns batch
                yield (np.array(batch_images), batch_labels)
            i += 1
        
        
    def transform(self, path, preprocessors=[]):
        """
        Loads image from `path` and preprocess it with functions inside `preprocessors`.

        # Args:
            path: valid path to an existing image.
            preprocessors: list of preprocessing functions(or classes with a `preprocess` method) that accepts and returns an image.

        # Returns:
            A preprocessed image.
        """
        preprocessors = preprocessors or []
        # load an image from path
        image = cv2.imread(path)
        image = cv2.resize(image, self.target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # apply preprocessor
        for preprocessor in preprocessors:
            image = preprocessor.preprocess(image)
        
        return image