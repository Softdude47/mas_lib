import os
import cv2

class SimpleDatasetWriter:
    def __init__(self, classes=[], parent_dir=None, *args, **kwargs):
        
        self.index = 0
        self.classes = classes
        self.parent_dir = parent_dir
        
        for cls in classes:
            os.makedirs(os.path.join(str(parent_dir), cls), exist_ok=True)
        
    def add(self, image, label, filename):
        """

        Args:
            image (array): image or array to be saved
            label (string): image label in our dataset
            filename (string): filename of the image in datasets
        """
        for cls in self.classes:
            if label in cls:
                full_path = os.path.sep.join([self.parent_dir, cls, filename])
                cv2.imwrite(full_path, image)
        
        self.index += 1
    
    def close(self,):
        pass