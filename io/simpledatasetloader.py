import cv2

class SimpleDatasetLoader:
    """ loads and preprocess the image with a set of prerocessors """
    def __init__(self, preprocessors:list = None):
        self.preprocessors = preprocessors
        if self.preprocessors == None:
            self.preprocessors = []
            
    def load(self, image_path):
        image = cv2.imread(image_path)
        label = image_path.split("/")[-2]
        
        for preprocessors in self.preprocessors:
            image = preprocessors.preprocess(image)
            
        return (image, label)