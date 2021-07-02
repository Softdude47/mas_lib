import cv2

class SimplePreprocessor:
    """ resizes the image to the specified width and hegiht. use only on square images """
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        # stores the target width and height
        # along with the interpolation method
        self.width = width
        self.height = height
        self.interpolation = interpolation
        
    def preprocess(self, image):
        # returns the image resized to it target width and height
        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)