import cv2
import imutils
from tensorflow.keras.preprocessing.image import img_to_array

class ImageToArray:
    """ converts image to keras array format """
    def __init__(self):
        pass
    
    def preprocess(self, image):
        return img_to_array(image)


class AspectAwarePreprocessor:
    """ converts image to desired dimension while maintaining the aspect ratio """
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        # store the target image width, height and method
        # to be used in resizing
        self.width = width
        self.height = height
        self.interpolation = interpolation
        
    def preprocess(self, image):
        # grab the image shape and initialize the
        # deltas to use when cropping
        (h, w) = image.shape[:2]
        dH = 0
        dW = 0
        
        # resize along the shortest dimension
        if w < h:
            image = imutils.resize(image=image, width=w, inter=self.interpolation)
            dH = int((image.shape[0] - self.height) / 2)
        else:
            image = imutils.resize(image=image, height=h, inter=self.interpolation)
            dW = int((image.shape[1] - self.width) / 2)
            
        # cropping the image
        (h, w) = image.shape[:2]
        image = image[dH : h - dH, dW : w - dW]
        
        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)