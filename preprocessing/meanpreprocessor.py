import cv2


class MeanPreprocessor:
    """ Performs Mean Subtraction(data normalization technique) on an image """
    def __init__(self, rMean, gMean, bMean):
        # stores the Mean values of blue, green and red 
        # channels of the image training dataset
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean
        
    def preprocess(self, image):
        # split image into its three channels
        (B, G, R) = cv2.split(image.astype("float"))
        
        # perform mean subtraction on the image
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean
        
        # merge the new channels in a single image
        return cv2.merge([B, G, R])