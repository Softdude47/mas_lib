from tensorflow.keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    """ converts image to keras readable format """
    def __init__(self):
        pass
    
    def preprocess(self, image):
        return img_to_array(image)