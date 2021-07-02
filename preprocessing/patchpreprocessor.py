from sklearn.feature_extraction.image import extract_patches_2d

class PatchPreprocessor:
    """ [data augmentation] extracts random cropped width x height patches """
    def __init__(self, width, height):
        # store the target width and height
        self.width = width
        self.height = height
        
    def preprocess(self, image):
        # returns a randomly cropped version of the image
        # which has the target width and height
        return extract_patches_2d(image=image, patch_size=(self.width, self.height), max_patches=1)[0]