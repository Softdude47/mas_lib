import cv2
import numpy as np


class CropPreprocessor:
    """ crops over sized and flips [if horizontal_flip=True] images """
    def __init__(self, width, height, horizontal_flip=True, interpolation=cv2.INTER_AREA):
        # stores the target width, height and the method used for resizing images
        self.width = width
        self.height = height
        self.interpolation = interpolation
        self.horizontal_flip = horizontal_flip
        
    def preprocess(self, image):
        # extracts image dimension and initializes a list
        # to store final results
        (h, w) = image.shape[:2]
        cropped_images = []
        
        # a list containing image cordinates(used in resizing)
        cords = [
            # [   top-left  ], [  bottom-right  ]
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h]
        ]
        
        # get the mid points of the cropped parts
        dH = int(0.5 * (h - self.height))
        dW = int(0.5 * (w - self.width))
        
        # form another cordinate from the mid point
        # of the cropped parts
        cords.append([dW, dH, w - dW, h - dH])
        
        # crop and resize the images
        # using the list of cordinates
        for (x1, y1, x2, y2) in cords:
            crop = image[y1 : y2, x1 : x2]
            crop = cv2.resize(crop, (self.wdith, self.height))
            cropped_images.append(crop)
        
        # make horizontal flip based on it value
        if self.horizontal_flip:
            flipped_images = [cv2.flip(c, 1) for c in cropped_images]
            cropped_images.extend(flipped_images)
        
        # array containing the cropped images
        return np.array(cropped_images)