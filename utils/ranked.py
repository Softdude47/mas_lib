import numpy as np


def rank5_accuracy(y_true, y_pred):
    """calculates and returns rank-1 and rank-5 accuracy

    Args:
        y_true (array): ground truth label
        y_pred (array): predicted label

    Returns:
        tuple: [int, int]
    """
    rank5 = 0
    rank1 = 0
    
    # loop over the ground truth and predicted label
    for (ground_truth, prediction) in (y_true, y_pred):
        # sort the predictions in decending order
        prediction = np.argsort(prediction, axis=0)
        
        # checks if the ground truth appeard
        # in the top 5 predictions
        if ground_truth in prediction[:5]:
            rank5 += 1
        
        # checks if the top most prediction
        # is the ground truth
        if ground_truth == prediction[0]:
            rank1 += 1
    
    # compute final rank-1 and rank-5 accuracies
    rank1 /= len(y_true)
    rank5 /= len(y_true)
    
    # returns a tuple of rank one and rank five
    return (rank1, rank5)