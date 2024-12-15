import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        # N is the number of column in the input matrix
        # alpha is the learning rate of our network
        # initialize the learning rate and weights
        self.alpha = alpha
        self.W = np.random.randn(N + 1) / np.sqrt(N)

    def step(self, x):
        # mimics the step function
        return 1 if x > 0 else 0
    
    def fit(self, x, y, epochs=100):
        # inserts an extra column to the trainiing data
        # this method allows us to easily treat the bias
        # as a trainable paramenter
        x = np.c_[x, np.ones((x.shape[0]))]
        for epoch in np.arange(0, epochs):
            # loop over the data and labels individually
            for (data, label) in zip(x, y):
                # compute prediction by applying the step
                # function on the result of dot multiplication
                # of the data and weight matrix
                predictions = self.step(np.dot(data, self.W))
                
                if predictions != label :
                    # calculates error in prediction and the
                    # error direction with respect to the weight
                    error = predictions - y
                    gradient = error * x

                    # corrects the weight to the direction direction
                    self.W += -self.alpha * gradient

    def predict(self, x, add_bias=True):
        # ensure the data is a matrix
        X = np.atleast_2d(x)

        if add_bias:
            # adds a column of ones so as to easily train
            # the weight(bias) matrix
            X = np.c_[X, np.ones((X.shape[0]))]

        # compute prediction by applying the step function to
        # the dot product of input data and weight matrix
        return self.step(X.dot(self.W))
