import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.layers = layers
        self.alpha = alpha
        self.__W = []

        # looping over the layers and initializing
        # weights for each layers(excluding the last tow)
        for i in np.arange(0, len(layers)-2):
            # adding the bias to the weight matrix in each layers
            # while connecting the current layer to the next
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.__W.append(w / np.sqrt(layers[i]))

        # initializing the weight matrix of the second to last
        # layer since the inputs needs a bias, while the output
        # which connects to the output layer doesn't
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.__W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return f"Neural Network: {'.'.join(i for i in self.layers)}"
    
    def __compute_loss(self, X, y):
        # calculates loss by halving the sum of squares of
        # the difference between the actual class and
        # our predicted class
        p = self.predict(X, add_bias=False)
        return 0.5 * np.sum((p - y)** 2)
    
    def __fit(self, X, y):
        # heres where the actual train goes on
        # first we ensure that the input X is a 2D
        # matrix and then adds it to a list of Activations
        A = [np.atleast_2d(X)]

        # FEED FORWARD
        # loop over the layers and use it associating wieght
        # to comute predictions
        for layer in np.arange(0, len(self.__W)):
            # computer dot product of data with
            # weight matrix in the current layer
            net = A[layer].dot(self.__W[layer])

            # pass the resulting matrix into the
            # sigmoid function
            sigmoid_applied = self.__sigmoid(net)
            A.append(sigmoid_applied)

        # BACKPROPAGATION

        # compute the error of predicted value from
        # the final layer
        error = A[-1] - y

        # create a list of deltas(gradients) where the
        # the first entry is the delta of our final output
        # i.e the derivative of our cost function
        # 0.5 * ((prediction - actual) ** 2) with respect to
        # our prediction which is our Sigmoid function
        D =[error * self.__sigmoid_deriv(A[-1])]

        # loop over each [interconnected] layers, except the
        # last two [interconnected] layers since we already
        # computed their deltas
        for layer in np.arange(len(A) - 2, 0, -1):
            # we compute the delta of the current layer
            # by multiplying the delta of the previous layer
            # with the weight matrix of the current layer
            # qnd then append it to our list of deltas
            delta = D[-1].dot(self.__W[layer].T)
            delta *= self.__sigmoid_deriv(A[layer])
            D.append(delta)

        # reverse our list of delta from order of
        # [last layer to first layer] to
        # [first layer to last layer]
        D.reverse()

        # WEIGHT UPDATE
        for layer in np.arange(0, len(self.__W)):
            # update the weight by adding the negativae product of
            # learning rate and the dot product of the current Activation
            # layer and the computed weight gradient i.e delta of that layer
            self.__W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def __sigmoid(self, X):
        # compute the sigmoid function of the input
        return 1.0 / (1 + np.exp(-X))
    
    def __sigmoid_deriv(self, X):
        # calculates the derivative of X assuming it
        # has already being passed through the sigmoid
        # function i.e X = Sigmoid(x)
        return X * (1 - X)
    
    def fit(self, X, y, epochs=100, verbose=500):
        # add a column of ones so as to be able to
        # perforom the bias trick which treats the
        # bias as a trainable parameter withing the 
        # weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            # loop over each data point and the corrensponding
            # labels and train on them
            for (data, label) in zip(X, y):
                self.__fit(data, label)

            # display informmation on the losses and current epoch
            if epoch == 0 or (epoch + 1) % verbose == 0:
                loss = self.__compute_loss(X, y)
                print(f"[INFO]: Epoch={epoch+1} loss={loss}")
            
    def predict(self, X, add_bias=True):
        # initialized the prediction as the input since
        # it would be passed through each layers in our
        # neural network till after the last layer where
        # it would eventually become a prediction
        p = np.atleast_2d(X)

        if add_bias:
            # adds a column of ones
            p = np.c_[p, np.ones((p.shape[0]))]

        # loop over each layers in the network
        for layer in np.arange(0, len(self.__W)):
            # passing the inpput through each layers
            # since the output of the previous layer
            # is the input of the current layer and the
            # output of the last layer is the final prediction
            p = self.__sigmoid(p.dot(self.__W[layer]))

        # returns the prediction
        return p
