import numpy as np


class LinearRegression:
    def __init__(self,learning_rate=0.01,iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.cost_history = []



    def predict(self,X):
        return X @self.weights + self.bias

    def cost_function(self,X,y):
        m = X.shape[0]
        return 1/(2 * m) * np.sum((self.predict(X) - y) **2)

    def gradient_descent(self,X,y):
        m = X.shape[0]
        for _ in  range(self.iterations):
            # Calculate predictions and update parameters
            error  = self.predict(X) - y
            dw = (X.T @ error) / m
            db = np.mean(error)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            cost = self.cost_function(X, y)
            self.cost_history.append(cost)



    def fit(self,X,y):

        _, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        #optimize the parameters
        self.gradient_descent(X, y)

        return self


    def score(self,X,y):
        predictions = self.predict(X)

        residual_sum = np.sum((y - predictions) ** 2)
        total_sum = np.sum((y - np.mean(y)) ** 2)

        return 1 - residual_sum / total_sum