
import numpy as np



class LogisticsRegression:
    def __init__(self,learning_rate=0.01, num_iterations=1000):
        self.bias = None
        self.weights = None
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.cost_history = []


    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))
    def predict_probability(self,X):
        z = X @self.weights + self.bias
        return self.sigmoid(z)

    def cost_function(self,X,y):
        m_features = X.shape[0]
        probabilities = self.predict_probability(X)
        #Prevent log(0)
        epsilon = 1e-15
        probabilities = np.clip(probabilities,
                                epsilon,
                                1 - epsilon
                                )
        cost = - (1/m_features) * np.sum(
            y * np.log(probabilities) +
            (1 - y) * np.log( 1- probabilities)
        )

        return cost


    def fit(self,x,y):
        m, num_features = x.shape
        self.weights = np.zeros(num_features)
        self.cost_history = []

        for _ in range(self.num_iterations):
            probabilities = self.predict_probability(X)
            error = probabilities - y
            dw = (x.T @ error) / m
            db = np.sum(error) / m
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            cost = self.cost_function(x, y)
            self.cost_history.append(cost)
            
            return self




