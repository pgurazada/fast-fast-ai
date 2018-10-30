import numpy as np
import pandas as pd

from numpy.random import seed
from sklearn.preprocessing import StandardScaler

class AdalineSGD:

    def __init__(self, eta=0.01, n_epochs=10,
                 shuffle=True, random_state=20130810):
        self.eta = eta
        self.n_epochs = n_epochs
        self.w_initialized = False
        self.shuffle = shuffle

        if random_state:
            seed(random_state)

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for epoch in range(self.n_epochs):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)

        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2

        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)
    
    def predict(self, X):
        return np.where(self.activation(X) >= 0, 1, -1)

if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    model_adaline_sgd = AdalineSGD(eta=0.01, n_epochs=10, random_state=20130810)
    model_adaline_sgd.fit(StandardScaler().fit_transform(X), y)

    print(model_adaline_sgd.cost_)
