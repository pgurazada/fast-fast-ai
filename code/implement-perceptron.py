import numpy as np
import pandas as pd


class Perceptron:

    '''Perceptron classifier

    '''

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])  # initialize weights to zeros

        for epoch in range(self.n_iter):
            errors = 0
            
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update * 1
                errors += int(update != 0.0)

            print('Number of misclassifications at epoch: ', epoch, ' = ', errors)            

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1 , -1)


if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    model_perceptron = Perceptron(eta=0.1, n_iter=10)
    model_perceptron.fit(X, y)
